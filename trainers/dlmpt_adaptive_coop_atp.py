import time
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import AverageMeter

from trainers.base_coop_atp import BaseCoOpATP


@TRAINER_REGISTRY.register()
class DLMPT_Adaptive_CoOp_ATP(BaseCoOpATP):
    """DL-MPT-Adaptive(CoOp+ATP) with task-adaptive prompt delta.

    TaskAdapter MLP generates a prompt shift from support-set visual features:
        delta = MLP(mean(F_support))
    The shift is temporarily added to the learnable context tokens during the
    meta-forward and restored afterwards, so the computation graph flows
    through the adapter while the base prompt parameters remain intact.
    """

    def __init__(self, cfg):
        self.hidden_dim = cfg.TRAINER.ADAPTIVE_DELTA.HIDDEN_DIM
        self.delta_weight = cfg.TRAINER.ADAPTIVE_DELTA.DELTA_WEIGHT
        super().__init__(cfg)

    def build_model(self):
        """Build CoOp/CoCoOp base model then add task-adaptive adapter."""
        super().build_model()

        if self.use_cocoop:
            raise NotImplementedError(
                "DLMPT_Adaptive_CoOp_ATP does not support CoCoOp yet"
            )

        prompt_learner = self._model.prompt_learner
        self.n_ctx = prompt_learner.n_ctx
        self.ctx_dim = prompt_learner.ctx.shape[-1]
        vis_dim = self._model.image_encoder.output_dim

        print(
            f"Building task adapter: "
            f"vis_dim={vis_dim}, hidden_dim={self.hidden_dim}, "
            f"n_ctx={self.n_ctx}, ctx_dim={self.ctx_dim}"
        )

        self.task_adapter = nn.Sequential(
            nn.Linear(vis_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_ctx * self.ctx_dim),
        )
        self.task_adapter.to(self.device)
        clip_dtype = self._model.text_encoder.transformer.parameters().__next__().dtype
        self.task_adapter = self.task_adapter.to(dtype=clip_dtype)

        # Near-zero init so delta starts small and does not destroy CLIP manifold
        for m in self.task_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.optim_adapter = build_optimizer(self.task_adapter, self.cfg.OPTIM)
        self.sched_adapter = build_lr_scheduler(self.optim_adapter, self.cfg.OPTIM)
        self.register_model(
            "task_adapter", self.task_adapter, self.optim_adapter, self.sched_adapter
        )

    def run_epoch(self):
        """Training loop: base batch + episodic meta regularisation."""
        self.set_model_mode("train")
        self.num_batches = min(len(self.train_loader_x), self.n_episodes)

        batch_time = AverageMeter()
        end = time.time()

        train_loader_iter = iter(self.train_loader_x)

        for self.batch_idx in range(self.num_batches):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader_x)
                batch = next(train_loader_iter)

            loss_summary = self.forward_backward(batch)

            batch_time.update(time.time() - end)
            end = time.time()

            meet_freq = (self.batch_idx + 1) % 20 == 0
            only_few = self.num_batches < 20
            if meet_freq or only_few or (self.batch_idx + 1) == self.num_batches:
                nb_remain = self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                msg = (
                    f"[DL-MPT-Adaptive] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_summary['loss_base']:.3f} "
                    f"L_meta={loss_summary['loss_meta']:.3f} "
                    f"L_total={loss_summary['loss_total']:.3f} "
                    f"acc_base={loss_summary['acc_base']:.1f}% "
                    f"acc_meta={loss_summary['acc_meta']:.1f}% "
                    f"λ={loss_summary['lambda']:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)
                self._log_progress(msg)

        self.update_lr()

    def forward_backward(self, batch):
        """Single training step with episodic task-adaptive regularisation.

        Steps
        -----
        1. Base CE loss on the standard training batch.
        2. Sample an N-way K-shot episode.
        3. Extract support visual features (frozen encoder).
        4. Adapter -> delta, reshape to (n_ctx, ctx_dim).
        5. Temporarily replace prompt_learner.ctx with ctx+delta.
        6. Meta CE loss on query set using the shifted prompt.
        7. Restore original ctx.
        8. Total loss = L_base + lambda * L_meta, backward, step.
        """
        img_base, label_base = self.parse_batch_train(batch)

        support_idxs, query_idxs = self._construct_episode()
        support_img = self.cached_images[support_idxs]
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        dtype = self._model.dtype

        with torch.no_grad():
            support_feat = self._model.image_encoder(support_img.type(dtype))
        support_feat = support_feat / support_feat.norm(dim=-1, keepdim=True)

        delta = self.task_adapter(support_feat.mean(dim=0))
        delta = delta.reshape(self.n_ctx, self.ctx_dim)

        prompt_learner = self._model.prompt_learner
        original_ctx_data = prompt_learner.ctx.data.clone()
        if prompt_learner.ctx.dim() == 3:
            delta = delta.unsqueeze(0).expand(prompt_learner.ctx.shape[0], -1, -1)
        # Ensure delta matches ctx dtype to avoid fp16/fp32 mismatch
        delta = delta.to(dtype=prompt_learner.ctx.dtype)
        prompt_learner.ctx.data.add_(self.delta_weight * delta)
        if self.scaler is not None:
            with autocast():
                output_base = self.model(img_base)
                loss_base = F.cross_entropy(output_base, label_base)

                image_features = self._model.image_encoder(query_img.type(dtype))
                prompts = self._model.prompt_learner()
                tokenized_prompts = self._model.tokenized_prompts
                text_features = self._model.text_encoder(prompts, tokenized_prompts)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = self._model.logit_scale.exp()
                output_meta = logit_scale * image_features @ text_features.t()
                loss_meta = F.cross_entropy(output_meta, query_label)
        else:
            output_base = self.model(img_base)
            loss_base = F.cross_entropy(output_base, label_base)

            image_features = self._model.image_encoder(query_img.type(dtype))
            prompts = self._model.prompt_learner()
            tokenized_prompts = self._model.tokenized_prompts
            text_features = self._model.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self._model.logit_scale.exp()
            output_meta = logit_scale * image_features @ text_features.t()
            loss_meta = F.cross_entropy(output_meta, query_label)

        acc_base = compute_accuracy(output_base, label_base)[0].item()
        acc_meta = compute_accuracy(output_meta, query_label)[0].item()

        prompt_learner.ctx.data.copy_(original_ctx_data)
        lam = self.current_lambda
        loss = loss_base + lam * loss_meta

        if self.scaler is not None:
            self.optim.zero_grad()
            self.optim_adapter.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.step(self.optim_adapter)
            self.scaler.update()
        else:
            self.optim.zero_grad()
            self.optim_adapter.zero_grad()
            loss.backward()
            self.optim.step()
            self.optim_adapter.step()

        return {
            "loss_base": loss_base.item(),
            "loss_meta": loss_meta.item(),
            "loss_total": loss.item(),
            "acc_base": acc_base,
            "acc_meta": acc_meta,
            "lambda": lam,
        }
