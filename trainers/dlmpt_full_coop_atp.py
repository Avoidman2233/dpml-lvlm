import time
import datetime
import random
from itertools import product

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import AverageMeter

from trainers.base_coop_atp import BaseCoOpATP
from trainers.episodic_utils import (
    attribute_similarity_matrix,
    sample_hard_episode,
)


@TRAINER_REGISTRY.register()
class DLMPT_Full_CoOp_ATP(BaseCoOpATP):
    """DL-MPT-Full(CoOp+ATP) combining all meta-learning components.

    Integrates:
        - FOMAML first-order inner-loop adaptation
        - Task-adaptive prompt delta via TaskAdapter MLP
        - Attribute-aware hard episode sampling
        - Attribute alignment loss for compositional reasoning
    """

    def __init__(self, cfg):
        # FOMAML params
        self.inner_lr = cfg.TRAINER.FOMAML.INNER_LR
        self.inner_steps = cfg.TRAINER.FOMAML.INNER_STEPS
        # Adaptive params
        self.hidden_dim = cfg.TRAINER.ADAPTIVE_DELTA.HIDDEN_DIM
        self.delta_weight = cfg.TRAINER.ADAPTIVE_DELTA.DELTA_WEIGHT
        # AttrSample params
        self.tau = cfg.TRAINER.ATTR_SAMPLE.TAU
        self.hard_ratio = cfg.TRAINER.ATTR_SAMPLE.HARD_RATIO
        # Align params
        self.lambda_attr = cfg.TRAINER.ATTR_ALIGN.LAMBDA_ATTR
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def build_model(self):
        """Build base model, task adapter, and attribute similarity structures."""
        super().build_model()

        if self.use_cocoop:
            raise NotImplementedError(
                "DLMPT_Full_CoOp_ATP does not support CoCoOp yet"
            )

        # --- Task Adapter (from DLMPT_Adaptive) ---
        prompt_learner = self._model.prompt_learner
        self.n_ctx = prompt_learner.n_ctx
        self.ctx_dim = prompt_learner.ctx.shape[-1]
        vis_dim = self._model.image_encoder.output_dim

        print(
            f"[Full] Building task adapter: "
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

        # Near-zero init so delta starts small
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

        # --- Attribute similarity matrix (from AttrSample + Align) ---
        classnames = self.dm.dataset.classnames
        n_cls = len(classnames)
        print(f"[Full] Building attribute similarity matrix for {n_cls} classes...")

        with torch.no_grad():
            prompts = self._model.prompt_learner()
            tokenized_prompts = self._model.tokenized_prompts
            attr_embeddings = self._model.text_encoder(prompts, tokenized_prompts)

        self.attr_embeddings = attr_embeddings.detach()
        sim_matrix = attribute_similarity_matrix(
            classnames, self.attr_embeddings
        )
        self.attr_sim_matrix = sim_matrix.float().detach()

        self.align_pairs = [
            (i, j)
            for i, j in product(range(n_cls), range(n_cls))
            if i < j and self.attr_sim_matrix[i, j].item() > self.tau
        ]
        print(
            f"[Full] Found {len(self.align_pairs)} attribute-similar pairs "
            f"(tau={self.tau})."
        )

        self.class_indices = {}
        for idx, label in enumerate(self.cached_labels.cpu().numpy()):
            label = int(label)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        print("[Full] Model build complete.")

    # ------------------------------------------------------------------
    # Meta-parameter helpers
    # ------------------------------------------------------------------
    def _get_meta_params(self):
        """All parameters participating in the meta-learning inner loop."""
        return list(self._model.prompt_learner.parameters()) + list(
            self.task_adapter.parameters()
        )

    def _set_meta_params(self, params):
        for p, new_val in zip(self._get_meta_params(), params):
            p.data.copy_(new_val.data)

    def _compute_loss(self, images, labels):
        output = self.model(images)
        loss = F.cross_entropy(output, labels)
        acc = compute_accuracy(output, labels)[0].item()
        return loss, acc

    def _compute_attr_loss(self):
        """Attribute alignment loss from DLMPT_Align."""
        if len(self.align_pairs) == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        all_prompts = self._model.prompt_learner()
        tokenized_prompts = self._model.tokenized_prompts
        all_text_feats = self._model.text_encoder(all_prompts, tokenized_prompts)

        loss_attr = torch.tensor(0.0, device=all_text_feats.device, dtype=torch.float32)
        for i, j in self.align_pairs:
            diff = all_text_feats[i].float() - all_text_feats[j].float()
            loss_attr = loss_attr + torch.norm(diff, p=2)

        loss_attr = loss_attr / len(self.align_pairs)
        loss_attr = loss_attr.to(dtype=self._model.dtype)
        return loss_attr

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def run_epoch(self):
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

                lam = self.current_lambda
                msg = (
                    f"[Full] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_summary['loss_base']:.3f} "
                    f"L_meta={loss_summary['loss_meta']:.3f} "
                    f"L_attr={loss_summary['loss_attr']:.3f} "
                    f"\u03bb={lam:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)
                self._log_progress(msg)

        self.update_lr()

    # ------------------------------------------------------------------
    # Forward / backward step
    # ------------------------------------------------------------------
    def forward_backward(self, batch):
        img_base, label_base = self.parse_batch_train(batch)

        # 1. Base classification loss
        if self.scaler is not None:
            with autocast():
                loss_base, acc_base = self._compute_loss(img_base, label_base)
        else:
            loss_base, acc_base = self._compute_loss(img_base, label_base)

        # 2. Attribute-aware episode sampling
        if random.random() < self.hard_ratio:
            support_idxs, query_idxs = sample_hard_episode(
                self.dm.dataset.classnames,
                self.attr_embeddings,
                self.n_way,
                self.k_support,
                self.k_query,
                self.class_indices,
                tau=self.tau,
                hard_ratio=1.0,
            )
        else:
            support_idxs, query_idxs = self._construct_episode()

        support_img = self.cached_images[support_idxs]
        support_label = self.cached_labels[support_idxs]
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        dtype = self._model.dtype

        # 3. Extract support visual features (frozen encoder)
        with torch.no_grad():
            support_feat = self._model.image_encoder(support_img.type(dtype))
        support_feat = support_feat / support_feat.norm(dim=-1, keepdim=True)

        # 4. FOMAML inner-loop with task-adaptive delta
        meta_params = self._get_meta_params()
        original_params = [p.clone().detach() for p in meta_params]
        fast_weights = [p.clone() for p in meta_params]
        prompt_learner = self._model.prompt_learner

        for _ in range(self.inner_steps):
            self._set_meta_params(fast_weights)

            # Task-adaptive delta from current adapter
            delta = self.task_adapter(support_feat.mean(dim=0))
            delta = delta.reshape(self.n_ctx, self.ctx_dim)

            # Temporarily inject delta into prompt
            saved_ctx_data = prompt_learner.ctx.data.clone()
            if prompt_learner.ctx.dim() == 3:
                delta = delta.unsqueeze(0).expand(prompt_learner.ctx.shape[0], -1, -1)
            delta = delta.to(dtype=prompt_learner.ctx.dtype)
            prompt_learner.ctx.data.add_(self.delta_weight * delta)

            # Support loss
            loss_support, _ = self._compute_loss(support_img, support_label)

            # Restore prompt before grad computation
            prompt_learner.ctx.data.copy_(saved_ctx_data)

            # First-order gradients w.r.t. meta_params
            grads = torch.autograd.grad(
                loss_support, meta_params, create_graph=False, allow_unused=True
            )
            fast_weights = [
                fw - self.inner_lr * (g if g is not None else torch.zeros_like(fw))
                for fw, g in zip(fast_weights, grads)
            ]

        # 5. Outer-loop meta loss with final fast_weights + delta
        self._set_meta_params(fast_weights)

        delta = self.task_adapter(support_feat.mean(dim=0))
        delta = delta.reshape(self.n_ctx, self.ctx_dim)

        saved_ctx_data = prompt_learner.ctx.data.clone()
        if prompt_learner.ctx.dim() == 3:
            delta = delta.unsqueeze(0).expand(prompt_learner.ctx.shape[0], -1, -1)
        delta = delta.to(dtype=prompt_learner.ctx.dtype)
        prompt_learner.ctx.data.add_(self.delta_weight * delta)

        loss_meta, acc_meta = self._compute_loss(query_img, query_label)

        prompt_learner.ctx.data.copy_(saved_ctx_data)

        meta_grads = torch.autograd.grad(
            loss_meta, meta_params, create_graph=False, allow_unused=True
        )

        # Restore original meta-parameters
        self._set_meta_params(original_params)

        # 6. Attribute alignment loss (outer loop only)
        if self.scaler is not None:
            with autocast():
                loss_attr = self._compute_attr_loss()
        else:
            loss_attr = self._compute_attr_loss()

        # 7. Total loss and backward
        lam = self.current_lambda
        loss_total = (
            loss_base
            + lam * loss_meta.detach()
            + self.lambda_attr * loss_attr
        )

        if self.scaler is not None:
            self.optim.zero_grad()
            self.optim_adapter.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optim)
            self.scaler.unscale_(self.optim_adapter)

            # Inject first-order meta gradients
            for p, mg in zip(meta_params, meta_grads):
                if mg is None:
                    continue
                mg = mg.to(dtype=p.data.dtype)
                if p.grad is not None:
                    p.grad.add_(lam * mg)
                else:
                    p.grad = (lam * mg).clone()

            self.scaler.step(self.optim)
            self.scaler.step(self.optim_adapter)
            self.scaler.update()
        else:
            self.optim.zero_grad()
            self.optim_adapter.zero_grad()
            loss_total.backward()

            # Inject first-order meta gradients
            for p, mg in zip(meta_params, meta_grads):
                if mg is None:
                    continue
                mg = mg.to(dtype=p.data.dtype)
                if p.grad is not None:
                    p.grad.add_(lam * mg)
                else:
                    p.grad = (lam * mg).clone()

            self.optim.step()
            self.optim_adapter.step()

        return {
            "loss_base": loss_base.item(),
            "loss_meta": loss_meta.item(),
            "loss_attr": loss_attr.item(),
            "loss_total": loss_total.item(),
            "acc_base": acc_base,
            "acc_meta": acc_meta,
        }
