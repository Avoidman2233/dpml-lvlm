import time
import datetime
from itertools import product

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter

from trainers.base_coop_atp import BaseCoOpATP
from trainers.episodic_utils import attribute_similarity_matrix


@TRAINER_REGISTRY.register()
class DLMPT_Align_CoOp_ATP(BaseCoOpATP):
    """DL-MPT-Align(CoOp+ATP) with attribute alignment loss.

    Constrains prompt embeddings of attribute-similar classes to be close,
    encouraging the model to learn attribute-compositional representations.
    """

    def __init__(self, cfg):
        self.lambda_attr = cfg.TRAINER.ATTR_ALIGN.LAMBDA_ATTR
        self.tau = cfg.TRAINER.ATTR_SAMPLE.TAU
        self.attr_align_freq = getattr(cfg.TRAINER.ATTR_ALIGN, "FREQ", 10)
        super().__init__(cfg)

    def build_model(self):
        super().build_model()

        classnames = self.dm.dataset.classnames
        n_cls = len(classnames)
        print(f"[Align] Building attribute similarity matrix for {n_cls} classes...")

        with torch.no_grad():
            if self.use_cocoop:
                vis_dim = self._model.prompt_learner.meta_net[0].in_features
                dummy = torch.zeros(
                    1, vis_dim, device=self.device, dtype=self._model.dtype
                )
                prompts = self._model.prompt_learner(dummy).squeeze(0)
            else:
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
        print(f"[Align] Found {len(self.align_pairs)} attribute-similar pairs (tau={self.tau}).")

    def run_epoch(self):
        self.set_model_mode("train")
        self.num_batches = len(self.train_loader_x)

        batch_time = AverageMeter()
        end = time.time()

        train_loader_iter = iter(self.train_loader_x)

        for self.batch_idx in range(self.num_batches):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader_x)
                batch = next(train_loader_iter)

            loss_base, loss_meta, loss_attr, loss_total, acc_base = self.forward_backward(batch)

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
                    f"[Align] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_base:.3f} L_meta={loss_meta:.3f} "
                    f"L_attr={loss_attr:.3f} "
                    f"\u03bb={lam:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)

        self.update_lr()

    def forward_backward(self, batch):
        img_base, label_base = self.parse_batch_train(batch)

        if self.scaler is not None:
            with autocast():
                output_base = self.model(img_base)
                loss_base = F.cross_entropy(output_base, label_base)
        else:
            output_base = self.model(img_base)
            loss_base = F.cross_entropy(output_base, label_base)

        acc_base = compute_accuracy(output_base, label_base)[0].item()

        support_idxs, query_idxs = self._construct_episode()
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        if self.scaler is not None:
            with autocast():
                output_meta = self.model(query_img)
                loss_meta = F.cross_entropy(output_meta, query_label)
        else:
            output_meta = self.model(query_img)
            loss_meta = F.cross_entropy(output_meta, query_label)

        compute_attr = (self.batch_idx % self.attr_align_freq == 0) and len(self.align_pairs) > 0

        if compute_attr:
            if self.scaler is not None:
                with autocast():
                    loss_attr = self._compute_attr_loss()
            else:
                loss_attr = self._compute_attr_loss()
        else:
            loss_attr = torch.tensor(0.0, device=self.device, dtype=loss_base.dtype)

        lam = self.current_lambda
        loss_total = loss_base + lam * loss_meta + self.lambda_attr * loss_attr

        if self.scaler is not None:
            self.optim.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.zero_grad()
            loss_total.backward()
            self.optim.step()

        return (
            loss_base.item(),
            loss_meta.item(),
            loss_attr.item(),
            loss_total.item(),
            acc_base,
        )

    def _compute_attr_loss(self):
        if len(self.align_pairs) == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if self.use_cocoop:
            vis_dim = self._model.prompt_learner.meta_net[0].in_features
            dummy = torch.zeros(
                1, vis_dim, device=self.device, dtype=self._model.dtype
            )
            all_prompts = self._model.prompt_learner(dummy).squeeze(0)
        else:
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
