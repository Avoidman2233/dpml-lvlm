import time
import datetime
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter

from trainers.base_coop_atp import BaseCoOpATP
from trainers.episodic_utils import (
    attribute_similarity_matrix,
    sample_hard_episode,
)


@TRAINER_REGISTRY.register()
class DLMPT_AttrSample_CoOp_ATP(BaseCoOpATP):
    """DL-MPT with Attribute-aware episodic sampling for CoOp+ATP.

    Samples hard-transfer episodes where classes share similar attributes
    (e.g., wolf/husky/fox share fur/tail/canine) to enhance compositional
    reasoning. Uses episodic regularization (no inner-loop).
    """

    def __init__(self, cfg):
        self.tau = cfg.TRAINER.ATTR_SAMPLE.TAU
        self.hard_ratio = cfg.TRAINER.ATTR_SAMPLE.HARD_RATIO
        super().__init__(cfg)

    def build_model(self):
        """Build base model and pre-compute attribute similarity matrix."""
        super().build_model()

        classnames = self.dm.dataset.classnames
        print(f"[AttrSample] Building attribute similarity matrix for {len(classnames)} classes...")

        with torch.no_grad():
            if self.use_cocoop:
                # CoCoOp prompt learner requires image features;
                # use a single dummy feature to get class-level prompts
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
        # Store as float32 to avoid fp16 precision issues in threshold comparisons
        self.attr_sim_matrix = sim_matrix.float().detach()

        # Build class -> indices mapping for hard episode sampling
        self.class_indices = {}
        for idx, label in enumerate(self.cached_labels.cpu().numpy()):
            label = int(label)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        print("[AttrSample] Attribute similarity matrix built.")

    def run_epoch(self):
        """Run one training epoch with attribute-aware episodic sampling."""
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

            loss_base, loss_meta, loss_total, acc_base = self.forward_backward(batch)

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
                    f"[AttrSample] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_base:.3f} L_meta={loss_meta:.3f} "
                    f"L_total={loss_total:.3f} acc_base={acc_base:.1f}% "
                    f"\u03bb={lam:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)

        self.update_lr()

    def forward_backward(self, batch):
        """Forward/backward for one batch with episodic regularization.

        Returns:
            loss_base (float), loss_meta (float), loss_total (float), acc_base (float)
        """
        img_base, label_base = self.parse_batch_train(batch)

        # --- Base classification loss ---
        if self.scaler is not None:
            with autocast():
                output_base = self.model(img_base)
                loss_base = F.cross_entropy(output_base, label_base)
        else:
            output_base = self.model(img_base)
            loss_base = F.cross_entropy(output_base, label_base)

        acc_base = compute_accuracy(output_base, label_base)[0].item()

        # --- Attribute-aware episode sampling ---
        if random.random() < self.hard_ratio:
            support_idxs, query_idxs = sample_hard_episode(
                self.dm.dataset.classnames,
                self.attr_embeddings,
                self.n_way,
                self.k_support,
                self.k_query,
                self.class_indices,
                tau=self.tau,
                hard_ratio=1.0,  # Force hard since we already decided
            )
        else:
            support_idxs, query_idxs = self._construct_episode()

        # --- Meta loss: episodic regularization on query set (no inner-loop) ---
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        if self.scaler is not None:
            with autocast():
                output_meta = self.model(query_img)
                loss_meta = F.cross_entropy(output_meta, query_label)
        else:
            output_meta = self.model(query_img)
            loss_meta = F.cross_entropy(output_meta, query_label)

        lam = self.current_lambda
        loss_total = loss_base + lam * loss_meta

        # --- Backward ---
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
            loss_total.item(),
            acc_base,
        )
