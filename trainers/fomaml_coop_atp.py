import time
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter

from trainers.base_coop_atp import BaseCoOpATP


@TRAINER_REGISTRY.register()
class FOMAML_CoOp_ATP(BaseCoOpATP):
    def __init__(self, cfg):
        self.inner_lr = cfg.TRAINER.FOMAML.INNER_LR
        self.inner_steps = cfg.TRAINER.FOMAML.INNER_STEPS
        super().__init__(cfg)

    def _get_prompt_params(self):
        return list(self._model.prompt_learner.parameters())

    def _set_prompt_params(self, params):
        for p, new_val in zip(self._get_prompt_params(), params):
            p.data.copy_(new_val.data)

    def _compute_loss(self, images, labels):
        output = self.model(images)
        loss = F.cross_entropy(output, labels)
        acc = compute_accuracy(output, labels)[0].item()
        return loss, acc

    def forward_backward(self, batch):
        img_base, label_base = self.parse_batch_train(batch)

        if self.scaler is not None:
            with autocast():
                loss_base, acc_base = self._compute_loss(img_base, label_base)
        else:
            loss_base, acc_base = self._compute_loss(img_base, label_base)

        support_idxs, query_idxs = self._construct_episode()
        support_img = self.cached_images[support_idxs]
        support_label = self.cached_labels[support_idxs]
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        prompt_params = self._get_prompt_params()
        original_params = [p.clone().detach() for p in prompt_params]

        fast_weights = [p.clone() for p in prompt_params]
        for _ in range(self.inner_steps):
            self._set_prompt_params(fast_weights)

            loss_support, _ = self._compute_loss(support_img, support_label)
            grads = torch.autograd.grad(
                loss_support, prompt_params, create_graph=False, allow_unused=True
            )
            fast_weights = [
                fw - self.inner_lr * (g if g is not None else torch.zeros_like(fw))
                for fw, g in zip(fast_weights, grads)
            ]

        self._set_prompt_params(fast_weights)
        loss_meta, acc_meta = self._compute_loss(query_img, query_label)

        meta_grads = torch.autograd.grad(
            loss_meta, prompt_params, create_graph=False, allow_unused=True
        )
        meta_grads = [g if g is not None else torch.zeros_like(p) for p, g in zip(prompt_params, meta_grads)]

        self._set_prompt_params(original_params)

        lam = self.current_lambda
        loss_total = loss_base + lam * loss_meta.detach()

        if self.scaler is not None:
            self.optim.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optim)
            for p, mg in zip(prompt_params, meta_grads):
                if p.grad is not None:
                    p.grad.add_(lam * mg)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.zero_grad()
            loss_total.backward()
            for p, mg in zip(prompt_params, meta_grads):
                if p.grad is not None:
                    p.grad.add_(lam * mg)
                else:
                    p.grad = lam * mg.clone()
            self.optim.step()

        return {
            "loss_base": loss_base.item(),
            "loss_meta": loss_meta.item(),
            "loss_total": loss_total.item(),
            "acc_base": acc_base,
            "acc_meta": acc_meta,
        }

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
                    f"[FOMAML] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_summary['loss_base']:.3f} "
                    f"L_meta={loss_summary['loss_meta']:.3f} "
                    f"λ={lam:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)
                self._log_progress(msg)

        self.update_lr()
