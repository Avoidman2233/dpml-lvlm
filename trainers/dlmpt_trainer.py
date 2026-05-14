import time
import os.path as osp
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.episodic_sampler import EpisodicSampler
from dassl.data.data_manager import DatasetWrapper, build_transform

from trainers.coop_atp import load_clip_to_cpu


@TRAINER_REGISTRY.register()
class DLMPTTrainer(TrainerX):
    def __init__(self, cfg):
        self.lambda_ = cfg.TRAINER.DLMPT.LAMBDA
        self.n_way = cfg.TRAINER.DLMPT.N_WAY
        self.k_support = cfg.TRAINER.DLMPT.K_SUPPORT
        self.k_query = cfg.TRAINER.DLMPT.K_QUERY
        self.n_episodes = cfg.TRAINER.DLMPT.N_EPISODES
        self.warmup_epochs = cfg.TRAINER.DLMPT.WARMUP_EPOCHS
        self.refine_epochs = cfg.TRAINER.DLMPT.REFINE_EPOCHS
        self.refine_lambda = cfg.TRAINER.DLMPT.REFINE_LAMBDA
        self.use_cocoop = cfg.TRAINER.DLMPT.COCOOP
        super().__init__(cfg)

    @property
    def current_lambda(self):
        if self.epoch < self.warmup_epochs:
            return 0.0
        if self.epoch >= self.refine_epochs:
            return self.refine_lambda
        return self.lambda_

    @property
    def _model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def build_data_loader(self):
        dm = DataManager(self.cfg)
        data_source = dm.dataset.train_x
        tfm_train = build_transform(self.cfg, is_train=True)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm
        self.tfm_train = tfm_train

        self.episodic_sampler = EpisodicSampler(
            data_source,
            n_way=self.n_way,
            k_support=self.k_support,
            k_query=self.k_query,
            n_episodes=self.n_episodes,
        )

        print(f"Preloading {len(data_source)} training images to GPU...")
        loader = DataLoader(
            DatasetWrapper(self.cfg, data_source, transform=tfm_train, is_train=True),
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        all_imgs = []
        all_labels = []
        for batch in tqdm(loader, desc="Caching images"):
            all_imgs.append(batch["img"])
            all_labels.append(batch["label"])
        self.cached_images = torch.cat(all_imgs).to(self.device)
        self.cached_labels = torch.cat(all_labels).to(self.device)
        mem_mb = self.cached_images.element_size() * self.cached_images.numel() // 1024 ** 2
        print(f"Cached {len(self.cached_images)} images ({mem_mb}MB)")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        if self.use_cocoop:
            from trainers.cocoop_atp import load_clip_to_cpu, CustomCLIP
            prec_key = "COCOOP"
            cfg.defrost()
            cfg.TRAINER.COCOOP.N_CTX = cfg.TRAINER.DLMPT.COCOOP_N_CTX
            cfg.freeze()
        else:
            from trainers.coop_atp import load_clip_to_cpu, CustomCLIP
            prec_key = "COOP"

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if getattr(cfg.TRAINER, prec_key).PREC == "fp32" or getattr(cfg.TRAINER, prec_key).PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if getattr(cfg.TRAINER, prec_key).PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def run_epoch(self):
        self.set_model_mode("train")
        self.num_batches = min(len(self.train_loader_x), self.n_episodes)

        batch_time = AverageMeter()
        end = time.time()

        train_loader_iter = iter(self.train_loader_x)
        episodic_iter = iter(self.episodic_sampler)

        for self.batch_idx in range(self.num_batches):
            try:
                base_batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader_x)
                base_batch = next(train_loader_iter)

            try:
                support_idxs, query_idxs = next(episodic_iter)
            except StopIteration:
                episodic_iter = iter(self.episodic_sampler)
                support_idxs, query_idxs = next(episodic_iter)

            img_base, label_base = self.parse_batch_train(base_batch)

            if self.use_cocoop:
                def _cocoop_base():
                    imf_base = self._model.image_encoder(img_base.type(self._model.dtype))
                    imf_base = imf_base / imf_base.norm(dim=-1, keepdim=True)
                    prompts_base = self._model.prompt_learner(imf_base)
                    logits_base = []
                    for pts_i, imf_i in zip(prompts_base, imf_base):
                        tf_i = self._model.text_encoder(pts_i, self._model.tokenized_prompts)
                        tf_i = tf_i / tf_i.norm(dim=-1, keepdim=True)
                        logits_base.append(self._model.logit_scale.exp() * imf_i @ tf_i.t())
                    output_base = torch.stack(logits_base)
                    loss_base = F.cross_entropy(output_base, label_base)
                    return output_base, loss_base

                if self.scaler is not None:
                    with autocast():
                        output_base, loss_base = _cocoop_base()
                else:
                    output_base, loss_base = _cocoop_base()
            else:
                if self.scaler is not None:
                    with autocast():
                        output_base = self.model(img_base)
                        loss_base = F.cross_entropy(output_base, label_base)
                else:
                    output_base = self.model(img_base)
                    loss_base = F.cross_entropy(output_base, label_base)

            acc_base = compute_accuracy(output_base, label_base)[0].item()

            loss_meta, acc_meta = self._proto_meta_loss(support_idxs, query_idxs)

            lam = self.current_lambda
            loss = loss_base + lam * loss_meta

            if self.scaler is not None:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

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
                    f"[DL-MPT] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                    f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                    f"L_base={loss_base.item():.3f} L_meta={loss_meta.item():.3f} "
                    f"L_total={loss.item():.3f} acc_base={acc_base:.1f}% "
                    f"acc_meta={acc_meta:.1f}% λ={lam:.2f} "
                    f"lr={self.get_current_lr():.2e} eta={eta}"
                )
                print(msg)
                self._log_progress(msg)

        self.update_lr()

    def _proto_meta_loss(self, support_idxs, query_idxs):
        if self.use_cocoop:
            return self._proto_meta_loss_cocoop(support_idxs, query_idxs)
        return self._proto_meta_loss_coop(support_idxs, query_idxs)

    def _proto_meta_loss_coop(self, support_idxs, query_idxs):
        support_img = self.cached_images[support_idxs]
        support_label = self.cached_labels[support_idxs]
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        unique = torch.unique(support_label)
        label_map = {orig.item(): new for new, orig in enumerate(unique)}
        s_lab = torch.tensor(
            [label_map[l.item()] for l in support_label],
            device=self.device, dtype=torch.long,
        )
        q_lab = torch.tensor(
            [label_map[l.item()] for l in query_label],
            device=self.device, dtype=torch.long,
        )

        dtype = self._model.dtype

        vf = self._model.image_encoder(support_img.type(dtype))
        vf = F.normalize(vf, dim=-1)
        vis_protos = torch.stack(
            [F.normalize(vf[s_lab == c].mean(0), dim=-1) for c in range(self.n_way)]
        )

        prompts = self._model.prompt_learner()
        tokenized_prompts = self._model.tokenized_prompts
        text_features = self._model.text_encoder(prompts.type(dtype), tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)
        text_protos = text_features[unique]

        prototypes = F.normalize((vis_protos + text_protos) / 2, dim=-1)

        qf = self._model.image_encoder(query_img.type(dtype))
        qf = F.normalize(qf, dim=-1)
        sim = qf @ prototypes.T
        loss = F.cross_entropy(sim, q_lab)
        acc = (sim.argmax(-1) == q_lab).float().mean().item() * 100

        return loss, acc

    def _proto_meta_loss_cocoop(self, support_idxs, query_idxs):
        support_img = self.cached_images[support_idxs]
        support_label = self.cached_labels[support_idxs]
        query_img = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        unique = torch.unique(support_label)
        label_map = {orig.item(): new for new, orig in enumerate(unique)}
        s_lab = torch.tensor(
            [label_map[l.item()] for l in support_label],
            device=self.device, dtype=torch.long,
        )
        q_lab = torch.tensor(
            [label_map[l.item()] for l in query_label],
            device=self.device, dtype=torch.long,
        )

        dtype = self._model.dtype
        n_way = len(unique)

        vf = self._model.image_encoder(support_img.type(dtype))
        vf = F.normalize(vf, dim=-1)
        vis_protos = torch.stack(
            [F.normalize(vf[s_lab == c].mean(0), dim=-1) for c in range(n_way)]
        )

        prompts = self._model.prompt_learner(vf)
        text_emb_list = []
        for i in range(len(support_img)):
            pts_i = prompts[i]
            text_feat_i = self._model.text_encoder(pts_i, self._model.tokenized_prompts)
            text_feat_i = F.normalize(text_feat_i, dim=-1)
            class_idx = support_label[i].item()
            text_emb_list.append(text_feat_i[class_idx])

        text_features = torch.stack(text_emb_list)
        text_protos = torch.stack(
            [F.normalize(text_features[s_lab == c].mean(0), dim=-1) for c in range(n_way)]
        )

        prototypes = F.normalize((vis_protos + text_protos) / 2, dim=-1)

        qf = F.normalize(self._model.image_encoder(query_img.type(dtype)), dim=-1)
        sim = qf @ prototypes.T
        loss = F.cross_entropy(sim, q_lab)
        acc = (sim.argmax(-1) == q_lab).float().mean().item() * 100

        return loss, acc
