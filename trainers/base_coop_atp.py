import time
import os.path as osp
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dassl.engine import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.episodic_sampler import EpisodicSampler
from dassl.data.data_manager import DatasetWrapper, build_transform

from trainers.coop_atp import (
    load_clip_to_cpu,
    TextEncoder,
    CUSTOM_TEMPLATES,
    PromptLearner,
    CustomCLIP,
)


class BaseCoOpATP(TrainerX):
    """Shared base class for all CoOp+ATP variant trainers.

    Subclasses must override:
        - run_epoch()
        - forward_backward()
    """

    def __init__(self, cfg):
        # Extract episodic / lambda-scheduling hyperparameters
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

    # ------------------------------------------------------------------
    # Lambda scheduling
    # ------------------------------------------------------------------
    @property
    def current_lambda(self):
        if self.epoch < self.warmup_epochs:
            return 0.0
        if self.epoch >= self.refine_epochs:
            return self.refine_lambda
        return self.lambda_

    # ------------------------------------------------------------------
    # Unwrap DataParallel
    # ------------------------------------------------------------------
    @property
    def _model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # ------------------------------------------------------------------
    # Config check
    # ------------------------------------------------------------------
    def check_cfg(self, cfg):
        if self.use_cocoop:
            assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]
        else:
            assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    # ------------------------------------------------------------------
    # Data loading + episodic sampler + image cache
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Model construction (CoOp or CoCoOp)
    # ------------------------------------------------------------------
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        if self.use_cocoop:
            from trainers.cocoop_atp import load_clip_to_cpu as _load_clip, CustomCLIP as _CustomCLIP
            prec_key = "COCOOP"
            cfg.defrost()
            cfg.TRAINER.COCOOP.N_CTX = cfg.TRAINER.DLMPT.COCOOP_N_CTX
            cfg.freeze()
        else:
            from trainers.coop_atp import load_clip_to_cpu as _load_clip, CustomCLIP as _CustomCLIP
            prec_key = "COOP"

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = _load_clip(cfg)

        if getattr(cfg.TRAINER, prec_key).PREC == "fp32" or getattr(cfg.TRAINER, prec_key).PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = _CustomCLIP(cfg, classnames, clip_model)

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

    # ------------------------------------------------------------------
    # Checkpoint loading (CoOp+ATP token-vector aware)
    # ------------------------------------------------------------------
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Delete fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            if "token_middle1" in state_dict:
                del state_dict["token_middle1"]
            if "token_middle2" in state_dict:
                del state_dict["token_middle2"]
            if "token_middle3" in state_dict:
                del state_dict["token_middle3"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Inference (delegates to model forward)
    # ------------------------------------------------------------------
    def model_inference(self, input):
        return self.model(input)

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    # ------------------------------------------------------------------
    # Episodic helper
    # ------------------------------------------------------------------
    def _construct_episode(self):
        """Sample one N-way K-shot episode from the episodic sampler."""
        if not hasattr(self, "_episodic_iter"):
            self._episodic_iter = iter(self.episodic_sampler)
        try:
            support_idxs, query_idxs = next(self._episodic_iter)
        except StopIteration:
            self._episodic_iter = iter(self.episodic_sampler)
            support_idxs, query_idxs = next(self._episodic_iter)
        return support_idxs, query_idxs

    # ------------------------------------------------------------------
    # Abstract methods — must be overridden by subclasses
    # ------------------------------------------------------------------
    def run_epoch(self):
        raise NotImplementedError(
            "Subclasses of BaseCoOpATP must implement run_epoch()"
        )

    def forward_backward(self, batch):
        raise NotImplementedError(
            "Subclasses of BaseCoOpATP must implement forward_backward()"
        )
