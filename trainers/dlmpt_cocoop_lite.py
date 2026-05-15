"""
DL-MPT CoCoOp Lite — 继承 CoCoOp_ATP trainer，仅覆盖数据加载和 episodic loop。
最小的代码改动 = 最小的 bug 可能性。
"""
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter
from dassl.data import DataManager
from dassl.data.episodic_sampler import EpisodicSampler
from dassl.data.data_manager import DatasetWrapper, build_transform

from trainers.cocoop_atp import CoCoOp_ATP


@TRAINER_REGISTRY.register()
class DLMPTCoCoOpLite(CoCoOp_ATP):
    """DL-MPT for CoCoOp, with on-demand image loading."""

    def __init__(self, cfg):
        # Read DL-MPT config BEFORE super().__init__
        self.lambda_ = cfg.TRAINER.DLMPT.LAMBDA
        self.n_way = cfg.TRAINER.DLMPT.N_WAY
        self.k_support = cfg.TRAINER.DLMPT.K_SUPPORT
        self.k_query = cfg.TRAINER.DLMPT.K_QUERY
        self.n_episodes = cfg.TRAINER.DLMPT.N_EPISODES
        self.warmup_epochs = cfg.TRAINER.DLMPT.WARMUP_EPOCHS
        self.refine_epochs = cfg.TRAINER.DLMPT.REFINE_EPOCHS
        self.refine_lambda = cfg.TRAINER.DLMPT.REFINE_LAMBDA
        super().__init__(cfg)

    @property
    def current_lambda(self):
        if self.epoch < self.warmup_epochs: return 0.0
        if self.epoch >= self.refine_epochs: return self.refine_lambda
        return self.lambda_

    def build_data_loader(self):
        super().build_data_loader()
        dm = DataManager(self.cfg)
        data_source = dm.dataset.train_x
        tfm_train = build_transform(self.cfg, is_train=True)

        self.episodic_sampler = EpisodicSampler(
            data_source, n_way=self.n_way, k_support=self.k_support,
            k_query=self.k_query, n_episodes=self.n_episodes)

        print(f"Preloading {len(data_source)} training images to GPU...")
        loader = DataLoader(
            DatasetWrapper(self.cfg, data_source, transform=tfm_train, is_train=True),
            batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        all_imgs, all_labels = [], []
        for batch in tqdm(loader, desc="Caching images"):
            all_imgs.append(batch["img"])
            all_labels.append(batch["label"])
        self.cached_images = torch.cat(all_imgs).to(self.device)
        self.cached_labels = torch.cat(all_labels).to(self.device)
        mem_mb = self.cached_images.element_size() * self.cached_images.numel() // 1024 ** 2
        print(f"Cached {len(self.cached_images)} images ({mem_mb}MB)")

    def run_epoch(self):
        """Override: dual-loop with CoCoOp base path + proto meta path."""
        self.set_model_mode("train")
        self.num_batches = min(len(self.train_loader_x), self.n_episodes)
        batch_time = AverageMeter(); end = time.time()

        train_iter = iter(self.train_loader_x)
        episodic_iter = iter(self.episodic_sampler)

        for self.batch_idx in range(self.num_batches):
            try: base_batch = next(train_iter)
            except StopIteration: train_iter = iter(self.train_loader_x); base_batch = next(train_iter)
            try: support_idxs, query_idxs = next(episodic_iter)
            except StopIteration: episodic_iter = iter(self.episodic_sampler); support_idxs, query_idxs = next(episodic_iter)

            # Path 1: CoCoOp base (reuse parent's forward_backward)
            loss_summary = self.forward_backward(base_batch)
            loss_base = sum(loss_summary.values()) if isinstance(loss_summary, dict) else loss_summary
            acc_base = loss_summary.get("acc", 0) if isinstance(loss_summary, dict) else 0

            # Path 2: Episodic proto
            loss_meta, acc_meta = self._proto_meta_loss(support_idxs, query_idxs)

            lam = self.current_lambda
            if lam > 0:
                (lam * loss_meta).backward()
                self.model_backward_and_update(loss_base)

            if (self.batch_idx + 1) % 20 == 0 or self.batch_idx < 2:
                eta_sec = batch_time.avg * (self.num_batches - self.batch_idx - 1) if hasattr(batch_time, 'avg') else 0
                msg = (f"[DL-MPT Lite] epoch={self.epoch+1:02d}/{self.max_epoch:02d} "
                       f"batch={self.batch_idx+1:03d}/{self.num_batches:03d} "
                       f"L_base={loss_base.item() if hasattr(loss_base,'item') else loss_base:.3f} "
                       f"L_meta={loss_meta.item():.3f} "
                       f"acc_meta={acc_meta:.1f}% λ={lam:.2f} "
                       f"lr={self.get_current_lr():.2e}")
                print(msg)
                self._log_progress(msg)

            batch_time.update(time.time() - end); end = time.time()

        self.update_lr()

    def _proto_meta_loss(self, support_idxs, query_idxs):
        support_img = self.cached_images[support_idxs]
        support_label = self.cached_labels[support_idxs]
        query_img   = self.cached_images[query_idxs]
        query_label = self.cached_labels[query_idxs]

        unique = torch.unique(support_label)
        label_map = {orig.item(): new for new, orig in enumerate(unique)}
        s_lab = torch.tensor([label_map[l.item()] for l in support_label], device=self.device)
        q_lab = torch.tensor([label_map[l.item()] for l in query_label],   device=self.device)

        dtype = self.model.dtype; n_way = len(unique)

        vf = self.model.image_encoder(support_img.type(dtype))
        vis_protos = torch.stack([F.normalize(vf[s_lab == c].mean(0), dim=-1) for c in range(n_way)])

        im_features = self.model.image_encoder(support_img.type(dtype))
        prompts = self.model.prompt_learner(im_features)
        text_emb_list = []
        for i in range(len(support_img)):
            orig_label = support_label[i].item()
            pts_i = prompts[i]  # (n_cls, n_tokens, dim), no batch dim
            text_feat_i = self.model.text_encoder(pts_i, self.model.tokenized_prompts)
            text_emb_list.append(F.normalize(text_feat_i, dim=-1)[orig_label])
        text_features = torch.stack(text_emb_list)
        text_protos = torch.stack([F.normalize(text_features[s_lab == c].mean(0), dim=-1) for c in range(n_way)])

        prototypes = F.normalize((vis_protos + text_protos) / 2, dim=-1)
        qf = F.normalize(self.model.image_encoder(query_img.type(dtype)), dim=-1)
        sim = qf @ prototypes.T
        return F.cross_entropy(sim, q_lab), (sim.argmax(-1) == q_lab).float().mean().item() * 100
