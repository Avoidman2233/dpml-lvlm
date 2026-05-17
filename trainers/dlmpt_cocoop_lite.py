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


def _cocoop_forward(model, images, labels):
    dtype = model.dtype
    imf = model.image_encoder(images.type(dtype))
    imf = F.normalize(imf, dim=-1)
    prompts = model.prompt_learner(imf)
    logits = []
    for pts_i, imf_i in zip(prompts, imf):
        tf_i = model.text_encoder(pts_i, model.tokenized_prompts)
        tf_i = F.normalize(tf_i, dim=-1)
        logits.append(model.logit_scale.exp() * imf_i @ tf_i.t())
    output = torch.stack(logits)
    return output, F.cross_entropy(output, labels)


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

            # Path 1: CoCoOp base classification
            img_base, label_base = self.parse_batch_train(base_batch)
            if self.cfg.TRAINER.COCOOP.PREC == "amp" and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output_base, loss_base = _cocoop_forward(self.model, img_base, label_base)
            else:
                output_base, loss_base = _cocoop_forward(self.model, img_base, label_base)
            acc_base = compute_accuracy(output_base, label_base)[0].item()

            # Path 2: Episodic proto
            loss_meta, acc_meta = self._proto_meta_loss(support_idxs, query_idxs)

            lam = self.current_lambda
            loss = loss_base + lam * loss_meta
            self.model_backward_and_update(loss)

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
        pl = self.model.prompt_learner
        tok = self.model.tokenized_prompts

        vf = self.model.image_encoder(support_img.type(dtype))
        vis_protos = torch.stack([F.normalize(vf[s_lab == c].mean(0), dim=-1) for c in range(n_way)])

        im_feat = self.model.image_encoder(support_img.type(dtype))
        text_embs = []
        for i in range(len(support_img)):
            cls = support_label[i].item()
            bias = pl.meta_net(im_feat[i:i+1]).unsqueeze(1)
            ctx_i = pl.ctx.unsqueeze(0) + bias

            if pl.atp_num == 1:
                prompt = torch.cat([
                    pl.token_prefix[cls:cls+1], pl.ctx_att1.unsqueeze(0),
                    pl.token_middle1[cls:cls+1], ctx_i, pl.token_suffix[cls:cls+1],
                ], dim=1)
            else:
                prompt = torch.cat([
                    pl.token_prefix[cls:cls+1], pl.ctx_att1.unsqueeze(0),
                    pl.token_middle1[cls:cls+1], pl.ctx_att2.unsqueeze(0),
                    pl.token_middle2[cls:cls+1], ctx_i, pl.token_suffix[cls:cls+1],
                ], dim=1)
            tf = self.model.text_encoder(prompt, tok[cls:cls+1])
            text_embs.append(F.normalize(tf, dim=-1)[0])

        text_features = torch.stack(text_embs)
        text_protos = torch.stack([F.normalize(text_features[s_lab == c].mean(0), dim=-1) for c in range(n_way)])

        prototypes = F.normalize((vis_protos + text_protos) / 2, dim=-1)
        qf = F.normalize(self.model.image_encoder(query_img.type(dtype)), dim=-1)
        sim = qf @ prototypes.T
        return F.cross_entropy(sim, q_lab), (sim.argmax(-1) == q_lab).float().mean().item() * 100
