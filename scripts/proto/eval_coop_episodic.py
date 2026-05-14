#!/usr/bin/env python3
"""5-way episodic evaluation for ProtoATP-CoOp checkpoint."""
import argparse, random, sys, os
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Dassl.pytorch"))
os.chdir(PROJECT_ROOT)

import train
from dassl.config import get_cfg_default
from dassl.data import DataManager
from dassl.data.episodic_sampler import EpisodicSampler
from dassl.data.data_manager import build_transform, DatasetWrapper
from dassl.utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

import trainers.coop_atp


def make_cfg(data_dir, dataset):
    cfg = get_cfg_default()
    train.extend_cfg(cfg)
    cfg.merge_from_file(f"configs/trainers/CoOp/vit_b16.yaml")
    cfg.merge_from_file(f"configs/datasets/{dataset}.yaml")
    cfg.DATASET.ROOT = data_dir
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    cfg.TRAINER.ATPROMPT.USE_ATPROMPT = True
    cfg.SEED = 1
    train.choose_attribute_for_atprompt(cfg)
    return cfg


def episodic_eval_coop(model, cfg, n_way, k_support, k_query, n_episodes, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    dm = DataManager(cfg)
    data_source = dm.dataset.test
    tfm_test = build_transform(cfg, is_train=False)
    sampler = EpisodicSampler(
        data_source, n_way=n_way, k_support=k_support,
        k_query=k_query, n_episodes=n_episodes
    )
    loader = DataLoader(
        DatasetWrapper(cfg, data_source, transform=tfm_test, is_train=False),
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    all_imgs, all_labels = [], []
    for batch in tqdm(loader, desc="Caching test images"):
        all_imgs.append(batch["img"])
        all_labels.append(batch["label"])
    cached_imgs = torch.cat(all_imgs).to(device)
    cached_labels = torch.cat(all_labels).to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for support_idxs, query_idxs in tqdm(sampler, desc=f"Eval {n_way}w{k_support}s"):
            support_img = cached_imgs[support_idxs]
            support_label = cached_labels[support_idxs]
            query_img = cached_imgs[query_idxs]
            query_label = cached_labels[query_idxs]

            unique_labels = torch.unique(support_label)
            label_map = {orig.item(): new for new, orig in enumerate(unique_labels)}
            query_label_remapped = torch.tensor(
                [label_map[l.item()] for l in query_label],
                device=device, dtype=torch.long
            )
            dtype = model.dtype

            visual_features = model.image_encoder(support_img.type(dtype))
            visual_features = F.normalize(visual_features, dim=-1)
            visual_protos = []
            for c in unique_labels:
                mask = support_label == c
                visual_protos.append(F.normalize(
                    visual_features[mask].mean(0).unsqueeze(0), dim=-1
                ).squeeze(0))
            visual_protos = torch.stack(visual_protos)

            prompts = model.prompt_learner()
            text_features = model.text_encoder(prompts.type(dtype), model.tokenized_prompts)
            text_features = F.normalize(text_features, dim=-1)
            text_protos = text_features[unique_labels]

            prototypes = (visual_protos + text_protos) / 2.0
            prototypes = F.normalize(prototypes, dim=-1)

            query_features = model.image_encoder(query_img.type(dtype))
            query_features = F.normalize(query_features, dim=-1)
            sim = query_features @ prototypes.T * 10.0
            pred = sim.argmax(dim=-1)
            correct += (pred == query_label_remapped).sum().item()
            total += len(query_label)
    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="stanford_cars")
    parser.add_argument("--data-dir", type=str, default="/home/avoidman2233/Desktop/LVLM/DATA")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-support", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--k-query", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = make_cfg(args.data_dir, args.dataset)
    set_random_seed(args.seed)
    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    print(f"Dataset: {args.dataset}, {len(classnames)} new classes")

    clip_model = trainers.coop_atp.load_clip_to_cpu(cfg)
    clip_model.float()
    model = trainers.coop_atp.CustomCLIP(cfg, classnames, clip_model)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.prompt_learner.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Checkpoint loaded: {args.checkpoint}")

    for k in args.k_support:
        acc = episodic_eval_coop(
            model, cfg, args.n_way, k, args.k_query,
            args.n_episodes, args.seed, device
        )
        print(f"  {args.n_way}-way  K={k}: {acc:.2f}%")
