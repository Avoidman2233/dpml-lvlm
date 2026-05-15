#!/usr/bin/env python
"""DL-MPT Episodic Evaluation (Protocol B)."""
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, random, os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import defaultdict
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper, build_transform
from torch.utils.data import DataLoader
from clip import clip
import train as _t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='stanford_cars')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--output', default='/tmp/dlmpt_episodic_results.txt')
    args = parser.parse_args()

    # Load token embedding
    ckpt_path = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
    jit_model = torch.jit.load(ckpt_path, map_location='cpu').eval()
    sd = jit_model.state_dict()
    token_embed = torch.nn.Embedding(sd['token_embedding.weight'].shape[0], sd['token_embedding.weight'].shape[1]).cuda()
    token_embed.weight.data.copy_(sd['token_embedding.weight'])
    del jit_model, sd

    # Build config
    cfg = get_cfg_default(); _t.extend_cfg(cfg)
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    cfg.DATASET.ROOT = os.path.join(_proj_root, '..', 'DATA')
    cfg.DATASET.NUM_SHOTS = 16; cfg.DATASET.SUBSAMPLE_CLASSES = 'new'
    cfg.SEED = args.seed; cfg.TRAINER.NAME = 'CoOp_ATP'
    cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'; cfg.INPUT.SIZE = (224, 224)
    cfg.TRAINER.COOP.N_CTX = 2; cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
    cfg.TRAINER.ATPROMPT.USE_ATPROMPT = True; cfg.DATALOADER.NUM_WORKERS = 4
    cfg.merge_from_file(f'configs/datasets/{args.dataset}.yaml')
    cfg.merge_from_file('configs/trainers/CoOp/vit_b16.yaml')
    _t.choose_attribute_for_atprompt(cfg)

    # Build trainer and load checkpoint
    trainer = build_trainer(cfg)
    trainer.load_model(args.checkpoint, epoch=25)
    model = trainer.model; model.eval()
    pl = model.prompt_learner; dtype = model.dtype

    # Load new class data
    dm = DataManager(cfg); ds = dm.dataset.train_x
    tfm = build_transform(cfg, is_train=False)
    loader = DataLoader(DatasetWrapper(cfg, ds, transform=tfm, is_train=False),
                       batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    imgs, labs = [], []
    for b in loader: imgs.append(b['img']); labs.append(b['label'])
    images = torch.cat(imgs).cuda(); labels = torch.cat(labs).cuda()
    li = defaultdict(list)
    for i, l in enumerate(labels): li[l.item()].append(i)
    ul = sorted(li.keys()); nc = len(ul)

    results = []
    for K in [1, 3, 5]:
        accs = []; nw = min(5, nc)
        for ep in range(args.n_episodes):
            sel = random.sample(ul, nw)
            si, sl_, qi, ql_ = [], [], [], []
            for ni, ol in enumerate(sel):
                idxs = li[ol]; random.shuffle(idxs)
                for s in idxs[:K]: si.append(images[s]); sl_.append(ni)
                for q in idxs[K:K+10]: qi.append(images[q]); ql_.append(ni)
            sup = torch.stack(si); qry = torch.stack(qi)
            sll = torch.tensor(sl_).cuda(); qll = torch.tensor(ql_).cuda()
            with torch.no_grad():
                vf = model.image_encoder(sup.type(dtype))
                vp = torch.stack([F.normalize(vf[sll==c].mean(0), dim=-1) for c in range(nw)])
                cnames = [dm.lab2cname[l].replace('_',' ') for l in sel]
                prompts_text = ['X X luxury X X ' + name + '.' for name in cnames]
                tok = torch.cat([clip.tokenize(p) for p in prompts_text]).cuda()
                emb = token_embed(tok).type(dtype)
                emb[:,1:3,:] = pl.ctx_att1.unsqueeze(0).expand(len(cnames),-1,-1)
                emb[:,4:6,:] = pl.ctx.unsqueeze(0).expand(len(cnames),-1,-1)
                tf = F.normalize(model.text_encoder(emb, tok), dim=-1)
                protos = F.normalize((vp+tf)/2, dim=-1)
                qf = F.normalize(model.image_encoder(qry.type(dtype)), dim=-1)
                accs.append((qf@protos.T).argmax(-1).eq(qll).float().mean().item())
        mean = sum(accs)/len(accs)*100
        std = (sum((a*100-mean)**2 for a in accs)/len(accs))**0.5
        results.append(f'DL-MPT episodic K={K}: {mean:.1f}% ± {std:.1f}%')

    with open(args.output, 'w') as f:
        for r in results: f.write(r + '\n')
    for r in results: print(r)

if __name__ == '__main__':
    main()
