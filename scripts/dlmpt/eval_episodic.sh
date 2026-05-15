#!/bin/bash
set -e
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$PROJ/../DATA"
PY=python

DATASET="stanford_cars"; SEED=1; CHECKPOINT=""; N_EPISODES=200; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --n-episodes) N_EPISODES="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) shift;;
    esac
done

[ -z "$CHECKPOINT" ] && { echo "ERROR: --checkpoint required"; exit 1; }

RESULT_FILE="output/dlmpt/${DATASET}/seed${SEED}/episodic_results.txt"
mkdir -p "$(dirname "$RESULT_FILE")"

echo "=== DL-MPT Episodic Evaluation ===" | tee "$RESULT_FILE"
echo "Dataset: $DATASET  Seed: $SEED  Episodes: $N_EPISODES" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

for K in 1 3 5; do
    NW=$K; [ $NW -gt 10 ] && NW=10
    CMD="$PY -c \"
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import defaultdict
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper, build_transform
from torch.utils.data import DataLoader
from clip import clip
import train as _t

# Load token embedding
ckpt_path = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
jit = torch.jit.load(ckpt_path, map_location='cpu').eval()
sd = jit.state_dict()
token_embed = torch.nn.Embedding(sd['token_embedding.weight'].shape[0], sd['token_embedding.weight'].shape[1]).cuda()
token_embed.weight.data.copy_(sd['token_embedding.weight'])
del jit, sd

cfg = get_cfg_default()
_t.extend_cfg(cfg)
cfg.DATASET.ROOT = '$DATA'
cfg.DATASET.NUM_SHOTS = 16
cfg.DATASET.SUBSAMPLE_CLASSES = 'new'
cfg.SEED = $SEED
cfg.TRAINER.NAME = 'CoOp_ATP'
cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
cfg.INPUT.SIZE = (224, 224)
cfg.INPUT.TRANSFORMS = ('random_resized_crop', 'random_flip', 'normalize')
cfg.TRAINER.COOP.N_CTX = 2
cfg.TRAINER.COOP.CSC = False
cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
cfg.TRAINER.ATPROMPT.USE_ATPROMPT = True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.merge_from_file(f'configs/datasets/${DATASET}.yaml')
cfg.merge_from_file('configs/trainers/CoOp/vit_b16.yaml')
_t.choose_attribute_for_atprompt(cfg)

trainer = build_trainer(cfg)
from trainers.meta_pretrainer import MetaPretrainer
MetaPretrainer.load_meta_checkpoint(trainer, '$CHECKPOINT')
model = trainer.model
model.eval()
pl = model.prompt_learner
dtype = model.dtype

dm = DataManager(cfg)
ds = dm.dataset.train_x
tfm = build_transform(cfg, is_train=False)
loader = DataLoader(DatasetWrapper(cfg, ds, transform=tfm, is_train=False), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
imgs, labs = [], []
for b in loader:
    imgs.append(b['img'])
    labs.append(b['label'])
images = torch.cat(imgs).cuda()
labels = torch.cat(labs).cuda()
li = defaultdict(list)
for i, l in enumerate(labels):
    li[l.item()].append(i)
ul = sorted(li.keys())
nc = len(ul)

n_way = min($NW, nc)
accs = []
for ep in range($N_EPISODES):
    sel = random.sample(ul, n_way)
    si, sl_, qi, ql_ = [], [], [], []
    for ni, ol in enumerate(sel):
        idxs = li[ol]
        random.shuffle(idxs)
        for s in idxs[:$K]:
            si.append(images[s])
            sl_.append(ni)
        for q in idxs[$K:min($K+10, len(idxs))]:
            qi.append(images[q])
            ql_.append(ni)
    sup = torch.stack(si)
    qry = torch.stack(qi)
    sll = torch.tensor(sl_).cuda()
    qll = torch.tensor(ql_).cuda()
    with torch.no_grad():
        vf = model.image_encoder(sup.type(dtype))
        vp = torch.stack([F.normalize(vf[sll==c].mean(0), dim=-1) for c in range(n_way)])
        cnames = [dm.lab2cname[l].replace('_', ' ') for l in sel]
        attr_texts = []
        if hasattr(pl, 'atp_num'):
            for n in range(1, pl.atp_num + 1):
                attr_texts.append(getattr(cfg.TRAINER.ATPROMPT, f'ATT{n}_TEXT', ''))
        attr_str = ' '.join(attr_texts) if attr_texts else 'luxury'
        prompts_text = [f'X X {attr_str} X X {name}.' for name in cnames]
        tok = torch.cat([clip.tokenize(p) for p in prompts_text]).cuda()
        emb = token_embed(tok).type(dtype)
        emb[:, 1:3, :] = pl.ctx_att1.unsqueeze(0).expand(len(cnames), -1, -1)
        if hasattr(pl, 'ctx_att2') and pl.atp_num >= 2:
            emb[:, 4:6, :] = pl.ctx_att2.unsqueeze(0).expand(len(cnames), -1, -1)
            emb[:, 7:9, :] = pl.ctx.unsqueeze(0).expand(len(cnames), -1, -1)
        else:
            emb[:, 4:6, :] = pl.ctx.unsqueeze(0).expand(len(cnames), -1, -1)
        tf = F.normalize(model.text_encoder(emb, tok), dim=-1)
        protos = F.normalize((vp + tf) / 2, dim=-1)
        qf = F.normalize(model.image_encoder(qry.type(dtype)), dim=-1)
        accs.append((qf @ protos.T).argmax(-1).eq(qll).float().mean().item())

mean = sum(accs) / len(accs) * 100
std = (sum((a * 100 - mean) ** 2 for a in accs) / len(accs)) ** 0.5
print(f'DLMPT episodic K=${K}: {mean:.1f}% ± {std:.1f}%')
\"" 2>&1
    
    if $DRY_RUN; then
        echo "[DRY-RUN] K=$K"
    else
        eval "$CMD" | tee -a "$RESULT_FILE"
    fi
done

echo "Results saved to: $RESULT_FILE"
