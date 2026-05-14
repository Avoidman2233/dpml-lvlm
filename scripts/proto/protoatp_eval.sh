#!/bin/bash
# ProtoATP Few-Shot Evaluation Script
# Usage: bash scripts/proto/protoatp_eval.sh --dataset stanford_cars [--seed 1] [--checkpoint <path>] [--k-shots "1 3 5"] [--n-episodes 200]
#
# Evaluates a ProtoATP meta-trained checkpoint on new classes using
# episodic few-shot protocol (N-way K-shot prototype inference, zero gradient).

set -euo pipefail

PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA_ROOT=/home/avoidman2233/Desktop/LVLM/DATA
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED=1
CHECKPOINT=""
DATASET=""
OUTPUT_ROOT=output/protoatp
K_SHOTS="1 3 5"
N_EPISODES=200
N_WAY=5
INNER_STEPS=2
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --dataset <name> --checkpoint <path> [options]

Options:
  --dataset NAME       Dataset name (required)
  --checkpoint PATH    Meta checkpoint .pth.tar file (required)
  --seed N             Random seed (default: $SEED)
  --output DIR         Output root directory (default: $OUTPUT_ROOT)
  --k-shots "1 3 5"    Space-separated K values (default: "$K_SHOTS")
  --n-way N            Number of classes per episode (default: $N_WAY)
  --n-episodes N       Episodes per K value (default: $N_EPISODES)
  --dry-run            Print commands without executing
  -h, --help           Show this help

Example (auto-detect checkpoint):
  $(basename "$0") --dataset stanford_cars --seed 1

Example (explicit checkpoint):
  $(basename "$0") --dataset eurosat --checkpoint output/protoatp/stanford_cars/seed1/meta/prompt_learner/model.pth.tar-20
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2"; shift 2 ;;
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --output)       OUTPUT_ROOT="$2"; shift 2 ;;
        --k-shots)      K_SHOTS="$2"; shift 2 ;;
        --n-way)        N_WAY="$2"; shift 2 ;;
        --n-episodes)   N_EPISODES="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── validation ─────────────────────────────────────────────
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"; usage
fi

# Auto-detect checkpoint if not specified
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="${OUTPUT_ROOT}/${DATASET}/seed${SEED}/meta/prompt_learner/model.pth.tar-20"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT"
        echo "  Specify with --checkpoint or train first with protoatp_coop_train.sh"
        exit 1
    fi
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"; exit 1
fi

# ── output paths ──────────────────────────────────────────
EVAL_DIR="${OUTPUT_ROOT}/${DATASET}/seed${SEED}/eval"
mkdir -p "$EVAL_DIR"

# ── Python eval snippet ───────────────────────────────────
PYTHON_EVAL=$(cat <<'PYEOF'
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, random, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import defaultdict
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper, build_transform
from torch.utils.data import DataLoader
from clip import clip
import train as _t

# Load config
cfg = get_cfg_default(); _t.extend_cfg(cfg)
cfg.DATASET.ROOT = DATA_ROOT
cfg.DATASET.NUM_SHOTS = 16; cfg.DATASET.SUBSAMPLE_CLASSES = 'new'
cfg.SEED = SEED; cfg.TRAINER.NAME = 'CoOp_ATP'
cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
cfg.INPUT.SIZE = (224, 224)
cfg.INPUT.TRANSFORMS = ('random_resized_crop', 'random_flip', 'normalize')
cfg.TRAINER.COOP.N_CTX = N_CTX; cfg.TRAINER.COOP.CSC = False
cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
cfg.TRAINER.ATPROMPT.USE_ATPROMPT = True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.merge_from_file(f'configs/datasets/{DS_FILE}.yaml')
cfg.merge_from_file('configs/trainers/CoOp/vit_b16.yaml')
_t.choose_attribute_for_atprompt(cfg)

trainer = build_trainer(cfg)
from trainers.meta_pretrainer import MetaPretrainer
MetaPretrainer.load_meta_checkpoint(trainer, CHECKPOINT)
model = trainer.model; model.eval(); pl = model.prompt_learner; dtype = model.dtype

# Load token_embed
ckpt_p = '/home/avoidman2233/.cache/clip/ViT-B-16.pt'
jit = torch.jit.load(ckpt_p, map_location='cpu').eval(); sd = jit.state_dict()
token_embed = torch.nn.Embedding(sd['token_embedding.weight'].shape[0], sd['token_embedding.weight'].shape[1]).cuda()
token_embed.weight.data.copy_(sd['token_embedding.weight']); del jit, sd

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

print(f'Dataset: {DS_NAME} | New classes: {nc} | N_way: {N_WAY} | Episodes: {N_EPISODES}')

results = {}
for K in K_SHOTS_LIST:
    accs = []; nw = min(N_WAY, nc)
    for ep in range(N_EPISODES):
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
            # Build attribute prompt (handles 1-3 attributes)
            parts = ATTR_PARTS + ['X X ' + name + '.' for name in cnames]
            ptexts = [f'X X {ATTR_STR} X X {name}.' for name in cnames]
            tok = torch.cat([clip.tokenize(p) for p in ptexts]).cuda()
            emb = token_embed(tok).type(dtype)
            # Insert ctx_att and ctx (positions depend on attr count)
            pos = 1
            for ai in range(ATTR_NUM):
                catt = getattr(pl, f'ctx_att{ai+1}', None)
                if catt is not None:
                    natt = catt.shape[0]
                    emb[:, pos:pos+natt, :] = catt.unsqueeze(0).expand(len(cnames), -1, -1)
                    pos += natt + 1  # +1 for the attr word token
            emb[:, pos:pos+N_CTX, :] = pl.ctx.unsqueeze(0).expand(len(cnames), -1, -1)
            tf = F.normalize(model.text_encoder(emb, tok), dim=-1)
            protos = F.normalize((vp + tf) / 2, dim=-1)
            qf = F.normalize(model.image_encoder(qry.type(dtype)), dim=-1)
            accs.append((qf @ protos.T).argmax(-1).eq(qll).float().mean().item())
    mean = sum(accs) / len(accs) * 100
    std = (sum((a*100 - mean)**2 for a in accs) / len(accs)) ** 0.5
    results[f'K={K}'] = f'{mean:.1f}% \u00b1 {std:.1f}%'
    print(f'  {results[f"K={K}"]}')

# Save results
os.makedirs(EVAL_DIR, exist_ok=True)
with open(f'{EVAL_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {EVAL_DIR}/results.json')
PYEOF
)

# ── execute ───────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " ProtoATP Few-Shot Evaluation"
echo "═══════════════════════════════════════════════════════════"
echo " Dataset:    $DATASET"
echo " Checkpoint: $CHECKPOINT"
echo " K-shots:    $K_SHOTS"
echo " N-way:      $N_WAY"
echo " Episodes:   $N_EPISODES"
echo " Seed:       $SEED"
echo " Output:     $EVAL_DIR"
echo "═══════════════════════════════════════════════════════════"

if $DRY_RUN; then
    echo "[DRY-RUN] Would evaluate $DATASET with K-shots: $K_SHOTS"
    exit 0
fi

# Build Python command with variable substitution
eval_cmd="$PYTHON -c \"$(echo "$PYTHON_EVAL" | sed \
    -e "s|DATA_ROOT|'$DATA_ROOT'|g" \
    -e "s|SEED|$SEED|g" \
    -e "s|CHECKPOINT|'$CHECKPOINT'|g" \
    -e "s|DS_FILE|'$DATASET'|g" \
    -e "s|DS_NAME|'$DATASET'|g" \
    -e "s|N_WAY|$N_WAY|g" \
    -e "s|N_EPISODES|$N_EPISODES|g" \
    -e "s|EVAL_DIR|'$EVAL_DIR'|g" \
    -e "s|K_SHOTS_LIST|$K_SHOTS|g" \
    -e "s|N_CTX|\${DS_N_CTX[$DATASET]:-2}|g" \
)\""

# Get N_CTX and attribute info for this dataset
_n_ctx="${DS_N_CTX[$DATASET]:-2}"
_n_way="${DS_N_WAY[$DATASET]:-5}"

# Actually run
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0 $PYTHON -c "
import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, random, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import defaultdict
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper, build_transform
from torch.utils.data import DataLoader
from clip import clip
import train as _t

cfg = get_cfg_default(); _t.extend_cfg(cfg)
cfg.DATASET.ROOT = '$DATA_ROOT'
cfg.DATASET.NUM_SHOTS = 16; cfg.DATASET.SUBSAMPLE_CLASSES = 'new'
cfg.SEED = $SEED; cfg.TRAINER.NAME = 'CoOp_ATP'
cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
cfg.INPUT.SIZE = (224, 224)
cfg.INPUT.TRANSFORMS = ('random_resized_crop', 'random_flip', 'normalize')
cfg.TRAINER.COOP.N_CTX = $_n_ctx; cfg.TRAINER.COOP.CSC = False
cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
cfg.TRAINER.ATPROMPT.USE_ATPROMPT = True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.merge_from_file('configs/datasets/${DATASET}.yaml')
cfg.merge_from_file('configs/trainers/CoOp/vit_b16.yaml')
_t.choose_attribute_for_atprompt(cfg)

trainer = build_trainer(cfg)
from trainers.meta_pretrainer import MetaPretrainer
MetaPretrainer.load_meta_checkpoint(trainer, '$CHECKPOINT')
model = trainer.model; model.eval(); pl = model.prompt_learner; dtype = model.dtype

ckpt_p = '/home/avoidman2233/.cache/clip/ViT-B-16.pt'
jit = torch.jit.load(ckpt_p, map_location='cpu').eval(); sd = jit.state_dict()
token_embed = torch.nn.Embedding(sd['token_embedding.weight'].shape[0], sd['token_embedding.weight'].shape[1]).cuda()
token_embed.weight.data.copy_(sd['token_embedding.weight']); del jit, sd

dm = DataManager(cfg); ds = dm.dataset.train_x
tfm = build_transform(cfg, is_train=False)
loader = DataLoader(DatasetWrapper(cfg, ds, transform=tfm, is_train=False), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
imgs, labs = [], []
for b in loader: imgs.append(b['img']); labs.append(b['label'])
images = torch.cat(imgs).cuda(); labels = torch.cat(labs).cuda()
li = defaultdict(list)
for i, l in enumerate(labels): li[l.item()].append(i)
ul = sorted(li.keys()); nc = len(ul)
print(f'Classes: {nc} | N_way: ${_n_way} | Episodes: ${N_EPISODES}')

attr1 = cfg.TRAINER.ATPROMPT.ATT1_TEXT
attr2 = getattr(cfg.TRAINER.ATPROMPT, 'ATT2_TEXT', '')
attr3 = getattr(cfg.TRAINER.ATPROMPT, 'ATT3_TEXT', '')
attr_words = ' '.join([w for w in [attr1, attr2, attr3] if w])
print(f'Attributes: {attr_words}')

for K in ${K_SHOTS}:
    accs = []; nw = min(${_n_way}, nc)
    for ep in range(${N_EPISODES}):
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
            ptexts = ['X X ' + attr_words + ' X X ' + name + '.' for name in cnames]
            tok = torch.cat([clip.tokenize(p) for p in ptexts]).cuda()
            emb = token_embed(tok).type(dtype)
            emb[:,1:3,:] = pl.ctx_att1.unsqueeze(0).expand(len(cnames),-1,-1)
            # Insert remaining ctx_att if present
            pos = 4
            if hasattr(pl, 'ctx_att2') and pl.atp_num >= 2:
                emb[:,pos:pos+pl.ctx_att2.shape[0],:] = pl.ctx_att2.unsqueeze(0).expand(len(cnames),-1,-1)
                pos += pl.ctx_att2.shape[0] + 1
            if hasattr(pl, 'ctx_att3') and pl.atp_num >= 3:
                emb[:,pos:pos+pl.ctx_att3.shape[0],:] = pl.ctx_att3.unsqueeze(0).expand(len(cnames),-1,-1)
                pos += pl.ctx_att3.shape[0] + 1
            emb[:,pos:pos+${_n_ctx},:] = pl.ctx.unsqueeze(0).expand(len(cnames),-1,-1)
            tf = F.normalize(model.text_encoder(emb, tok), dim=-1)
            protos = F.normalize((vp + tf) / 2, dim=-1)
            qf = F.normalize(model.image_encoder(qry.type(dtype)), dim=-1)
            accs.append((qf @ protos.T).argmax(-1).eq(qll).float().mean().item())
    mean = sum(accs) / len(accs) * 100
    std = (sum((a*100 - mean)**2 for a in accs) / len(accs)) ** 0.5
    print(f'RESULT K={K}: {mean:.1f}% std={std:.1f}%')
" 2>&1 | grep -E "RESULT|Classes|Attributes|Error|Traceback"