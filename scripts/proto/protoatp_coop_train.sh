#!/bin/bash
# ProtoATP-CoOp Training Script
# Usage: bash scripts/proto/protoatp_coop_train.sh --dataset stanford_cars [--seed 1] [--epochs 20] [--output output/protoatp]
#
# Fixed few-shot learner with attribute-enhanced prototypes (CoOp backbone).
# Trains prompt_learner (ctx + ctx_att) via episode-based prototypical loss
# on base classes; evaluates via prototype inference on new classes.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────
PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA_ROOT=/home/avoidman2233/Desktop/LVLM/DATA
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED=1
EPOCHS=20
OUTPUT_ROOT=output/protoatp
N_EPISODES=200
K_SUPPORT=3
K_QUERY=10
INNER_STEPS=2
N_WAY=""
N_CTX=""
DATASET=""
DRY_RUN=false
QUIET=false

# ── dataset registry ──────────────────────────────────────
# Format: dataset_name|n_way|n_ctx|attrs
declare -A DS_N_WAY DS_N_CTX DS_ATTRS

_register() {
    local ds=$1 nw=$2 nc=$3; shift 3
    DS_N_WAY[$ds]=$nw
    DS_N_CTX[$ds]=$nc
    DS_ATTRS[$ds]="$*"
}

_register "stanford_cars"    20  2  "luxury"
_register "oxford_pets"      10  2  "playfulness" "energy"
_register "oxford_flowers"   20  2  "color" "habitat" "growth"
_register "eurosat"           3  2  "habitat"
_register "dtd"              10  4  "pattern" "color" "design"
_register "fgvc_aircraft"    20  2  "design" "range"
_register "food101"          20  2  "flavor" "preparation"
_register "ucf101"           20  2  "precision"
_register "caltech101"       20  2  "shape" "size"
_register "sun397"           20  2  "function"

# ── argument parsing ──────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --dataset <name> [options]

Options:
  --dataset NAME       Dataset name (required)
                         One of: ${!DS_N_WAY[*]}
  --seed N             Random seed (default: $SEED)
  --epochs N           Meta-training epochs (default: $EPOCHS)
  --output DIR         Output root directory (default: $OUTPUT_ROOT)
  --n-episodes N       Episodes per epoch (default: $N_EPISODES)
  --k-support N        Support samples per class (default: $K_SUPPORT)
  --k-query N          Query samples per class (default: $K_QUERY)
  --dry-run            Print commands without executing
  --quiet              Suppress verbose output
  -h, --help           Show this help

Example:
  $(basename "$0") --dataset stanford_cars --seed 1 --epochs 20
  $(basename "$0") --dataset oxford_pets --seed 2 --epochs 10 --dry-run
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --output)       OUTPUT_ROOT="$2"; shift 2 ;;
        --n-episodes)   N_EPISODES="$2"; shift 2 ;;
        --k-support)    K_SUPPORT="$2"; shift 2 ;;
        --k-query)      K_QUERY="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --quiet)        QUIET=true; shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── validation ─────────────────────────────────────────────
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"
    usage
fi

N_WAY="${DS_N_WAY[$DATASET]:-}"
N_CTX="${DS_N_CTX[$DATASET]:-}"

if [[ -z "$N_WAY" ]]; then
    echo "ERROR: Unknown dataset '$DATASET'. Available: ${!DS_N_WAY[*]}"
    exit 1
fi

# ── output paths ──────────────────────────────────────────
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/seed${SEED}/meta"
mkdir -p "$OUT_DIR"

# ── build command ─────────────────────────────────────────
CMD="cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 $PYTHON train.py \
    --root $DATA_ROOT \
    --seed $SEED \
    --trainer ProtoTrainer \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir $OUT_DIR \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True \
    TRAINER.COOP.N_CTX $N_CTX \
    TRAINER.PROTO.METHOD CoOp \
    TRAINER.PROTO.MODE full \
    TRAINER.PROTO.N_WAY $N_WAY \
    TRAINER.PROTO.K_SUPPORT $K_SUPPORT \
    TRAINER.PROTO.K_QUERY $K_QUERY \
    TRAINER.PROTO.N_EPISODES $N_EPISODES \
    TRAINER.PROTO.TEMPERATURE 10.0 \
    OPTIM.MAX_EPOCH $EPOCHS \
    OPTIM.WARMUP_EPOCH 3 \
    DATALOADER.NUM_WORKERS 4"

# ── execute ───────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " ProtoATP-CoOp Training"
echo "═══════════════════════════════════════════════════════════"
echo " Dataset:       $DATASET"
echo " N_WAY:         $N_WAY"
echo " N_CTX:         $N_CTX"
echo " Attributes:    ${DS_ATTRS[$DATASET]}"
echo " Seed:          $SEED"
echo " Epochs:        $EPOCHS"
echo " Episodes/epoch: $N_EPISODES"
echo " K_support:     $K_SUPPORT"
echo " K_query:       $K_QUERY"
echo " Output:        $OUT_DIR"
echo "═══════════════════════════════════════════════════════════"

if $DRY_RUN; then
    echo "[DRY-RUN] $CMD"
else
    eval "$CMD" 2>&1 | {
        if $QUIET; then
            grep -E "accuracy:|error:|ProtoATP|epoch \[|DONE|Error|Traceback" || true
        else
            cat
        fi
    }
    echo ""
    echo "Training complete. Checkpoint: $OUT_DIR/prompt_learner/model.pth.tar-${EPOCHS}"
    echo "Progress log:    $OUT_DIR/progress.log"
fi
