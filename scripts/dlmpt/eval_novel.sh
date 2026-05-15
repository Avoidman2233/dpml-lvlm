#!/bin/bash
set -e
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$PROJ/../DATA"
PY=python

DATASET="stanford_cars"; SEED=1; MODEL_DIR=""; LOAD_EPOCH=200; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --model-dir) MODEL_DIR="$2"; shift 2;;
        --load-epoch) LOAD_EPOCH="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) shift;;
    esac
done

[ -z "$MODEL_DIR" ] && { echo "ERROR: --model-dir required"; exit 1; }

cd "$PROJ"
CMD="$PY train.py --root $DATA --seed $SEED --trainer CoOp_ATP \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir /tmp/dlmpt_eval_novel \
    --model-dir $MODEL_DIR --load-epoch $LOAD_EPOCH --eval-only \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.ATPROMPT.USE_ATPROMPT True TRAINER.COOP.N_CTX 2 DATALOADER.NUM_WORKERS 4"

if $DRY_RUN; then echo "$CMD"; exit 0; fi
eval "$CMD" 2>&1 | grep -E "accuracy:|correct:|total:"
