#!/bin/bash
# DL-MPT + CoCoOp Training Script
# Usage: bash scripts/dlmpt/train_cocoop.sh --dataset stanford_cars --seed 1 --lambda 0.2
set -e

PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA=/home/avoidman2233/Desktop/LVLM/DATA
PROJ=/home/avoidman2233/Desktop/LVLM/ATPrompt

DATASET="stanford_cars"; SEED=1; LAMBDA=0.2; OUTPUT=""; DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --lambda) LAMBDA="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) shift;;
    esac
done

[ -z "$OUTPUT" ] && OUTPUT="output/dlmpt_cocoop/${DATASET}/seed${SEED}/lambda${LAMBDA}"

cd "$PROJ"
mkdir -p "$OUTPUT"

CMD="CUDA_VISIBLE_DEVICES=0 $PY train.py \
    --root $DATA --seed $SEED --trainer DLMPTTrainer \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/dlmpt/vit_b16_cocoop.yaml \
    --output-dir $OUTPUT \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True \
    TRAINER.DLMPT.LAMBDA $LAMBDA \
    TRAINER.DLMPT.COCOOP True \
    DATALOADER.NUM_WORKERS 4"

if $DRY_RUN; then echo "$CMD"; exit 0; fi

echo "[DL-MPT CoCoOp] dataset=$DATASET seed=$SEED lambda=$LAMBDA output=$OUTPUT"
echo "[DL-MPT CoCoOp] start: $(date '+%Y-%m-%d %H:%M:%S')"
eval "$CMD" 2>&1 | tee "${OUTPUT}/train_stdout.log"
echo "[DL-MPT CoCoOp] done:  $(date '+%Y-%m-%d %H:%M:%S')"
