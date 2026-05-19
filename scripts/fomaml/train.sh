#!/bin/bash
# FOMAML Training Script
# Usage: bash scripts/fomaml/train.sh --dataset stanford_cars --seed 1
set -e

PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python

DATASET=""
SEED=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed)    SEED="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[ -z "$DATASET" ] && { echo "ERROR: --dataset required"; exit 1; }
[ -z "$SEED" ] && { echo "ERROR: --seed required"; exit 1; }
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="output/fomaml/${DATASET}/seed${SEED}"

mkdir -p "$OUTPUT_DIR"

CMD="$PYTHON train.py \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed ${SEED} \
    --trainer FOMAML_CoOp_ATP \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/fomaml/vit_b16.yaml \
    --output-dir ${OUTPUT_DIR}"

echo "[FOMAML TRAIN] dataset=$DATASET seed=$SEED output=$OUTPUT_DIR"
echo "[FOMAML TRAIN] start: $(date '+%Y-%m-%d %H:%M:%S')"
eval "$CMD" > "${OUTPUT_DIR}/train.log" 2>&1
echo "[FOMAML TRAIN] done:  $(date '+%Y-%m-%d %H:%M:%S')"
