#!/bin/bash
# Reptile+L2 Training Script
# Usage: bash scripts/reptile_l2/train.sh --dataset stanford_cars --seed 1
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
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="output/reptile_l2/${DATASET}/seed${SEED}"

mkdir -p "$OUTPUT_DIR"

CMD="$PYTHON train.py \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed ${SEED} \
    --trainer Reptile_L2_CoOp_ATP \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/reptile_l2/vit_b16.yaml \
    --output-dir ${OUTPUT_DIR}"

echo "[REPTILE_L2 TRAIN] dataset=$DATASET seed=$SEED output=$OUTPUT_DIR"
echo "[REPTILE_L2 TRAIN] start: $(date '+%Y-%m-%d %H:%M:%S')"
eval "$CMD" > "${OUTPUT_DIR}/train.log" 2>&1
echo "[REPTILE_L2 TRAIN] done:  $(date '+%Y-%m-%d %H:%M:%S')"
