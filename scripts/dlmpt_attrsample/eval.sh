#!/bin/bash
# DL-MPT-AttrSample Novel Class Evaluation Script
# Usage: bash scripts/dlmpt_attrsample/eval.sh --dataset stanford_cars --seed 1 --load-epoch 25
set -e

PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python

DATASET=""
SEED=""
EPOCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed)    SEED="$2"; shift 2;;
        --load-epoch) EPOCH="$2"; shift 2;;
        *) shift;;
    esac
done

[ -z "$DATASET" ] && { echo "ERROR: --dataset required"; exit 1; }
[ -z "$SEED" ] && { echo "ERROR: --seed required"; exit 1; }
[ -z "$EPOCH" ] && { echo "ERROR: --load-epoch required"; exit 1; }

CMD="$PYTHON train.py \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed ${SEED} \
    --trainer DLMPT_AttrSample_CoOp_ATP \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/dlmpt_attrsample/vit_b16.yaml \
    --model-dir output/dlmpt_attrsample/${DATASET}/seed${SEED}/ \
    --load-epoch ${EPOCH} --eval-only \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.COOP.N_CTX 2 \
    TRAINER.ATPROMPT.USE_ATPROMPT True"

echo "[DL-MPT-ATTRSAMPLE EVAL]  dataset=$DATASET seed=$SEED epoch=$EPOCH"
eval "$CMD" 2>&1
