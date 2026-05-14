#!/bin/bash
# Meta-pretraining on base classes (single dataset)
# Usage: bash scripts/meta/meta_pretrain.sh <dataset>
DATA=../DATA
TRAINER=MetaPretrainer
CFG=vit_b16
SHOTS=16
DATASET=$1

for SEED in 1 2 3; do
    DIR=output/meta_pretrain/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/meta/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
done
