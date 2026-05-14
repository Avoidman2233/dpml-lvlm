#!/bin/bash
# Cross-dataset meta-training on ImageNet base classes
# Usage: bash scripts/meta/cross_meta_train.sh [--seed N] [--output DIR]

SEED=1
DIR=output/maml_fewshot/imagenet_meta

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEED=$2
            shift 2
            ;;
        --output)
            DIR=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA=/home/avoidman2233/Desktop/LVLM/DATA

${PYTHON} train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer MetaPretrainer \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/meta/imagenet_meta.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base
