#!/bin/bash
# Evaluate on new classes after meta-pretraining + fine-tuning
# Usage: bash scripts/meta/meta_eval.sh <dataset>
DATA=/root/prompt_dataset
TRAINER=CoOp_ATP
CFG=vit_b16
SHOTS=16
DATASET=$1

for SEED in 1 2 3; do
    DIR=output/meta_finetune/${DATASET}/${TRAINER}/vit_b16_${SHOTS}shots/nctx16_cscFalse_ctpend/seed${SEED}
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir output/evaluation/meta/${DATASET}/seed${SEED} \
        --model-dir ${DIR} \
        --load-epoch 100 \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES new \
        TRAINER.ATPROMPT.USE_ATPROMPT True \
        TRAINER.ATPROMPT.N_ATT1 8 \
        TRAINER.ATPROMPT.N_ATT2 8 \
        TRAINER.ATPROMPT.N_ATT3 8
done
