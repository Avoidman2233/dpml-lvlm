#!/bin/bash
# Fine-tune after meta-pretraining
# Usage: bash scripts/meta/meta_finetune.sh <dataset>
DATA=../DATA
TRAINER=CoOp_ATP
CFG=vit_b16
SHOTS=16
CSC=False
CTP=end
NCTX=16
DATASET=$1

for SEED in 1 2 3; do
    META_DIR=output/meta_pretrain/${DATASET}/vit_b16_${SHOTS}shots/seed${SEED}
    DIR=output/meta_finetune/${DATASET}/${TRAINER}/vit_b16_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${META_DIR} \
        --load-epoch 20 \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.ATPROMPT.USE_ATPROMPT True \
        TRAINER.ATPROMPT.N_ATT1 8 \
        TRAINER.ATPROMPT.N_ATT2 8 \
        TRAINER.ATPROMPT.N_ATT3 8
done
