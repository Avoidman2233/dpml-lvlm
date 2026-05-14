#!/bin/bash
# ProtoATP-CoCoOp meta-training on Stanford Cars
# Instance-conditional prompts for prototype computation (FIXED)
set -e

PYTHON=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA=/home/avoidman2233/Desktop/LVLM/DATA
export CUDA_VISIBLE_DEVICES=0

$PYTHON train.py \
    --root ${DATA} \
    --seed 1 \
    --trainer ProtoTrainer \
    --dataset-config-file configs/datasets/stanford_cars.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/protoatp/stanford_cars/seed1/cocoop_meta \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True \
    TRAINER.COOP.N_CTX 2 \
    TRAINER.COCOOP.N_CTX 2 \
    TRAINER.PROTO.METHOD CoCoOp \
    TRAINER.PROTO.N_WAY 20 \
    TRAINER.PROTO.N_EPISODES 200 \
    TRAINER.PROTO.K_SUPPORT 3 \
    TRAINER.PROTO.K_QUERY 10 \
    OPTIM.MAX_EPOCH 20 \
    OPTIM.WARMUP_EPOCH 3 \
    DATALOADER.NUM_WORKERS 4
