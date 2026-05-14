#!/bin/bash
# Multi-process meta-pretrain runner - maximizes GPU utilization
# Usage: bash scripts/meta/run_parallel.sh <dataset> <n_way> <num_processes>

DS=${1:-eurosat}
NW=${2:-5}
NP=${3:-2}
ROOT=/home/avoidman2233/Desktop/LVLM/DATA
AT=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
OUT=output/meta_v3/${DS}

mkdir -p $OUT

echo "=== $DS N_WAY=$NW Process=$NP ==="

for S in $(seq 1 $NP); do
    CUDA_VISIBLE_DEVICES=0 $AT train.py \
        --trainer MetaPretrainer --root $ROOT --seed $S \
        --dataset-config-file configs/datasets/${DS}.yaml \
        --config-file configs/trainers/meta/vit_b16.yaml \
        --output-dir ${OUT}/seed${S} \
        DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.META.N_WAY $NW TRAINER.META.K_SUPPORT 1 \
        TRAINER.META.K_QUERY 15 TRAINER.META.INNER_STEPS 3 \
        TRAINER.META.N_EPISODES 200 OPTIM.MAX_EPOCH 30 \
        DATALOADER.NUM_WORKERS 2 \
        > ${OUT}/seed${S}/log.txt 2>&1 &
    echo "  Started seed $S (PID $!)"
done

echo "Waiting for all processes..."
wait
echo "=== ALL SEEDS DONE ==="
for S in $(seq 1 $NP); do
    [ -f ${OUT}/seed${S}/prompt_learner/model.pth.tar-30 ] && echo "seed$S OK" || echo "seed$S FAIL"
done