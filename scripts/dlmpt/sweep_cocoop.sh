#!/bin/bash
# DL-MPT(CoCoOp+ATP) 全数据集 Sweep (7 datasets × 3 seeds = 21 runs)
# Usage: bash scripts/dlmpt/sweep_cocoop.sh [--dry-run]
set -e

PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA=/home/avoidman2233/Desktop/LVLM/DATA
PROJ=/home/avoidman2233/Desktop/LVLM/ATPrompt

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

# Per-dataset config: data_dir, config_stem, N_WAY, N_CTX
# config_stem is the yaml filename prefix (e.g., "food101" → configs/datasets/food101.yaml)
declare -A DS_NWAY DS_NCTX DS_CFG
DS_NWAY[stanford_cars]=20;   DS_NCTX[stanford_cars]=2;  DS_CFG[stanford_cars]="stanford_cars"
DS_NWAY[eurosat]=3;         DS_NCTX[eurosat]=2;        DS_CFG[eurosat]="eurosat"
DS_NWAY[oxford_pets]=10;    DS_NCTX[oxford_pets]=2;    DS_CFG[oxford_pets]="oxford_pets"
DS_NWAY[oxford_flowers]=20; DS_NCTX[oxford_flowers]=2; DS_CFG[oxford_flowers]="oxford_flowers"
DS_NWAY[ucf101]=20;         DS_NCTX[ucf101]=2;         DS_CFG[ucf101]="ucf101"
DS_NWAY[dtd]=10;            DS_NCTX[dtd]=4;            DS_CFG[dtd]="dtd"
DS_NWAY[food101]=20;        DS_NCTX[food101]=2;        DS_CFG[food101]="food101"
DS_NWAY[fgvc_aircraft]=20;  DS_NCTX[fgvc_aircraft]=2;  DS_CFG[fgvc_aircraft]="fgvc_aircraft"

DATASETS=(stanford_cars eurosat oxford_pets oxford_flowers ucf101 dtd food101 fgvc_aircraft)
SEEDS=(1 2 3)
LAMBDA=0.2
LOG_FILE="$PROJ/output/dlmpt_cocoop/sweep.log"

cd "$PROJ"

# Header
printf "%-22s %5s %5s %5s  %s\n" "DATASET" "N_WAY" "CTX" "SEED" "STATUS" | tee "$LOG_FILE"
printf "%s\n" "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"

TOTAL=$(( ${#DATASETS[@]} * ${#SEEDS[@]} ))
CURRENT=0

for DS in "${DATASETS[@]}"; do
    NW=${DS_NWAY[$DS]}
    NCTX=${DS_NCTX[$DS]}
    
    for S in "${SEEDS[@]}"; do
        CFG="${DS_CFG[$DS]}"
        OUT="output/dlmpt_cocoop/${DS}/seed${S}/lambda${LAMBDA}"
        CMD="CUDA_VISIBLE_DEVICES=0 $PY train.py \
            --root $DATA --seed $S --trainer DLMPTCoCoOpLite \
            --dataset-config-file configs/datasets/${CFG}.yaml \
            --config-file configs/trainers/dlmpt/vit_b16_cocoop.yaml \
            --output-dir $OUT \
            DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
            TRAINER.ATPROMPT.USE_ATPROMPT True \
            TRAINER.DLMPT.LAMBDA $LAMBDA \
            TRAINER.DLMPT.N_WAY $NW \
            TRAINER.DLMPT.N_EPISODES 100 \
            TRAINER.DLMPT.K_SUPPORT 3 TRAINER.DLMPT.K_QUERY 3 \
            TRAINER.DLMPT.COCOOP_N_CTX $NCTX \
            OPTIM.MAX_EPOCH 25 DATALOADER.NUM_WORKERS 0 \
            DATALOADER.TRAIN_X.BATCH_SIZE 4"
        
        CURRENT=$((CURRENT + 1))
        PAD_DS=$(printf "%-22s" "$DS")
        echo -n "[$CURRENT/$TOTAL] $PAD_DS  NW=$NW CTX=$NCTX seed=$S ... " | tee -a "$LOG_FILE"
        
        if $DRY_RUN; then
            echo "DRY-RUN"
        else
            mkdir -p "$OUT"
            if eval "$CMD" > "$OUT/train_stdout.log" 2>&1; then
                echo "OK" | tee -a "$LOG_FILE"
            else
                echo "FAIL" | tee -a "$LOG_FILE"
            fi
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
OK_COUNT=$(grep -c "OK$" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Done: $OK_COUNT/$TOTAL OK | $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
