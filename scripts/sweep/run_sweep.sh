#!/bin/bash
# FT epoch sweep orchestration script.
# Usage:
#   bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "5 10 20 50 100"
#   bash scripts/sweep/run_sweep.sh --dataset eurosat --ft-epochs "10 20" --seeds "1 2" --method maml --dry-run

PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
PROJECT_ROOT=/home/avoidman2233/Desktop/LVLM/ATPrompt
DATA_ROOT=/home/avoidman2233/Desktop/LVLM/DATA

DATASET=""
FT_EPOCHS=""
SEEDS="1 2 3"
METHOD="baseline"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)   DATASET="$2";   shift 2 ;;
        --ft-epochs) FT_EPOCHS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2";     shift 2 ;;
        --method)    METHOD="$2";    shift 2 ;;
        --dry-run)   DRY_RUN=true;   shift ;;
        *) echo "Error: Unknown option $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$FT_EPOCHS" ]]; then
    echo "Error: --dataset and --ft-epochs are required"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

for EPOCH in $FT_EPOCHS; do
    for SEED in $SEEDS; do
        if [[ "$METHOD" == "baseline" ]]; then
            CMD="$PY train.py --trainer CoOp_ATP --root $DATA_ROOT \
                --seed $SEED --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoOp/vit_b16.yaml \
                --output-dir output/sweep/${DATASET}/seed${SEED}/baseline_ft${EPOCH} \
                DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                OPTIM.MAX_EPOCH ${EPOCH} DATALOADER.NUM_WORKERS 4"
        elif [[ "$METHOD" == "maml" ]]; then
            CMD="$PY train_meta_pipeline.py --root $DATA_ROOT \
                --dataset ${DATASET} --seeds ${SEED} --ft-epochs ${EPOCH} \
                --output-dir output/sweep/${DATASET}/seed${SEED}/maml_ft${EPOCH}"
        else
            echo "Error: Unknown method '$METHOD'. Use 'baseline' or 'maml'."
            exit 1
        fi

        if $DRY_RUN; then
            echo "$CMD"
        else
            echo "Running: $CMD"
            eval "$CMD"
        fi
    done
done
