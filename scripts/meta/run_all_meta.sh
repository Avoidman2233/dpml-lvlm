#!/bin/bash
# ============================================================
# Meta-Pretrain: All 7 datasets, 3 seeds parallel each
# Run: bash scripts/meta/run_all_meta.sh
# ============================================================
cd /home/avoidman2233/Desktop/LVLM/ATPrompt
AT=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
ROOT=/home/avoidman2233/Desktop/LVLM/DATA

declare -A NW
NW[dtd]=15
NW[oxford_pets]=10
NW[oxford_flowers]=30
NW[fgvc_aircraft]=30
NW[ucf101]=30
NW[food101]=30
NW[stanford_cars]=40

DATASETS="dtd oxford_pets oxford_flowers fgvc_aircraft ucf101 food101 stanford_cars"
TOTAL=0; for DS in $DATASETS; do TOTAL=$((TOTAL+1)); done
CURRENT=0

echo "========================================="
echo " Meta-Pretrain: $TOTAL datasets"
echo " Start: $(date)"
echo "========================================="

for DS in $DATASETS; do
    CURRENT=$((CURRENT+1))
    NWV=${NW[$DS]}
    echo ""
    echo "========== [$CURRENT/$TOTAL] $DS (N_WAY=$NWV) =========="

    mkdir -p output/meta_v4/${DS}
    PIDS=()
    for S in 1 2 3; do
        mkdir -p output/meta_v4/${DS}/seed${S}
        $AT train.py \
            --trainer MetaPretrainer --root $ROOT --seed $S \
            --dataset-config-file configs/datasets/${DS}.yaml \
            --config-file configs/trainers/meta/vit_b16.yaml \
            --output-dir output/meta_v4/${DS}/seed${S} \
            DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
            TRAINER.META.N_WAY $NWV TRAINER.META.K_SUPPORT 1 \
            TRAINER.META.K_QUERY 15 TRAINER.META.INNER_STEPS 2 \
            TRAINER.META.N_EPISODES 200 OPTIM.MAX_EPOCH 20 \
            TRAIN.PRINT_FREQ 20 DATALOADER.NUM_WORKERS 4 \
            2>&1 | sed "s/^/[S${S}] /" &
        PIDS+=($!)
    done

    echo "  3 seeds running: PIDs ${PIDS[*]}"
    echo "  Waiting..."
    for PID in ${PIDS[@]}; do wait $PID; done
    echo ""

    OK=0
    for S in 1 2 3; do
        CK=output/meta_v4/${DS}/seed${S}/prompt_learner/model.pth.tar-20
        if [ -f "$CK" ]; then echo "  $DS seed$S ✅"; OK=$((OK+1))
        else echo "  $DS seed$S ❌ MISSING"; fi
    done
    echo "  $DS: $OK/3 complete"
done

echo ""
echo "========== ALL DONE — $(date) =========="
for DS in $DATASETS; do
    OK=0; for S in 1 2 3; do
        [ -f output/meta_v4/${DS}/seed${S}/prompt_learner/model.pth.tar-20 ] && OK=$((OK+1))
    done
    echo "  $DS: $OK/3"
done
