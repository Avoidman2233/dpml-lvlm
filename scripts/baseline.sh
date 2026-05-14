#!/bin/bash
PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
cd /home/avoidman2233/Desktop/LVLM/ATPrompt
DATASETS=${@:-eurosat dtd oxford_pets oxford_flowers fgvc_aircraft ucf101 food101 stanford_cars}
for DS in $DATASETS; do
    echo "=== $DS ==="
    for S in 1 2 3; do
        $PY train.py --trainer CoOp_ATP --root /home/avoidman2233/Desktop/LVLM/DATA \
            --seed $S --dataset-config-file configs/datasets/${DS}.yaml \
            --config-file configs/trainers/CoOp/vit_b16.yaml \
            --output-dir output/baseline/${DS}/seed${S} \
            DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
            TRAINER.ATPROMPT.USE_ATPROMPT True \
            OPTIM.MAX_EPOCH 100 DATALOADER.NUM_WORKERS 4
    done
done
