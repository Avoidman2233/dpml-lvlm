#!/bin/bash

DATA=../DATA

CFG=vit_b16  # config file
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

TRAINER=AttributeCompute
CTP=end

NCTX=4
SEED=1

for DATASET in  eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101
do
if [ $DATASET = "oxford_pets" ]; then
        ATT1_TEXT=affection
        ATT2_TEXT=energy
        ATT3_TEXT=intelligence
        ATT4_TEXT=gentleness
        ATT5_TEXT=calmness
        echo 'pets'
elif [ $DATASET = "stanford_cars" ]; then
        ATT1_TEXT=engine
        ATT2_TEXT=design
        ATT3_TEXT=interior
        ATT4_TEXT=performance
        ATT5_TEXT=size
        echo 'cars'
elif [ $DATASET = 'oxford_flowers' ]; then
        ATT1_TEXT=color
        ATT2_TEXT=shape
        ATT3_TEXT=fragrance
        ATT4_TEXT=lifespan
        ATT5_TEXT=habitat
        echo 'flowers'
elif [ $DATASET = 'food101' ]; then
        ATT1_TEXT=shape
        ATT2_TEXT=color
        ATT3_TEXT=characteristic
        ATT4_TEXT=habit
        ATT5_TEXT=function
        echo 'food'
elif [ $DATASET = 'fgvc_aircraft' ]; then
        ATT1_TEXT=fuselage
        ATT2_TEXT=engines
        ATT3_TEXT=wings
        ATT4_TEXT=range
        ATT5_TEXT=capacity
        echo 'fgvc'
elif [ $DATASET = 'dtd' ]; then
        ATT1_TEXT=lines
        ATT2_TEXT=spots
        ATT3_TEXT=holes
        ATT4_TEXT=weaves
        ATT5_TEXT=textures
        echo 'dtd'
elif [ $DATASET = 'eurosat' ]; then
        ATT1_TEXT=vegetation
        ATT2_TEXT=water
        ATT3_TEXT=infrastructure
        ATT4_TEXT=color
        ATT5_TEXT=shape
        echo 'eurosat'
elif [ $DATASET = 'ucf101' ]; then
        ATT1_TEXT=movement
        ATT2_TEXT=equipment
        ATT3_TEXT=skill
        ATT4_TEXT=purpose
        ATT5_TEXT=coordination
        echo 'ucf'
else
        echo 'no value'
fi
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/search_attribute

        CUDA_VISIBLE_DEVICES=0 python train_select_attribute.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoOp/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.COOP.N_ATT1 ${NCTX} \
                TRAINER.COOP.N_ATT2 ${NCTX} \
                TRAINER.COOP.N_ATT3 ${NCTX} \
                TRAINER.COOP.N_ATT4 ${NCTX} \
                TRAINER.COOP.N_ATT5 ${NCTX} \
                TRAINER.COOP.ATT1_TEXT ${ATT1_TEXT} \
                TRAINER.COOP.ATT2_TEXT ${ATT2_TEXT} \
                TRAINER.COOP.ATT3_TEXT ${ATT3_TEXT} \
                TRAINER.COOP.ATT4_TEXT ${ATT4_TEXT} \
                TRAINER.COOP.ATT5_TEXT ${ATT5_TEXT} \
                OPTIM.MAX_EPOCH 40
done