#!/bin/bash
set -euo pipefail

PYTHON="/home/avoidman2233/miniconda3/envs/atprompt/bin/python"
DATA_DIR="/home/avoidman2233/Desktop/LVLM/DATA"

# Defaults
CHECKPOINT=""
DATASET=""
SEEDS="1 2 3"
N_EPISODES=200
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --seeds)      SEEDS="$2";      shift 2 ;;
        --n-episodes) N_EPISODES="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$DATASET" ]]; then
    echo "Usage: $0 --checkpoint <path> --dataset <name> [--seeds \"1 2 3\"] [--n-episodes 200] [--dry-run]"
    exit 1
fi

K_VALUES=(1 3 5 10)

for SEED in $SEEDS; do
    for K in "${K_VALUES[@]}"; do
        OUT_DIR="output/maml_fewshot/${DATASET}/seed${SEED}/kshot${K}"
        CMD="${PYTHON} -c \"from trainers.meta_tester import MetaTester; acc = MetaTester.evaluate('${CHECKPOINT}', '${DATASET}', '${DATA_DIR}', n_way=min(${K}, num_new_classes), k_support=${K}, k_query=10, n_episodes=${N_EPISODES}, inner_steps=2, seed=${SEED}); print(f'K=${K}: {acc}')\""

        if $DRY_RUN; then
            echo "mkdir -p ${OUT_DIR}"
            echo "${CMD} > ${OUT_DIR}/accuracy.txt"
        else
            mkdir -p "${OUT_DIR}"
            echo "Running K=${K}, seed=${SEED}..."
            eval "${CMD}" > "${OUT_DIR}/accuracy.txt"
            echo "  -> saved to ${OUT_DIR}/accuracy.txt"
        fi
    done
done
