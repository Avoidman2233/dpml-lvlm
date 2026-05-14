#!/bin/bash
PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
cd /home/avoidman2233/Desktop/LVLM/ATPrompt
$PY train_meta_pipeline.py --root /home/avoidman2233/Desktop/LVLM/DATA --datasets "$@"
