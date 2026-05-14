# Learnings

## 2026-05-13: FT Epoch Sweep Scripts

### Files created
- `scripts/sweep/run_sweep.sh` — Bash orchestrator for FT epoch sweeps. Supports `--dataset`, `--ft-epochs`, `--seeds`, `--method` (baseline/maml), `--dry-run`.
- `scripts/sweep/aggregate.py` — Python script that walks sweep output dirs, finds `accuracy: XX.XX%` in log files, and writes CSV with columns `dataset, seed, method, ft_epoch, accuracy`.

### File modified
- `train_meta_pipeline.py` — Added `--ft-epochs` (int, default None) and `--dataset` (str, single-dataset alias for `--datasets`) argparse arguments. `phase_ft()` and `run_dataset()` accept `ft_epochs` param; when provided, it overrides `OPTIM.MAX_EPOCH` and the eval checkpoint epoch (`fep`).

### Key outputs
- Baseline export path: `output/sweep/{dataset}/seed{seed}/baseline_ft{epoch}`
- MAML export path: `output/sweep/{dataset}/seed{seed}/maml_ft{epoch}`
- Aggregate: `output/sweep/{dataset}_summary.csv`

### Verification
- `bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "5 10" --seeds "1" --dry-run` produces correct commands for both baseline and maml methods.
- All files pass Python AST parsing and bash syntax checks.

## 2026-05-14: Integration Test (Stanford Cars)

### Results
- Meta-pretraining: 3 epochs, stable loss 1.43-1.84, no NaN/Inf, test accuracy 64.27%
- FT: 5 epochs, stable loss 1.31-2.04, no NaN/Inf, test accuracy 69.37%
- All stability improvements verified: gradient clipping, K_SUPPORT=3, warmup=3

### Bugs discovered and fixed during test
1. `train.py` extend_cfg() missing `cfg.TRAINER.META.GRAD_ACCUM = 2` — caused KeyError when loading meta config
2. `trainers/meta_pretrainer.py` line 144 used `cfg` instead of `self.cfg` in build_data_loader() — caused NameError

### Commands used
- Meta: `python train.py --trainer MetaPretrainer ... OPTIM.MAX_EPOCH 3 TRAINER.META.N_EPISODES 50`
- FT: `python train.py --trainer CoOp_ATP ... --resume <meta_ckpt> OPTIM.MAX_EPOCH 5`
- Important: `--resume` must come BEFORE config overrides (opts) in argparse

### Evidence saved to
- `.sisyphus/evidence/task-10-integration.log`
