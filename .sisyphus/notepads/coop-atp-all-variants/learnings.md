
## Task 1: Create BaseCoOpATP (base_coop_atp.py)

**Date**: 2026-05-18

### What was done
- Created `trainers/base_coop_atp.py` with `BaseCoOpATP(TrainerX)` base class.
- Extracted common logic from `coop_atp.py` and `dlmpt_trainer.py`.

### Key design decisions
- **Imports over copy**: Instead of copying `PromptLearner`, `CustomCLIP`, `TextEncoder`, `load_clip_to_cpu`, and `CUSTOM_TEMPLATES` into the base class, we import them from `trainers.coop_atp`. This avoids code duplication and ensures any fixes to the original classes propagate automatically.
- **No TRAINER_REGISTRY registration**: The base class is intentionally NOT decorated with `@TRAINER_REGISTRY.register()` — only leaf trainers register.
- **Abstract methods**: `run_epoch()` and `forward_backward()` raise `NotImplementedError` to force subclasses to implement their own training loops.
- **DL-MPT config namespace**: All variants share the `cfg.TRAINER.DLMPT.*` hyperparameter namespace (lambda, n_way, k_support, k_query, n_episodes, warmup_epochs, refine_epochs, refine_lambda, cocoop flag).
- **CoOp / CoCoOp dual support**: `build_model()` branches on `self.use_cocoop` and imports the appropriate `load_clip_to_cpu` + `CustomCLIP` from `trainers.cocoop_atp` or `trainers.coop_atp`.
- **Image caching**: `build_data_loader()` preloads the entire training set to GPU (same as `dlmpt_trainer.py`) to avoid repeated disk I/O during episodic sampling.
- **Checkpoint loading**: `load_model()` deletes fixed token vectors (`token_prefix`, `token_suffix`, `token_middle1/2/3`) before loading state dict, matching `coop_atp.py` behavior.
- **Episodic helper**: `_construct_episode()` wraps `self.episodic_sampler` iterator management so subclasses can simply call it without worrying about `StopIteration`.

### Notes on task requirements vs. codebase reality
- `parse_model` does not exist anywhere in the Dassl.pytorch framework or this project. We omitted it since there is no such method to implement.
- `before_train`, `after_train`, `save_model`, and `model_inference` are fully functional via inheritance from `TrainerX` / `TrainerBase`. We explicitly defined `model_inference` and `parse_batch_train` in the base class for clarity; the others are inherited.

### Verification
- `python -c "import trainers.base_coop_atp; print('OK')"` succeeds in the `atprompt` conda environment.
- `BaseCoOpATP` is confirmed NOT present in `TRAINER_REGISTRY`.
- All required methods (`check_cfg`, `build_model`, `load_model`, `model_inference`, `build_data_loader`, `before_train`, `after_train`, `save_model`, `run_epoch`, `forward_backward`, `_construct_episode`, `current_lambda`, `_model`) are present.

## Task 2: Create episodic_utils.py and update train.py

**Date**: 2026-05-18

### What was done
- Created `trainers/episodic_utils.py` with `EpisodicTaskSampler`, `attribute_similarity_matrix`, and `sample_hard_episode`.
- Added 7 new imports to `train.py` (base_coop_atp + 6 variant trainers) after the existing `trainers.dlmpt_trainer` import.
- Added 5 new config blocks to `extend_cfg()`: FOMAML, REPTILE, ADAPTIVE_DELTA, ATTR_SAMPLE, ATTR_ALIGN (DL-MPT-Full has no dedicated config block since it uses DLMPT namespace + all other configs).

### Key design decisions
- **No DL-MPT-Full config block**: The full variant combines all other variants' features, so it doesn't need its own config namespace — it reads from FOMAML, REPTILE, ADAPTIVE_DELTA, ATTR_SAMPLE, ATTR_ALIGN directly.
- **Existing pattern preserved**: Config headers follow the existing `# Config for X` comment style used throughout `extend_cfg()`.
- **Import order**: base_coop_atp imported first (dependency for all variants), followed by the 6 variant trainers.
- **Notepad file append**: Always use `edit` with oldString matching the last line to append, not `write` to avoid overwriting.

### Verification
- `conda run -n atprompt python -c "from trainers.episodic_utils import EpisodicTaskSampler; print('OK')"` → OK
- Both files pass `python -m py_compile` syntax check.
- All 7 imports confirmed present in `train.py` (lines 36-42).
- All 5 new config blocks confirmed present in `train.py` (lines 187-211).
- `from train import extend_cfg` verification requires Tasks 3-8 to complete first (module-level imports in train.py fail when trainer modules don't exist). This is expected — will pass once variant trainers are created.
