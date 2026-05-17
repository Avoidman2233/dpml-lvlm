# DL-MPT-Adapter-B Learnings

## 2026-05-16 Plan Execution Complete

### Key Learnings
1. **Subagent scope creep**: Subagents tend to modify files beyond their assigned scope. Always verify git diff after each delegation and revert unintended changes immediately.
2. **Python environment**: The project requires `/home/avoidman2233/miniconda3/envs/atprompt/bin/python` — default `python` doesn't have torch installed.
3. **CrossAttentionAdapter shape**: `kv.expand(vis_patches.size(0), -1, -1)` correctly broadcasts single-class text anchor to batch dimension for MultiheadAttention.
4. **Patch token extraction**: `_extract_patch_tokens` reimplements ViT forward up to transformer output, excluding `ln_post` and `proj`. Works because backbone is frozen (`@torch.no_grad()`).
5. **Git isolation**: Original files (`dlmpt_trainer.py`, `dlmpt_cocoop_lite.py`, `configs/trainers/dlmpt/`, `scripts/dlmpt/`) must remain untouched. Verified via `git diff --name-only HEAD`.

### Decisions
- Adapter trained jointly with prompt (single-phase), not separate phases
- Only visual-side adapter, text prototypes unchanged
- No explicit alignment loss — L_meta (cross-entropy) is sufficient
- Inherited from TrainerX directly, not DLMPTTrainer (complete isolation)

### Gotchas
- `register_model` name changed to `"prompt_learner+adapter"` to reflect both trainable modules
- Optimizer receives combined params: `list(self.model.prompt_learner.parameters()) + list(self.adapter.parameters())`
- Progress log prefix: `[DL-MPT-Adapter-B]` (not `[DL-MPT]`)
