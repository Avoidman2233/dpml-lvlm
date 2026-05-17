# Draft: DL-MPT + MaPLe + ATP Implementation

## Requirements (confirmed from handoff)
- Create DL-MPT(MapLe+ATP) trainer: `trainers/dlmpt_maple_lite.py`
  - Inherit from `MaPLe_ATP` (not CoCoOp_ATP)
  - Copy pattern from `dlmpt_cocoop_lite.py` (190 lines)
  - `_proto_meta_loss()` identical to CoOp version (MaPLe.forward(image) returns logits — same API as CoOp)
- Create config: `configs/trainers/dlmpt/vit_b16_maple.yaml`
  - Copy from `vit_b16_cocoop.yaml`, change MaPLe-specific params (N_CTX=2, DEPTH=9)
- Create training script: `scripts/dlmpt/train_maple.sh`
- Create Protocol A eval script: `scripts/dlmpt/eval_maple_novel.sh`
- Protocol B eval: reuse `scripts/dlmpt/eval_episodic.py`

## Technical Decisions
- MaPLe uses deep prompts (PROMPT_DEPTH=9) but forward is same as CoOp — just call `self.model(img)` to get logits
- MaPLe has NO Meta-Net → no per-image conditional text — `_proto_meta_loss` can use simple prompt_learner() call
- Use "lite" pattern (inherit from MaPLe_ATP, override `run_epoch` with dual-loop)
- Suppress `print(prompts)` noise with sys.stdout redirect pattern
- Proto loop: use prompt_learner() for ALL classes (like CoOp version), not per-class construction — simpler since MaPLe has no meta_net
- Output: `> log 2>&1` NOT `| tee` (P3 rule)

## Research Findings
- `trainers/dlmpt_cocoop_lite.py` (190 lines): inherits CoCoOp_ATP, overrides __init__, build_data_loader, run_epoch, _cocoop_forward, _proto_meta_loss
- `trainers/dlmpt_trainer.py` (328 lines): standalone implementation with both CoOp and CoCoOp modes, has _proto_meta_loss_coop() that calls prompt_learner() for ALL classes
- `trainers/maple_atp.py` (481 lines): MaPLe with ATP, has MultiModalPromptLearner, MaPLe_ATP class with normal TrainerX.forward_model() that calls model(image)
- Config for CoCoOp: vit_b16_cocoop.yaml uses DLMPTTrainer with COCOOP=True
- Config for MaPLe base: vit_b16_c2_ep5_batch4_2ctx.yaml with N_CTX=2, PROMPT_DEPTH=9
- Train script for CoCoOp: train_cocoop.sh uses trainer=DLMPTCoCoOpLite, batch_size=4, num_workers=0
- Key pitfall: MaPLe's TextEncoder takes 3 args (prompts, tokenized_prompts, compound_prompts_deeper_text) — different from CoOp/CoCoOp
- For proto loss: MaPLe prompt_learner() returns ALL-class prompts → can use the CoOp _proto_meta_loss approach

## Scope Boundaries
- INCLUDE: trainer, config, train script, Protocol A eval script
- EXCLUDE: No changes to dlmpt_trainer.py, no sweep scripts, no cross-dataset training
- First target: stanford_cars dataset, single seed (seed 1)

## Test Strategy
- No test framework exists (ML research project)
- Agent-Executed QA ONLY
- QA: dry-run script, verify import, verify structure matches pattern

## Open Questions
- MaPLe_ATP.forward_model() uses model(image) which calls forward() — need to verify return shape
- MaPLe's text encoder needs compound_prompts_deeper_text — need to check if prompt_learner returns required structure
- Need to check if MaPLe_ATP has parse_batch_train (for base loss)
