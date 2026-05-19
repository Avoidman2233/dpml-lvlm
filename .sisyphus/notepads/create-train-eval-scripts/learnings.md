
## Task 11: Create train.sh/eval.sh for 3 DL-MPT variants

Created 6 scripts following the fomaml pattern (hardcoded PYTHON, > train.log 2>&1, no tee).
Each variant uses its own trainer name and config path:
- dlmpt_adaptive: DLMPT_Adaptive_CoOp_ATP → configs/trainers/dlmpt_adaptive/vit_b16.yaml
- dlmpt_attrsample: DLMPT_AttrSample_CoOp_ATP → configs/trainers/dlmpt_attrsample/vit_b16.yaml
- dlmpt_align: DLMPT_Align_CoOp_ATP → configs/trainers/dlmpt_align/vit_b16.yaml

Key differences from the old dlmpt scripts:
- No --lambda/--n-way args (these aren't episodic-regularization variants)
- PYTHON is hardcoded absolute path (fomaml style) not relative PROJ/DATA pattern
- Log output uses `> train.log 2>&1` (redirect) not `tee`
- All 6 scripts pass bash -n syntax check
