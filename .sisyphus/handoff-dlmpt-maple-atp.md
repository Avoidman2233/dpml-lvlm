## DL-MPT Method Summary

**命名约定**: 我们的基线方法均为含属性增强的版本。DL-MPT 注入 CoOp+ATP / CoCoOp+ATP / MaPLe+ATP 中。

| 正确名称 | 含义 |
|---------|------|
| `DL-MPT(CoOp+ATP)` | DL-MPT based on CoOp with ATPrompt |
| `DL-MPT(CoCoOp+ATP)` | DL-MPT based on CoCoOp with ATPrompt |
| `DL-MPT(MaPLe+ATP)` | DL-MPT based on MaPLe with ATPrompt |

DL-MPT (Dual-loop Meta Prompt Tuning) adds an episodic proto loss to standard CLIP prompt learning:

```
L_total = L_base + λ * L_meta

L_base:  Standard cross-entropy classification (CoOp / CoCoOp / MaPLe forward)
L_meta:  N-way K-shot episodic prototype loss (cosine similarity on fused prototypes)

Three training phases:
  Warmup (λ=0, epoch 0-4):     Only L_base, stabilize prompt
  Joint  (λ=0.2, epoch 5-19):  Dual-loop training
  Refine (λ=0.5, epoch 20-25): Increase meta weight
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `trainers/dlmpt_trainer.py` | DL-MPT for CoOp mode (works, +4.88% on Cars) | 328 |
| `trainers/dlmpt_cocoop_lite.py` | DL-MPT for CoCoOp mode (adapted, see results) | 190 |
| `trainers/maple_atp.py` | MaPLe+ATP reference implementation | 481 |
| `configs/trainers/dlmpt/vit_b16.yaml` | CoOp config | |
| `configs/trainers/dlmpt/vit_b16_cocoop.yaml` | CoCoOp config | |
| `scripts/dlmpt/train.sh` | CoOp training | |
| `scripts/dlmpt/train_cocoop.sh` | CoCoOp training | |
| `scripts/dlmpt/sweep_cocoop.sh` | CoCoOp 8-dataset sweep | |
| `scripts/dlmpt/eval_novel.sh` | Protocol A (zero-shot) | |
| `scripts/dlmpt/eval_episodic.py` | Protocol B (episodic) | |

## Results Summary

### DL-MPT(CoOp+ATP) — Stanford Cars only
| Protocol | Result | vs CoOp+ATP |
|----------|:---:|:---:|
| A: Novel Zero-shot | **71.43%** | +4.88% |
| B: Episodic K=1/3/5 | 94.7/98.1/98.7 | competitive |

### DL-MPT(CoCoOp+ATP) — 8 datasets
| Dataset | Ours | vs CoCoOp+ATP (paper) |
|---------|:---:|:---:|
| stanford_cars | 73.63 | +0.16 |
| fgvc_aircraft | 34.03 | +0.88 |
| oxford_pets | 96.85 | -1.04 |
| food101 | 90.94 | -0.22 |
| ucf101 | 71.50 | -1.65 |
| oxford_flowers | 71.39 | -2.20 |
| dtd | 50.00 | -6.89 |
| eurosat | 52.72 | -16.78 |

**Why CoCoOp didn't gain**: CoCoOp's Meta-Net already provides instance-conditional regularization → episodic regularization is redundant.

## Next: DL-MPT + MaPLe + ATP

### Why MaPLe
- MaPLe has NO Meta-Net → episodic regularization should provide NEW value
- MaPLe uses deep prompts (multiple layers) but forward is same as CoOp
- Paper MaPLe+ATP Novel on Cars: 73.84% (target to beat)

### Implementation Plan

**Step 1**: Create `trainers/dlmpt_maple_lite.py`
- Copy pattern from `dlmpt_cocoop_lite.py` (190 lines)
- Import from `trainers.maple_atp` instead of `trainers.cocoop_atp`
- Inherit from `MaPLe_ATP` instead of `CoCoOp_ATP`

**Step 2**: The `_proto_meta_loss()` can be IDENTICAL to the CoOp version
- MaPLe.forward(image) returns logits — same API as CoOp
- No per-image conditional text needed (unlike CoCoOp)
- Use the same `self.model.prompt_learner()` and `self.model.text_encoder()`

**Step 3**: The MaPLe PromptLearner generates prompts for ALL classes
- proto loop may need the 98-class text_encoder → memory concern
- But we can use the ClassNamePrompt construction approach (same as CoCoOp fix)

**Step 4**: Config
- Copy `vit_b16_cocoop.yaml` → `vit_b16_maple.yaml`
- Change: `TRAINER.MAPLE.N_CTX=2`, `TRAINER.MAPLE.DEPTH=9`

### Critical Pitfalls to Avoid

1. **OOM from 98-class text_encoder in proto loop**: For proto, construct 1-class prompts using PromptLearner's token buffers (NOT manual tokenization). See `_proto_meta_loss` in `dlmpt_cocoop_lite.py` for the correct pattern.
2. **Protocol prompt construction BUG**: NEVER manually tokenize "X X attr X X name." and replace positions. The CLIP bpe tokenization places the attribute word token at unexpected positions. ALWAYS use PromptLearner's own `token_prefix`/`token_middle1`/`token_suffix` buffers indexed by class label, concatenated with learned `ctx`/`ctx_att`. Pattern:
```python
prompt = torch.cat([
    pl.token_prefix,                      # SOS token
    pl.ctx_att1.unsqueeze(0),             # learned attr tokens
    pl.token_middle1[class_idx:class_idx+1],  # attribute word
    ctx_instance,                         # instance-conditional ctx  
    pl.token_suffix[class_idx:class_idx+1],   # class name + EOS
], dim=1)
text_feat = text_encoder(prompt, tokenized[class_idx:class_idx+1])
```
3. **`print(prompts)` noise**: Use `sys.stdout` redirect in `__init__` (lines 29-35 of lite version)
3. **`parse_batch_train` returning 3 values**: Override to return 2 (input, label)
4. **Token embedding access**: Load from cached CLIP checkpoint, not from TextEncoder
5. **Config freeze**: `cfg.defrost()` before setting N_CTX, `cfg.freeze()` after
6. **output redirection**: Use `> log 2>&1`, NOT `| tee` (pipe buffer causes GPU memory bloat)

### Key Commands

```bash
# Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 \
    --trainer DLMPTMapleLite \
    --dataset-config-file configs/datasets/stanford_cars.yaml \
    --config-file configs/trainers/dlmpt/vit_b16_maple.yaml \
    --output-dir output/dlmpt_maple/stanford_cars/seed1 \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True TRAINER.DLMPT.LAMBDA 0.2 \
    TRAINER.DLMPT.N_WAY 20 TRAINER.DLMPT.N_EPISODES 100 \
    OPTIM.MAX_EPOCH 25 DATALOADER.NUM_WORKERS 0 \
    DATALOADER.TRAIN_X.BATCH_SIZE 4

# Protocol A eval
python train.py --root DATA --seed 1 --trainer MaPLe_ATP \
    --dataset-config-file configs/datasets/stanford_cars.yaml \
    --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --model-dir output/dlmpt_maple/stanford_cars/seed1 --load-epoch 25 --eval-only \
    DATASET.SUBSAMPLE_CLASSES new TRAINER.ATPROMPT.USE_ATPROMPT True

# Protocol B eval
python scripts/dlmpt/eval_episodic.py \
    --checkpoint output/dlmpt_maple/stanford_cars/seed1 \
    --dataset stanford_cars --seed 1
```

### Environment
- Python: `/home/avoidman2233/miniconda3/envs/atprompt/bin/python`
- PyTorch: 2.1.2, CUDA 11.8
- Data root: `/home/avoidman2233/Desktop/LVLM/DATA`
- Project root: `/home/avoidman2233/Desktop/LVLM/ATPrompt`
