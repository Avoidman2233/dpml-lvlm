# DL-MPT-Adapter-A(CoOp+ATP) — Gated MLP Adapter 变体实现

## TL;DR

> **Quick Summary**: 方案A — 用轻量 Gated MLP Adapter（~490K params）替换方案B的 Cross-Attention Adapter（3.15M params）。Bottleneck MLP + text-guided gating + residual，更适配 few-shot 场景。

> **Deliverables**:
> - `trainers/dlmpt_adapter_a_trainer.py` — GatedMLPAdapter + DLMPTAdapterATrainer
> - `configs/trainers/dlmpt-adapter-a/vit_b16.yaml`
> - `scripts/dlmpt-adapter-a/train_dlm_coop_atp.sh`, `eval_novel.sh`, `eval_episodic.sh`, `sweep.sh`
> - `train.py` +1 import, `trainers/__init__.py` +1 import

> **与方案B的唯一差异**: Adapter 类不同（GatedMLPAdapter vs CrossAttentionAdapter），其余全部复刻。

---

## Architecture: GatedMLPAdapter

```
vis_patches (B, 196, 768)                    text_anchor (512,)
        │                                          │
   LayerNorm(768)                            Linear(512→768)
        │                                     sigmoid
   Bottleneck MLP:                               │
   768→64→ReLU→768                        gate_weights (768,)
        │                                          │
        └──── element-wise × ──────────────────────┘
                     │
              + residual (vis_patches)
                     │
              mean pool → (B, 768)
                     │
              Linear(768→512) → (B, 512)
                     │
              L2 normalize → adapted prototype
```

**参数量**: ~490K (vs 3.15M for cross-attention)

---

## Implementation Plan

### Task 1: Core Trainer
- Copy `trainers/dlmpt_adapter_b_trainer.py` → `trainers/dlmpt_adapter_a_trainer.py`
- Replace `CrossAttentionAdapter` → `GatedMLPAdapter` (as defined above)
- Replace `DLMPTAdapterBTrainer` → `DLMPTAdapterATrainer`
- Replace `@TRAINER_REGISTRY.register()` name to `DLMPTAdapterATrainer`
- Replace `CrossAttentionAdapter()` → `GatedMLPAdapter()` in build_model
- Replace `[DL-MPT-Adapter-B]` → `[DL-MPT-Adapter-A]` in log prefix
- Keep `register_model("prompt_learner", ...)` (same as fixed B)

### Task 2: Config
- Copy `configs/trainers/dlmpt-adapter-b/vit_b16.yaml` → `configs/trainers/dlmpt-adapter-a/vit_b16.yaml`
- Change TRAINER.NAME to `DLMPTAdapterATrainer`
- Change DLMPT_ADAPTER_B → DLMPT_ADAPTER_A
- Add BOTTLENECK_DIM: 64

### Task 3: Scripts
- Copy `scripts/dlmpt-adapter-b/` → `scripts/dlmpt-adapter-a/`
- Replace all `dlmpt-adapter-b` → `dlmpt-adapter-a` in paths
- Replace `DLMPTAdapterBTrainer` → `DLMPTAdapterATrainer`

### Task 4: Registration
- train.py: add `import trainers.dlmpt_adapter_a_trainer`
- trainers/__init__.py: add `import trainers.dlmpt_adapter_a_trainer`

### Task 5: Verification
- Import + shape test
- 1-epoch Stanford Cars seed1 smoke test