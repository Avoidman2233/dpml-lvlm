# Experiment Design: Meta-Pretraining for ATPrompt

## Pipeline Overview

Three-phase experimental protocol:

1. **Meta-Pretrain**: MAML episodes on base classes (single-dataset or cross-dataset)
2. **Standard Fine-tune**: Load meta checkpoint, fine-tune on target base classes
3. **Evaluation**: Test on held-out new classes

## Baselines

| ID | Method | Description |
|----|--------|-------------|
| B1 | CoOp_ATP | Standard prompt learning with attributes, no meta |
| B2 | CoOp (vanilla) | Prompt learning without ATP, no meta |
| B3 | Zero-shot CLIP | CLIP zero-shot classification |
| B4 | CoCoOp_ATP | Conditional prompt learning with ATP |
| B5 | MaPLe_ATP | Multi-modal prompt learning with ATP |
| B6 | DePT_ATP | Decomposed prompt learning with ATP |

## Methods Under Test

| ID | Method | Description |
|----|--------|-------------|
| M1 | Meta-only | Meta-pretrained only, skip fine-tune, eval directly |
| M2 | Meta + Fine-tune | Full pipeline: meta pretrain then standard fine-tune |
| M3 | Meta-FOMAML + Fine-tune | First-order MAML ablation, then fine-tune |

## Ablation Studies

| ID | Parameter | Values |
|----|-----------|--------|
| A1 | Inner loop steps | 1, 3, 5, 10 |
| A2 | N-way | 3, 5, 10 |
| A3 | K-support shots | 1, 3, 5 |
| A4 | Meta-training epochs | 5, 10, 20, 50 |
| A5 | Gradient order | Second-order vs First-order (FOMAML) |
| A6 | Attribute usage | With ATP vs Without ATP |
| A7 | Prompt method | CoOp vs CoCoOp vs MaPLe vs DePT |
| A8 | Data scope | Single-dataset vs Cross-dataset meta-training |

## Datasets

Training on base split (16-shot), evaluation on new split (zero-shot from learned prompts):

- caltech101, oxford_pets, stanford_cars, oxford_flowers, food101
- fgvc_aircraft, sun397, dtd, eurosat, ucf101
- imagenet, imagenet_sketch, imagenetv2, imagenet_a, imagenet_r

## Metrics

- **Base Accuracy**: Accuracy on base classes after fine-tuning
- **New Accuracy**: Accuracy on new classes (zero-shot)
- **Harmonic Mean**: H = 2 * base * new / (base + new)

## Result Table

| Method | Dataset | Base Acc (%) | New Acc (%) | H-Mean (%) |
|--------|---------|-------------|-------------|------------|
| CoOp_ATP (baseline) | caltech101 | - | - | - |
| CoOp_ATP (baseline) | oxford_pets | - | - | - |
| CoOp_ATP (baseline) | stanford_cars | - | - | - |
| CoOp_ATP (baseline) | oxford_flowers | - | - | - |
| CoOp_ATP (baseline) | food101 | - | - | - |
| CoOp_ATP (baseline) | fgvc_aircraft | - | - | - |
| CoOp_ATP (baseline) | sun397 | - | - | - |
| CoOp_ATP (baseline) | dtd | - | - | - |
| CoOp_ATP (baseline) | eurosat | - | - | - |
| CoOp_ATP (baseline) | ucf101 | - | - | - |
| Meta + Fine-tune | caltech101 | - | - | - |
| Meta + Fine-tune | oxford_pets | - | - | - |
| Meta + Fine-tune | stanford_cars | - | - | - |
| Meta + Fine-tune | oxford_flowers | - | - | - |
| Meta + Fine-tune | food101 | - | - | - |
| Meta + Fine-tune | fgvc_aircraft | - | - | - |
| Meta + Fine-tune | sun397 | - | - | - |
| Meta + Fine-tune | dtd | - | - | - |
| Meta + Fine-tune | eurosat | - | - | - |
| Meta + Fine-tune | ucf101 | - | - | - |

## Reproducibility

- **Seeds**: 1, 2, 3 (report mean ± std)
- **Backbone**: ViT-B/16 CLIP (frozen)
- **Hardware**: Single NVIDIA GPU (24GB+ VRAM recommended)
- **Environment**: PyTorch 1.x+, CUDA 11.x, Dassl.pytorch
- **Meta-training time**: ~2-4 hours per dataset on single GPU
- **Fine-tuning time**: Same as standard CoOp_ATP (~1 hour per dataset)

## Running Experiments

```bash
# Phase 1: Meta-pretraining
bash scripts/meta/meta_pretrain.sh caltech101

# Phase 2: Fine-tuning
bash scripts/meta/meta_finetune.sh caltech101

# Phase 3: Evaluation
bash scripts/meta/meta_eval.sh caltech101
```
