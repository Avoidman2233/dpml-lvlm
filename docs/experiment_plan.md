# 元任务微调实验计划

> **项目**：基于元任务微调与属性辅助推理的 LVLM 小样本学习框架  
> **硬件**：NVIDIA RTX 4060 Ti 16GB  
> **日期**：2026-05-09

---

## 1. 研究问题

| ID | 问题 |
|----|------|
| RQ1 | MAML 元预训练能否提升属性辅助 Prompt Learning 的小样本分类性能？ |
| RQ2 | 元预训练带来的提升在不同数据集上是否一致？ |
| RQ3 | 跨数据集元预训练是否优于单数据集元预训练？ |
| RQ4 | 二阶 MAML 相比一阶近似（FOMAML）是否有显著增益？ |
| RQ5 | 内循环步数、N-way、K-shot 等超参数如何影响最终性能？ |
| RQ6 | 元预训练在不同 Prompt Learning 方法（CoOp/CoCoOp/MaPLe/DePT）上的适用性？ |

---

## 2. 数据集

### 2.1 实验数据集（11 个）

| 数据集 | 类别数 | 领域 | 选择原因 |
|--------|--------|------|----------|
| EuroSAT | 10 | 卫星遥感 | 最小最快，验证用 |
| DTD | 47 | 纹理 | 细粒度，挑战性 |
| Caltech101 | 101 | 通用物体 | 经典 benchmark |
| Oxford Pets | 37 | 宠物 | 细粒度 |
| Oxford Flowers | 102 | 花卉 | 细粒度 |
| Stanford Cars | 196 | 车辆 | 细粒度，类别多 |
| UCF101 | 101 | 动作识别 | 动态场景 |
| Food101 | 101 | 食物 | 细粒度 |
| FGVC Aircraft | 100 | 飞行器 | 细粒度 |
| SUN397 | 397 | 场景 | 类别多，难度大 |
| ImageNet | 1000 | 通用 | 大规模预训练源 |

### 2.2 数据集划分

遵循 CoOp 协议的 **base-to-new** 划分：

```
每个数据集:
├── base classes (50%):  用于训练（meta-pretrain + fine-tune）
└── new classes (50%):   用于评估（zero-shot）
```

### 2.3 跨数据集设置

```
源数据集（meta-training）:
  └── ImageNet base classes (500 类)

目标数据集（fine-tune + eval）:
  └── 其余 10 个数据集
```

---

## 3. Baseline 方法

### 3.1 必选 Baseline

| ID | 方法 | 说明 |
|----|------|------|
| B1 | Zero-shot CLIP | 完全不训练，CLIP 零样本分类 |
| B2 | CoOp (vanilla) | 标准 Context Optimization，无属性，无 meta |
| B3 | CoOp_ATP | 属性辅助 CoOp，无 meta（**主 baseline**）|
| B4 | CoCoOp_ATP | 属性辅助 CoCoOp，无 meta |
| B5 | MaPLe_ATP | 属性辅助 MaPLe，无 meta |

### 3.2 可选 Baseline

| ID | 方法 | 条件 |
|----|------|------|
| B6 | DePT_ATP | 时间允许时 |
| B7 | Linear Probe CLIP | 对比传统 fine-tune |

---

## 4. 实验方法

### 4.1 核心方法

| ID | 方法 | 配置 |
|----|------|------|
| M1 | **Meta-Pretrain + Fine-tune** (主力) | 单数据集 meta → fine-tune |
| M2 | **Cross-Dataset Meta + Fine-tune** | ImageNet meta → 下游 fine-tune |
| M3 | **Meta-Only** (消融) | 仅 meta-pretrain，跳过 fine-tune |

### 4.2 消融变体

| ID | 变体 | 默认 | 对比值 |
|----|------|------|--------|
| A1 | Inner Steps | 5 | 1, 3, **5**, 10 |
| A2 | N-Way | 5 | 3, **5**, 10 |
| A3 | K-Support | 1 | **1**, 3, 5 |
| A4 | Meta Epochs | 20 | 5, 10, **20**, 50 |
| A5 | Second-Order | True | **True**, False (FOMAML) |
| A6 | ATP Usage | True | **True**, False (纯 CoOp meta) |
| A7 | Meta LR | 0.01 | 0.001, 0.005, **0.01**, 0.05 |

---

## 5. 实验矩阵

### 5.1 主实验（RQ1, RQ2）

**固定配置**：16-shot, ViT-B/16, seed × 3

| 方法 | EuroSAT | DTD | Caltech101 | Pets | Flowers | Cars | UCF101 | Food101 | Aircraft | SUN397 |
|------|---------|-----|------------|------|---------|------|--------|---------|----------|--------|
| B1: Zero-shot CLIP | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| B2: CoOp vanilla | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| B3: CoOp_ATP | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| M1: Meta+Fine-tune | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| M3: Meta-Only | ● | ● | ● | - | - | - | - | - | - | - |

**统计**：10 数据集 × 5 方法 × 3 seeds = **150 次实验**

### 5.2 跨数据集实验（RQ3）

| 方法 | EuroSAT | DTD | Caltech101 | Pets | Flowers |
|------|---------|-----|------------|------|---------|
| B3: CoOp_ATP | ● | ● | ● | ● | ● |
| M1: Single-Dataset Meta | ● | ● | ● | ● | ● |
| M2: Cross-Dataset Meta | ● | ● | ● | ● | ● |

**统计**：5 数据集 × 3 方法 × 3 seeds = **45 次实验**

### 5.3 方法泛化实验（RQ6）

在 **3 个代表性数据集** 上测试不同 Prompt Learning 方法：

| Meta Method | EuroSAT | Caltech101 | Stanford Cars |
|-------------|---------|------------|---------------|
| CoOp_ATP | ● | ● | ● |
| CoCoOp_ATP | ● | ● | ● |
| MaPLe_ATP | ● | ● | ● |

**统计**：3 数据集 × 3 方法 × 3 seeds = **27 次实验**

### 5.4 消融实验（RQ4, RQ5）

在 **EuroSAT + Caltech101** 上：

| 消融 | 变量 | 实验次数 |
|------|------|----------|
| A1: Inner Steps | {1,3,5,10} | 4 × 2 × 3 = 24 |
| A2: N-Way | {3,5,10} | 3 × 2 × 3 = 18 |
| A3: K-Support | {1,3,5} | 3 × 2 × 3 = 18 |
| A5: FOMAML | {True, False} | 2 × 2 × 3 = 12 |
| A6: ATP Usage | {True, False} | 2 × 2 × 3 = 12 |

**统计**：约 **84 次实验**

---

## 6. 超参数配置

### 6.1 固定超参数（所有实验）

| 参数 | 值 |
|------|-----|
| Backbone | ViT-B/16 CLIP |
| Input Size | 224 × 224 |
| Precision | fp16 |
| N_CTX (context tokens) | 16 |
| Optimizer | SGD, lr=0.002 |
| LR Scheduler | Cosine |
| Warmup Epochs | 1 |
| Weight Decay | 5e-4 |

### 6.2 Meta-Pretrain 默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| N_WAY | 5 | 每 episode 类别数 |
| K_SUPPORT | 1 | 每类 support 样本数 |
| K_QUERY | 3 | 每类 query 样本数 |
| INNER_LR | 0.01 | 内循环学习率 |
| INNER_STEPS | 5 | 内循环 SGD 步数 |
| N_EPISODES | 100 | 每 epoch 的 episode 数 |
| MAX_EPOCH | 20 | 元训练 epoch 数 |
| SECOND_ORDER | True | 使用二阶梯度 |

### 6.3 Fine-tune 配置

| 参数 | 值 |
|------|-----|
| NUM_SHOTS | 16 |
| MAX_EPOCH | 100 |
| LR | 0.002 |

---

## 7. 评估指标

### 7.1 主要指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Base Accuracy** | - | 在 base 类上的分类准确率 |
| **New Accuracy** | - | 在 new 类上的 zero-shot 准确率 |
| **Harmonic Mean (H)** | H = 2 × Base × New / (Base + New) | 综合衡量 base-new 权衡 |

### 7.2 统计报告

- 所有结果报告 **mean ± std**（基于 3 个 seed）
- 使用 H-Mean 排序方法
- 最优结果 **加粗**，次优结果 _斜体_

### 7.3 显著性检验（可选）

- 对主结果做 paired t-test（p < 0.05）
- 标注统计显著优于 baseline 的结果

---

## 8. 实验结果模板

```markdown
### EuroSAT (10 classes, 16-shot)

| Method | Base Acc (%) | New Acc (%) | H-Mean (%) |
|--------|-------------|-------------|------------|
| Zero-shot CLIP | - | - | - |
| CoOp vanilla | - | - | - |
| CoOp_ATP | - | - | - |
| Meta-Only | - | - | - |
| **Meta+Fine-tune** | **-** | **-** | **-** |
| _Cross-Dataset Meta_ | _-_ | _-_ | _-_ |
```

---

## 9. 运行脚本

### 9.1 单数据集主实验流程

```bash
# 每个数据集的完整流程
DS="eurosat"  # 替换为其他数据集

# Phase 1: Meta-Pretrain
python train.py \
    --trainer MetaPretrainer \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed 1 \
    --dataset-config-file configs/datasets/${DS}.yaml \
    --config-file configs/trainers/meta/vit_b16.yaml \
    --output-dir output/meta_pretrain/${DS}/seed1 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    DATALOADER.NUM_WORKERS 4

# Phase 2: Fine-tune
python train.py \
    --trainer CoOp_ATP \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed 1 \
    --dataset-config-file configs/datasets/${DS}.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/meta_finetune/${DS}/seed1 \
    --model-dir output/meta_pretrain/${DS}/seed1 \
    --load-epoch 20 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True \
    OPTIM.MAX_EPOCH 100

# Phase 3: Evaluate
python train.py \
    --trainer CoOp_ATP \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed 1 \
    --dataset-config-file configs/datasets/${DS}.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/eval_meta/${DS}/seed1 \
    --model-dir output/meta_finetune/${DS}/seed1 \
    --load-epoch 100 \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.ATPROMPT.USE_ATPROMPT True
```

### 9.2 跨数据集实验

```bash
# 1. 在 ImageNet base 类上 meta-training
python train.py \
    --trainer MetaPretrainer \
    --root /home/avoidman2233/Desktop/LVLM/DATA \
    --seed 1 \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/meta/vit_b16.yaml \
    --output-dir output/meta_pretrain/imagenet_cross/seed1 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.META.N_EPISODES 200 \
    OPTIM.MAX_EPOCH 50

# 2. 在下游数据集上 fine-tune
for DS in eurosat dtd caltech101 oxford_pets; do
    python train.py \
        --trainer CoOp_ATP \
        --root /home/avoidman2233/Desktop/LVLM/DATA \
        --seed 1 \
        --dataset-config-file configs/datasets/${DS}.yaml \
        --config-file configs/trainers/CoOp/vit_b16.yaml \
        --output-dir output/cross_finetune/${DS}/seed1 \
        --model-dir output/meta_pretrain/imagenet_cross/seed1 \
        --load-epoch 50 \
        DATASET.NUM_SHOTS 16 \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.ATPROMPT.USE_ATPROMPT True
done
```

### 9.3 消融实验示例

```bash
# 测试不同 inner_steps
for STEPS in 1 3 5 10; do
    python train.py \
        --trainer MetaPretrainer \
        --root /home/avoidman2233/Desktop/LVLM/DATA \
        --seed 1 \
        --dataset-config-file configs/datasets/eurosat.yaml \
        --config-file configs/trainers/meta/vit_b16.yaml \
        --output-dir output/ablation/eurosat/inner_steps_${STEPS} \
        DATASET.NUM_SHOTS 16 \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.META.INNER_STEPS ${STEPS} \
        OPTIM.MAX_EPOCH 5
done
```

---

## 10. 时间估算

| 阶段 | 数据集数 | 每集时间 | 总时间 |
|------|----------|----------|--------|
| Meta-Pretrain（单数据集） | 10 | ~2h | ~20h |
| Fine-tune | 10 | ~1h | ~10h |
| Evaluation | 10 | ~10min | ~2h |
| Baselines (B1-B5) | 10 × 5 | ~0.5-1h | ~30h |
| Ablation (EuroSAT+Caltech101) | 84 | ~0.5h | ~40h |
| Cross-dataset Meta | 1 + 5 | ~3h + 5h | ~8h |
| **总计（全量）** | | | **~110 GPU-hours** |

**精简方案**（优先发表）：
- 5 个代表性数据集 × 主实验 → ~50 GPU-hours
- 可在 4060 Ti 上 3-4 天完成

---

## 11. 预期结果

### 11.1 假设

- **H1**：Meta+Fine-tune 的 H-Mean 显著高于 CoOp_ATP baseline
- **H2**：跨数据集 meta 优于单数据集 meta（更多样化的 episode）
- **H3**：二阶 MAML 优于一阶 FOMAML（差距较小但一致）
- **H4**：内循环步数 3-5 最优（太少欠拟合，太多过拟合）
- **H5**：Meta-pretrain 对 CoCoOp/MaPLe/DePT 同样有效

### 11.2 预期提升幅度

基于文献中 MAML + CLIP 的先例：

| 指标 | 预期提升 |
|------|----------|
| New Accuracy | +1~3% |
| H-Mean | +1~2% |
| 稳定性 (std) | 降低 0.5~1% |

---

## 12. 日程建议

| Day | 任务 |
|-----|------|
| Day 1 | 解压所有数据集，跑 2-3 个 Baselines（B1-B3），确认环境正常 |
| Day 2 | 跑完所有 Baselines（B4-B5）|
| Day 3 | Meta-Pretrain 3 个数据集 (EuroSAT, DTD, Caltech101) |
| Day 4 | Meta-Pretrain 剩余 7 个数据集 + Fine-tune + Eval |
| Day 5 | 消融实验（A1-A3）|
| Day 6 | 消融实验（A4-A7）+ 方法泛化（CoCoOp, MaPLe）|
| Day 7 | 跨数据集实验 + 补充实验 + 整理结果 |

---

## 13. 结果记录位置

- 主实验结果填入 `docs/EXPERIMENT_DESIGN.md` 的结果表格
- 消融结果记录在 `output/ablation/*/log.txt`
- 所有 checkpoint 在 `output/meta_pretrain/` 和 `output/meta_finetune/`
- 评估日志在 `output/eval_meta/`
