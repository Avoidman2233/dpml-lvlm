# DL-MPT: Dual-loop Meta Prompt Tuning

## 面向「基于元任务微调与属性辅助推理的小样本学习」

---

## 一、设计原则

### P1: 脚本即真理

**所有训练、评估流程必须固化为可重复调用的脚本文件。** 禁止 agent 动态生成临时 Python 代码。每次实验的输出路径、超参数、随机种子由脚本固定。

```
scripts/dlmpt/
├── train.sh              # 单次训练入口
├── eval_novel.sh         # Novel class zero-shot 评估
├── eval_episodic.sh      # Episodic K-shot 评估
├── run_sweep.sh          # 批量 sweep（数据集 × λ × seed）
└── aggregate.py          # 结果汇总
```

### P2: 进度可见

每个训练迭代在 **stdout + progress.log** 中同时输出格式化进度信息，格式固定：

```
[DL-MPT] epoch=X/Y batch=Z/W | L_base=0.XXX L_meta=0.XXX L_total=0.XXX | acc_base=XX.X% acc_meta=XX.X% | lr=X.XXe-X | eta=HH:MM:SS
```

禁止使用 `python -c "..."` 临时运行。必须通过固定脚本文件执行。

### P3: 双环联合

训练过程同时存在两条优化路径，**不是串行（Proto→checkpoint→FT），而是并行（每 iter 同时计算两个 loss）**：

```
for each training iteration:
    batch_base = next(base_loader)       # 标准 batch
    episode = sampler.sample(N,K)         # episodic task
    
    loss_base = CE(model(batch_base))     # 基础分类
    loss_meta = ProtoLoss(episode)        # 元任务泛化
    
    loss = loss_base + λ * loss_meta      # 联合
    loss.backward()
```

### P4: 元环用 Proto 范式

**不使用 MAML 风格的 inner-loop SGD。** 我们已验证 MAML 在 CLIP 上失败。元环使用原型网络方式：support 取均值做 prototype，query 做 cosine 分类。零梯度、纯前向。

### P5: λ 可配置

λ（meta regularization coefficient）通过 config 文件控制，不硬编码。支持 sweep。

### P6: 评估独立

训练脚本只负责训练和保存 checkpoint。评估由独立的 eval 脚本完成，确保评估协议不被训练过程污染。

---

## 二、核心方法

### 2.1 DL-MPT 算法

```
Algorithm: Dual-loop Meta Prompt Tuning

Input:  CLIP ViT-B/16 (frozen), prompt_learner (trainable)
        Base class data, N_way, K_support, K_query, λ

For each epoch:
  For each iteration:
    
    # --- Path 1: Base Classification ---
    images_base, labels_base = next(base_dataloader)
    logits_base = model(images_base)           # CoOp_ATP forward
    L_base = CrossEntropy(logits_base, labels_base)
    
    # --- Path 2: Episodic Meta ---
    episode = EpisodicSampler.sample(N_way, K_support, K_query)
    support_img, support_labels, query_img, query_labels = episode
    
    # Proto-style meta (NO inner-loop SGD)
    for each class c:
        visual_proto_c = ImageEncoder(support_img[c]).mean(0)
        text_proto_c = TextEncoder(ctx_att + "{attr}" + ctx + "{class_name}")
        proto_c = (visual_proto_c + text_proto_c) / 2
    
    query_features = ImageEncoder(query_img)
    sim = cosine_similarity(query_features, prototypes)  # (Q, N)
    L_meta = CrossEntropy(sim, query_labels)
    
    # --- Joint Optimization ---
    L_total = L_base + λ * L_meta
    L_total.backward()
    optimizer.step()
```

### 2.2 训练阶段

| 阶段 | Epochs | λ | 说明 |
|------|--------|---|------|
| Warmup | 0-4 | 0.0 | 仅 L_base，稳定 prompt 初始化 |
| Joint | 5-19 | 0.2 | 联合训练，主要阶段 |
| Refine | 20-24 | 0.5 | 提升 meta 权重 |

### 2.3 评估协议

**协议 A：Base-to-Novel Zero-shot**
```
在 base 类上训练 → 在 novel 类上 zero-shot 全量分类
对标论文 CoOp+ATP
启动: bash scripts/dlmpt/eval_novel.sh
```

**协议 B：Episodic K-shot**
```
在 base 类上训练 → 在 novel 类上 N-way K-shot episodic 评估
对标小样本学习的标准协议
启动: bash scripts/dlmpt/eval_episodic.sh
```

---

## 三、实现方案

### 3.1 新增文件

| 文件 | 功能 | 行数估计 |
|------|------|---------|
| `trainers/dlmpt_trainer.py` | DLMPTTrainer 类 | ~250 |
| `configs/trainers/dlmpt/vit_b16.yaml` | DL-MPT 配置 | ~40 |
| `scripts/dlmpt/train.sh` | 训练入口 | ~50 |
| `scripts/dlmpt/eval_novel.sh` | Novel zero-shot 评估 | ~40 |
| `scripts/dlmpt/eval_episodic.sh` | Episodic K-shot 评估 | ~50 |
| `scripts/dlmpt/run_sweep.sh` | 批量 sweep | ~60 |
| `scripts/dlmpt/aggregate.py` | 结果汇总 | ~80 |

### 3.2 DLMPTTrainer 核心设计

```python
@TRAINER_REGISTRY.register()
class DLMPTTrainer(TrainerX):
    """
    Dual-loop Meta Prompt Tuning.
    
    Config keys:
        cfg.TRAINER.DLMPT.LAMBDA: float = 0.2
        cfg.TRAINER.DLMPT.N_WAY: int = 20
        cfg.TRAINER.DLMPT.K_SUPPORT: int = 3
        cfg.TRAINER.DLMPT.K_QUERY: int = 10
        cfg.TRAINER.DLMPT.N_EPISODES: int = 100
        cfg.TRAINER.DLMPT.WARMUP_EPOCHS: int = 5
        cfg.TRAINER.DLMPT.REFINE_EPOCHS: int = 20
        cfg.TRAINER.DLMPT.REFINE_LAMBDA: float = 0.5
    """
    
    def build_data_loader(self):
        # 标准 DataLoader (复用 CoOp_ATP)
        # EpisodicSampler (复用已有)
        
    def build_model(self):
        # 复用 CoOp_ATP 的 CustomCLIP
        
    def run_epoch(self):
        # 双环核心逻辑
        for batch_idx, (base_batch, episode) in enumerate(
                zip(self.train_loader_x, self.episodic_sampler)):
            
            # Path 1: Base classification
            img_base, label_base = base_batch
            output_base = self.model(img_base)
            loss_base = F.cross_entropy(output_base, label_base)
            
            # Path 2: Episodic meta (proto-style)
            loss_meta, acc_meta = self._proto_meta_loss(episode)
            
            # Joint
            loss = loss_base + self.current_lambda * loss_meta
            loss.backward()
            
            # 格式化进度输出
            self._log_progress(batch_idx, loss_base, loss_meta, acc_meta)
    
    def _proto_meta_loss(self, episode):
        """Proto-style meta loop. 零 inner-loop SGD."""
        support_img, support_label, query_img, query_label = episode
        prototypes = self._compute_prototypes(support_img, support_label)
        query_features = F.normalize(self.model.image_encoder(query_img))
        sim = query_features @ prototypes.T
        loss = F.cross_entropy(sim, query_label)
        acc = compute_accuracy(sim, query_label)[0]
        return loss, acc
    
    @property
    def current_lambda(self):
        if self.epoch < self.warmup_epochs:
            return 0.0
        elif self.epoch >= self.refine_epochs:
            return self.refine_lambda
        else:
            return self.lambda_
```

### 3.3 进度输出格式

每个 batch 输出一行到 stdout + progress.log：

```
[DL-MPT] epoch=06/25 batch=032/100 L_base=0.852 L_meta=0.314 L_total=0.915 acc_base=71.2% acc_meta=86.7% λ=0.20 lr=1.23e-03 eta=00:45:12
```

**格式规范**：
- `[DL-MPT]` 固定前缀
- `epoch=XX/YY` epoch 计数（XX 从 1 开始，YY 为 total）
- `batch=XXX/YYY` batch 计数
- `L_base=X.XXX` base loss（3 位小数）
- `L_meta=X.XXX` meta loss（3 位小数）
- `L_total=X.XXX` 总 loss（3 位小数）
- `acc_base=XX.X%` base 分类准确率
- `acc_meta=XX.X%` meta 分类准确率
- `λ=X.XX` 当前 λ 值
- `lr=X.XXe-X` 学习率
- `eta=HH:MM:SS` 预计剩余时间

**输出频率**：每 PRINT_FREQ=20 batch 输出一次，epoch 开始时输出 `[EPOCH X/Y] HH:MM:SS`。

### 3.4 脚本示例

#### scripts/dlmpt/train.sh

```bash
#!/bin/bash
# DL-MPT Training
# Usage: bash scripts/dlmpt/train.sh --dataset stanford_cars --seed 1 --lambda 0.2

PY=/home/avoidman2233/miniconda3/envs/atprompt/bin/python
DATA=/home/avoidman2233/Desktop/LVLM/DATA
PROJ=/home/avoidman2233/Desktop/LVLM/ATPrompt

# Defaults
DATASET="stanford_cars"
SEED=1
LAMBDA=0.2
OUTPUT=""

# Parse args (simplified: use positional or getopt)
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --lambda) LAMBDA="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

[ -z "$OUTPUT" ] && OUTPUT="output/dlmpt/${DATASET}/seed${SEED}/lambda${LAMBDA}"

cd "$PROJ"
mkdir -p "$OUTPUT"

echo "[DL-MPT TRAIN] dataset=$DATASET seed=$SEED λ=$LAMBDA output=$OUTPUT"
echo "[DL-MPT TRAIN] start: $(date '+%Y-%m-%d %H:%M:%S')"

CUDA_VISIBLE_DEVICES=0 "$PY" train.py \
    --root "$DATA" \
    --seed "$SEED" \
    --trainer DLMPTTrainer \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/dlmpt/vit_b16.yaml" \
    --output-dir "$OUTPUT" \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True \
    TRAINER.C

...(output truncated for display)
---

## 四、实验设计

### 4.1 核心对比

| 方法 | 训练方式 | Novel 评估 |
|------|---------|-----------|
| CoOp (A) | 标准 CE | Zero-shot（全量） |
| CoOp_ATP (B) | 标准 CE + 属性词 | Zero-shot（全量） |
| ProtoATP (C) | Proto-only | 原型推断 |
| **DL-MPT (D)** | **Dual-loop** | **Zero-shot + 原型** |

### 4.2 研究问题

| RQ | 问题 | 对比 |
|----|------|------|
| RQ1 | Dual-loop 是否优于单环 CoOp_ATP？ | D > B (zero-shot) |
| RQ2 | Dual-loop 是否保留 episodic 泛化？ | D > C (episodic K-shot) |
| RQ3 | λ 的最优值？ | λ sweep {0.05, 0.1, 0.2, 0.3, 0.5} |
| RQ4 | 训练阶段效果分解？ | Warmup/Joint/Refine ablation |
| RQ5 | 种子间方差是否降低？ | D std vs B std |

### 4.3 λ Sweep

| λ | 期望效果 |
|---|---------|
| 0.0 | 等同 CoOp_ATP（下界） |
| 0.05 | 弱正则化 |
| 0.2 | **默认值** |
| 0.5 | 激进 meta |

---

## 五、执行路线图

| Phase | 内容 | GPU |
|-------|------|-----|
| 1 | DLMPTTrainer + scripts + Stanford Cars 单 seed 验证 | ~2h |
| 2 | λ sweep × Stanford Cars | ~3h |
| 3 | 最优 λ × 5 数据集 × 3 seeds | ~5h |
| 4 | Ablation + 报告 | ~2h |

---

## 六、成功判定

DL-MPT 成功条件（≥1 条满足）:
1. D > B novel zero-shot in ≥3/5 datasets
2. D > C episodic K-shot in ≥3/5 datasets  
3. D std < B std in ≥4/5 datasets
4. D(1-shot) > B(1-shot) in ≥3/5 datasets

全部不满足 → 方法无效，放弃此方向。

---

## 七、与 ChatGPT 原方案的差异

| 维度 | ChatGPT | 本方案完善 |
|------|:---:|:---:|
| Meta loop | MAML inner-loop | **Proto-style（零梯度）** |
| 脚本化 | 未提及 | **全部固定脚本** |
| 进度输出 | 未定义 | **固定格式 + progress.log** |
| Go/No-Go | 未定义 | **明确量化标准** |

核心保留：**Dual-loop 结构**（L_base+λL_meta）和**三层训练阶段**（Warmup→Joint→Refine）。
