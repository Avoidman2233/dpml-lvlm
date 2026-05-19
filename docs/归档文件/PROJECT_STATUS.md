# ATPrompt 项目进展说明

## 项目目标

**主命题**：基于元任务微调与属性辅助推理的小样本学习

在 CLIP ViT-B/16 (frozen backbone) 的 Prompt Learning 框架下，验证：
1. GPT-4o 生成的属性描述词（shape/color/material/luxury/habitat 等）能否通过增强文本 Prompt 提升小样本分类性能
2. 元任务微调（MAML / Metric-based ProtoNet）能否与属性辅助推理产生协同效应

## 方法演进

### 第一阶段：CoOp_ATP（属性辅助 CoOp）— 已复现论文

在标准 CoOp Prompt Learning 的基础上，在文本 Prompt 中注入属性词：
- `"X X luxury X X {class_name}."`（Stanford Cars）
- `"X X habitat X X {class_name}."`（EuroSAT）

**结果**：5/8 数据集达到或超过论文复现水平，验证属性词有效。

### 第二阶段：MAML + CoOp_ATP（失败）

在 base classes 上用 MAML episode-based 训练 prompt_learner，期望 meta-initialization 改善标准 fine-tuning。

**结果**：仅在 Stanford Cars (+1.1%) 上有效，EuroSAT (-22%)、DTD (-7%) 负向。根因：CLIP prompt learning 参数空间极小（N_CTX=2，仅 1024 参数），优化 landscape 近乎凸，MAML 的"更好初始化"无实质增益。

### 第三阶段：ProtoATP（Metric-based, 当前方法）

基于 Prototypical Networks (Snell et al., 2017) 思想，利用 CLIP 原生的 cosine similarity 度量空间，设计多模态原型（视觉原型 + 属性增强文本原型）进行 few-shot 分类。

**ProtoATP 方法**：
```
训练: Episode-based (N-way K-shot)
  For each class c:
    visual_proto_c = ImageEncoder(support_c).mean(0)
    text_proto_c  = TextEncoder(ctx_att + "{attr_word}" + ctx + "{class_name}")
    proto_c = (visual_proto_c + text_proto_c) / 2
  classify query: cosine_similarity(query_embedding, proto_c)

推理: zero-gradient prototype inference (纯前向，无梯度更新)
```

**对内消融验证**（Stanford Cars, 5-way episodic）：
| 消融 | K=1 | 说明 |
|------|-----|------|
| ProtoATP (full) | 94.5% | 视觉+文本+属性 |
| Proto (vanilla) | 85.4% | 纯视觉原型，无文本 |
| D_noattr | 87.4% | 视觉+文本，但文本不含属性词 |
| CoOp_ATP zero-shot | 88.6% | CLIP 零样本 5-way 子集分类 |

属性增强文本原型带来 +9.1% (K=1) 的内部消融增益。

## 公平对比实验（CoOp_ATP K-shot vs ProtoATP）

**协议**：同一 data split，同一 5-way (或 3-way) episodic 评估。

**CoOp_ATP baseline**：在 **所有 novel class** 上做 K-shot 标准 fine-tuning（10 epoch），然后在 episodic 评估时取对应 5 类的 logit 子集。

**ProtoATP**：在 base class 上 episodic meta-training（20 epoch），然后在 novel class 上用原型推断（零梯度）。

### Stanford Cars (98 novel classes, 5-way)

| K | ProtoATP | CoOp_ATP K-shot | Delta |
|---|----------|-----------------|-------|
| 1 | 93.6% ± 7.5 | **98.9%** ± 1.5 | -5.3% |
| 3 | 97.6% ± 3.2 | **98.7%** ± 2.0 | -1.1% |
| 5 | 98.1% ± 2.9 | **99.1%** ± 1.8 | -1.0% |

### Oxford Pets (18 novel classes, 5-way)

| K | ProtoATP | CoOp_ATP K-shot | Delta |
|---|----------|-----------------|-------|
| 1 | 92.4% ± 7.3 | **98.8%** ± 2.1 | -6.4% |
| 3 | 98.1% ± 3.2 | **99.6%** ± 1.2 | -1.5% |
| 5 | 98.9% ± 2.0 | **99.2%** ± 1.6 | -0.3% |

### EuroSAT (5 novel classes, 3-way)

| K | ProtoATP | CoOp_ATP K-shot | Delta |
|---|----------|-----------------|-------|
| 1 | 75.3% ± 16.7 | **92.0%** ± 6.3 | -16.7% |
| 3 | **80.4%** ± 15.1 | 77.8% ± 13.6 | **+2.7%** |
| 5 | 83.8% ± 13.3 | **90.2%** ± 8.7 | -6.4% |

**14/15 对比中 CoOp_ATP K-shot 胜出。**

## 代码基础设施

| 文件 | 功能 |
|------|------|
| `trainers/proto_trainer.py` | ProtoTrainer — episode-based 原型网络训练 |
| `trainers/meta_pretrainer.py` | MetaPretrainer — MAML meta-training |
| `trainers/coop_atp.py` | CoOp_ATP — 属性增强 Prompt Learning |
| `trainers/meta_tester.py` | MetaTester — MAML few-shot 评估 |
| `Dassl/dassl/data/curriculum_episodic_sampler.py` | 课程学习 Episode Sampler |
| `scripts/sweep/` | 自动化 sweep 脚本 |

## 硬件约束

- GPU: NVIDIA RTX 4060 Ti 16GB
- CPU: AMD 5600, RAM: 32GB
- Backbone: CLIP ViT-B/16 (frozen)

## 当前核心困境

**CoOp_ATP 标准 K-shot 微调在所有数据集和 K 值上系统性地优于 ProtoATP。**

可能原因：
1. CLIP ViT-B/16 特征极强，简单的 prompt 微调已接近上限
2. 原型网络在 3-5 类的小规模 episode 上无法体现度量学习优势
3. ProtoATP meta-training 的目标（最大化 N-way 原型分离）与最终评估的多样性存在 gap
4. 10 epoch SGD（CoOp_ATP）vs 零梯度前向推理（ProtoATP）存在不对称的计算优势

## 已排除的方向

- MAML（optimization-based meta-learning）：与 CLIP prompt learning 参数空间极小天然不匹配
- Curriculum EpisodicSampler：实现但未产生显著增益

## 寻求的建议

1. 在 CLIP 强 backbone 下，metric-based meta-learning 是否天然劣势？是否需要弱 backbone (RN50) 验证？
2. ProtoATP 的训练目标（原型分离）是否应改为与 CoOp_ATP 一致的交叉熵分类？
3. 属性词的使用方式是否应该更激进（如 per-class 自定义属性，而非 per-dataset）？
4. 小样本学习在 CLIP 时代是否应该换一个范式？例如：不追求 0-gradient inference，而是探索更高效的 few-shot 微调策略？
5. 是否有其他 metric-based 方法（如 MatchingNet、RelationNet）在 CLIP 上取得过正面的结果？
