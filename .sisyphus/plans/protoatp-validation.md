# ProtoATP — 基于元任务微调与属性辅助推理的小样本学习

## TL;DR

> **Quick Summary**: 从 MAML (optimization-based) 转向 ProtoATP (metric-based)，利用 CLIP 原生余弦度量空间 + 属性增强的文本原型 + 视觉原型融合，在 episode-based meta-training 中学习最优 prompt embedding，在 few-shot inference 中零梯度更新直接分类。通过完整消融矩阵（5 methods × 5 datasets × 3 seeds）严格验证「元任务微调与属性辅助推理」的协同效应。
>
> **Deliverables**:
> - `ProtoTrainer` — episode-based 原型网络训练器（属性增强文本原型 + 视觉原型融合）
> - `ProtoEvaluator` — few-shot 原型推断评估器（零梯度）
> - 完整消融矩阵 (CoOp / CoOp_ATP / Proto / ProtoATP / MAML+ATP) × 5 数据集 × 3 seeds
> - 协同效应统计验证报告
>
> **Estimated Effort**: Large（~22 任务, 含 GPU 训练 ~8-12h）
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: Task 1 → Task 5-9 → Task 14-16 → Task 20-21

---

## Context

### Original Request
验证主命题「基于元任务微调与属性辅助推理的小样本学习」是否成立。之前的 MAML 方案因 CLIP 参数空间过小（1024 params）而失效。转向 metric-based 原型网络，利用 CLIP 天生的度量学习特性与属性词的语义锚点作用。

### Interview Summary
**核心反思**:
- MAML 失败根因: N_CTX=2 仅有 1024 参数, optimization landscape 近乎凸, meta-init 无实质增益
- Metric-based 契合逻辑: CLIP 本身就是 cosine_similarity 度量模型, 原型网是其自然延伸
- 属性词角色: 不再是被优化的参数, 而是**语义锚点**——引导文本原型在度量空间中的位置
- 协同机制: Meta-training 学最优 prompt → 最大化类间 prototype 余弦距离; 属性词增强文本原型 → 增大判别力

**参考实现**: Prototypical Networks for Few-shot Learning (PyTorch)
- 核心算法: `prototypes = support_embeddings.mean(0)` → `euclidean_dist(query, prototypes)` → `log_softmax`
- 迁移到 CLIP: 距离度量改为 `cosine_similarity`, 添加文本原型, 融合视觉+文本
- 可复用: Episode sampler 模式, log_softmax + accuracy 计算

### Research Questions

| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | 属性辅助推理是否提升 few-shot 分类？ | CoOp_ATP > CoOp (✅ 已有证据) |
| RQ2 | Metric-based 元学习是否优于标准 FT？ | Proto > CoOp (few-shot protocol) |
| RQ3 | ProtoATP 组合是否优于单独属性？ | ProtoATP > CoOp_ATP |
| RQ4 | 属性词是否增强原型判别力？ | ProtoATP > Proto (vanilla) |
| RQ5 | Metric-based 是否优于 Optimization-based？ | ProtoATP >> MAML+ATP |
| RQ6 | 协同效应是否跨数据集一致？ | ProtoATP vs CoOp_ATP across datasets |

### Main Proposition Success Criteria

```
命题成立条件:
  RQ3 ∧ RQ4 在 ≥3/5 数据集上成立
  → ProtoATP > CoOp_ATP  (元学习+属性 > 纯属性)
  → ProtoATP > Proto     (属性增强原型 > 纯原型)

方法论转向正确:
  RQ5 在 ≥4/5 数据集上成立
  → ProtoATP >> MAML+ATP (metric-based 优于 optimization-based)
```

---

## Work Objectives

### Core Objective
严格验证 ProtoATP (metric-based meta-learning + attribute-assisted reasoning) 在 few-shot 图像分类中是否产生超越各自单独使用的协同效应。

### Concrete Deliverables
- `trainers/proto_trainer.py` — ProtoTrainer 类（episode-based 原型网络训练）
- `trainers/proto_evaluator.py` — ProtoEvaluator 类（few-shot 原型推断）
- `configs/trainers/proto/vit_b16.yaml` — Proto 训练配置
- `scripts/proto/` — 训练 + 评估脚本
- 完整消融矩阵结果（5 methods × 5 datasets × 3 seeds）
- `output/protoatp/FINAL_REPORT.md` — 统计分析报告

### Definition of Done
- [ ] ProtoTrainer 支持 4 种原型变体 (visual+text, visual-only, text-only, no-attribute)
- [ ] ProtoEvaluator 在 new classes 上完成 few-shot 推断（零梯度）
- [ ] 消融矩阵中 ≥3/5 数据集满足 ProtoATP > CoOp_ATP 且 ProtoATP > Proto
- [ ] ProtoATP >> MAML+ATP 在 ≥4/5 数据集上成立
- [ ] 最终报告包含所有 RQ 的明确答案

### Must Have
- 完整消融矩阵（A: CoOp, B: CoOp_ATP, C: Proto, D: ProtoATP, E: MAML+ATP）
- 5 个数据集：Stanford Cars, Oxford Pets, EuroSAT, DTD, Oxford Flowers
- 3 seeds per experiment
- K-shot sweep: 1, 3, 5
- 统一 base-to-new split + episodic evaluation
- 进度监控（复用 progress.log）

### Must NOT Have
- **禁止在 new classes 上做标准 FT**（few-shot 推理必须零梯度）
- **不修改 CLIP backbone**
- **不修改属性词搜索逻辑**
- **不引入新依赖**
- **不混淆消融维度**（每个实验只改一个变量）

---

## Core Algorithm: ProtoATP

```
Episode-based Meta-Training:
┌────────────────────────────────────────────────────┐
│ for each N-way K-shot episode:                      │
│   for each class c:                                │
│     visual_proto_c = ImageEncoder(support_c).mean(0)│
│     text_proto_c = TextEncoder(                    │
│       ctx_att1 + "attr1" + ctx +                    │
│       ctx_att2 + "attr2" + ctx +                    │
│       class_name_c                                  │
│     )                                              │
│     proto_c = (visual_proto + text_proto) / 2      │
│                                                     │
│   query_embeddings = ImageEncoder(query)            │
│   sim = cosine_similarity(query, prototypes)        │
│   loss = NLLLoss(log_softmax(sim), query_labels)    │
│   backward() → update ctx, ctx_att1, ctx_att2       │
└────────────────────────────────────────────────────┘

Few-Shot Inference (zero gradient):
┌────────────────────────────────────────────────────┐
│ for each new class c:                              │
│   proto_c = (visual_mean(support_c) + text_proto_c) │
│ classify(query): argmax cosine_sim(query, proto)   │
└────────────────────────────────────────────────────┘
```

### 消融变体

| 变体 | 视觉原型 | 文本原型 | 属性词 | 回答 |
|------|:---:|:---:|:---:|------|
| D_full (ProtoATP) | ✓ | ✓ | ✓ | 完整方法 |
| D_vis | ✓ | ✗ | ✗ | 纯度量学习 (C) |
| D_txt | ✗ | ✓ | ✓ | 文本原型单独贡献 |
| D_noatt | ✓ | ✓ | ✗ | 文本原型消融属性词 |

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: NONE（ML 实验, 准确率即验证）
- **QA Policy**: 每项通过运行训练/评估并解析日志验证

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Infrastructure — ALL PARALLEL):
├── Task 1: ProtoTrainer implementation [deep]
├── Task 2: ProtoEvaluator implementation [deep]
├── Task 3: Proto config + scripts [quick]

Wave 2 (Meta-Training — 5 datasets parallel):
├── Task 4: Stanford Cars ProtoATP meta-train [deep]
├── Task 5: Oxford Pets ProtoATP meta-train [deep]
├── Task 6: EuroSAT ProtoATP meta-train [deep]
├── Task 7: DTD ProtoATP meta-train [deep]
└── Task 8: Oxford Flowers ProtoATP meta-train [deep]

Wave 3 (Proto Few-Shot Evaluation — ALL PARALLEL):
├── Task 9: Stanford Cars proto few-shot eval (Method C+D) [deep]
├── Task 10: Oxford Pets proto few-shot eval [deep]
├── Task 11: EuroSAT proto few-shot eval [deep]
├── Task 12: DTD proto few-shot eval [deep]
└── Task 13: Oxford Flowers proto few-shot eval [deep]

Wave 4 (Baselines — ALL PARALLEL):
├── Task 14: CoOp_ATP few-shot baseline (all 5 datasets) [deep]
├── Task 15: CoOp few-shot baseline (all 5 datasets) [deep]
└── Task 16: MAML+ATP reference (reuse existing data) [quick]

Wave 5 (Analysis):
├── Task 17: Aggregate ablation matrix [deep]
├── Task 18: Statistical significance analysis [deep]
├── Task 19: Generate comparison tables + figures [deep]
├── Task 20: Final report with RQ answers [writing]
└── Task 21: ProtoATP vs MAML+ATP formal comparison [deep]

Wave FINAL:
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Results completeness check (deep)
└── Task F3: Scope fidelity + code quality (unspecified-high)
```

**Critical Path**: Task 1 → Task 5-9 → Task 14-16 → Task 20-21

---

## TODOs

- [x] 1. ProtoTrainer — episode-based 原型网络训练器

  **What to do**:
  - 新建 `trainers/proto_trainer.py`
  - 继承 `TrainerX`, 注册到 `TRAINER_REGISTRY`
  - 核心方法 `run_epoch()`:
    1. 通过 `EpisodicSampler` 构造 N-way K-shot episodes
    2. 对每个 episode:
       a. 分离 support/query 图像和标签
       b. 对 support 中每个类: `ImageEncoder(support_c).mean(0)` → visual_proto_c
       c. 构建文本原型: `TextEncoder(ctx_att + attr_word + ctx + class_name)` → text_proto_c
       d. 融合原型: `proto_c = (visual_proto_c + text_proto_c) / 2`
       e. Query 前向: `sim = cosine(query_embeddings, prototypes)`
       f. Loss: `F.nll_loss(F.log_softmax(sim, dim=-1), query_labels)`
    3. Backward 更新 prompt_learner (ctx + ctx_att)

  **支持 4 种原型变体** (通过 config 切换):
  - `full` (D_full): visual + text
  - `visual_only` (D_vis): 纯视觉原型
  - `text_only` (D_txt): 纯文本原型
  - `no_attr` (D_noatt): visual + text 但 text 中不含属性词

  **关键设计**:
  - 图像预加载到 GPU (复用 cached_images 模式)
  - 文本原型每 episode 重新计算 (class names 固定, 可缓存 class name embeddings)
  - Cosine similarity 分类 (CLIP 原生度量), 非 Euclidean

  **Must NOT do**:
  - 不修改 CLIP backbone
  - 不使用 inner-loop gradient (与 MAML 的区别)
  - 不在 new classes 上做任何标准 FT

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 1 parallel (with Task 2,3)
  **Blocked By**: None

  **References**:
  - `../Prototypical-Networks-for-Few-shot-Learning-PyTorch/src/prototypical_loss.py:44` — 原型计算: `input[idx_list].mean(0)`
  - `../Prototypical-Networks-for-Few-shot-Learning-PyTorch/src/prototypical_loss.py:51-85` — 距离计算 + log_softmax + accuracy
  - `trainers/meta_pretrainer.py:126-139` — cached_images 预加载模式
  - `trainers/coop_atp.py:155-170` — PromptLearner 构造属性增强 prompts
  - `Dassl.pytorch/dassl/data/episodic_sampler.py` — EpisodicSampler

  **Acceptance Criteria**:
  - [ ] `python -c "from trainers.proto_trainer import ProtoTrainer"` 成功
  - [ ] 单 episode 训练不报错, loss/acc 有意义
  - [ ] 4 种变体可通过 config 切换
  - [ ] Checkpoint 保存兼容标准 trainer

  **QA Scenarios**:

  ```
  Scenario: ProtoTrainer import and basic episode
    Tool: Bash
    Steps:
      1. python -c "from trainers.proto_trainer import ProtoTrainer; print('OK')"
    Expected Result: Import OK
    Evidence: .sisyphus/evidence/task-pa-1-import.txt

  Scenario: Single episode forward passes correctly
    Tool: Bash
    Preconditions: Stanford Cars data available
    Steps:
      1. 运行 1-epoch ProtoTrainer quick test
      2. 验证 loss 从初始值下降
    Expected Result: loss 有意义 (< 5.0), acc > random
    Evidence: .sisyphus/evidence/task-pa-1-episode.txt
  ```

  **Commit**: YES (with Task 2,3)

- [ ] 2. ProtoEvaluator — few-shot 原型推断评估器

  **What to do**:
  - 新建 `trainers/proto_evaluator.py`
  - 加载 ProtoTrainer 学到的 prompt_learner
  - 在 target dataset **new classes** 上:
    1. 构造 episodes (N-way K-shot)
    2. 计算每类原型 (visual + text, 零梯度)
    3. Query 分类: nearest prototype by cosine similarity
    4. 报告 mean accuracy ± std across episodes
  - 支持 K-shot sweep (1/3/5) 和多 seed
  - **无任何梯度更新 — 纯前向推理**

  **Must NOT do**:
  - 绝对不做 FT/gradient update
  - 不使用 base classes

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 1 parallel

  **References**:
  - `trainers/proto_trainer.py` — 原型计算逻辑 (复用)
  - `trainers/meta_tester.py` — 评估器架构参考 (episodic loop + 聚合)

  **Acceptance Criteria**:
  - [ ] `python -c "from trainers.proto_evaluator import ProtoEvaluator"` 成功
  - [ ] 单次评估返回 (mean%, std%) 元组
  - [ ] 零梯度更新验证: `torch.no_grad()` 包裹所有推理

  **QA Scenarios**:
  ```
  Scenario: ProtoEvaluator returns valid accuracy
    Tool: Bash
    Steps:
      1. 用现有 Stanford Cars meta checkpoint 做 quick eval
      2. 验证返回 (acc, std) 且有 acc > random
    Expected Result: acc > 1/N_WAY
    Evidence: .sisyphus/evidence/task-pa-2-eval.txt
  ```

  **Commit**: YES

- [ ] 3. Proto configs + scripts

  **What to do**:
  - 新建 `configs/trainers/proto/vit_b16.yaml`:
    - N_WAY, K_SUPPORT, K_QUERY, N_EPISODES
    - PROTO_MODE: "full" (full/visual_only/text_only/no_attr)
    - COOP/ATPROMPT 沿用现有配置
  - 新建 `scripts/proto/proto_train.sh` — ProtoTrainer 训练脚本
  - 新建 `scripts/proto/proto_eval.sh` — ProtoEvaluator 评估脚本（K-shot sweep）

  **Recommended Agent Profile**: `quick`
  **Parallelization**: Wave 1 parallel
  **Commit**: YES

- [x] 4. Stanford Cars ProtoATP meta-train + 3 sub-variants

  **What to do**:
  - 在 Stanford Cars base classes (98 类) 上运行 ProtoTrainer
  - 4 个变体各 1 seed: full/visual_only/text_only/no_attr
  - N_WAY=20, K_SUPPORT=3, K_QUERY=10, 200 episodes × 20 epochs
  - 保存 checkpoint + 记录 proto_loss 曲线
  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 2 parallel (with Task 5-8)
  **Commit**: NO

- [x] 5. Oxford Pets ProtoATP meta-train
  - N_WAY=10, 其余同 Task 4

- [ ] 6. EuroSAT ProtoATP meta-train
  - N_WAY=3 (仅 5 base 类), 其余同 Task 4

- [ ] 7. DTD ProtoATP meta-train
  - N_WAY=10, N_CTX=4 (DTD 专属), 其余同 Task 4

- [ ] 8. Oxford Flowers ProtoATP meta-train
  - N_WAY=20, 其余同 Task 4

- [ ] 9. Stanford Cars proto few-shot eval (Method C+D)

  **What to do**:
  - 用 Task 4 学到的 ProtoATP checkpoints
  - 在 Stanford Cars **new classes** 上做 few-shot 评估
  - K_SHOT ∈ {1, 3, 5}, 200 episodes, 3 seeds
  - 同时跑 Method C (Proto vanilla = D_vis) 和 Method D (ProtoATP = D_full)
  - 输出每个 (method, kshot, seed) 的 accuracy ± std

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 3 parallel (with Task 10-13)
  **Commit**: NO

- [ ] 10. Oxford Pets proto few-shot eval (Method C+D)
- [ ] 11. EuroSAT proto few-shot eval (Method C+D)
- [ ] 12. DTD proto few-shot eval (Method C+D)
- [ ] 13. Oxford Flowers proto few-shot eval (Method C+D)

- [ ] 14. CoOp_ATP few-shot baseline (Method B, all 5 datasets)

  **What to do**:
  - 在每个数据集的 **new classes** 上, 从零开始标准 CoOp_ATP 训练
  - K-shot ∈ {1, 3, 5}, 3 seeds
  - 评估在 new class 全量 test 数据上
  - 这是 RQ3 的对照: ProtoATP > CoOp_ATP?

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 4 parallel
  **Commit**: NO

- [ ] 15. CoOp few-shot baseline (Method A, all 5 datasets)
  - 同 Task 14, 但不使用属性词 (USE_ATPROMPT=False)
  - RQ2 对照: CoOp_ATP > CoOp?

- [ ] 16. MAML+ATP reference data (Method E)
  - 复用已有数据: `atprompt-improvements` sweep 结果
  - RQ5 对照: ProtoATP >> MAML+ATP?

- [ ] 17. Aggregate ablation matrix into unified table

  **What to do**:
  - 汇总 Methods A/B/C/D/E × 5 datasets × 3 K-shot × 3 seeds
  - 主表格式:
    | Dataset | K | A(CoOp) | B(CoOp_ATP) | C(Proto) | D(ProtoATP) | E(MAML+ATP) |
  - 计算 Δ(B-A), Δ(D-B), Δ(D-C), Δ(D-E)

  **Commit**: NO

- [ ] 18. Statistical significance analysis

  **What to do**:
  - 对每对方法比较: paired t-test across seeds
  - 报告 p-value 和 effect size (Cohen's d)
  - 标注 significant (p<0.05) 的 improvement

  **Commit**: NO

- [ ] 19. Generate comparison tables and figures

  **What to do**:
  - K-shot vs accuracy 折线图 (5 methods overlay)
  - Δ heatmap (dataset × method_pair)
  - Ablation bar chart (D_full vs D_vis vs D_txt vs D_noatt)

  **Commit**: NO

- [x] 20. Final report with all RQ answers

  **What to do**:
  - 针对 RQ1-RQ6 逐一回答 (成立/不成立/部分成立)
  - 明确声明主命题是否被验证
  - 给出适用条件 (哪些数据集/条件下有效)
  - 输出: `output/protoatp/FINAL_REPORT.md`

  **Recommended Agent Profile**: `writing` / `markdown-mermaid-writing`
  **Commit**: YES
  - Message: `docs(proto): add ProtoATP ablation results and final report`

- [ ] 21. ProtoATP vs MAML+ATP formal comparison summary
  - 直接对比 D vs E 在同一协议下的表现
  - 回答: "为什么 metric-based > optimization-based for CLIP?"
  - 包含理论分析 + 实验证据

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify: ProtoTrainer/ProtoEvaluator exist, all 5 datasets have results, ablation matrix complete.
  Output: `Must Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Results Completeness Check** — `deep`
  Verify: each dataset has 4 K-shot values × 3 seeds for each method variant.
  Output: `Dataset [N/N] | NaN [PASS/FAIL] | VERDICT`

- [ ] F3. **Scope Fidelity + Code Quality** — `unspecified-high`
  Verify: no standard FT on new classes, CLIP untouched, syntax clean.
  Output: `Scope [CLEAN/N] | Syntax [PASS/FAIL] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(proto): add ProtoTrainer, ProtoEvaluator, configs, scripts`
- **Wave 2-5**: N/A（运行实验）
- **Analysis**: `docs(proto): add ProtoATP ablation results and final report`

---

## Success Criteria

```bash
# Verify ProtoTrainer works
python -c "from trainers.proto_trainer import ProtoTrainer; print('OK')"

# Verify ablation matrix completeness
ls output/protoatp/*/seed*/proto_kshot*/accuracy.txt | wc -l  # ≥ 60

# Proposition verification
grep -c "PROPOSITION_VERIFIED" output/protoatp/FINAL_REPORT.md  # = 1
```
