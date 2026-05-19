# CoOp+ATP 全方案变体实现计划

## TL;DR

> **Quick Summary**: 在现有 DL-MPT(CoOp+ATP) 框架上，实现 6 个独立命名的 CoOp+ATP 改进变体（FOMAML 严格元学习、Reptile+L2、Task-adaptive Delta、Attr-aware Sampling、Attr Alignment Loss、全组合），每个变体拥有独立的命名空间、配置文件、训练/评估脚本。
>
> **Deliverables**:
> - 6 个独立 trainer 文件（`trainers/`）
> - 6 个独立 config YAML（`configs/trainers/{method}/`）
> - 12 个 bash 脚本（每个变体 `train.sh` + `eval.sh`）
> - 1 个共享基类 `BaseCoOpATP`（提取公共逻辑，避免 6 份重复代码）
> - 1 个 `extend_cfg` 更新（注册 6 个新方法的配置默认值）
> - 1 个共享 episodic sampler 工具
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves（基类 + 配置 → 5 变体并行 → 全组合 + 脚本）
> **Critical Path**: Task 1（基类）→ Task 3-7（5 变体并行）→ Task 8（全组合）→ Task 9-14（脚本并行）→ Task 15（验证）

---

## Context

### Original Request
用户要求：阅读 `docs/来自chatgpt的设计方案3.md` 与 `README.md`，完成「所有方案的 CoOp-ATP 实现」。经访谈确认，具体需求为：

1. 实现 6 个独立命名的 CoOp+ATP 改进变体
2. **禁止任何全量训练**（仅做快速冒烟验证）
3. 所有新实现放在独立命名空间中（不修改现有 trainer）

### Interview Summary

**Key Discussions**:
- **变体范围**: 每轮访谈逐步收窄至 6 个变体（FOMAML、Reptile+L2、DL-MPT-Adaptive、DL-MPT-AttrSample、DL-MPT-Align、DL-MPT-Full）
- **Reptile+L2 保留**: 纯 Reptile 已在失败变体文档中被否定（Novel -4.2%），但 Reptile+L2 保留作为独立方法
- **组织方式**: 每个改进独立命名（遵循 P0 命名规范），而非单一方法的多配置开关
- **命名空间**: 新 trainer 文件 + 新 config 文件 + 新 scripts 子目录（不修改现有实现）
- **验证方式**: 快速冒烟验证（stanford_cars 3-5 epoch，验证 loss 下降/无 NaN/checkpoint 保存）+ 评估脚本 dry-run

**Research Findings**:
- 现有代码库基于 Dassl.pytorch（自定义 fork），TRAINER_REGISTRY 注册机制
- DL-MPT 当前为 episodic regularization（**无 inner-loop**），需在此基础上加 true meta-learning
- 所有现有 ATP trainer 共享 `PromptLearner`（含 ATP 属性词注入）、`CustomCLIP`（冻结 backbone）、`TextEncoder` 模式
- 配置系统: yacs CfgNode + YAML 文件 + CLI `key=value` 覆盖
- 脚本规范: `scripts/{method}/train.sh` + `eval_novel.sh`，训练日志用 `> log 2>&1`（禁止 `tee`）

### Metis Review

**Identified Gaps** (addressed):
- **代码共享策略**: 决定创建共享基类 `BaseCoOpATP`，提取 PromptLearner/ATP 构建、CLIP 加载、forward、build_model、eval 逻辑。6 个变体仅 override 训练循环（`run_epoch`/`forward_backward`）
- **冒烟测试成功标准**: 确认为三项检查——① loss 递减（不发散）② 无 NaN/Inf ③ checkpoint 文件成功保存（`.pth` 存在且可加载）
- **超参数默认值**: 从 ChatGPT 设计方案和 DL-MPT 成熟配置中提取，见各任务细则
- **变体间依赖**: DL-MPT-Full 依赖所有 5 个组件就绪后才能组装，因此放在 Wave 3
- **共享 sampler**: 提取 `EpisodicTaskSampler` 为独立工具类，供 FOMAML、Reptile+L2、DL-MPT-AttrSample 共用

---

## Work Objectives

### Core Objective
在 ATPrompt 项目中实现 ChatGPT 设计方案3 提出的全部 CoOp+ATP 改进变体（与现有 DL-MPT 区分），每个变体作为独立命名的方法，具备完整的训练/评估通路，且仅需冒烟验证（不执行全量训练）。

### Concrete Deliverables
- `trainers/base_coop_atp.py` — 共享基类（~300 行）
- `trainers/fomaml_coop_atp.py` — FOMAML(CoOp+ATP)
- `trainers/reptile_l2_coop_atp.py` — Reptile+L2(CoOp+ATP)
- `trainers/dlmpt_adaptive_coop_atp.py` — DL-MPT-Adaptive(CoOp+ATP)
- `trainers/dlmpt_attrsample_coop_atp.py` — DL-MPT-AttrSample(CoOp+ATP)
- `trainers/dlmpt_align_coop_atp.py` — DL-MPT-Align(CoOp+ATP)
- `trainers/dlmpt_full_coop_atp.py` — DL-MPT-Full(CoOp+ATP)
- `trainers/episodic_utils.py` — 共享 episodic 工具（EpisodicTaskSampler, attribute_similarity 等）
- `configs/trainers/{method}/vit_b16.yaml` — 每个变体独立配置 ×6
- `scripts/{method}/train.sh` + `scripts/{method}/eval.sh` — 每个变体独立脚本 ×12
- `train.py` — 新增 6 个 import + `extend_cfg` 新增 6 个配置块

### Definition of Done
- [ ] 所有 6 个 trainer 通过 `@TRAINER_REGISTRY.register()` 成功注册
- [ ] 每个变体在 `stanford_cars` 上冒烟训练 3-5 epoch：loss 递减 + 无 NaN + checkpoint 可保存
- [ ] 每个变体的 `eval.sh` 可独立运行 protocol A 评估（dry-run 验证通路）
- [ ] 所有文件符合 P0 命名规范和 P2 目录规范

### Must Have
- 共享基类 `BaseCoOpATP`，避免 6 份重复的 PromptLearner/CLIP 加载/optim 构建代码
- 每个变体独立的 `TRAINER.DLMPT.*` 或 `TRAINER.{METHOD}.*` 配置块
- 每个变体独立的输出目录 `output/{method}/{dataset}/seed{S}/`
- 训练脚本必须使用 `> log 2>&1`（禁止 `tee`）

### Must NOT Have (Guardrails)
- **禁止修改** 任何现有 trainer 文件（`coop_atp.py`、`dlmpt_trainer.py`、`cocoop_atp.py` 等）
- **禁止修改** 任何现有 config YAML（`configs/trainers/dlmpt/vit_b16.yaml` 等）
- **禁止执行** 任何全量训练（>5 epoch 或 >1 dataset 或 >1 seed）
- **禁止引入** pytest/unittest 框架
- **禁止** `| tee` 管道（内存堆积）— 已通过 P3 规范阻止
- **禁止** 临时 `python -c` — 所有流程固化为 `.sh` 脚本（P1）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — 所有验证由执行 agent 完成。

### Test Decision
- **Infrastructure exists**: NO（无 pytest/unittest）
- **Automated tests**: None
- **Framework**: N/A
- **Agent-Executed QA**: PRIMARY — 每个任务通过 bash 冒烟命令验证

### QA Policy
每任务包含 agent-executed QA scenarios：
- **冒烟训练**: bash 运行 train.sh → 检查 log 中 loss 递减、无 NaN、checkpoint 存在
- **评估通路**: bash 运行 eval.sh（`--eval-only`）→ 检查输出中 accuracy 字段、无 Python traceback
- **证据保存**: 日志文件保存到 `.sisyphus/evidence/task-{N}-*.log`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1（基础建设 - 先决条件）:
├── Task 1: 共享基类 BaseCoOpATP [deep]
├── Task 2: 共享 episodic 工具 + extend_cfg + train.py 注册 [quick]

Wave 2（5 个变体并行实现 - MAX PARALLEL）:
├── Task 3: FOMAML(CoOp+ATP) [deep]
├── Task 4: Reptile+L2(CoOp+ATP) [deep]
├── Task 5: DL-MPT-Adaptive(CoOp+ATP) [deep]
├── Task 6: DL-MPT-AttrSample(CoOp+ATP) [deep]
└── Task 7: DL-MPT-Align(CoOp+ATP) [deep]

Wave 3（全组合）:
└── Task 8: DL-MPT-Full(CoOp+ATP) [deep]

Wave 4（配置 + 脚本 - MAX PARALLEL）:
├── Task 9:  Config YAML ×6 [quick]
├── Task 10: Scripts FOMAML + Reptile+L2 [quick]
├── Task 11: Scripts Adaptive + AttrSample + Align [quick]
└── Task 12: Scripts Full [quick]

Wave FINAL（冒烟验证 + 审查 - 4 并行）:
├── Task 13: 冒烟验证（所有 6 变体 stanford_cars 3-5 epoch）[unspecified-high]
├── Task 14: 计划遵从审计 [oracle]
├── Task 15: 代码质量审查 [unspecified-high]
└── Task 16: 手动冒烟 QA [unspecified-high]

Critical Path: Task 1 → Tasks 3-7 → Task 8 → Tasks 9-12 → Tasks 13-16
Parallel Speedup: ~70% vs sequential (5 variants built simultaneously in Wave 2)
Max Concurrent: 5 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 3,4,5,6,7,8 | 1 |
| 2 | - | 3,4,5,6,7,9-12 | 1 |
| 3 | 1,2 | 8 | 2 |
| 4 | 1,2 | 8 | 2 |
| 5 | 1,2 | 8 | 2 |
| 6 | 1,2 | 8 | 2 |
| 7 | 1,2 | 8 | 2 |
| 8 | 3,4,5,6,7 | 9-12 | 3 |
| 9 | 8 | 13 | 4 |
| 10 | 8 | 13 | 4 |
| 11 | 8 | 13 | 4 |
| 12 | 8 | 13 | 4 |
| 13 | 9,10,11,12 | - | FINAL |
| 14 | 13 | - | FINAL |
| 15 | 13 | - | FINAL |
| 16 | 13 | - | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 2 — T1 → `deep`, T2 → `quick`
- **Wave 2**: 5 — T3-T7 → `deep`（5 并行）
- **Wave 3**: 1 — T8 → `deep`
- **Wave 4**: 4 — T9-T12 → `quick`（4 并行）
- **FINAL**: 4 — T13 → `unspecified-high`, T14 → `oracle`, T15 → `unspecified-high`, T16 → `unspecified-high`

---

## TODOs

- [x] 1. 共享基类 `BaseCoOpATP`（提取公共逻辑）

  **What to do**:
  - 创建 `trainers/base_coop_atp.py`
  - 从 `coop_atp.py` 和 `dlmpt_trainer.py` 中提取公共代码：
    - `load_clip_to_cpu(cfg)` — CLIP 模型加载（已有，直接 import）
    - `TextEncoder` — CLIP text encoder wrapper（已有，直接 import 或复制）
    - `CUSTOM_TEMPLATES` — 数据集模板（已有）
    - `BasePromptLearner` — 提取 `PromptLearner.__init__` 中的 ctx/ATP 构建逻辑
    - `BaseCustomCLIP` — 提取 `CustomCLIP.forward` 和 encoder 管理
    - `build_model()` — CLIP 加载 + CustomCLIP 构建 + 冻结 backbone + optimizer/scheduler
    - `load_model()` — checkpoint 加载逻辑
    - `model_inference()` — eval only 推理（protocol A）
  - 类设计：`BaseCoOpATP(TrainerX)` with:
    - `check_cfg()`, `build_model()`, `load_model()`, `model_inference()` — 直接实现
    - `run_epoch()` — 声明 `raise NotImplementedError`（子类 override）
    - `forward_backward()` — 声明 `raise NotImplementedError`（子类 override）
  - 提取 `_construct_episode()` 辅助方法供子类使用（从 base classes 随机采样 N-way K-shot episode）
  - 保持 `@TRAINER_REGISTRY.register()` 在子类而非基类（基类仅做抽象）

  **Must NOT do**:
  - 不要修改 `trainers/coop_atp.py` 或 `trainers/dlmpt_trainer.py`
  - 不要在基类中注册到 TRAINER_REGISTRY
  - 不要让基类包含任何 inner-loop / meta-learning 逻辑

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要在充分理解现有 3 个 trainer 代码基础上提取抽象，涉及继承设计和接口定义
  - **Skills**: `[]`
    - 不需要特殊 skill，纯 Python/Dassl 代码组织

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2 并行）
  - **Parallel Group**: Wave 1（with Task 2）
  - **Blocks**: Tasks 3,4,5,6,7,8
  - **Blocked By**: None

  **References**:
  - `trainers/coop_atp.py:1-500` — 完整的 PromptLearner + CustomCLIP + CoOp_ATP 类，学习 ATP 注入和 build_model 模式
  - `trainers/dlmpt_trainer.py:1-328` — DL-MPT run_epoch + forward_backward + λ 调度模式
  - `trainers/cocoop_atp.py` — CoCoOp+ATP 变体，了解 PromptLearner 在不同方法中的差异
  - `trainers/attributecompute.py` — 共享 ATP 工具函数（_get_class_attributes 等）
  - `Dassl.pytorch/dassl/engine/trainer.py` — TrainerX 基类接口（run_epoch, forward_backward 签名）

  **Acceptance Criteria**:
  - [ ] `trainers/base_coop_atp.py` 文件存在且无语法错误（`python -c "import trainers.base_coop_atp"` 通过）
  - [ ] 基类包含 `build_model`, `load_model`, `model_inference` 三个完整实现
  - [ ] `run_epoch` 和 `forward_backward` 抛出 `NotImplementedError`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: 基类可被 import 且不触发注册
    Tool: Bash
    Steps:
      1. python -c "import trainers.base_coop_atp; print('IMPORT_OK')"
      2. python -c "from dassl.engine import TRAINER_REGISTRY; print('BaseCoOpATP' in [x.__name__ for x in TRAINER_REGISTRY._model_registry.values()])"
    Expected Result: IMPORT_OK 打印，第二个命令打印 False（基类不注册）
    Evidence: .sisyphus/evidence/task-1-import.log

  Scenario: 基类 build_model 方法签名正确
    Tool: Bash
    Steps:
      1. python -c "from trainers.base_coop_atp import BaseCoOpATP; import inspect; sig = inspect.signature(BaseCoOpATP.build_model); print(sig)"
    Expected Result: 显示 (self) 签名
    Evidence: .sisyphus/evidence/task-1-signature.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-1-import.log`
  - [ ] `.sisyphus/evidence/task-1-signature.log`

  **Commit**: YES
  - Message: `feat(trainers): add BaseCoOpATP shared base class`
  - Files: `trainers/base_coop_atp.py`

- [x] 2. 共享 episodic 工具 + extend_cfg 注册 + train.py 导入

  **What to do**:
  - 创建 `trainers/episodic_utils.py`，放置：
    - `EpisodicTaskSampler`: 从 base class dataset 中随机采样 N-way K-shot episode（返回 support_images, support_labels, query_images, query_labels）
    - `attribute_similarity_matrix(classnames, attr_vectors)`: 计算类间属性 cosine 相似度矩阵（用于 Task 6 AttrSample）
    - `sample_hard_episode(classnames, attr_vectors, N, K, Q)`: 按属性相似度构造 hard-transfer episode
  - 更新 `train.py` 的 `extend_cfg`：
    - 新增 `cfg.TRAINER.FOMAML` 配置块：`INNER_LR=0.01`, `INNER_STEPS=3`, `OUTER_LR=0.002`
    - 新增 `cfg.TRAINER.REPTILE` 配置块：`INNER_LR=0.01`, `INNER_STEPS=3`, `EPSILON=0.1`, `L2_REG=0.02`
    - 新增 `cfg.TRAINER.ADAPTIVE_DELTA` 配置块：`HIDDEN_DIM=256`, `DELTA_WEIGHT=0.5`
    - 新增 `cfg.TRAINER.ATTR_SAMPLE` 配置块：`TAU=0.5`, `HARD_RATIO=0.5`
    - 新增 `cfg.TRAINER.ATTR_ALIGN` 配置块：`LAMBDA_ATTR=0.01`
    - 所有新增块包含 `LAMBDA=0.2`, `N_WAY=20`, `K_SUPPORT=3`, `K_QUERY=10`, `N_EPISODES=100`（与 DL-MPT 一致的超参数）
  - 更新 `train.py` 的 import 段：添加 6 个新 trainer 的 import 语句

  **Must NOT do**:
  - 不要删除或修改任何现有的 `extend_cfg` 配置块
  - 不要在 `choose_attribute_for_atprompt` 中添加逻辑（新方法复用现有属性选择）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 配置注册和导入是机械操作，清晰明确
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 1 并行）
  - **Parallel Group**: Wave 1（with Task 1）
  - **Blocks**: Tasks 3,4,5,6,7,9-12
  - **Blocked By**: None

  **References**:
  - `train.py:83-178` — 现有 `extend_cfg` 模式（DL-MPT, ATPROMPT, COOP 等配置块定义）
  - `train.py:1-36` — 现有 import 语句
  - `trainers/dlmpt_trainer.py:200-250` — DL-MPT 的 `_sample_episode` 方法（episodic sampler 参考实现）
  - `configs/trainers/dlmpt/vit_b16.yaml:1-50` — DL-MPT 默认超参数值

  **Acceptance Criteria**:
  - [ ] `trainers/episodic_utils.py` 存在且 `EpisodicTaskSampler` 可 import
  - [ ] `train.py` 中 6 个新 trainer import 无 ModuleNotFoundError
  - [ ] `python -c "from train import extend_cfg; from yacs.config import CfgNode; c=CfgNode(); extend_cfg(c); print(c.TRAINER.FOMAML.INNER_LR)"` 输出 `0.01`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: extend_cfg 注册失败时返回可见错误
    Tool: Bash
    Steps:
      1. python -c "from train import extend_cfg; from yacs.config import CfgNode as CN; c=CN(); extend_cfg(c); assert hasattr(c.TRAINER, 'FOMAML'), 'FOMAML not registered'; assert hasattr(c.TRAINER, 'REPTILE'), 'REPTILE not registered'; print('ALL_CFG_OK')"
    Expected Result: 打印 ALL_CFG_OK（所有 6 个配置块均存在）
    Evidence: .sisyphus/evidence/task-2-cfg.log

  Scenario: EpisodicTaskSampler 功能性检查
    Tool: Bash
    Steps:
      1. python -c "from trainers.episodic_utils import EpisodicTaskSampler; print('IMPORT_OK')"
    Expected Result: IMPORT_OK
    Evidence: .sisyphus/evidence/task-2-sampler.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-2-cfg.log`
  - [ ] `.sisyphus/evidence/task-2-sampler.log`

  **Commit**: YES
  - Message: `feat(cfg): add episodic utils and config blocks for 6 new variants`
  - Files: `trainers/episodic_utils.py`, `train.py`

- [x] 3. FOMAML(CoOp+ATP) — 严格一阶 MAML 元学习

  **What to do**:
  - 创建 `trainers/fomaml_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `run_epoch()` 和 `forward_backward()`
  - **核心逻辑**（per batch）:
    1. 从 base classes 采样 N-way K-shot episode（使用 `EpisodicTaskSampler`）
    2. **Inner-loop**: 在 support set 上对 prompt_learner 参数做 1-3 步 SGD：
       - `fast_weights = current_weights - inner_lr * ∇L_ce(support_logits, support_labels)`
       - 只更新 `prompt_learner` 参数（冻结 encoder）
       - 使用 `torch.autograd.grad` 手动计算梯度
    3. **Outer-loop**: 用 fast_weights 在 query set 上计算 `L_meta = CE(query)`
    4. **Joint**: `L_total = L_base(batch) + λ * L_meta`
    5. 执行 standard backward on `L_total`
  - **FOMAML 实现**: 仅一阶（不用 `create_graph=True`），直接使用 `torch.autograd.grad` 手动计算 inner-loop 梯度
  - 保留与 DL-MPT 一致的 λ 调度（Warmup epoch 1-5 λ=0, Joint epoch 6-20 λ=0.2, Refine epoch 21-25 λ=0.5）
  - prompt_learner fast_weights 的 clone/detach 管理在 `run_epoch` 中完成

  **Must NOT do**:
  - 不要使用二阶梯度（`create_graph=True`）— 仅 FOMAML
  - 不要更新 CLIP backbone 参数
  - 不要使用 `higher` 库（轻量实现，直接用 autograd）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 涉及 autograd 内部机制、inner-loop/outer-loop 分离、FOMAML 算法正确性
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 4,5,6,7 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `trainers/dlmpt_trainer.py:200-328` — DL-MPT run_epoch 完整参考（episode 构造、λ 调度、loss 计算流程）
  - `trainers/base_coop_atp.py` — 基类接口（build_model, run_epoch 签名）
  - `trainers/coop_atp.py:83-284` — PromptLearner 参数结构（ctx, ctx_att1/2/3 — 哪些参数需要 inner-loop 更新）
  - `trainers/coop_atp.py:313-500` — CoOp_ATP TrainerX 完整实现（forward_backward, test 阶段参考）
  - ChatGPT 设计文档: `docs/来自chatgpt的设计方案3.md:252-267` — Inner-loop FOMAML 公式和机制

  **Acceptance Criteria**:
  - [ ] `trainers/fomaml_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `FOMAML_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: FOMAML 3-epoch 冒烟训练通过
    Tool: Bash
    Preconditions: DATA=/home/avoidman2233/Desktop/LVLM/DATA, stanford_cars 数据集可访问
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer FOMAML_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/fomaml/vit_b16.yaml --output-dir output/fomaml_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task3_fomaml.log 2>&1
      2. python -c "log=open('/tmp/task3_fomaml.log').read(); has_nan='nan' in log.lower(); losses=log.count('L_base'); print(f'Loss entries: {losses}, NaN detected: {has_nan}')"
      3. ls output/fomaml_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth
    Expected Result: Loss entries > 0, NaN detected: False, checkpoint_epoch_3.pth 存在
    Failure Indicators: Python traceback in log, 'nan' in log, checkpoint 不存在
    Evidence: .sisyphus/evidence/task-3-fomaml.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-3-fomaml.log`

  **Commit**: YES
  - Message: `feat(trainers): add FOMAML(CoOp+ATP) with first-order inner-loop meta-learning`
  - Files: `trainers/fomaml_coop_atp.py`

- [x] 4. Reptile+L2(CoOp+ATP) — Reptile 风格元学习 + L2 正则

  **What to do**:
  - 创建 `trainers/reptile_l2_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `run_epoch()` 和 `forward_backward()`
  - **核心逻辑**:
    1. 每 batch 采样 N-way K-shot episode
    2. **Inner-loop**: 在 support set 上对 prompt_learner 做 inner_steps 步 SGD：
       - 每步计算 L_support = CE(support)，但加入 L2 reg 惩罚 reg * ||θ_current - θ_original||²
       - θ_new = θ_current - inner_lr * ∇(L_support + L2_penalty)
    3. **Reptile update**: θ_final = θ_original + epsilon * (θ_adapted - θ_original)
    4. 同时计算标准 CoOp+ATP 的 L_base 在一个独立 batch 上
    5. L_total = L_base + lambda * Reptile_meta_loss
  - L2 正则系数 reg=0.02（来自失败变体 #8 的经验值）
  - Reptile epsilon=0.1，inner_lr=0.01
  - 保留 DL-MPT λ 调度

  **Must NOT do**:
  - 不要省略 L2 reg（纯 Reptile 已被证明退化）
  - 不要使用二阶梯度
  - 不要在同一 batch 上同时做 base 和 meta

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Reptile 算法 + L2 reg 的实现正确性、与 DL-MPT λ 调度的融合
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 3,5,6,7 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `docs/失败的dlmpt变体1.md:78-84` — Reptile+L2 实验记录（reg=0.02, epsilon, inner_lr 经验值）
  - `trainers/fomaml_coop_atp.py` — 同 wave 构建，参考 inner-loop 实现模式（注意区别：Reptile 用原始参数做外环更新）
  - `trainers/base_coop_atp.py` — 基类接口

  **Acceptance Criteria**:
  - [ ] `trainers/reptile_l2_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `Reptile_L2_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Reptile+L2 3-epoch 冒烟训练通过
    Tool: Bash
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer Reptile_L2_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/reptile_l2/vit_b16.yaml --output-dir output/reptile_l2_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task4_reptile.log 2>&1
      2. python -c "log=open('/tmp/task4_reptile.log').read(); print('NaN' if 'nan' in log.lower() else 'CLEAN')"
      3. ls output/reptile_l2_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth && echo "CKPT_OK"
    Expected Result: CLEAN, CKPT_OK
    Evidence: .sisyphus/evidence/task-4-reptile.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-4-reptile.log`

  **Commit**: YES
  - Message: `feat(trainers): add Reptile+L2(CoOp+ATP) with inner-loop L2 constraint`
  - Files: `trainers/reptile_l2_coop_atp.py`

- [x] 5. DL-MPT-Adaptive(CoOp+ATP) — Task-adaptive Prompt Delta

  **What to do**:
  - 创建 `trainers/dlmpt_adaptive_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `forward_backward()` 以注入 task-adaptive delta
  - **核心创新**: ΔP_task = MLP(mean(F_support))（ChatGPT 设计方案3 §三 的核心）
  - **实现步骤**:
    1. 在 `build_model()` 中创建轻量 TaskAdapter:
       - `self.task_adapter = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_ctx * ctx_dim))`（512=CLIP visual dim, 256 hidden, 输出 reshape 成 prompt delta）
       - 将它加入 optimizer 和 names_to_update
    2. 在 episodic regularization（沿用 DL-MPT episodic reg）的 forward 中：
       - 用 image_encoder 提取 support images 的视觉特征
       - support_feat_mean = F_support.mean(dim=0) → delta_prompt = self.task_adapter(support_feat_mean).reshape(n_ctx, ctx_dim)
       - P_task = ctx + ctx_att + delta_prompt（注入 task-adaptive shift）
    3. 在 query 上计算 meta loss（仍是 episodic reg，无 inner-loop）
    4. Base loop 也使用 task-adapted prompt（但 delta 由 base batch 自身生成）
  - **参数管理**:
    - delta_weight（默认 0.5）控制 delta 的注入强度：P_task = base_prompt + delta_weight * delta_prompt
    - TaskAdapter 参数 ~= 512×256 + 256×128 ≈ 164K

  **Must NOT do**:
  - 不要在 base loop 中使用 episodic 构造的 delta（base 应独立生成自己的 delta）
  - 不要让 delta 过大导致训练不稳（用 delta_weight 和较小的 lr 控制）
  - 不要使用 inner-loop（这不是 FOMAML，仍为 episodic regularization）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 新 MLP 模块设计、delta 注入点选择、与 DL-MPT episodic reg 的融合
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 3,4,6,7 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `docs/来自chatgpt的设计方案3.md:338-367` — Task-adaptive Prompt Delta 详细描述（ΔP_task = MLP(mean(F_support))）
  - `trainers/dlmpt_trainer.py:150-250` — DL-MPT forward_backward 中 episodic reg 的现有实现（需要注入 delta 的位置）
  - `trainers/base_coop_atp.py` — PromptLearner 接口（ctx, ctx_att, forward 返回 prompt embeddings）
  - `docs/失败的dlmpt变体1.md:10-22` — Gated MLP/Refiner 失败经验（delta 应在 prompt 级操作，不碰 patch/visual tokens）

  **Acceptance Criteria**:
  - [ ] `trainers/dlmpt_adaptive_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `DLMPT_Adaptive_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DL-MPT-Adaptive 3-epoch 冒烟训练通过
    Tool: Bash
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer DLMPT_Adaptive_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/dlmpt_adaptive/vit_b16.yaml --output-dir output/dlmpt_adaptive_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task5_adaptive.log 2>&1
      2. python -c "log=open('/tmp/task5_adaptive.log').read(); print('NaN' if 'nan' in log.lower() else 'CLEAN'); print('L_base entries:', log.count('L_base'))"
      3. ls output/dlmpt_adaptive_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth && echo "CKPT_OK"
    Expected Result: CLEAN, L_base entries > 0, CKPT_OK
    Evidence: .sisyphus/evidence/task-5-adaptive.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-5-adaptive.log`

  **Commit**: YES
  - Message: `feat(trainers): add DL-MPT-Adaptive(CoOp+ATP) with task-adaptive prompt delta`
  - Files: `trainers/dlmpt_adaptive_coop_atp.py`

- [x] 6. DL-MPT-AttrSample(CoOp+ATP) — Attribute-aware Episodic Sampling

  **What to do**:
  - 创建 `trainers/dlmpt_attrsample_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `run_epoch()` 中的 episode 构造逻辑
  - **核心创新**: 按属性相似度构造 episode（ChatGPT 设计方案3 §五）
  - **实现步骤**:
    1. 在 `build_model()` 中构建属性向量矩阵：
       - 提取每个 base class 的 ATP 属性文本 embedding：`F_attr_c = text_encoder(attribute_prompt_c)`
       - 构建 attribute_similarity 矩阵：`S[i,j] = cos(F_attr_i, F_attr_j)`
    2. 在 `run_epoch()` 中：
       - 按概率 p=hard_ratio 选择 hard-transfer episode（属性相近但类别不同）
       - 按概率 1-p 选择随机 episode（标准 DL-MPT 采样）
       - hard episode 构造：选一个 anchor 类，找 `S[anchor, j] > τ` 的非自身类作为 N-way 候选
    3. 其余逻辑与 DL-MPT 完全一致（episodic regularization, λ 调度）
  - 使用 `trainers/episodic_utils.py` 中的 `sample_hard_episode()` 工具函数
  - hard_ratio 默认 0.5，τ 默认 0.5

  **Must NOT do**:
  - 不要在推理阶段使用 attr-aware sampling（仅训练阶段）
  - 不要让 hard_ratio=1.0（保留随机性有助于泛化）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 属性相似度矩阵构建、hard episode 采样策略设计、与 DL-MPT episodic reg 集成
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 3,4,5,7 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `docs/来自chatgpt的设计方案3.md:396-467` — Attribute-aware Sampling 详细描述（属性相似度定义、hard-transfer episode 构造）
  - `trainers/dlmpt_trainer.py:150-200` — DL-MPT episode 构造位置（需要替换采样逻辑的地方）
  - `trainers/episodic_utils.py:sample_hard_episode()` — 共享工具函数（在 Task 2 中创建）
  - `trainers/coop_atp.py:83-284` — ATP 属性 prompt 构造（ctx_att1/2/3 → text prompt → text_encoder，用于计算 attribute_similarity 矩阵）

  **Acceptance Criteria**:
  - [ ] `trainers/dlmpt_attrsample_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `DLMPT_AttrSample_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DL-MPT-AttrSample 3-epoch 冒烟训练通过
    Tool: Bash
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer DLMPT_AttrSample_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/dlmpt_attrsample/vit_b16.yaml --output-dir output/dlmpt_attrsample_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task6_attrsample.log 2>&1
      2. python -c "log=open('/tmp/task6_attrsample.log').read(); print('NaN' if 'nan' in log.lower() else 'CLEAN')"
      3. ls output/dlmpt_attrsample_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth && echo "CKPT_OK"
    Expected Result: CLEAN, CKPT_OK
    Evidence: .sisyphus/evidence/task-6-attrsample.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-6-attrsample.log`

  **Commit**: YES
  - Message: `feat(trainers): add DL-MPT-AttrSample(CoOp+ATP) with attribute-aware episode sampling`
  - Files: `trainers/dlmpt_attrsample_coop_atp.py`

- [x] 7. DL-MPT-Align(CoOp+ATP) — Attribute Alignment Loss

  **What to do**:
  - 创建 `trainers/dlmpt_align_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `forward_backward()` 以加入 L_attr
  - **核心创新**: 约束属性相近类别的 prompt embedding 相互接近（ChatGPT 设计方案3 §六）
  - **实现步骤**:
    1. 在 `build_model()` 中构建与 Task 6 相同的 attribute_similarity 矩阵
    2. 在 `forward_backward()` 中：
       - 获取所有 base classes 的 prompt embeddings: `P_c = prompt_learner()[c]`
       - 对每对 (i,j) 且 `S[i,j] > τ`：`L_attr += ||P_i - P_j||_2`
       - `L_total = L_base + λ_meta * L_meta + λ_attr * L_attr`
    3. L_meta 仍为 DL-MPT episodic regularization
    4. λ_attr 默认 0.01，τ 默认 0.5
  - 为避免 L_attr 计算开销，每 epoch 只重新计算一次（或每隔 N batch）

  **Must NOT do**:
  - 不要对 S[i,j] < τ 的类对施加 L_attr（无属性相关的类别不应被约束在一起）
  - 不要让 λ_attr 过大（0.01 已是上限，防止 prompt 坍缩到单一方向）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 新 loss 项设计、属性矩阵复用、loss 权重平衡
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 3,4,5,6 并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `docs/来自chatgpt的设计方案3.md:486-507` — L_attr 详细定义（L_attr = ||P_i - P_j||_2 when S(i,j) > τ）
  - `trainers/dlmpt_trainer.py:100-150` — DL-MPT loss 计算位置（需要注入 L_attr 的地方）
  - `trainers/coop_atp.py:83-284` — PromptLearner.forward() 返回所有类别的 prompt embeddings

  **Acceptance Criteria**:
  - [ ] `trainers/dlmpt_align_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `DLMPT_Align_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DL-MPT-Align 3-epoch 冒烟训练通过
    Tool: Bash
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer DLMPT_Align_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/dlmpt_align/vit_b16.yaml --output-dir output/dlmpt_align_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task7_align.log 2>&1
      2. python -c "log=open('/tmp/task7_align.log').read(); print('NaN' if 'nan' in log.lower() else 'CLEAN')"
      3. ls output/dlmpt_align_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth && echo "CKPT_OK"
    Expected Result: CLEAN, CKPT_OK
    Evidence: .sisyphus/evidence/task-7-align.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-7-align.log`

  **Commit**: YES
  - Message: `feat(trainers): add DL-MPT-Align(CoOp+ATP) with attribute alignment loss`
  - Files: `trainers/dlmpt_align_coop_atp.py`

- [x] 8. DL-MPT-Full(CoOp+ATP) — 全组件组合

  **What to do**:
  - 创建 `trainers/dlmpt_full_coop_atp.py`
  - 继承 `BaseCoOpATP`，override `run_epoch()` 和 `build_model()`
  - **全组合组件**:
    1. **FOMAML inner-loop**（来自 Task 3）: support adaptation + query evaluation
    2. **Task-adaptive Prompt Delta**（来自 Task 5）: ΔP = MLP(mean(F_support))
    3. **Attribute-aware Sampling**（来自 Task 6）: hard-transfer episode 构造
    4. **Attribute Alignment Loss**（来自 Task 7）: L_attr = ||P_i - P_j||_2
  - **Loss**: `L_total = L_base + λ_meta * L_meta + λ_attr * L_attr`
  - **集成前确认**:
    - TaskAdapter MLP（来自 Task 5）和 FOMAML inner-loop 不冲突（adapter 参数也参与 inner-loop 更新）
    - AttrSample 构造的 episode 作为 FOMAML 的 meta-task episode
    - L_attr 在外环计算（不参与 inner-loop）
  - 所有超参数从各自组件的配置块读取

  **Must NOT do**:
  - 不要新造轮子 — 直接从各组件 trainer 复制/import 核心函数（减少不必要重复）
  - 不要让训练时间超过 2x DL-MPT（组合多种优化可能耗时，但应可控）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解全部 5 个组件如何协同工作，正确组装不冲突
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖全部 5 个组件）
  - **Parallel Group**: Wave 3（单独）
  - **Blocks**: Tasks 9-12
  - **Blocked By**: Tasks 3,4,5,6,7

  **References**:
  - `trainers/fomaml_coop_atp.py` — FOMAML inner-loop 实现（主要 meta-learning 机制）
  - `trainers/dlmpt_adaptive_coop_atp.py` — TaskAdapter MLP 和 delta 注入点
  - `trainers/dlmpt_attrsample_coop_atp.py` — attribute similarity 矩阵构建和 hard episode 采样
  - `trainers/dlmpt_align_coop_atp.py` — L_attr 计算逻辑
  - `docs/来自chatgpt的设计方案3.md:508-513` — 最终 Loss 公式（L = L_base + λ1*L_meta + λ2*L_attr）

  **Acceptance Criteria**:
  - [ ] `trainers/dlmpt_full_coop_atp.py` 存在且 import 不报错
  - [ ] `@TRAINER_REGISTRY.register()` 成功注册为 `DLMPT_Full_CoOp_ATP`
  - [ ] 冒烟训练（stanford_cars, 3 epoch）: loss 递减 + 无 NaN + checkpoint 可保存
  - [ ] 日志中应出现 L_base, L_meta, L_attr 三项 loss

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: DL-MPT-Full 3-epoch 冒烟训练通过
    Tool: Bash
    Steps:
      1. python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer DLMPT_Full_CoOp_ATP --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/dlmpt_full/vit_b16.yaml --output-dir output/dlmpt_full_coop_atp/stanford_cars/seed1/ TEST.NO_TEST True OPTIM.MAX_EPOCH 3 > /tmp/task8_full.log 2>&1
      2. python -c "log=open('/tmp/task8_full.log').read(); print('NaN' if 'nan' in log.lower() else 'CLEAN'); print('L_base:', log.count('L_base'), 'L_meta:', log.count('L_meta'), 'L_attr:', log.count('L_attr'))"
      3. ls output/dlmpt_full_coop_atp/stanford_cars/seed1/checkpoint_epoch_3.pth && echo "CKPT_OK"
    Expected Result: CLEAN, all three losses present (>0 each), CKPT_OK
    Evidence: .sisyphus/evidence/task-8-full.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-8-full.log`

  **Commit**: YES
  - Message: `feat(trainers): add DL-MPT-Full(CoOp+ATP) combining all meta-learning components`
  - Files: `trainers/dlmpt_full_coop_atp.py`

- [x] 9. Config YAML ×6（每个变体独立配置文件）

  **What to do**:
  - 创建 6 个 config YAML 文件（参考 `configs/trainers/dlmpt/vit_b16.yaml`）:
    - `configs/trainers/fomaml/vit_b16.yaml` — 含 `TRAINER.FOMAML.*` 默认值
    - `configs/trainers/reptile_l2/vit_b16.yaml` — 含 `TRAINER.REPTILE.*` 默认值
    - `configs/trainers/dlmpt_adaptive/vit_b16.yaml` — 含 `TRAINER.ADAPTIVE_DELTA.*` 默认值
    - `configs/trainers/dlmpt_attrsample/vit_b16.yaml` — 含 `TRAINER.ATTR_SAMPLE.*` 默认值
    - `configs/trainers/dlmpt_align/vit_b16.yaml` — 含 `TRAINER.ATTR_ALIGN.*` 默认值
    - `configs/trainers/dlmpt_full/vit_b16.yaml` — 含所有配置块组合
  - 每个 YAML 包含:
    - `TRAINER.NAME` → 对应的注册名（如 `FOMAML_CoOp_ATP`）
    - `TRAINER.DLMPT.*` 通用超参数（LAMBDA, N_WAY, K_SUPPORT, WARMUP_EPOCHS 等）
    - `TRAINER.COOP.*`（N_CTX=2, CSC=False, CLASS_TOKEN_POSITION="end", PREC="fp16"）
    - `TRAINER.ATPROMPT.*`（USE_ATPROMPT=True, ATT_NUM=3, 通用属性默认值）
    - `MODEL.*`, `INPUT.*`, `OPTIM.*`, `DATALOADER.*`（与 dlmpt/vit_b16.yaml 一致）
    - 特化配置块（如 `TRAINER.FOMAML.INNER_LR=0.01, INNER_STEPS=3`）
  - 创建必要的子目录 `configs/trainers/{method}/`

  **Must NOT do**:
  - 不要修改现有 `configs/trainers/dlmpt/vit_b16.yaml`
  - 不要在 YAML 中使用相对路径或环境变量

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML 文件编辑，遵循模板，无逻辑复杂度
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 10,11,12 并行）
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 13
  - **Blocked By**: Task 8

  **References**:
  - `configs/trainers/dlmpt/vit_b16.yaml` — 完整模板（所有通用字段 + DL-MPT 特有配置）
  - `configs/trainers/CoOp/vit_b16.yaml` — CoOp 标准配置参考（COOP 块的默认值）
  - `train.py:83-178` — extend_cfg 中定义的所有配置默认值

  **Acceptance Criteria**:
  - [ ] 6 个 YAML 文件存在且可被 yacs 解析
  - [ ] `python -c "from dassl.config import get_cfg_default; c=get_cfg_default(); c.merge_from_file('configs/trainers/fomaml/vit_b16.yaml'); print(c.TRAINER.NAME)"` 输出 `FOMAML_CoOp_ATP`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: 所有 6 个 config 可解析
    Tool: Bash
    Steps:
      1. python -c "
      from dassl.config import get_cfg_default
      import train
      configs = ['fomaml', 'reptile_l2', 'dlmpt_adaptive', 'dlmpt_attrsample', 'dlmpt_align', 'dlmpt_full']
      for c in configs:
        cfg = get_cfg_default()
        train.extend_cfg(cfg)
        cfg.merge_from_file(f'configs/trainers/{c}/vit_b16.yaml')
        print(f'{c}: OK -> {cfg.TRAINER.NAME}')
      "
    Expected Result: 所有 6 行输出 `{name}: OK -> {TRAINER_NAME}`
    Evidence: .sisyphus/evidence/task-9-configs.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-9-configs.log`

  **Commit**: YES（与 Tasks 10-12 合并）
  - Message: `feat(configs,scripts): add configs and scripts for 6 new CoOp+ATP variants`

- [x] 10. Scripts: FOMAML + Reptile+L2（train.sh + eval.sh）

  **What to do**:
  - 创建目录和脚本（参考 `scripts/dlmpt/train.sh` 和 `scripts/dlmpt/eval_novel.sh`）:
    - `scripts/fomaml/train.sh` — train FOMAML(CoOp+ATP) on single dataset
    - `scripts/fomaml/eval.sh` — protocol A eval on novel classes
    - `scripts/reptile_l2/train.sh` — train Reptile+L2(CoOp+ATP)
    - `scripts/reptile_l2/eval.sh` — protocol A eval
  - 每个 `train.sh`:
    - 参数: `--dataset`, `--seed`, `--lambda`（可选，默认从 YAML 读取）
    - 输出重定向: `> log 2>&1`（P3）
    - 进度格式: 遵循 P4 格式 `[{METHOD}] epoch=XX/25 ...`
  - 每个 `eval.sh`:
    - 参数: `--dataset`, `--seed`, `--load-epoch`
    - 使用 `--eval-only` flag + `DATASET.SUBSAMPLE_CLASSES new`
  - 创建 `scripts/fomaml/` 和 `scripts/reptile_l2/` 子目录

  **Must NOT do**:
  - 不要使用 `| tee` 管道（P3）
  - 不要在脚本中硬编码绝对路径（用 `--root` 参数）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: bash 脚本模板化，参考已有脚本
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Tasks 9,11,12 并行）
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 13
  - **Blocked By**: Task 8

  **References**:
  - `scripts/dlmpt/train.sh` — 完整训练脚本模板
  - `scripts/dlmpt/eval_novel.sh` — 完整评估脚本模板
  - `README.md:148-159` — 训练命令示例（参数格式参考）

  **Acceptance Criteria**:
  - [ ] 4 个脚本文件存在且可执行
  - [ ] `bash scripts/fomaml/train.sh --help` 或类似方式显示参数说明

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: 脚本语法检查
    Tool: Bash
    Steps:
      1. bash -n scripts/fomaml/train.sh && echo "FOMAML_TRAIN_OK"
      2. bash -n scripts/fomaml/eval.sh && echo "FOMAML_EVAL_OK"
      3. bash -n scripts/reptile_l2/train.sh && echo "REPTILE_TRAIN_OK"
      4. bash -n scripts/reptile_l2/eval.sh && echo "REPTILE_EVAL_OK"
    Expected Result: 4 行 OK 输出
    Evidence: .sisyphus/evidence/task-10-scripts.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/task-10-scripts.log`

  **Commit**: 合并到 Task 9 的 commit

- [x] 11. Scripts: Adaptive + AttrSample + Align（train.sh + eval.sh）

  **What to do**:
  - 与 Task 10 相同模式，为以下 3 个方法创建脚本:
    - `scripts/dlmpt_adaptive/train.sh` + `scripts/dlmpt_adaptive/eval.sh`
    - `scripts/dlmpt_attrsample/train.sh` + `scripts/dlmpt_attrsample/eval.sh`
    - `scripts/dlmpt_align/train.sh` + `scripts/dlmpt_align/eval.sh`
  - 创建对应的 3 个子目录
  - 脚本参数、输出格式、重定向规则与 Task 10 一致

  **Must NOT do**:
  - 同上（禁止 `tee`，禁止硬编码路径）

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Parallel Group**: Wave 4（与 9,10,12 并行）
  - **Blocks**: Task 13
  - **Blocked By**: Task 8

  **Acceptance Criteria**:
  - [ ] 6 个脚本文件存在且语法正确（`bash -n` 全部通过）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: 脚本语法检查 ×6
    Tool: Bash
    Steps: bash -n for all 6 scripts
    Expected Result: 全部 OK
    Evidence: .sisyphus/evidence/task-11-scripts.log
  ```

  **Commit**: 合并到 Task 9 的 commit

- [x] 12. Scripts: DL-MPT-Full（train.sh + eval.sh）

  **What to do**:
  - `scripts/dlmpt_full/train.sh` + `scripts/dlmpt_full/eval.sh`
  - 与 Task 10/11 相同模式，但 DL-MPT-Full 的 train.sh 需要额外传递 L_attr 和自适应 delta 相关参数
  - 创建 `scripts/dlmpt_full/` 子目录

  **Must NOT do**: 同上

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Parallel Group**: Wave 4（与 9,10,11 并行）
  - **Blocks**: Task 13
  - **Blocked By**: Task 8

  **Acceptance Criteria**:
  - [ ] 2 个脚本文件存在且语法正确（`bash -n` 通过）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Full 脚本语法检查
    Tool: Bash
    Steps: bash -n for both scripts
    Expected Result: 全部 OK
    Evidence: .sisyphus/evidence/task-12-scripts.log
  ```

  **Commit**: 合并到 Task 9 的 commit

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **冒烟验证 — 全部 6 变体** — `unspecified-high`

  对每个变体执行快速冒烟验证（stanford_cars, 3 epoch, seed 1）:
  1. `FOMAML_CoOp_ATP`: 运行 train.sh → 检查 log 无 NaN、loss 递减、checkpoint 存在
  2. `Reptile_L2_CoOp_ATP`: 同上
  3. `DLMPT_Adaptive_CoOp_ATP`: 同上
  4. `DLMPT_AttrSample_CoOp_ATP`: 同上
  5. `DLMPT_Align_CoOp_ATP`: 同上
  6. `DLMPT_Full_CoOp_ATP`: 同上（额外验证 L_base/L_meta/L_attr 三项 loss 均出现）
  对每个变体运行 eval.sh dry-run:
  - `--eval-only --load-epoch 3` 验证 protocol A 通路不报错
  输出: `[N/6] 冒烟通过 | [N/6] eval 通过 | VERDICT`

  > ⚠️ 注意: 每个变体训练约需 5-10 分钟 GPU 时间 ×6 ≈ 30-60 分钟。如果显存不足（24GB），降低 N_EPISODES 或 batch_size。

- [x] F2. **计划遵从审计** — `oracle`

  阅读计划 end-to-end：
  - 每个 "Must Have" 检查实现是否存在（read 对应源文件）
  - 每个 "Must NOT Have" 搜索受禁模式（grep 确认未修改 `coop_atp.py`、`dlmpt_trainer.py`、现有 configs）
  - 检查 evidence 文件存在（`.sisyphus/evidence/task-*.log`）
  - 检查所有 6 个 trainer 在 TRAINER_REGISTRY 中注册
  输出: `Must Have [N/N] | Must NOT Have [N/N] | Evidence [N/N] | VERDICT: APPROVE/REJECT`

- [x] F3. **代码质量审查** — `unspecified-high`

  检查所有新增文件:
  - Python 语法: `python -m py_compile` 通过
  - 无 `as any`/`@ts-ignore`（Python 代码检查无 `type: ignore` 滥用、无裸 `except:`）
  - 无 `print(debug)` 残留、无可疑注释掉的代码块
  - 命名规范: 所有 trainer 名遵循 P0 约定
  - 目录规范: 所有文件位置遵循 P2 约定
  输出: `Compile [N/N] | Clean [N/N] | Naming [N/N] | VERDICT`

- [x] F4. **范围忠诚检查** — `deep`

  对每个 task 的 "What to do" vs 实际 diff:
  - 验证 spec 中要求的内容全部实现（无遗漏）
  - 验证 spec 外来内容未侵入（无 scope creep）
  - 检查 "Must NOT do" 遵从情况
  - 检测交叉污染: Task N 未意外修改 Task M 的文件
  - 标记未记录的变更
  输出: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Wave | Tasks | Message | Files |
|------|-------|---------|-------|
| 1 | 1 | `feat(trainers): add BaseCoOpATP shared base class` | `trainers/base_coop_atp.py` |
| 1 | 2 | `feat(cfg): add episodic utils and config blocks for 6 new variants` | `trainers/episodic_utils.py`, `train.py` |
| 2 | 3 | `feat(trainers): add FOMAML(CoOp+ATP) with first-order inner-loop meta-learning` | `trainers/fomaml_coop_atp.py` |
| 2 | 4 | `feat(trainers): add Reptile+L2(CoOp+ATP) with inner-loop L2 constraint` | `trainers/reptile_l2_coop_atp.py` |
| 2 | 5 | `feat(trainers): add DL-MPT-Adaptive(CoOp+ATP) with task-adaptive prompt delta` | `trainers/dlmpt_adaptive_coop_atp.py` |
| 2 | 6 | `feat(trainers): add DL-MPT-AttrSample(CoOp+ATP) with attribute-aware episode sampling` | `trainers/dlmpt_attrsample_coop_atp.py` |
| 2 | 7 | `feat(trainers): add DL-MPT-Align(CoOp+ATP) with attribute alignment loss` | `trainers/dlmpt_align_coop_atp.py` |
| 3 | 8 | `feat(trainers): add DL-MPT-Full(CoOp+ATP) combining all meta-learning components` | `trainers/dlmpt_full_coop_atp.py` |
| 4 | 9-12 | `feat(configs,scripts): add configs and scripts for 6 new CoOp+ATP variants` | 6 YAML + 12 .sh files |

---

## Success Criteria

### Verification Commands
```bash
# 验证所有 trainer 正确注册
python -c "
from dassl.engine import TRAINER_REGISTRY
expected = ['FOMAML_CoOp_ATP', 'Reptile_L2_CoOp_ATP', 'DLMPT_Adaptive_CoOp_ATP',
            'DLMPT_AttrSample_CoOp_ATP', 'DLMPT_Align_CoOp_ATP', 'DLMPT_Full_CoOp_ATP']
for name in expected:
    assert name in TRAINER_REGISTRY._model_registry, f'{name} not registered'
    print(f'{name}: REGISTERED')
"

# 验证所有 config 可解析
for method in fomaml reptile_l2 dlmpt_adaptive dlmpt_attrsample dlmpt_align dlmpt_full; do
  python -c "
from dassl.config import get_cfg_default; import train
cfg = get_cfg_default(); train.extend_cfg(cfg)
cfg.merge_from_file('configs/trainers/$method/vit_b16.yaml')
print('$method: OK')
"
done
```

### Final Checklist
- [ ] All 6 trainer classes registered in TRAINER_REGISTRY
- [ ] All 6 config YAML files parse correctly
- [ ] All 12 bash scripts pass `bash -n` syntax check
- [ ] All 6 variants pass stanford_cars 3-epoch smoke test (loss decreasing + no NaN + checkpoint saved)
- [ ] All "Must NOT Have" constraints verified (no modification to existing files)
- [ ] Zero files outside `.sisyphus/` and approved project directories modified

---
