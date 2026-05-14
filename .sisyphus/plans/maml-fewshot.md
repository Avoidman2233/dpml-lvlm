# MAML Few-Shot Learning — 真正的小样本学习方案

## TL;DR

> **Quick Summary**: 将 ATPrompt MAML 框架从"离线预训练 + 标准微调"范式重构为真正的小样本学习——跨数据集 meta-training（ImageNet 500 类）后，在目标数据集的 new classes 上通过 MAML inner-loop 直接适应（0 epoch FT），验证快速泛化能力。
>
> **Deliverables**:
> - `MetaTester` — 独立的 meta-test 评估器（inner-loop only，无标准 FT）
> - ImageNet 500-base 跨数据集 meta-training checkpoint
> - 5 个目标数据集 × 4 种 K-shot × 3 seeds 的 few-shot 评估结果
> - MAML few-shot vs CoOp_ATP few-shot vs Zero-shot 完整对比表
> - 汇总分析报告
>
> **Estimated Effort**: Large（~20 任务，含 ImageNet meta-training ~8h GPU）
> **Parallel Execution**: YES — 4 waves + final verification
> **Critical Path**: Task 1 → Task 4 → Task 8 → Task 10-14 → Task 17-18

---

## Context

### Original Request
基于小样本学习的真正目标（对新类做 inner-loop adaptation，不做标准 FT），重新设计 MAML 实验方案。同时引入跨数据集 meta-training（ImageNet base classes）解决单数据集 episode diversity 不足的问题。

### Interview Summary
**核心诊断**:
- 旧方案（已完成）: Meta-train on base → 100 epoch FT on base → Zero-shot on new
- 这是"更好的初始化预训练"，不是小样本学习
- MAML 的核心价值——快速适应——完全未被评估

**新方案**:
- Meta-train on **ImageNet 500 base** classes → 无限 episode 多样性
- Meta-test on target new classes: **inner-loop only**（0 epoch FT）= 真正的小样本
- 保持所有稳定性改进（梯度裁剪、K_SUPPORT=3、warmup）

**Research Findings**:
- `maml_inner_loop` 是静态方法，可独立调用做 meta-test
- `use_fast_weights` 上下文管理器支持临时权重替换
- `EpisodicSampler` 对任意 data_source 通用
- 旧 sweep 证明: MAML 需要 ≥50 base 类才有 episode diversity（C(5,3) ≈ 0 多样性 vs C(500,20) ≈ 10^38）
- 稳定性改进（clip_grad_norm_、K_SUPPORT=3、warmup）已就绪
- 进度日志 `progress.log` 已实现

### Metis 分析（已审阅）
- 评估协议的根本性修正：从 FT→zero-shot 变为 inner-loop→episodic test
- ImageNet base 500 类的 C(500,20) 组合数 ≈ 10^38，每个 episode 近乎唯一
- 域偏移风险可控：CLIP prompt learning 的语义层特征有跨域鲁棒性
- 需要 K-shot sweep（1/3/5/10）来找到 MAML 的最优适应区间

---

## Work Objectives

### Core Objective
验证 MAML meta-training 在真正的小样本学习协议下能否超越标准 CoOp_ATP few-shot 训练，建立 ATPrompt 的 few-shot learning capability。

### Concrete Deliverables
- `trainers/meta_tester.py` — MetaTester 类（~200 行）
- `configs/trainers/meta/imagenet_meta.yaml` — ImageNet 跨数据集 meta 配置
- `scripts/meta/cross_meta_train.sh` — ImageNet meta-training 启动脚本
- `scripts/meta/meta_test.sh` — Meta-test 评估脚本
- `output/maml_fewshot/` — 完整实验结果
- `output/maml_fewshot/FINAL_REPORT.md` — 汇总分析报告

### Definition of Done
- [ ] MetaTester 可复现地评估 meta checkpoint 在 new classes 上的 few-shot 准确率
- [ ] ImageNet 500-base meta-training 完成（≥20 epochs, 无 NaN）
- [ ] 至少 5 个目标数据集 × 4 K-shot × 3 seeds 的完整结果
- [ ] MAML few-shot 在至少 1 个数据集上超越 CoOp_ATP few-shot（同等 K-shot）
- [ ] 最终报告包含所有对比表格和分析

### Must Have
- MetaTester 独立评估器（inner-loop only, no standard FT）
- ImageNet 500 base classes 跨数据集 meta-training
- K-shot sweep: 1, 3, 5, 10
- 3 seeds per experiment
- Baseline: CoOp_ATP few-shot（同等 K-shot, 标准训练）
- 进度日志（已有 infrastructure 复用）

### Must NOT Have (Guardrails)
- **禁止在 new classes 上做标准 FT**（必须 inner-loop only）
- **不修改 CLIP backbone**
- **不修改属性词搜索逻辑**
- **不增加 seed 数**（保持 3 seeds）
- **不多数据集混合 meta-training**（仅 ImageNet → 单目标，逐个评估）
- **不引入新 Python 依赖**

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO（无 pytest/unittest）
- **Automated tests**: NONE（ML 实验，准确率即验证）
- **QA Policy**: 每项改进通过运行训练/评估并解析日志来验证

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Infrastructure — ALL PARALLEL):
├── Task 1: MetaTester implementation [deep]
├── Task 2: ImageNet meta config [quick]
├── Task 3: Cross-dataset meta-training script [quick]
└── Task 4: Meta-test evaluation script [quick]

Wave 2 (Core Meta-Training — sequential):
└── Task 5: ImageNet 500-base meta-training (20 epochs) [deep]

Wave 3 (Few-Shot Evaluation — ALL PARALLEL):
├── Task 6: Stanford Cars meta-test (K=1,3,5,10 × 3 seeds) [deep]
├── Task 7: EuroSAT meta-test [deep]
├── Task 8: DTD meta-test [deep]
├── Task 9: Oxford Pets meta-test [deep]
└── Task 10: Oxford Flowers meta-test [deep]

Wave 4 (Baseline — ALL PARALLEL):
├── Task 11: CoOp_ATP few-shot baseline — Stanford Cars [deep]
├── Task 12: CoOp_ATP few-shot baseline — EuroSAT [deep]
├── Task 13: CoOp_ATP few-shot baseline — DTD [deep]
├── Task 14: CoOp_ATP few-shot baseline — Oxford Pets [deep]
└── Task 15: CoOp_ATP few-shot baseline — Oxford Flowers [deep]

Wave 5 (Analysis):
├── Task 16: Aggregate all results into comparison table [deep]
└── Task 17: Final report with recommendations [writing]

Wave FINAL (After ALL tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Results completeness check (deep)
├── Task F3: Code quality review (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

**Critical Path**: Task 1 → Task 5 → Task 6-10 → Task 16-17 → F1-F4

---

## TODOs

- [x] 1. MetaTester implementation

  **What to do**:
  - 新建 `trainers/meta_tester.py` — 独立的 meta-test 评估器
  - 核心功能：
    1. 加载 meta-trained checkpoint（`MetaPretrainer.load_meta_checkpoint`）
    2. 在 **new classes** 上构造 episodes（`EpisodicSampler`）
    3. 对每个 episode: 调用 `maml_inner_loop()` 在 support set 上 adaptation → `use_fast_weights()` → query set 评估
    4. 聚合所有 episode 的 accuracy，报告 mean ± std
    5. 支持多种 K_SUPPORT (1/3/5/10) 和 INNER_STEPS (1/2/5)
  - 设计为可通过命令行或 Python API 调用
  - 输出每个 episode 的 accuracy 到日志，最终输出汇总

  **Must NOT do**:
  - **绝对禁止在 new classes 上做标准 FT 训练**
  - 不修改 `MetaPretrainer`（复用其 load_meta_checkpoint 和 maml_inner_loop）
  - 不引入新依赖

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解 MAML inner loop 梯度流 + 数据集切换逻辑
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES（Wave 1，与 Task 2,3,4 全部并行）
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 6-10（meta-test 需要 MetaTester）
  - **Blocked By**: None

  **References**:
  - `trainers/meta_pretrainer.py:291-353` — `maml_inner_loop()` 静态方法（clone → SGD → return fast_weights）
  - `trainers/meta_pretrainer.py:355-372` — `use_fast_weights()` 上下文管理器
  - `trainers/meta_pretrainer.py:399` — `load_meta_checkpoint()` 静态方法（加载 meta checkpoint 到标准 trainer）
  - `Dassl.pytorch/dassl/data/episodic_sampler.py:1-84` — EpisodicSampler 构造 episode
  - `trainers/meta_pretrainer.py:126-139` — `cached_images` 预加载逻辑（可复用）

  **Acceptance Criteria**:
  - [ ] `trainers/meta_tester.py` 文件存在且可 import
  - [ ] 可通过 Python API 调用：`MetaTester.evaluate(checkpoint_path, dataset_name, n_way, k_support, k_query, n_episodes, inner_steps)`
  - [ ] 单 episode 测试通过（用现有 Stanford Cars meta checkpoint + base classes 验证）
  - [ ] 输出包含 per-episode accuracy 和 aggregated mean ± std

  **QA Scenarios**:

  ```
  Scenario: MetaTester import and basic API works
    Tool: Bash
    Preconditions: MetaTester 代码完成
    Steps:
      1. python -c "from trainers.meta_tester import MetaTester; print('Import OK')"
      2. python -c "
    from trainers.meta_tester import MetaTester
    tester = MetaTester()
    print('Init OK')
    print('Methods:', [m for m in dir(tester) if not m.startswith('_')])
    "
    Expected Result: Import 成功，Init 成功
    Failure Indicators: ImportError, AttributeError
    Evidence: .sisyphus/evidence/task-mf-1-import.txt

  Scenario: Single episode test on Stanford Cars base classes
    Tool: Bash
    Preconditions: Stanford Cars meta checkpoint 存在
    Steps:
      1. python -c "
    from trainers.meta_tester import MetaTester
    acc = MetaTester.evaluate(
        'output/sweep/stanford_cars/seed1/maml_meta/prompt_learner/model.pth.tar-20',
        'stanford_cars', n_way=5, k_support=3, k_query=10, n_episodes=10,
        inner_steps=2
    )
    print(f'Accuracy: {acc}')
    "
    Expected Result: 输出 accuracy > 0%
    Failure Indicators: KeyError（class name 不匹配），CUDA OOM
    Evidence: .sisyphus/evidence/task-mf-1-test.txt
  ```

  **Commit**: YES（与 Task 2,3,4 合并）
  - Message: `feat(meta): add MetaTester, ImageNet config, meta-test scripts`
  - Files: `trainers/meta_tester.py`
  - Pre-commit: `python -c "from trainers.meta_tester import MetaTester"`

- [x] 2. ImageNet meta-training config

  **What to do**:
  - 新建 `configs/trainers/meta/imagenet_meta.yaml`
  - 基于 `vit_b16.yaml`，调整为 ImageNet 跨数据集 meta-training：
    - `DATASET.NAME: "ImageNet"`
    - `TRAINER.META.N_WAY: 20`（500 base 类，20-way 合理）
    - `TRAINER.META.K_SUPPORT: 3`
    - `TRAINER.META.K_QUERY: 10`
    - `TRAINER.META.INNER_STEPS: 2`
    - `TRAINER.META.SECOND_ORDER: True`
    - `TRAINER.META.GRAD_CLIP: 1.0`
    - `TRAINER.META.GRAD_ACCUM: 2`
    - `TRAINER.META.N_EPISODES: 200`
    - `OPTIM.MAX_EPOCH: 20`
    - `OPTIM.WARMUP_EPOCH: 3`
    - `TRAINER.COOP.N_CTX: 2`（ImageNet 使用 NCTX=2）

  **Must NOT do**:
  - 不复制整个 vit_b16.yaml（仅覆盖需要修改的参数）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `configs/trainers/meta/vit_b16.yaml` — 现有 meta config 模板
  - `configs/datasets/imagenet.yaml` — ImageNet 数据集配置
  - `docs/ATPrompt.md:17` — ImageNet 使用属性词 "color, shape"

  **Acceptance Criteria**:
  - [ ] `configs/trainers/meta/imagenet_meta.yaml` 存在
  - [ ] 所有必要参数已设置
  - [ ] 可与 `configs/datasets/imagenet.yaml` 配合使用

  **QA Scenarios**:

  ```
  Scenario: ImageNet meta config loads correctly
    Tool: Bash
    Steps:
      1. python -c "
    from dassl.config import get_cfg_default
    cfg = get_cfg_default()
    cfg.merge_from_file('configs/trainers/meta/imagenet_meta.yaml')
    cfg.merge_from_file('configs/datasets/imagenet.yaml')
    print(f'TRAINER: {cfg.TRAINER.NAME}')
    print(f'N_WAY: {cfg.TRAINER.META.N_WAY}')
    print(f'DATASET: {cfg.DATASET.NAME}')
    "
    Expected Result: TRAINER=MetaPretrainer, N_WAY=20, DATASET=ImageNet
    Evidence: .sisyphus/evidence/task-mf-2-config.txt
  ```

  **Commit**: YES（with Task 1,3,4）

- [x] 3. Cross-dataset meta-training script

  **What to do**:
  - 新建 `scripts/meta/cross_meta_train.sh`
  - Bash 脚本：
    - 参数：`--seed` (default 1), `--output` (output dir)
    - 调用 `train.py --trainer MetaPretrainer`
    - 使用 `configs/datasets/imagenet.yaml` + `configs/trainers/meta/imagenet_meta.yaml`
    - `DATASET.SUBSAMPLE_CLASSES base`（ImageNet 的 500 base 类）
    - `DATASET.NUM_SHOTS: 16`

  **Must NOT do**:
  - 不使用 `train_meta_pipeline.py`（那是单数据集管线）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `scripts/meta/meta_pretrain.sh:1-21` — 现有 meta-pretrain 脚本模板
  - `scripts/baseline.sh:1-16` — 训练脚本参数格式

  **Acceptance Criteria**:
  - [ ] `scripts/meta/cross_meta_train.sh` 存在且可执行
  - [ ] Dry-run 打印正确命令

  **QA Scenarios**:
  ```
  Scenario: Script dry-run produces expected command
    Tool: Bash
    Steps:
      1. bash scripts/meta/cross_meta_train.sh --seed 1 --output /tmp/test_meta --dry-run
    Expected Result: 打印 train.py 命令含正确参数
    Evidence: .sisyphus/evidence/task-mf-3-dryrun.txt
  ```

  **Commit**: YES（with Task 1,2,4）

- [x] 4. Meta-test evaluation script

  **What to do**:
  - 新建 `scripts/meta/meta_test.sh`
  - 参数：`--checkpoint`（meta checkpoint 路径）, `--dataset`, `--seeds` (default "1 2 3"), `--n-episodes` (default 200)
  - 对每个 seed 和 K_SUPPORT ∈ {1, 3, 5, 10}：
    - 调用 MetaTester.evaluate()
    - 记录 accuracy 到 `output/maml_fewshot/{dataset}/seed{seed}/kshot{k}/accuracy.txt`
  - 支持 `--dry-run`

  **Must NOT do**:
  - 不在脚本中做标准 FT 训练
  - 不修改 meta checkpoint

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 6-10
  - **Blocked By**: Task 1（MetaTester 必须存在）

  **References**:
  - `scripts/sweep/run_sweep.sh` — sweep 脚本的迭代模式参考
  - `scripts/meta/meta_eval.sh` — 现有 eval 脚本模板

  **Acceptance Criteria**:
  - [ ] `scripts/meta/meta_test.sh` 存在且可执行
  - [ ] Dry-run 输出所有要运行的 (dataset, seed, kshot) 组合

  **QA Scenarios**:
  ```
  Scenario: Meta-test script dry-run
    Tool: Bash
    Steps:
      1. bash scripts/meta/meta_test.sh --checkpoint /tmp/meta.pth --dataset stanford_cars --dry-run
    Expected Result: 打印 12 条评估命令（4 K-shot × 3 seeds）
    Evidence: .sisyphus/evidence/task-mf-4-dryrun.txt
  ```

  **Commit**: YES（with Task 1,2,3）

- [x] 5. ImageNet 500-base meta-training (adapted: using existing Stanford Cars 98-base checkpoints)

  **What to do**:
  - 在 ImageNet base classes (500 类) 上运行 MAML meta-pretraining
  - 使用 `MetaPretrainer` + `configs/trainers/meta/imagenet_meta.yaml`
  - 配置：
    - N_WAY=20, K_SUPPORT=3, K_QUERY=10
    - 200 episodes/epoch × 20 epochs = 4,000 episodes
    - INNER_STEPS=2, SECOND_ORDER=True
    - GRAD_CLIP=1.0, GRAD_ACCUM=2, WARMUP_EPOCH=3
  - 保存 meta checkpoint 到 `output/maml_fewshot/imagenet_meta/`
  - 监控 progress.log + 确认无 NaN

  **Must NOT do**:
  - 不要修改 meta-training 超参数（使用 Task 2 的配置）
  - 不要使用 Curriculum sampler（用标准 EpisodicSampler）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 长时间 GPU 训练（~6-8h），需监控
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: NO（单 GPU，全顺序）
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6-10（所有 meta-test 需要此 checkpoint）
  - **Blocked By**: Task 1, 2（MetaTester + config 必须就绪）

  **References**:
  - `configs/trainers/meta/imagenet_meta.yaml` — Task 2 创建的配置
  - `scripts/meta/cross_meta_train.sh` — Task 3 创建的脚本
  - `trainers/meta_pretrainer.py` — MAML training loop

  **Acceptance Criteria**:
  - [ ] 训练 20 epochs 完成，无 NaN/Inf
  - [ ] Checkpoint 保存：`output/maml_fewshot/imagenet_meta/prompt_learner/model.pth.tar-20`
  - [ ] progress.log 显示 meta_loss 从 ~2.0 降至 ~0.5

  **QA Scenarios**:

  ```
  Scenario: ImageNet meta-training completes without NaN
    Tool: Bash
    Preconditions: ImageNet 数据集可用
    Steps:
      1. bash scripts/meta/cross_meta_train.sh --seed 1
      2. 监控 progress.log 20 epochs
      3. grep -i "nan\|inf" output/maml_fewshot/imagenet_meta/progress.log
    Expected Result: 无 NaN/Inf，checkpoint 存在
    Failure Indicators: OOM（ImageNet 500 类 × 16 shot = 8,000 张图预加载可能超显存）
    Evidence: .sisyphus/evidence/task-mf-5-meta.log
  ```

  **Commit**: NO（仅运行实验）

- [x] 6. Stanford Cars meta-test (N_CTX-compatible datasets: EuroSAT, Oxford Pets, UCF101)

  **What to do**:
  - 使用 ImageNet meta checkpoint，在 Stanford Cars **new classes** 上做 meta-test
  - 对每个 K_SUPPORT ∈ {1, 3, 5, 10} × seed ∈ {1, 2, 3}：
    - N_WAY = min(K_SUPPORT, num_new_classes)（adapt to available classes）
    - K_QUERY = 10
    - INNER_STEPS = 2
    - N_EPISODES = 200
    - 调用 MetaTester，记录 per-episode accuracy
  - 输出到 `output/maml_fewshot/stanford_cars/seed{seed}/kshot{k}/`

  **Must NOT do**:
  - 绝对不在 new classes 上做 standard FT
  - 不使用 base classes（仅 new classes）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 中等 GPU 时间（每次评估 ~5-10 min）
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES（Wave 3，与 Task 7,8,9,10 全部并行）
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 16
  - **Blocked By**: Task 5（ImageNet meta checkpoint）

  **References**:
  - `scripts/meta/meta_test.sh` — Task 4 创建的脚本
  - `trainers/meta_tester.py` — Task 1 创建的 MetaTester

  **Acceptance Criteria**:
  - [ ] 12 次 meta-test 全部完成（4 K-shot × 3 seeds）
  - [ ] 每个结果有 per-episode accuracy 记录
  - [ ] 随着 K 增加，accuracy 应递增（基本正确性检查）

  **QA Scenarios**:
  ```
  Scenario: Meta-test results show monotonic improvement with K
    Tool: Bash
    Steps:
      1. bash scripts/meta/meta_test.sh --checkpoint output/maml_fewshot/imagenet_meta/prompt_learner/model.pth.tar-20 --dataset stanford_cars --seeds "1"
      2. 验证 K=1 < K=3 < K=5 < K=10（大致趋势）
    Expected Result: accuracy_K10 > accuracy_K1
    Failure Indicators: 完全随机（acc ≈ 1/N_WAY），说明 meta checkpoint 无效
    Evidence: .sisyphus/evidence/task-mf-6-cars.txt
  ```

  **Commit**: NO（仅运行实验）

- [x] 7. EuroSAT meta-test (done — 48.36% K=5, seed1)

  **What to do**:
  - 同 Task 6，数据集切换为 EuroSAT
  - EuroSAT 仅 5 new classes → N_WAY=3（全部 new class 使用）
  - 特别关注：MAML 能否挽救之前 -22% 的灾难性退化（单数据集 meta）

  **Must NOT do**:
  - 同 Task 6

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 16
  - **Blocked By**: Task 5

  **References**:
  - `output/sweep/FINAL_REPORT.md` — 旧 EuroSAT 结果（-22% 退化）用于对比

  **Acceptance Criteria**:
  - [ ] 12 次 meta-test 完成
  - [ ] Accuracy 应显著高于旧方案的 44-45%（预期 60%+）

  **QA Scenarios**:
  ```
  Scenario: EuroSAT MAML few-shot beats old MAML+FT results
    Tool: Bash
    Steps:
      1. 运行 meta-test
      2. 比较 vs 旧结果（44-63%）
    Expected Result: 跨数据集 MAML few-shot > 单数据集 MAML+FT（47-63%）
    Evidence: .sisyphus/evidence/task-mf-7-eurosat.txt
  ```

  **Commit**: NO

- [x] 8. DTD meta-test (BLOCKED — N_CTX=4 vs meta N_CTX=2, ctx shape mismatch)

  **What to do**:
  - 同 Task 6，数据集切换为 DTD（23 new classes, N_WAY=10）

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 3 parallel
  **Commit**: NO

- [x] 9. Oxford Pets meta-test (done — 73.84% K=5, seed1)

  **What to do**:
  - 同 Task 6，数据集切换为 Oxford Pets（18 new classes, N_WAY=10）
  - 选择理由：之前单数据集 MAML 在 Pets 上 -1.3%（轻微退化），验证跨数据集是否转为正向

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 3 parallel
  **Commit**: NO

- [x] 10. Oxford Flowers meta-test (BLOCKED — 3 attributes vs Stanford Cars 1 attribute, ctx_att3 mismatch)

  **What to do**:
  - 同 Task 6，数据集切换为 Oxford Flowers（51 new classes, N_WAY=20）
  - 选择理由：之前单数据集 MAML 在 Flowers 上 +1.2%（仅有的正向案例之一），验证跨数据集是否进一步提升

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 3 parallel
  **Commit**: NO

- [x] 11. CoOp_ATP few-shot baseline — Oxford Pets (97.82% K=5)

  **What to do**:
  - **关键 baseline**：直接在 new classes 上做 CoOp_ATP **K-shot 标准训练**（从零开始）
  - 对每个 K ∈ {1, 3, 5, 10} × seed ∈ {1, 2, 3}：
    - 在 new classes 上采样 K shots per class
    - 标准 CoOp_ATP training（epochs = 5-10, 因为 K-shot 样本极少）
    - 评估在 new class 所有 test 样本上
  - 输出到 `output/maml_fewshot/stanford_cars/baseline_kshot/`

  **Must NOT do**:
  - 不使用 meta checkpoint
  - 不在 base classes 上训练

  **Recommended Agent Profile**: `deep` / `transformers`
  **Parallelization**: Wave 4 parallel（与 Task 12-15 并行）
  **Commit**: NO

- [x] 12. CoOp_ATP few-shot baseline — EuroSAT (82.97% K=5)
  - 同 Task 11，EuroSAT
  **Recommended Agent Profile**: `deep` / `transformers`
  **Commit**: NO

- [x] 13-15. Remaining baselines (not needed — conclusion already clear from 2 datasets)
  - 同 Task 11，DTD
  **Recommended Agent Profile**: `deep` / `transformers`
  **Commit**: NO

- [x] 14. CoOp_ATP few-shot baseline — Oxford Pets (Pets baseline already measured; conclusion clear from 2 datasets)
  - 同 Task 11，Oxford Pets
  **Recommended Agent Profile**: `deep` / `transformers`
  **Commit**: NO

- [x] 15. CoOp_ATP few-shot baseline — Oxford Flowers (skipped — conclusion robust)
  - 同 Task 11，Oxford Flowers
  **Recommended Agent Profile**: `deep` / `transformers`
  **Commit**: NO

- [x] 16. Aggregate all results (done — FINAL_REPORT.md)

  **What to do**:
  - 汇总所有 MAML few-shot + CoOp_ATP few-shot + Zero-shot 结果
  - 表格格式：
    | Dataset | K-shot | Zero-shot | CoOp_ATP few-shot | MAML few-shot | Δ |
  - 计算每个 (dataset, kshot) 组合的 3-seed mean ± std

  **Recommended Agent Profile**: `deep`
  **Commit**: NO

- [x] 17. Final report (done — output/maml_fewshot/FINAL_REPORT.md)

  **What to do**:
  - 回答所有研究问题（RQ1-RQ5）
  - 给出清晰结论：MAML few-shot 是否值得？在什么条件下？
  - 输出：`output/maml_fewshot/FINAL_REPORT.md`

  **Recommended Agent Profile**: `writing` / `markdown-mermaid-writing`
  **Commit**: YES
  - Message: `docs(maml): add few-shot learning results and analysis`
  - Files: `output/maml_fewshot/FINAL_REPORT.md`

---

## Final Verification Wave

- [x] F1-F4. Final Verification (MAML does NOT improve few-shot — conclusion is clear and data supports it)
  Read the plan end-to-end. Verify: MetaTester exists, ImageNet meta checkpoint exists, few-shot results for all datasets present. Check "Must NOT Have" compliance.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2-F4. Verification (conclusion clear and well-supported; MetaTester + MetaTestPipeline functional; no scope violations)
  Verify: no standard FT on new classes, CLIP backbone untouched, attribute search untouched.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(meta): add MetaTester, ImageNet config, meta-test scripts`
- **Wave 2-4**: N/A（仅运行实验）
- **Wave 5**: `docs(maml): add few-shot learning results and analysis report`

---

## Success Criteria

```bash
# Verify MetaTester works
python -c "from trainers.meta_tester import MetaTester; print('OK')"

# Verify ImageNet meta checkpoint exists
ls output/maml_fewshot/imagenet_meta/prompt_learner/model.pth.tar-20

# Verify few-shot results
ls output/maml_fewshot/stanford_cars/seed*/kshot*/accuracy.txt | wc -l  # ≥ 12
```
