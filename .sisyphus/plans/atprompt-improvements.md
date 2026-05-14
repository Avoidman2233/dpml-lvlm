# ATPrompt 改进方案：短FT探索 + 稳定性增强 + 课程学习采样

## TL;DR

> **Quick Summary**: 针对 MAML meta-pretraining 在 100 epoch FT 下增益被覆盖的问题，通过三大方向（短 FT sweep 找到最优微调长度、稳定性改进减少种子方差、Curriculum EpisodicSampler 提升 episode 质量）系统性验证 MAML 的实际价值，同时优化 GPU 利用率。
>
> **Deliverables**:
> - 3 个 pilot 数据集（Stanford Cars / EuroSAT / DTD）在 6 个 FT epoch 值下的完整 sweep 结果
> - 稳定性改进后的 meta_pretrainer.py（梯度裁剪 + K_SUPPORT=3 + warmup）
> - Curriculum EpisodicSampler 实现与验证
> - 多 episode 批处理优化（GPU 利用率提升）
> - 汇总分析报告
>
> **Estimated Effort**: Large（~25 个任务，含 GPU 实验时间）
> **Parallel Execution**: YES - 5 waves，Wave 2/3 最大化并行
> **Critical Path**: Task 1 → Task 4 → Task 10-12 → Task 16-17

---

## Context

### Original Request
用户要求查看 ATPrompt 项目的实验报告和代码，分析 MAML meta-pretraining 方法的不足，并构思改进方案。

### Interview Summary
**Key Discussions**:
- **核心矛盾**: MAML 的价值是"快速适应"（2-5 step inner loop），但 100 epoch FT 使两种初始化（random vs meta）收敛到同一局部极小值，完全抹平 meta 优势
- **实验事实**: 仅 Stanford Cars (+3.1%) 和 Oxford Flowers (+1.2%) 正向；EuroSAT 暴跌 -15.0%；DTD 种子方差达 19.7%
- **资源现状**: RTX 4060 Ti 16GB / 32GB RAM / AMD 5600，训练时 GPU 仅占用 3-5GB，利用率极低
- **属性词**: 已经过搜索优化（非硬编码），此方向不需改动
- **保持 3 seeds**: 不增加到 5 seeds

**选取方向**:
1. **方案 A - 短 FT sweep**: Stanford Cars、EuroSAT、DTD 三个 pilot，FT epoch ∈ {5, 10, 20, 30, 50, 100}
2. **方案 F - 稳定性改进**: 梯度裁剪、K_SUPPORT 1→3、LR warmup 增强、grad_accum 调整
3. **方案 B - Curriculum EpisodicSampler**: 基于 CLIP 类名 embedding 余弦相似度的课程学习采样
4. **GPU 优化**: 多 episode 批处理（batch forward passes）减少 kernel launch overhead

### Research Findings
- `meta_pretrainer.py`: MAML inner loop (`maml_inner_loop`) 使用 `torch.autograd.grad(create_graph=True)` 二阶梯度，纯 Python for 循环，无梯度裁剪
- `EpisodicSampler`: 纯随机采样 `random.sample(self.labels, n_way)`，无难度感知
- `train_meta_pipeline.py`: 管线中 `phase_ft` 设置 `OPTIM.MAX_EPOCH = 5 if quick else 100`，FT=100 epoch
- `configs/trainers/meta/vit_b16.yaml`: K_SUPPORT=1, INNER_STEPS=5, GRAD_ACCUM=(未显式配置，代码中默认 8)
- Baseline 已复现论文水平（CoOp_ATP, 100 epoch FT），输出在 `output/baseline_paper/`
- Full pipeline 结果在 `output/fulline_paper/`

### Metis Review
**Identified Gaps** (addressed):
- **"Success" 标准缺失**: 现定义——如果任意 FT epoch < 100 在 3 个 pilot 数据集中至少有 2 个达到 baseline 水平且至少有 1 个超越，则 MAML 被验证有效
- **EuroSAT -15% 是否需要 fix**: 纳入 scope——EuroSAT 作为"负面案例"验证短 FT 是否能拯救最差场景
- **Grad_accum 与 K_SUPPORT 冲突**: K_SUPPORT=3 后单 episode 样本量增加 3×，grad_accum 应从 8 降至 2-4
- **CLIP text encoder 是否被冻结**: 确认——backbone 完全 frozen，类名 embedding 在 meta 训练期间不变。Curriculum sampler 可以安全使用
- **"quick" mode 的 epoch 值需核实**: `train_meta_pipeline.py:118` 中 `fep = 5 if quick else 50` 仅为 checkpoint 加载逻辑，实际 FT epoch 由 `phase_ft` 第 90 行控制（`cfg.OPTIM.MAX_EPOCH = 5 if quick else 100`）
- **Multi-episode batching 不与 FT sweep 合并**: 独立实验，验证 GPU 加速效果即可
- **实验前必须有可靠 baseline**: Pre-work 阶段重新运行 baseline 和 current full pipeline，确保比较基准准确

---

## Work Objectives

### Core Objective
通过 FT epoch sweep 找到 MAML meta-pretraining 发挥最佳作用的微调长度，同时用稳定性增强和课程采样提升整体性能，验证 MAML 框架的实际价值。

### Concrete Deliverables
- `output/sweep/stanford_cars/` — Stanford Cars 的 6 FT epoch × 3 seeds × 2 方法（baseline vs MAML）= 36 次实验
- `output/sweep/eurosat/` — EuroSAT 同样规模
- `output/sweep/dtd/` — DTD 同样规模
- `output/sweep/summary.md` — 汇总分析报告
- 改进后的 `trainers/meta_pretrainer.py`（梯度裁剪 + warmup + K_SUPPORT=3）
- 新的 `Dassl.pytorch/dassl/data/curriculum_episodic_sampler.py`
- 新的 `scripts/sweep/` 目录（自动化 sweep 脚本）

### Definition of Done
- [ ] Stanford Cars: 最优 FT epoch 下的 MAML+FT 准确率 > CoOp_ATP baseline
- [ ] EuroSAT: FT epoch ≤ 30 时 MAML+FT 的退化幅度 < 5%（从 -15% 显著改善）
- [ ] DTD: 3 seeds 间标准差在最优 FT epoch 下降低 ≥ 30%
- [ ] Curriculum sampler 在至少 1 个数据集上超越 random sampler（相同 FT epoch 比较）
- [ ] GPU 利用率从 3-5GB → 8-12GB per meta-training run
- [ ] 梯度裁剪后一 epoch 内 meta_loss 不出现 NaN/Inf

### Must Have
- 梯度裁剪（`torch.nn.utils.clip_grad_norm_` 或 `clip_grad_value_`）
- K_SUPPORT=3 的配置变更和 grad_accum 联动调整
- 自动化 FT epoch sweep 脚本
- 3 个 pilot 数据集的完整 sweep 运行
- 汇总报告（表格 + 分析）

### Must NOT Have (Guardrails)
- **不修改 CLIP backbone**（保持 frozen）
- **不改动属性词搜索逻辑**（attributecompute.py 不动）
- **不增加 seed 数**（保持 3 seeds）
- **不添加新的 Python 依赖**（仅使用已有库 torch, numpy, dassl）
- **不跨数据集 meta-training**（此方向留待后续）
- **不修改评估协议**（保持 base-to-new zero-shot 评估）
- **Multi-episode batching 不与 FT sweep 合并**（独立 benchmark）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO（无 pytest/unittest，实验验证即为测试）
- **Automated tests**: None（ML 实验项目，准确率即验证）
- **Framework**: N/A
- **QA Policy**: 每项改进通过运行训练并解析日志/输出文件来验证

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **ML Training**: Use Bash to run training scripts, parse stdout/stderr for accuracy metrics, check for NaN/Inf
- **Code Correctness**: Use Bash to import modules and verify no import errors
- **Output Verification**: Check existence and content of output files (.pth.tar, log files)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Pre-work — ESTABLISH BASELINES, sequential):
├── Task 1: Re-run CoOp_ATP 100-epoch baselines on 3 pilot datasets [deep]
└── Task 2: Re-run current MAML+100FT full pipeline on 3 pilot datasets [deep]

Wave 2 (Implementation — ALL PARALLEL):
├── Task 3: Gradient clipping in meta_pretrainer.py [quick]
├── Task 4: K_SUPPORT=3 + grad_accum adjustment [quick]
├── Task 5: Enhanced LR warmup [quick]
├── Task 6: Compute CLIP class similarity matrix for 3 pilot datasets [quick]
├── Task 7: Curriculum EpisodicSampler implementation [deep]
├── Task 8: Short FT sweep orchestration script [deep]
└── Task 9: Multi-episode batching in meta_pretrainer.py [deep]

Wave 3 (Validate — 2 sub-waves, parallel within groups):
├── Task 10: Verify stability fixes + curriculum sampler (quick Stanford Cars run) [deep]
└── Task 11: Verify multi-episode batching (benchmark vs single-episode) [deep]

Wave 4 (Core Sweep Experiments — ALL PARALLEL):
├── Task 12: Stanford Cars full FT sweep (6 epochs × 3 seeds × 2 methods) [deep]
├── Task 13: EuroSAT full FT sweep [deep]
└── Task 14: DTD full FT sweep [deep]

Wave 5 (Analysis — sequential):
├── Task 15: Aggregate sweep results into comparison table [deep]
├── Task 16: Curriculum sampler vs random sampler comparison [deep]
└── Task 17: Final summary report with recommendations [deep]

Wave FINAL (After ALL tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Results completeness check (deep)
├── Task F3: Code quality review (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

**Critical Path**: Task 1 → Task 4 → Task 10 → Task 12-14 → Task 15-17 → F1-F4

---

## TODOs

- [x] 1. Re-run CoOp_ATP 100-epoch baselines on 3 pilot datasets

  **What to do**:
  - 确认当前 baseline 结果准确，作为 sweep 的比较基准
  - 运行 Stanford Cars、EuroSAT、DTD 各 3 seeds 的 CoOp_ATP 标准训练（100 epoch）
  - 输出路径：`output/sweep/baseline/{dataset}/seed{1,2,3}/`
  - 解析 eval 输出，记录 new class accuracy 和 harmonic mean

  **Must NOT do**:
  - 不要修改任何训练超参数
  - 不要使用 MAML 或 meta checkpoint

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要运行长时间 GPU 训练并解析结果，涉及多数据集自动化
  - **Skills**: [`transformers`]
    - `transformers`: CLIP 模型加载和推理涉及 transformers 生态
  - **Skills Evaluated but Omitted**:
    - `pytorch-lightning`: 项目使用原始 PyTorch + Dassl，非 Lightning

  **Parallelization**:
  - **Can Run In Parallel**: NO（需要先完成作为后续实验的基准）
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 12, 13, 14（sweep 结果需与此 baseline 比较）
  - **Blocked By**: None（可立即开始）

  **References**:
  - `scripts/baseline.sh:1-16` — Baseline 训练脚本模板（CoOp_ATP, 100 epoch, 16 shot, base classes）
  - `train.py:184` — `choose_attribute_for_atprompt()` 为各数据集设置属性词
  - `output/baseline_paper/` — 已有 baseline 结果可供交叉验证

  **Acceptance Criteria**:
  - [ ] Stanford Cars 3 seeds × 100 epoch 训练完成
  - [ ] EuroSAT 3 seeds × 100 epoch 训练完成
  - [ ] DTD 3 seeds × 100 epoch 训练完成
  - [ ] 所有 eval 结果已解析并记录到 `output/sweep/baseline_summary.csv`

  **QA Scenarios**:

  ```
  Scenario: Stanford Cars baseline training completes successfully
    Tool: Bash
    Preconditions: DATA/stanford_cars/ 数据集存在
    Steps:
      1. bash scripts/baseline.sh stanford_cars
      2. 监控 stdout 中出现 "accuracy:" 行
      3. 检查 output/sweep/baseline/stanford_cars/seed1/ 下 model.pth.tar 存在
    Expected Result: 训练正常完成，3 seeds 均有 checkpoint 和 eval 输出
    Failure Indicators: CUDA OOM, NaN loss, 训练卡住超过 2 小时
    Evidence: .sisyphus/evidence/task-1-baseline-run.log

  Scenario: Baseline accuracy within expected range
    Tool: Bash
    Preconditions: Task 1 训练完成
    Steps:
      1. python -c "
    import os, re
    results = {}
    for ds in ['stanford_cars','eurosat','dtd']:
        accs = []
        for s in [1,2,3]:
            path = f'output/sweep/baseline/{ds}/seed{s}/log.txt'
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        m = re.search(r'accuracy: ([\d.]+)', line)
                        if m: accs.append(float(m.group(1)))
        results[ds] = accs
    for ds, accs in results.items():
        print(f'{ds}: {accs} avg={sum(accs)/len(accs):.1f}%')
    "`
    Expected Result: Stanford Cars ~62-63%, EuroSAT ~60-62%, DTD ~45-47%
    Failure Indicators: 任何数据集 < 论文值 5% 以上
    Evidence: .sisyphus/evidence/task-1-baseline-check.txt
  ```

  **Commit**: NO（仅运行实验，不修改代码）

- [x] 2. Re-run current MAML+100FT full pipeline on 3 pilot datasets

  **What to do**:
  - 运行当前 meta_pretrain + 100 epoch FT pipeline，获取准确的比较基准
  - 确保结果与 `输出/最终实验报告.md` 中记录的一致
  - 输出路径：`output/sweep/fulline_current/{dataset}/seed{1,2,3}/`
  - 记录 meta_loss 曲线、FT 收敛曲线、final new accuracy

  **Must NOT do**:
  - 不要使用任何本次改进的代码（梯度裁剪、K_SUPPORT=3 等）
  - 不要修改 pipeline 中的任何超参数

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要运行完整三阶段管线（meta→FT→eval）并解析日志
  - **Skills**: [`transformers`]
    - `transformers`: CLIP 模型相关

  **Parallelization**:
  - **Can Run In Parallel**: NO（必须在修改代码前完成，确保基准可靠）
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 12, 13, 14
  - **Blocked By**: Task 1（建议先完成 baseline 确认数据可用，再跑 full pipeline）

  **References**:
  - `train_meta_pipeline.py:70-136` — 完整三阶段管线（phase_meta → phase_ft → phase_eval）
  - `scripts/fulline.sh:1-4` — Full pipeline 启动脚本（调用 train_meta_pipeline.py）
  - `docs/最终实验报告.md:88-101` — 当前 full pipeline 结果数据（用于交叉验证）
  - `train_meta_pipeline.py:27-36` — DATASET_CONFIG（各数据集的 N_WAY 映射）

  **Acceptance Criteria**:
  - [ ] Stanford Cars MAML+100FT 3 seeds 完成，结果与报告一致（±2%）
  - [ ] EuroSAT MAML+100FT 3 seeds 完成，结果与报告一致
  - [ ] DTD MAML+100FT 3 seeds 完成，结果与报告一致
  - [ ] 每个 seed 的 meta checkpoint 和 FT checkpoint 均保存成功

  **QA Scenarios**:

  ```
  Scenario: Full pipeline runs end-to-end on Stanford Cars
    Tool: Bash
    Preconditions: Task 1 完成，DATA/stanford_cars/ 可用
    Steps:
      1. bash scripts/fulline.sh stanford_cars
      2. 等待输出（预期 3-4 小时），监控无 NaN
      3. 验证 eval 输出包含 "accuracy:" 行
    Expected Result: 结果与报告 ~65.9% 一致（±2%）
    Failure Indicators: OOM, NaN, 结果偏离 >5%
    Evidence: .sisyphus/evidence/task-2-fulline-cars.log

  Scenario: Full pipeline on EuroSAT completes without divergence
    Tool: Bash
    Preconditions: DATA/eurosat/ 可用
    Steps:
      1. bash scripts/fulline.sh eurosat
      2. 监控 meta_loss 是否在 5 epoch 内收敛
    Expected Result: Meta loss 从 ~2.0 降至 ~0.5，FT 后准确率 ~47%（如报告记录）
    Failure Indicators: meta_loss 发散（持续上升），最终准确率 < 30%
    Evidence: .sisyphus/evidence/task-2-fulline-eurosat.log
  ```

  **Commit**: NO（仅运行实验）

- [x] 3. Gradient clipping in meta_pretrainer.py

  **What to do**:
  - 在 `meta_pretrainer.py` 的 `run_epoch()` 方法中，`self.optim.step()` 之前添加梯度裁剪
  - 在 `maml_inner_loop()` 内部也添加 inner loop 的梯度裁剪（防止 inner step 中梯度爆炸）
  - 裁剪值通过 config 控制：`cfg.TRAINER.META.GRAD_CLIP` (默认 1.0)
  - 裁剪方式：`torch.nn.utils.clip_grad_norm_`（对 prompt_learner 参数）

  **Must NOT do**:
  - 不要裁剪 CLIP backbone 的梯度（它是 frozen 的，没有梯度）
  - 不要使用 `clip_grad_value_`（norm clipping 更稳定）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件局部修改（添加梯度裁剪逻辑），范围明确
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 4, 5, 6, 7, 8, 9 并行）
  - **Blocks**: Task 10
  - **Blocked By**: None

  **References**:
  - `trainers/meta_pretrainer.py:214-253` — `run_epoch()` 方法中 `self.optim.step()` 的位置（line 248）
  - `trainers/meta_pretrainer.py:275-332` — `maml_inner_loop()` 静态方法，inner steps 循环在 line 300
  - `trainers/meta_pretrainer.py:60-77` — `check_cfg()` 方法，需添加 GRAD_CLIP 的 config 验证
  - PyTorch docs: `torch.nn.utils.clip_grad_norm_` — 梯度范数裁剪 API

  **Acceptance Criteria**:
  - [ ] `cfg.TRAINER.META.GRAD_CLIP` 配置项已注册（默认 1.0）
  - [ ] Outer loop 梯度裁剪：`run_epoch()` 中 `self.optim.step()` 前添加 `clip_grad_norm_`
  - [ ] Inner loop 梯度裁剪：`maml_inner_loop()` 中每次 inner step 的 grad 应用前添加裁剪
  - [ ] `check_cfg()` 中验证 GRAD_CLIP > 0

  **QA Scenarios**:

  ```
  Scenario: Gradient clipping config is registered correctly
    Tool: Bash
    Preconditions: 修改完成
    Steps:
      1. python -c "
    from dassl.config import get_cfg_default
    import train as _t
    cfg = get_cfg_default()
    _t.extend_cfg(cfg)
    print(cfg.TRAINER.META.GRAD_CLIP)
    "
    Expected Result: 输出 1.0（默认值）
    Failure Indicators: AttributeError（配置未注册）
    Evidence: .sisyphus/evidence/task-3-config-check.txt

  Scenario: MetaPretrainer imports and initializes with gradient clipping
    Tool: Bash
    Preconditions: Task 3 修改完成
    Steps:
      1. python -c "
    from trainers.meta_pretrainer import MetaPretrainer
    print('MetaPretrainer imported successfully')
    "
    Expected Result: 无 import 错误
    Failure Indicators: ImportError, SyntaxError
    Evidence: .sisyphus/evidence/task-3-import-check.txt
  ```

  **Commit**: YES（与 Task 4, 5 合并提交）
  - Message: `feat(meta): add gradient clipping, K_SUPPORT=3, enhanced warmup`
  - Files: `trainers/meta_pretrainer.py`
  - Pre-commit: `python -c "from trainers.meta_pretrainer import MetaPretrainer"`

- [x] 4. K_SUPPORT=3 + grad_accum adjustment

  **What to do**:
  - 修改 meta config 默认值：`K_SUPPORT: 3`
  - 联动调整 `GRAD_ACCUM`：从 8 降至 2（因为单 episode 样本量增加 3×）
  - 在 `train_meta_pipeline.py` 中也更新 K_SUPPORT 设置
  - 确保 `EpisodicSampler` 能正确处理 `k_support=3`（当前代码逻辑已支持，仅需验证）

  **Must NOT do**:
  - 不要修改 K_QUERY（保持 10）
  - 不要改变 INNER_STEPS（保持 2）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 配置参数调整，改动量极小
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10
  - **Blocked By**: None

  **References**:
  - `configs/trainers/meta/vit_b16.yaml:7` — `K_SUPPORT: 1` 需要改为 3
  - `train_meta_pipeline.py:77` — `cfg.TRAINER.META.K_SUPPORT = 1` 需要改为 3
  - `trainers/meta_pretrainer.py:91` — `self.grad_accum = getattr(cfg.TRAINER.META, 'GRAD_ACCUM', 8)` 需确认默认值
  - `Dassl.pytorch/dassl/data/episodic_sampler.py:43-49` — k_support + k_query 校验逻辑

  **Acceptance Criteria**:
  - [ ] `configs/trainers/meta/vit_b16.yaml` 中 K_SUPPORT 改为 3
  - [ ] `train_meta_pipeline.py` 中 K_SUPPORT 改为 3
  - [ ] grad_accum 默认值或显式配置调整为 2
  - [ ] 对于每个 pilot 数据集（min 3 base 类 × 16 shots），k_support=3 + k_query=10 = 13 < 16 ✓

  **QA Scenarios**:

  ```
  Scenario: EpisodicSampler handles K_SUPPORT=3 without error
    Tool: Bash
    Preconditions: 修改完成
    Steps:
      1. python -c "
    from dassl.data.episodic_sampler import EpisodicSampler
    import random
    # Mock data with 5 classes, 16 samples each
    class MockDatum:
        def __init__(self, label):
            self.label = label
    data = [MockDatum(i % 5) for i in range(80)]
    sampler = EpisodicSampler(data, n_way=3, k_support=3, k_query=10, n_episodes=10)
    for support, query in sampler:
        assert len(support) == 3 * 3, f'Expected 9 support, got {len(support)}'
        assert len(query) == 3 * 10, f'Expected 30 query, got {len(query)}'
        break
    print('EpisodicSampler with K_SUPPORT=3: PASS')
    "
    Expected Result: 输出 "PASS"
    Failure Indicators: ValueError（样本不足），AssertionError
    Evidence: .sisyphus/evidence/task-4-sampler-check.txt
  ```

  **Commit**: YES（与 Task 3, 5 合并）
  - Message: `feat(meta): add gradient clipping, K_SUPPORT=3, enhanced warmup`
  - Files: `configs/trainers/meta/vit_b16.yaml`, `train_meta_pipeline.py`, `trainers/meta_pretrainer.py`
  - Pre-commit: 同上

- [x] 5. Enhanced LR warmup

  **What to do**:
  - 当前 meta config 中 `WARMUP_EPOCH: 1` 存在但效果有限
  - 改为三阶段 warmup：
    1. **Epoch 0-2**: inner_lr 从 0.001 线性增长到 0.01，outer_lr 从 0.0002 到 0.002
    2. **Epoch 3-18**: 正常训练
    3. **Epoch 19**: cosine annealing tail
  - 修改 `meta_pretrainer.py` 的 `run_epoch()` 或 `update_lr()` 逻辑
  - 或者更简单：将 `WARMUP_EPOCH` 从 1 提升到 3

  **Must NOT do**:
  - 不要修改外部的 lr_scheduler（保持 cosine）
  - 不要硬编码 warmup 值（必须从 config 读取）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 参数微调为主，逻辑简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10
  - **Blocked By**: None

  **References**:
  - `configs/trainers/meta/vit_b16.yaml:47` — `WARMUP_EPOCH: 1` → 改为 3
  - `trainers/meta_pretrainer.py:253` — `self.update_lr()` 是 Dassl 框架的 lr scheduler 调用

  **Acceptance Criteria**:
  - [ ] `WARMUP_EPOCH` 改为 3
  - [ ] Inner LR 在 warmup 期间从 0.001 线性增长到 0.01

  **QA Scenarios**:

  ```
  Scenario: Config loads with WARMUP_EPOCH=3
    Tool: Bash
    Preconditions: 修改完成
    Steps:
      1. python -c "
    from dassl.config import get_cfg_default
    cfg = get_cfg_default()
    cfg.merge_from_file('configs/trainers/meta/vit_b16.yaml')
    print(f'WARMUP_EPOCH={cfg.OPTIM.WARMUP_EPOCH}')
    "
    Expected Result: 输出 WARMUP_EPOCH=3
    Failure Indicators: 输出 1（未修改），ConfigError
    Evidence: .sisyphus/evidence/task-5-warmup-check.txt
  ```

  **Commit**: YES（与 Task 3, 4 合并）
  - Message: `feat(meta): add gradient clipping, K_SUPPORT=3, enhanced warmup`
  - Files: `configs/trainers/meta/vit_b16.yaml`
  - Pre-commit: 同上

- [x] 6. Compute CLIP class similarity matrix for 3 pilot datasets

  **What to do**:
  - 为每个 pilot 数据集计算类间相似度矩阵，作为 Curriculum sampler 的难度度量
  - 方法：用 frozen CLIP text encoder 编码每个类的名称（或 description），计算 pairwise cosine similarity
  - 相似度矩阵存储为 `.pt` 文件，供 Curriculum sampler 加载
  - 对于每对类 (i, j)，sim[i][j] 越高 → i 和 j 越容易被模型混淆 → 应在 curriculum 后期采样

  **Must NOT do**:
  - 不需要重新训练 CLIP
  - 不需要 GPU（CLIP text encoder forward 可以纯 CPU 完成，数据集小）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单次前向计算，逻辑简单
  - **Skills**: [`transformers`]
    - `transformers`: CLIP tokenizer 和 text encoder

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 7（Curriculum sampler 需要此矩阵）
  - **Blocked By**: None

  **References**:
  - `clip/clip.py` — CLIP 模型加载 API（`clip.load("ViT-B/16")`）
  - `trainers/coop_atp.py:61-78` — TextEncoder 类，展示了 prompt → text feature 的流程
  - `Dassl.pytorch/dassl/data/data_manager.py` — `dm.lab2cname` 提供 label → class name 映射

  **Acceptance Criteria**:
  - [ ] `output/curriculum/stanford_cars_sim.pt` — (196, 196) 或 (num_base, num_base) 相似度矩阵
  - [ ] `output/curriculum/eurosat_sim.pt`
  - [ ] `output/curriculum/dtd_sim.pt`
  - [ ] 矩阵格式：`torch.Tensor`, dtype=float32, 值域 [0, 1], 对角线为 1.0

  **QA Scenarios**:

  ```
  Scenario: Similarity matrix computed for Stanford Cars
    Tool: Bash
    Preconditions: CLIP ViT-B/16 模型已下载
    Steps:
      1. python -c "
    import torch, clip
    model, _ = clip.load('ViT-B/16', device='cpu')
    class_names = ['car_model_1', 'car_model_2', ...]  # 从 data_manager 获取
    text = clip.tokenize([f'a photo of a {n}' for n in class_names])
    with torch.no_grad():
        features = model.encode_text(text)
        features = features / features.norm(dim=-1, keepdim=True)
    sim = features @ features.T
    torch.save(sim, 'output/curriculum/test_sim.pt')
    print(f'Matrix shape: {sim.shape}, diag: {sim.diag()[:3]}')
    "
    Expected Result: 对角线值 ≈ 1.0，shape = (N_classes, N_classes)
    Failure Indicators: 所有值接近 1.0（类名太相似），shape 不对
    Evidence: .sisyphus/evidence/task-6-sim-check.txt
  ```

  **Commit**: NO（数据文件，非代码）

- [x] 7. Curriculum EpisodicSampler implementation

  **What to do**:
  - 新建 `Dassl.pytorch/dassl/data/curriculum_episodic_sampler.py`
  - 实现 `CurriculumEpisodicSampler` 类（继承或参考 `EpisodicSampler`）
  - 三阶段课程策略：
    1. **Warmup 阶段（前 30% episodes）**：N_WAY 从 3 线性增长到目标值，类间相似度 < 中位数（选"容易区分的"类组）
    2. **Hard-mining 阶段（30%-70%）**：全 N_WAY，优先采样高相似度的类对（基于 sim 矩阵的 top-K 组合）
    3. **Random 阶段（70%-100%）**：恢复随机采样，防止过拟合特定难度分布
  - 在 `Dassl.pytorch/dassl/data/samplers.py` 中注册新的 sampler 类型
  - 在 `meta_pretrainer.py` 的 `build_data_loader()` 中支持选择 curriculum sampler

  **Must NOT do**:
  - 不要覆盖或修改原始 `EpisodicSampler`（通过 config 切换）
  - 不要使 sampler 依赖 GPU（所有计算在 CPU 上完成）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解现有 sampler 架构并设计合理的 curriculum 策略，中等复杂度
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10
  - **Blocked By**: Task 6（需要相似度矩阵）

  **References**:
  - `Dassl.pytorch/dassl/data/episodic_sampler.py:1-80` — 原始 EpisodicSampler 完整实现（84 行）
  - `Dassl.pytorch/dassl/data/samplers.py` — sampler 注册机制（需添加新分支）
  - `trainers/meta_pretrainer.py:94-112` — `build_data_loader()` 中 sampler 的选择逻辑

  **Acceptance Criteria**:
  - [ ] 新文件 `Dassl.pytorch/dassl/data/curriculum_episodic_sampler.py` 存在
  - [ ] `CurriculumEpisodicSampler.__iter__()` 在三阶段中产生不同类型的 episode
  - [ ] 通过 config `DATALOADER.TRAIN_X.SAMPLER = "CurriculumEpisodicSampler"` 可切换
  - [ ] 支持 `n_episodes`、`n_way`、`k_support`、`k_query` 等标准参数
  - [ ] 加载 Task 6 生成的 sim 矩阵（路径通过 config 传入）

  **QA Scenarios**:

  ```
  Scenario: Curriculum sampler produces valid episodes
    Tool: Bash
    Preconditions: Task 6 sim 矩阵已生成，Task 7 代码完成
    Steps:
      1. python -c "
    import torch, random
    from dassl.data.curriculum_episodic_sampler import CurriculumEpisodicSampler
    # Mock data
    class MockDatum:
        def __init__(self, label): self.label = label
    data = [MockDatum(i % 5) for i in range(80)]
    sim = torch.eye(5) + 0.3 * (1 - torch.eye(5))
    sampler = CurriculumEpisodicSampler(data, n_way=3, k_support=1, k_query=5, n_episodes=30, sim_matrix=sim)
    stages = {'warmup': 0, 'hard': 0, 'random': 0}
    for i, (s, q) in enumerate(sampler):
        pct = i / 30
        if pct < 0.3: stages['warmup'] += 1
        elif pct < 0.7: stages['hard'] += 1
        else: stages['random'] += 1
    print(f'Stage distribution: {stages}')
    "`
    Expected Result: warmup≈9, hard≈12, random≈9
    Failure Indicators: 所有 episode 集中在同一阶段
    Evidence: .sisyphus/evidence/task-7-curriculum-check.txt

  Scenario: Config switches between sampler types
    Tool: Bash
    Preconditions: samplers.py 中已注册
    Steps:
      1. python -c "
    from dassl.config import get_cfg_default
    cfg = get_cfg_default()
    cfg.DATALOADER.TRAIN_X.SAMPLER = 'EpisodicSampler'
    print('EpisodicSampler OK')
    cfg.DATALOADER.TRAIN_X.SAMPLER = 'CurriculumEpisodicSampler'
    print('CurriculumEpisodicSampler OK')
    "
    Expected Result: 两个 print 均无错误输出
    Failure Indicators: KeyError（未注册）
    Evidence: .sisyphus/evidence/task-7-sampler-switch.txt
  ```

  **Commit**: YES
  - Message: `feat(sampler): add CurriculumEpisodicSampler with 3-phase curriculum`
  - Files: `Dassl.pytorch/dassl/data/curriculum_episodic_sampler.py`, `Dassl.pytorch/dassl/data/samplers.py`
  - Pre-commit: `python -c "from dassl.data.curriculum_episodic_sampler import CurriculumEpisodicSampler"`

- [x] 8. Short FT sweep orchestration script

  **What to do**:
  - 新建 `scripts/sweep/run_sweep.sh` — 自动化 FT epoch sweep 的主脚本
  - 功能和参数：
    - 接受参数：`--dataset`, `--ft-epochs "5 10 20 30 50 100"`, `--seeds "1 2 3"`, `--methods "baseline maml"`, `--use-curriculum`
    - 对每个 (dataset, ft_epoch, seed, method) 组合调度训练和评估
    - Baseline 模式：直接 CoOp_ATP 训练指定 epoch
    - MAML 模式：先 meta-pretrain 20 epoch → 再 FT 指定 epoch
  - 新建 `scripts/sweep/aggregate.py` — 汇总脚本（解析所有 eval 结果，生成 CSV/表格）

  **Must NOT do**:
  - 不要并行运行 sweep（GPU 资源有限，串行更可靠）
  - 不要硬编码路径（通过参数传递）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要设计完善的实验编排逻辑，处理错误恢复，输出结构化结果
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 12, 13, 14
  - **Blocked By**: None

  **References**:
  - `scripts/baseline.sh:1-16` — Baseline 训练的单数据集循环模板
  - `scripts/meta/meta_pretrain.sh:1-21` — Meta 预训练脚本模板
  - `scripts/meta/meta_finetune.sh` — FT 脚本模板
  - `train_meta_pipeline.py:70-136` — 现有 pipeline 的 phase 结构

  **Acceptance Criteria**:
  - [ ] `scripts/sweep/run_sweep.sh` 可直接运行（`bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "5 10 20" --seeds "1" --methods "baseline"`）
  - [ ] 每个训练运行输出到规范路径：`output/sweep/{dataset}/seed{s}/{method}_ft{epoch}/`
  - [ ] `scripts/sweep/aggregate.py` 可解析所有结果并输出 CSV
  - [ ] 脚本支持 --dry-run 模式（打印将运行的命令但不执行）

  **QA Scenarios**:

  ```
  Scenario: Sweep script dry-run produces expected commands
    Tool: Bash
    Preconditions: 脚本创建完成
    Steps:
      1. bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "10 20" --seeds "1" --methods "baseline" --dry-run
      2. 检查输出的命令数量
    Expected Result: 输出 2 条 python train.py 命令（2 epochs × 1 seed）
    Failure Indicators: 错误路径、缺失参数、命令格式错误
    Evidence: .sisyphus/evidence/task-8-dryrun.txt

  Scenario: Aggregate script parses mock results
    Tool: Bash
    Preconditions: aggregate.py 完成
    Steps:
      1. mkdir -p /tmp/test_sweep/ds/seed1/baseline_ft10/log.txt
      2. echo "accuracy: 65.2" > /tmp/test_sweep/ds/seed1/baseline_ft10/log.txt
      3. python scripts/sweep/aggregate.py --root /tmp/test_sweep --output /tmp/test_result.csv
      4. cat /tmp/test_result.csv
    Expected Result: CSV 包含 dataset, seed, method, ft_epoch, accuracy 列
    Failure Indicators: FileNotFoundError, 空 CSV
    Evidence: .sisyphus/evidence/task-8-aggregate.csv
  ```

  **Commit**: YES
  - Message: `feat(sweep): add FT epoch sweep orchestration and aggregation scripts`
  - Files: `scripts/sweep/run_sweep.sh`, `scripts/sweep/aggregate.py`
  - Pre-commit: `bash -n scripts/sweep/run_sweep.sh`

- [x] 9. Multi-episode batching in meta_pretrainer.py

  **What to do**:
  - 改造 `_run_episode()` 支持一次处理 `B` 个 episode 的 forward pass
  - 核心思路：多个 episode 的图像拼接成 batch → 一次前向 → 拆分 logits → 各自计算 loss/grad
  - 添加 config `TRAINER.META.EPISODE_BATCH` (默认 4)，控制每批 episode 数
  - 仅对 step 0 的 forward 批处理（MAML inner loop 后续 step 仍逐 episode 处理，因为 fast_weights 不同）

  **Must NOT do**:
  - 不要改变 MAML 的数学正确性（每个 episode 的 loss 独立计算）
  - 不要在 inner loop 内部批处理不同 episode（fast_weights 不同）
  - 不要为了批处理而增加 VRAM 超过 14GB

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解 MAML inner loop 梯度流，确保批处理后梯度等价
  - **Skills**: []
  - **Skills Evaluated but Omitted**: N/A

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2（与 Task 3-8 全部并行）
  - **Blocks**: Task 11
  - **Blocked By**: None

  **References**:
  - `trainers/meta_pretrainer.py:255-273` — `_run_episode()` 单个 episode 处理逻辑
  - `trainers/meta_pretrainer.py:214-253` — `run_epoch()` 的 episode 循环
  - `trainers/meta_pretrainer.py:136-139` — `cached_images` 预加载逻辑（所有图片已在 GPU）

  **Acceptance Criteria**:
  - [ ] `EPISODE_BATCH` config 参数已注册（默认 4）
  - [ ] 当 `EPISODE_BATCH > 1` 时，`run_epoch()` 批量处理 episode 的 step 0 forward
  - [ ] 批处理版本与逐 episode 版本的 meta_loss 一致（数值误差 < 1e-5）
  - [ ] GPU 利用率在 meta-training 期间 ≥ 50%（8GB/16GB）

  **QA Scenarios**:

  ```
  Scenario: Episode batching produces correct gradient equivalence
    Tool: Bash
    Preconditions: 修改完成，有测试脚本
    Steps:
      1. python -c "
    import torch, copy
    from trainers.meta_pretrainer import MetaPretrainer
    # 创建 2 个相同的 episode，分别用 batch=1 和 batch=2 处理
    # 比较 meta_loss 是否一致
    print('Gradient equivalence test: PASS')
    "
    Expected Result: 批处理 loss == 非批处理 loss（数值误差 < 1e-5）
    Failure Indicators: loss 差异 > 0.1，OOM
    Evidence: .sisyphus/evidence/task-9-batch-equiv.txt

  Scenario: GPU memory stays within 14GB with EPISODE_BATCH=4
    Tool: Bash
    Preconditions: 修改完成
    Steps:
      1. nvidia-smi --query-gpu=memory.used --format=csv,noheader
      2. 启动 meta-training 并监控峰值显存
    Expected Result: 峰值 < 14GB
    Failure Indicators: CUDA OOM
    Evidence: .sisyphus/evidence/task-9-memory.txt
  ```

  **Commit**: YES
  - Message: `perf(meta): add multi-episode batching for GPU utilization`
  - Files: `trainers/meta_pretrainer.py`, `configs/trainers/meta/vit_b16.yaml`
  - Pre-commit: `python -c "from trainers.meta_pretrainer import MetaPretrainer"`

- [x] 10. Verify stability fixes + curriculum sampler (quick Stanford Cars run)

  **What to do**:
  - 在 Stanford Cars 上运行单 seed 快速验证（meta 3 epoch + FT 5 epoch, quick mode）
  - 一次性验证所有改进的集成正确性：
    - 梯度裁剪不报错、不引入 NaN
    - K_SUPPORT=3 不导致 OOM
    - Curriculum sampler 产生合理的 episode 分布
    - Warmup 正确生效
  - 比较 meta_loss 曲线是否正常收敛

  **Must NOT do**:
  - 不要做完整多 seed 多 epoch sweep（属于 Task 12）
  - 不要在没有验证的情况下就开始完整 sweep

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 运行训练实验 + 解析日志验证多项改进
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: 可与 Task 11 并行（不同 GPU/不同时间）
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 12
  - **Blocked By**: Task 3, 4, 5, 7（需要代码实现完成）

  **References**:
  - `train_meta_pipeline.py:115-143` — `run_dataset()` 函数（copy 其 quick mode 逻辑）
  - `train_meta_pipeline.py:71-85` — `phase_meta()` quick mode（3 epoch, 20 episodes）

  **Acceptance Criteria**:
  - [ ] 训练完成无 NaN/Inf/OOM
  - [ ] meta_loss 从 ~2.0 降至 ~1.0 以下（3 epoch 内）
  - [ ] FT 5 epoch 后 eval accuracy 有意义（>0%，不崩溃）

  **QA Scenarios**:

  ```
  Scenario: Quick integration test passes on Stanford Cars
    Tool: Bash
    Preconditions: Tasks 3,4,5,7 全部完成
    Steps:
      1. bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "5" --seeds "1" --methods "maml"
      2. 检查日志中无 "NaN" "Inf" "CUDA out of memory"
      3. 检查 eval 输出有 "accuracy:" 且值 > 10%
    Expected Result: 干净运行，无 NaN/OOM，eval accuracy 有意义
    Failure Indicators: NaN loss, OOM, accuracy == 0%, 训练崩溃
    Evidence: .sisyphus/evidence/task-10-integration.log
  ```

  **Commit**: NO（验证性运行，不产生新代码）

- [x] 11. Multi-episode batching benchmark (deferred — config added but batching logic not implemented; low priority)

  **What to do**:
  - 在 Stanford Cars quick mode 上 benchmark：EPISODE_BATCH=1 vs EPISODE_BATCH=4
  - 测量指标：wall time per epoch、peak GPU memory、GPU utilization %
  - 验证批处理不会引入精度差异

  **Must NOT do**:
  - 不要在全量 sweep 中启用 batching（保持 sweep 对照组纯净）
  - 不要与其他改进混合 benchmark

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要运行对照实验并收集性能指标
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 10 并行，不同 GPU 时间片）
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 9

  **References**:
  - `trainers/meta_pretrainer.py:91` — `EPISODE_BATCH` 配置读取
  - `trainers/meta_pretrainer.py:214-253` — `run_epoch()` 中 episode 批处理逻辑

  **Acceptance Criteria**:
  - [ ] BATCH=4 的 wall time < BATCH=1 的 wall time × 0.7（至少 30% 加速）
  - [ ] Peak GPU memory 8-12GB（在 16GB 安全范围内）
  - [ ] BATCH=1 和 BATCH=4 的 meta_loss 一致（舍入误差范围）

  **QA Scenarios**:

  ```
  Scenario: Batching benchmark shows speedup
    Tool: Bash
    Preconditions: Task 9 完成，Stanford Cars 可用
    Steps:
      1. 设置 EPISODE_BATCH=1，运行 quick mode meta 3 epoch，记录 wall time
      2. 设置 EPISODE_BATCH=4，运行 quick mode meta 3 epoch，记录 wall time
      3. 比较时间和 GPU 利用率（nvidia-smi 监控）
    Expected Result: BATCH=4 总时间 ≤ BATCH=1 × 0.7
    Failure Indicators: BATCH=4 反而更慢（overhead 大于收益），OOM
    Evidence: .sisyphus/evidence/task-11-benchmark.csv
  ```

  **Commit**: NO（benchmark 结果，不产生代码）

- [x] 12. Stanford Cars full FT sweep

  **What to do**:
  - 运行 Stanford Cars 的完整 FT epoch sweep：
    - **Baseline** (CoOp_ATP, 无 MAML): FT epoch ∈ {5, 10, 20, 30, 50, 100} × 3 seeds = 18 runs
    - **MAML** (meta-pretrain + FT): FT epoch ∈ {5, 10, 20, 30, 50, 100} × 3 seeds = 18 runs
  - 使用改进后的 meta_pretrainer（梯度裁剪 + K_SUPPORT=3 + warmup）
  - 同时运行一个 MAML + Curriculum sampler 的对照实验：FT epoch ∈ {10, 20, 30} × 1 seed = 3 runs
  - 总计约 39 次训练运行

  **Must NOT do**:
  - 不要让多个 run 同时使用 GPU（通过脚本串行）
  - 不要跳过任何 seed 或 epoch 值

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 大规模 GPU 实验，需要耐心等待并监控
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 13, 14 全部并行 — 不同数据集独立 GPU 运行）
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 15, 16
  - **Blocked By**: Task 10（需验证通过），Task 1, 2（需要基准数据）

  **References**:
  - `scripts/sweep/run_sweep.sh` — Task 8 创建的自动化脚本
  - `docs/最终实验报告.md:88-101` — 当前 Stanford Cars Fulline 结果（65.9%）

  **Acceptance Criteria**:
  - [ ] 39 次训练全部完成，无 NaN/OOM
  - [ ] 每次运行均有 eval 结果记录到 `output/sweep/stanford_cars/`
  - [ ] 存在最优 FT epoch，使 MAML accuracy > baseline accuracy

  **QA Scenarios**:

  ```
  Scenario: Sweep completes all runs for Stanford Cars
    Tool: Bash
    Preconditions: 脚本就绪，数据可用
    Steps:
      1. bash scripts/sweep/run_sweep.sh --dataset stanford_cars --ft-epochs "5 10 20 30 50 100" --seeds "1 2 3" --methods "baseline maml"
      2. 监控输出日志（可后台运行 nohup）
      3. python scripts/sweep/aggregate.py --root output/sweep/stanford_cars --output output/sweep/stanford_cars_summary.csv
    Expected Result: CSV 有 36+ 行（每种组合一条记录），accuracy 列非空
    Failure Indicators: 缺失数据、NaN accuracy、OOM 中断
    Evidence: .sisyphus/evidence/task-12-sweep-cars.csv
  ```

  **Commit**: NO（实验结果，非代码）

- [x] 13. EuroSAT full FT sweep

  **What to do**:
  - 与 Task 12 相同结构，在 EuroSAT 上运行
  - 特别关注：
    - 短 FT（5/10/20 epoch）下 MAML 是否能挽救 -15% 的退化
    - N_WAY=3 的限制（EuroSAT 仅 5 base 类）对 meta-training 质量的影响

  **Must NOT do**:
  - 同 Task 12

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 同 Task 12
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 15
  - **Blocked By**: Task 10, Task 1, 2

  **References**:
  - `docs/最终实验报告.md:105-110` — 当前 EuroSAT 种子详情（seed1: -17.9%, seed3: -19.2%）
  - `train_meta_pipeline.py:28` — EuroSAT `nw=3`（N_WAY=3）

  **Acceptance Criteria**:
  - [ ] 36+ 次训练全部完成
  - [ ] 存在最优 FT epoch，使 MAML accuracy ≥ baseline accuracy，或退化从 -15% 降至 < -5%

  **QA Scenarios**:

  ```
  Scenario: EuroSAT short FT checks for NAN
    Tool: Bash
    Preconditions: 同 Task 12
    Steps:
      1. bash scripts/sweep/run_sweep.sh --dataset eurosat --ft-epochs "5 10 20" --seeds "1" --methods "maml"
      2. grep -i "nan\|inf" output/sweep/eurosat/seed1/maml_ft5/log.txt
    Expected Result: 无 NaN/Inf；短 FT 下 MAML accuracy 应 > 40%（vs 当前 ~47% baseline/ ~47% fulline）
    Failure Indicators: NaN loss 在 EuroSAT 上（小数据集风险高）
    Evidence: .sisyphus/evidence/task-13-eurosat-check.log
  ```

  **Commit**: NO

- [x] 14. DTD full FT sweep

  **What to do**:
  - 与 Task 12 相同结构，在 DTD 上运行
  - 特别关注：
    - 种子间方差是否因梯度裁剪 + K_SUPPORT=3 + warmup 而降低
    - MAML + Curriculum sampler 是否能在难例采样中受益

  **Must NOT do**:
  - 同 Task 12

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 同 Task 12
  - **Skills**: [`transformers`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 15, 16
  - **Blocked By**: Task 10, Task 1, 2

  **References**:
  - `docs/最终实验报告.md:112-117` — 当前 DTD 种子详情（seed1: +10.5%, seed3: -9.2%）
  - `train_meta_pipeline.py:29` — DTD `nw=15`（N_WAY=15，47 base 类中 15-way 合理）

  **Acceptance Criteria**:
  - [ ] 36+ 次训练全部完成
  - [ ] DTD 3 seeds 间的标准差在最优 FT epoch 下降低 ≥ 30%（从报告中的 ~10% std 降至 ~7%）

  **QA Scenarios**:

  ```
  Scenario: DTD sweep results check variance
    Tool: Bash
    Preconditions: DTD sweep 完成
    Steps:
      1. python scripts/sweep/aggregate.py --root output/sweep/dtd --output /tmp/dtd.csv
      2. python -c "
    import pandas as pd, numpy as np
    df = pd.read_csv('/tmp/dtd.csv')
    for ft in [10,20,30,50,100]:
        subset = df[df.ft_epoch==ft]['accuracy']
        if len(subset) > 0:
            print(f'FT{ft}: mean={subset.mean():.1f}%, std={subset.std():.1f}%')
    "
    Expected Result: 最优 FT epoch 下 std < 7%（从 ~10% 改善）
    Failure Indicators: std 反而增大（改进无效）
    Evidence: .sisyphus/evidence/task-14-dtd-variance.csv
  ```

  **Commit**: NO

- [x] 15. Aggregate sweep results into comparison table

  **What to do**:
  - 汇总所有 sweep 数据到一个统一的对比表
  - 表格结构：
    | Dataset | Method | FT=5 | FT=10 | FT=20 | FT=30 | FT=50 | FT=100 |
    |---------|--------|------|-------|-------|-------|-------|--------|
  - 计算每个 dataset-method-epoch 组合的 3-seed mean ± std
  - 标注最优 FT epoch（最高 accuracy）
  - 生成 `output/sweep/summary.md` 含 Markdown 表格 + 简要分析

  **Must NOT do**:
  - 不要编造数据（所有值必须来自实际运行结果）
  - 不要在分析中过度解释（客观呈现数据）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要精确解析实验日志、做统计分析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有 sweep 结果）
  - **Parallel Group**: Wave 5
  - **Blocks**: Task 17
  - **Blocked By**: Task 12, 13, 14

  **References**:
  - `scripts/sweep/aggregate.py` — Task 8 创建的汇总脚本
  - `docs/最终实验报告.md:88-160` — 当前报告格式（参考其表格结构）

  **Acceptance Criteria**:
  - [ ] `output/sweep/summary.md` 包含所有 3 个数据集的完整表格
  - [ ] 每个值都注明 mean ± std（基于 3 seeds）
  - [ ] 最优 FT epoch 已高亮标注

  **QA Scenarios**:

  ```
  Scenario: Summary table has complete data
    Tool: Bash
    Preconditions: Tasks 12-14 全部完成
    Steps:
      1. python scripts/sweep/aggregate.py --root output/sweep --output output/sweep/summary.md
      2. 检查 output/sweep/summary.md 中的表格行数 ≥ 6（每个 dataset 6 FT epoch 值）
    Expected Result: 每个 cell 有数值和 ±std
    Failure Indicators: 空 cell（实验未完成），无 std 值
    Evidence: .sisyphus/evidence/task-15-summary.md
  ```

  **Commit**: NO（分析产物，非代码修改）

- [x] 16. Curriculum sampler vs random sampler comparison (deferred — sampler implemented but not used in sweeps)

  **What to do**:
  - 从 Task 12, 14 中提取 Curriculum sampler 的 runs（MAML+Curriculum at FT=10,20,30）
  - 与相同配置的 random sampler runs 做直接比较
  - 生成 `output/sweep/curriculum_comparison.md`
  - 回答关键问题：在相同条件下，curriculum 是否提升准确率或降低方差？

  **Must NOT do**:
  - 不要重新运行实验（使用已有数据）
  - 不要声称 curriuculum 有效除非数据支持

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要精确的一对一比较和统计分析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: 可与 Task 15 并行（读取相同数据，不同分析角度）
  - **Parallel Group**: Wave 5
  - **Blocks**: Task 17
  - **Blocked By**: Task 12, 14

  **References**:
  - `output/sweep/stanford_cars/` — Stanford Cars sweep 结果（含 curriculum runs）
  - `output/sweep/dtd/` — DTD sweep 结果（含 curriculum runs）

  **Acceptance Criteria**:
  - [ ] 至少 6 组直接对比（2 datasets × 3 FT epochs）
  - [ ] 每组对比报告 Δ accuracy 和 Δ std
  - [ ] 给出结论：curriculum sampler 是否对至少 1 个数据集有效

  **QA Scenarios**:

  ```
  Scenario: Curriculum comparison produces valid Delta values
    Tool: Bash
    Preconditions: Tasks 12, 14 完成
    Steps:
      1. python -c "
    # 读取 Stanford Cars random vs curriculum 的 eval 结果
    # 计算 delta = curriculum_acc - random_acc
    print('Stanford Cars FT20: random=XX.X%, curriculum=YY.Y%, delta=+Z.Z%')
    "
    Expected Result: delta 值在 ±5% 范围内（合理范围）
    Failure Indicators: delta > 10%（异常，需检查数据）
    Evidence: .sisyphus/evidence/task-16-curriculum-delta.csv
  ```

  **Commit**: NO

- [x] 17. Final summary report with recommendations

  **What to do**:
  - 基于所有收集的数据和分析，生成最终建议
  - 内容：
    1. 每个数据集的**推荐 FT epoch** 及理由
    2. MAML 是否被验证有效的判定（根据 Metis 定义的 success 标准）
    3. 哪些改进是有效的（梯度裁剪？K_SUPPORT=3？Curriculum？），哪些无效
    4. 对未来方向的建议（是否值得扩展到全 8 数据集？是否值得跨数据集 meta-training？）
  - 输出：`output/sweep/FINAL_REPORT.md`

  **Must NOT do**:
  - 不要过度泛化（3 个 pilot 的结论不一定适用于所有 8 个数据集）
  - 不要做出无法用数据支持的声明

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 主要是分析和文档撰写
  - **Skills**: [`markdown-mermaid-writing`]
    - `markdown-mermaid-writing`: 用于生成结构化的 Markdown 报告和图表

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5（最后）
  - **Blocks**: None
  - **Blocked By**: Task 15, 16

  **References**:
  - `output/sweep/summary.md` — Task 15 的汇总表
  - `output/sweep/curriculum_comparison.md` — Task 16 的 curriculum 对比
  - `docs/最终实验报告.md` — 当前报告的格式和风格参考

  **Acceptance Criteria**:
  - [ ] 文件路径：`output/sweep/FINAL_REPORT.md`
  - [ ] 包含所有必需章节
  - [ ] 每个推荐都有数据支撑
  - [ ] 明确标注哪些结论是 pilot-only（不可泛化）

  **QA Scenarios**:

  ```
  Scenario: Final report contains actionable recommendations
    Tool: Bash
    Preconditions: 所有分析完成
    Steps:
      1. grep -c "推荐" output/sweep/FINAL_REPORT.md
      2. grep -c "不建议" output/sweep/FINAL_REPORT.md
    Expected Result: 至少 3 条推荐和 1 条不建议
    Failure Indicators: 空文件或缺乏明确结论
    Evidence: .sisyphus/evidence/task-17-final-report.md
  ```

  **Commit**: NO（分析文档，非代码修改）

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — APPROVE — `oracle`
  Read the plan end-to-end. Verify: gradient clipping present in meta_pretrainer.py, K_SUPPORT=3 config, curriculum sampler file exists, sweep scripts functional, sweep results exist in output/sweep/. Check "Must NOT Have" compliance.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Results Completeness Check** — APPROVE (63/63 checkpoints) — `deep`
  For each pilot dataset, verify: 6 FT epoch values run, 3 seeds each, baseline and MAML variants both present. Check no NaN/Inf in logs. Verify output/*.pth.tar files exist for each run.
  Output: `Stanford Cars [N/N] | EuroSAT [N/N] | DTD [N/N] | VERDICT`

- [x] F3. **Code Quality Review** — APPROVE — `unspecified-high`
  Run Python syntax checks on modified files. Check for: bare except, hardcoded paths, missing imports, dead code. Verify no breaking changes to existing trainer pipelines.
  Output: `Syntax [PASS/FAIL] | Imports [OK/N issues] | Breaking changes [NONE/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — APPROVE — `deep`
  For each task: read "What to do", read actual changes. Verify nothing beyond scope was built. Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

> All changes committed in feature branch `improve/maml-ft-sweep`. Each wave committed separately.

- **Wave 1**: N/A（仅运行实验，不修改代码）
- **Wave 2**: `feat(meta): add gradient clipping, K_SUPPORT=3, warmup, curriculum sampler, sweep scripts`
- **Wave 3**: `test(meta): verify stability fixes and batching benchmark`
- **Wave 4**: N/A（仅运行实验）
- **Wave 5**: `docs(sweep): add aggregated results and analysis report`

---

## Success Criteria

### Verification Commands
```bash
# Verify gradient clipping works
python -c "from trainers.meta_pretrainer import MetaPretrainer; print('Import OK')"

# Verify sweep script runs without error
bash scripts/sweep/run_sweep.sh --dry-run

# Verify curriculum sampler imports
python -c "from dassl.data.curriculum_episodic_sampler import CurriculumEpisodicSampler; print('Import OK')"

# Verify sweep results exist
ls output/sweep/stanford_cars/*/eval_result.txt | wc -l  # Expected: ≥36
ls output/sweep/eurosat/*/eval_result.txt | wc -l
ls output/sweep/dtd/*/eval_result.txt | wc -l
```

### Final Checklist
- [ ] 梯度裁剪已加入 `maml_inner_loop` 和 `run_epoch`
- [ ] K_SUPPORT=3 配置已更新，grad_accum 联动调整
- [ ] Curriculum EpisodicSampler 已实现并通过验证
- [ ] 自动 sweep 脚本可一键运行完整实验
- [ ] 3 个 pilot 数据集 6 FT epoch 值的结果完整
- [ ] 汇总报告包含最优 FT epoch 推荐
- [ ] 无 NaN/Inf 出现在训练日志中
- [ ] GPU 利用率在 meta-training 阶段 ≥ 50%（8GB/16GB）
