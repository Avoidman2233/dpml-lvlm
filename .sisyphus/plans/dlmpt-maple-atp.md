# DL-MPT + MaPLe + ATP 实现计划

## TL;DR

> **Quick Summary**: 实现 DL-MPT(MaPLe+ATP) — 将 Dual-loop Meta Prompt Tuning 注入 MaPLe+ATP，在 CLIP ViT-B/16 frozen backbone 上验证 episodic proto 正则化能否提升 MaPLe 的小样本图像分类性能。
>
> **Deliverables**:
> - `trainers/dlmpt_maple_lite.py` — 新训练器（继承 MaPLe_ATP）
> - `configs/trainers/dlmpt/vit_b16_maple.yaml` — 配置文件
> - `scripts/dlmpt/train_maple.sh` — 训练脚本
> - `scripts/dlmpt/eval_maple_novel.sh` — Protocol A 评估脚本
>
> **Estimated Effort**: Quick（4 个紧密耦合任务，Pattern 复用为主）
> **Parallel Execution**: NO — sequential（所有文件互相依赖，需按序创建）
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
来自 handoff（`.sisyphus/handoff-dlmpt-maple-atp.md`）：DL-MPT 已在 CoOp+ATP(+4.88%) 和 CoCoOp+ATP(持平) 上验证有效。下一步是将 DL-MPT 注入 MaPLe+ATP。MaPLe 没有 Meta-Net，episodic regularization 应提供 NEW value。目标：超越 MaPLe+ATP 在 Stanford Cars 上的 Novel 准确率 73.84%。

### Interview Summary
**Key Discussions**:
- DL-MPT 训练器：继承 MaPLe_ATP（非 CoCoOp_ATP），复用 `dlmpt_cocoop_lite.py` 的 dual-loop 模式
- `_proto_meta_loss()`：采用 CoOp 版本的 ALL-classes 模式（MaPLe.forward(image) 返回 logits，API 与 CoOp 相同）
- 关键差异：MaPLe 的 `prompt_learner()` 返回 4 个值（非 2 个），`text_encoder()` 和 `image_encoder()` 都接收 deep prompt 参数
- 输出：使用 `> log 2>&1` 而非 `| tee`（P3 设计原则）
- 命名：`DL-MPT(MaPLe+ATP)`，脚本文件名为 `train_maple.sh`

**Research Findings**:
- `prompt_learner()` 返回 `(prompts, shared_ctx, deep_text_prompts, deep_vision_prompts)` — 4 元组
- `text_encoder(prompts, tokenized_prompts, deep_text_prompts)` — 3 个参数
- `image_encoder(image, shared_ctx, deep_vision_prompts, None, None, None, None)` — 额外参数
- `CustomCLIP.forward(image, label=None)`: training 时返回 loss，eval 时返回 logits
- Base accuracy 通过 `model(img)`（无 label，返回 logits）获得
- `MaPLe_ATP.parse_batch_train` 已返回 2 个值（input, label），无需 override

### Metis Review
**Identified Gaps** (addressed):
- **测试策略**: 研究项目无单元测试框架 → Agent-Executed QA（import check、dry-run、结构审查）和 GPU 训练验证
- **OOM 边缘情况**: ALL-classes text_encoder 在 MaPLe 上可能内存更大 → 配置中降低 batch size 到 2，如遇 OOM 回退到 per-class 模式
- **checkpoint 加载**: MaPLe_ATP 已有 `load_model()` method，eval 脚本可复用
- **Protocol B 评估**: 复用现有 `scripts/dlmpt/eval_episodic.py`，无需新脚本

---

## Work Objectives

### Core Objective
创建 DL-MPT(MaPLe+ATP) 训练器，实现 dual-loop episodic 正则化，使 MaPLe+ATP 在 Stanford Cars 小样本分类上获得提升。

### Concrete Deliverables
- `trainers/dlmpt_maple_lite.py` — DL-MPT(MaPLe+ATP) 训练器
- `configs/trainers/dlmpt/vit_b16_maple.yaml` — 配置文件
- `scripts/dlmpt/train_maple.sh` — 训练脚本
- `scripts/dlmpt/eval_maple_novel.sh` — Protocol A 零样本 novel 类评估

### Definition of Done
- [ ] `python -c "from trainers.dlmpt_maple_lite import DLMPTMapleLite"` 导入成功
- [ ] `bash scripts/dlmpt/train_maple.sh --dataset stanford_cars --seed 1 --dry-run` 输出完整命令
- [ ] 代码结构与 `dlmpt_cocoop_lite.py` 一致（继承基类，override `__init__`/`build_data_loader`/`run_epoch`/`_proto_meta_loss`）
- [ ] 进度日志格式符合 P4 规范：`[DL-MPT Maple] epoch=01/25 batch=020/100 L_base=1.324 L_meta=2.863 acc_meta=75.0% λ=0.20 lr=1.88e-03`

### Must Have
- 继承 `MaPLe_ATP`（从 `trainers.maple_atp` 导入）
- 正确处理 MaPLe 的 4 值 `prompt_learner()` 返回
- 在 `_proto_meta_loss` 中将 deep prompts 传递给 `text_encoder` 和 `image_encoder`
- 输出重定向使用 `> log 2>&1`（不用 `| tee`）
- 命名正确：`DL-MPT(MaPLe+ATP)`，`train_maple.sh`，`DLMPTMapleLite`

### Must NOT Have (Guardrails)
- 不要修改 `trainers/maple_atp.py` 或 `trainers/dlmpt_trainer.py`
- 不要创建新的 `EpisodicSampler` 或 evaluation protocol
- 不要在 proto loop 中手动 tokenize（使用 prompt_learner 的 token buffers）
- 不要添加 CoCoOp 模式的 meta_net（MaPLe 无 Meta-Net）
- 不要使用 `| tee`（管道缓冲导致 GPU 显存堆积）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — 所有验证由 agent 执行。

### Test Decision
- **Infrastructure exists**: NO（ML 研究项目，无 pytest/vitest）
- **Automated tests**: None
- **Framework**: N/A
- **Agent-Executed QA**: 每个任务包含 import check、结构审查和 dry-run 验证

### QA Policy
每个任务包含 Agent-Executed QA Scenarios（见 TODO 模板）。证据保存到 `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`。

- **Python import**: Bash — 运行 `python -c "from ... import ..."`
- **Bash dry-run**: Bash — 运行脚本 `--dry-run` 验证输出
- **结构审查**: Bash grep — 检查关键 pattern 是否存在于文件中

---

## Execution Strategy

### Parallel Execution Waves

> 所有任务互相依赖（Config 引用 Trainer 名称，Script 引用 Config 和 Trainer），因此 sequential 执行。

```
Wave 1 (唯一 Wave — sequential 但独立验证):
├── Task 1: 创建 dlmpt_maple_lite.py 训练器 [quick]
├── Task 2: 创建 vit_b16_maple.yaml 配置 [quick]
├── Task 3: 创建 train_maple.sh 训练脚本 [quick]
└── Task 4: 创建 eval_maple_novel.sh 评估脚本 [quick]

Wave FINAL (After ALL tasks):
├── Task F1: Plan Compliance Audit (oracle)
├── Task F2: Code Quality Review (unspecified-high)
├── Task F3: Real Manual QA (unspecified-high)
└── Task F4: Scope Fidelity Check (deep)

Critical Path: Task 1 → Task 2 → Task 3 → Task 4 → F1-F4 → user okay
Parallel Speedup: N/A（所有任务 sequential）
Max Concurrent: 1（Sequential wave）
```

### Dependency Matrix

- **1**: - - 2, 3, 4
- **2**: 1 - 3
- **3**: 1, 2 - 4
- **4**: 1, 2, 3 - -

### Agent Dispatch Summary

- **1**: **4** - T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `quick`
- **FINAL**: **4** - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

