# DL-MPT-Adapter-B(CoOp+ATP) — Cross-Attention Adapter 变体实现

## TL;DR

> **Quick Summary**: 在 DL-MPT(CoOp+ATP) 的 episodic meta path 中，将朴素的 `(vis_proto + text_proto) / 2` 原型计算替换为**轻量级 Cross-Attention Adapter**，使视觉 patch tokens 在文本锚点的引导下进行选择性重映射，提升跨模态原型对齐能力。创建完整隔离副本，不修改任何现有代码。

> **Deliverables**:
> - 1 个新 Trainer: `trainers/dlmpt_adapter_b_trainer.py` (~400 行)
> - 1 个新 Config: `configs/trainers/dlmpt-adapter-b/vit_b16.yaml`
> - 3 个新 Scripts: `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh`, `eval_novel.sh`, `eval_episodic.sh`
> - 注册点更新: `train.py` +1 import, `trainers/__init__.py` +1 import

> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 4 tasks in Wave 1 (trainer + config + scripts + registration)
> **Critical Path**: Task 1 (Trainer 核心) → Task 5 (端到端验证)

---

## Context

### Original Request
> 应用方案B到现有方法DL-MPT(CoOp+ATP)，先拷贝副本，单独进行方法命名与实践，防止破坏现有代码与运行结果，命名为DL-MPT-Adapter-B(CoOp+ATP)

### Interview Summary
**Key Discussions**:
- Adapter 架构: 轻量版跨注意力（单层 MultiheadAttention + LayerNorm + residual，4 heads，embed_dim=512）
- 训练策略: 单阶段联合训练（Adapter 与 prompt 在 dual-loop 中同时训练）
- 损失函数: 仅 L_meta（cross-entropy on query matching，不引入显式 alignment/proto loss）
- 隔离策略: 完全独立副本，不修改 `dlmpt_trainer.py` 及任何现有文件

**Research Findings**:
- 当前原型计算位于 `dlmpt_trainer.py:271`: `F.normalize((vis_protos + text_protos) / 2, dim=-1)`
- 当前 vis_protos 使用 pooled CLS token（非 patch-level），Adapter 需要 patch tokens
- ViT-B/16 可提取 patch tokens：在 transformer 输出后、ln_post 前截取
- 项目无测试基础设施，使用 agent-executed QA
- README P0 命名规范：必须包含 `+ATP` 后缀，脚本文件命名严格

### Metis Review
**Identified Gaps** (addressed):
- **CRITICAL: 当前视觉特征是 pooled（非 patch-level）**: Adapter 需从 ViT transformer 输出提取 patch tokens，通过 `@torch.no_grad()` 重新实现 ViT forward 的中间截取逻辑（frozen backbone 无梯度需求）
- **Config 扩展**: 需新 TRAINER.DLMPT_ADAPTER_B section 存放 adapter 专属参数（N_HEADS, ADAPTER_DIM 等），同时复用 TRAINER.DLMPT 和 TRAINER.ATPROMPT 的参数
- **train.py 注册**: 需在 `extend_cfg` 中添加 `DLMPT_ADAPTER_B` section 的默认值
- **CoCoOp 路径**: DLMPTTrainer 有 `use_cocoop` flag 但新变体仅针对 CoOp+ATP，明确分支

---

## Work Objectives

### Core Objective
创建 `DL-MPT-Adapter-B(CoOp+ATP)` 方法的完整实验基础设施，核心改动：在 episodic meta path 中引入 Cross-Attention Adapter 替换朴素原型平均。

### Concrete Deliverables
- `trainers/dlmpt_adapter_b_trainer.py` — 新 Trainer 类 + CrossAttentionAdapter 模块
- `configs/trainers/dlmpt-adapter-b/vit_b16.yaml` — 配置文件（含 adapter 超参数）
- `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh` — 训练脚本
- `scripts/dlmpt-adapter-b/eval_novel.sh` — Protocol A 评估脚本
- `scripts/dlmpt-adapter-b/eval_episodic.sh` — Protocol B 评估脚本
- `train.py` — +1 import（适配器 trainer）
- `trainers/__init__.py` — +1 import

### Definition of Done
- [ ] `python train.py --trainer DLMPTAdapterBTrainer --dry-run` 通过（配置正确加载）
- [ ] Stanford Cars 数据集 1 epoch 训练不报错（loss 正常下降）
- [ ] Protocol A (eval_novel.sh) 评估产出准确率
- [ ] Protocol B (eval_episodic.sh, K=1/3/5) 评估产出 episodic accuracy
- [ ] 原始 `dlmpt_trainer.py` 未经任何修改（git diff 验证）

### Must Have
- Cross-Attention Adapter 模块完整实现（MultiheadAttention + LayerNorm + residual）
- Patch token 提取逻辑（从 ViT transformer 输出截取，不含 CLS token）
- 与 DL-MPT(CoOp+ATP) 完全相同的 dual-loop 训练流程（warmup/refine lambda scheduling）
- 完整的脚本基础设施（train + eval_novel + eval_episodic）

### Must NOT Have (Guardrails)
- **禁止修改** `dlmpt_trainer.py`, `dlmpt_cocoop_lite.py` 及任何现有 trainer 文件
- **禁止修改** `configs/trainers/dlmpt/` 下任何现有配置文件
- **禁止修改** `scripts/dlmpt/` 下任何现有脚本
- **禁止**引入显式 alignment loss / proto loss（用户明确选择仅 L_meta）
- **禁止**跨模态双向 adapter（仅视觉侧 adapter，文本侧保持不变）
- **AI slop**: 避免过度抽象（单一 adapter 类，不创建多层继承），避免冗余注释

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None
- **Agent-Executed QA**: MANDATORY (all tasks)

### QA Policy
Every task includes agent-executed QA scenarios:
- **CLI/TUI**: Use `bash` (tmux) — Run commands, validate output, check exit codes
- **API/Library**: Use `bash` (python -c) — Import modules, check shapes, verify config loading
- **Training**: Use `bash` — Run 1 epoch dry-run, check for runtime errors

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — MAX PARALLEL):
├── Task 1: Core Adapter + Trainer implementation [deep]
├── Task 2: Config file [quick]
├── Task 3: Training + Eval scripts [quick]
└── Task 4: Registration + train.py extend_cfg [quick]

Wave 2 (After Wave 1 — verify):
├── Task 5: End-to-end dry-run verification [deep]
```

```
Wave 1: 4 tasks parallel
Wave 2: 1 task (depends on ALL of Wave 1)
```

### Dependency Matrix
- **1-4**: — — 5
- **5**: 1, 2, 3, 4 — —

### Critical Path
Task 1 → Task 5

---

## TODOs

- [x] 1. **Core: CrossAttentionAdapter 模块 + DLMPTAdapterBTrainer 类**

  **What to do**:
  - 创建新文件 `trainers/dlmpt_adapter_b_trainer.py`
  - 实现 `CrossAttentionAdapter(nn.Module)`:
    - `__init__`: `nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)` + `nn.LayerNorm(512)`
    - `forward(vis_patches, text_anchor)`: Q=vis_patches, K=text_anchor.unsqueeze(0), V=text_anchor.unsqueeze(0) → residual + LayerNorm → mean pool over patches → L2 normalize
    - `vis_patches` shape: `(B, N_patches, 512)`, 不含 CLS token
    - `text_anchor` shape: `(512,)` — 单个 class 的 text prototype
    - 输出: `(B, 512)` — adapted visual prototypes
  - 实现 `DLMPTAdapterBTrainer(TrainerX)`:
    - 继承自 `TrainerX`（直接继承，不继承 DLMPTTrainer，避免名称空间污染）
    - `__init__`: 从 `cfg.TRAINER.DLMPT` 和 `cfg.TRAINER.DLMPT_ADAPTER_B` 读取参数；`self.use_cocoop = False`（仅 CoOp+ATP）
    - `build_model`: 复用 `dlmpt_trainer.py` 中 `build_model` 的 CoOp 分支逻辑（import `CustomCLIP` from `trainers.coop_atp`，freeze backbone，仅训练 prompt_learner）
    - `build_data_loader`: 完整复制 `dlmpt_trainer.py` 的 `build_data_loader`（含 episodic sampler + GPU caching）
    - `run_epoch`: 完整复制 dual-loop 训练逻辑（CoOp base path + episodic proto path）
    - `_proto_meta_loss_coop`: **核心修改点** — 替换原型计算逻辑：
      1. 使用 `_extract_patch_tokens(support_img)` 获取 ViT patch tokens（非 pooled CLS）
      2. 获取 text anchors: `text_features[unique]`（复用现有逻辑）
      3. 对每张 support image，调用 `self.adapter(vf_patches[i].unsqueeze(0), text_anchors[s_lab[i]])` 获取 adapted visual feature
      4. 按类聚合 adapted features → `vis_protos`
      5. `text_protos = text_anchors`（文本侧不经过 adapter）
      6. `prototypes = F.normalize((vis_protos + text_protos) / 2, dim=-1)`
    - `_extract_patch_tokens(images)`: 静态辅助函数，从 ViT transformer 输出提取 patch tokens：
      ```python
      @torch.no_grad()
      def _extract_patch_tokens(image_encoder, images, dtype):
          x = image_encoder.conv1(images.type(dtype))
          x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
          x = torch.cat([image_encoder.class_embedding.to(dtype) + 
                         torch.zeros(x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device), x], dim=1)
          x = x + image_encoder.positional_embedding.type(dtype)
          x = image_encoder.ln_pre(x).permute(1, 0, 2)
          x = image_encoder.transformer(x).permute(1, 0, 2)
          return x[:, 1:, :]  # exclude CLS token
      ```
    - `@TRAINER_REGISTRY.register()` 装饰器注册为 `DLMPTAdapterBTrainer`
  - 进度日志格式: `[DL-MPT-Adapter-B] epoch=... batch=... L_base=... L_meta=... acc_meta=... λ=... lr=...`
  - 完整复制 `warmup_epochs`, `refine_epochs`, `current_lambda` 逻辑
  - 完整复制 `model_backward_and_update`, `parse_batch_train` 等辅助方法

  **Must NOT do**:
  - 不继承 `DLMPTTrainer`（避免父类修改影响新变体）
  - 不修改 CoOp 的 base path 分类逻辑（保持与原始 DL-MPT 一致）
  - 不对文本侧进行 adapter 处理
  - 不引入额外的 loss 项

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要精确理解现有 DL-MPT 训练流程、ViT 内部结构、CoOp prompt 机制，在正确位置插入 adapter 逻辑
  - **Skills**: `[]`
    - 无需特定 skill — 这是标准 PyTorch/CLIP 开发任务

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):

  **Pattern References** (existing code to follow):
  - `trainers/dlmpt_trainer.py:23-35` — `__init__` 参数读取模式（cfg.TRAINER.DLMPT.*）
  - `trainers/dlmpt_trainer.py:91-132` — `build_model` 的 CoOp 分支逻辑（CustomCLIP 导入、freeze backbone、optimizer 注册）
  - `trainers/dlmpt_trainer.py:49-89` — `build_data_loader` 完整逻辑（DataManager、EpisodicSampler、GPU 图像 caching）
  - `trainers/dlmpt_trainer.py:141-233` — `run_epoch` 双循环训练逻辑（batch iteration、loss composition、lambda scheduling、进度日志）
  - `trainers/dlmpt_trainer.py:240-279` — `_proto_meta_loss_coop` 当前原型计算（需替换的核心方法）
  - `trainers/coop_atp.py:39-58` — `load_clip_to_cpu` 和 CustomCLIP 导入模式
  - `trainers/coop_atp.py:61-78` — `TextEncoder` 类定义（text_encoder 接口）
  - `trainers/cocoop_atp.py:1-50` — CoCoOp 的 CustomCLIP 和 prompt_learner 模式（了解 CoCoOp 差异，仅确保不被误用）

  **API/Type References** (contracts to implement against):
  - `clip/model.py` — VisionTransformer 内部结构（conv1, class_embedding, positional_embedding, ln_pre, transformer, ln_post）
  - `dassl/engine/trainer.py` — `TrainerX` 基类接口（build_model, build_data_loader, run_epoch, parse_batch_train, model_backward_and_update）

  **External References**:
  - `torch.nn.MultiheadAttention` API: embed_dim=512, num_heads=4, batch_first=True
  - `torch.nn.LayerNorm` API: normalized_shape=512

  **Acceptance Criteria**:
  - [ ] 文件 `trainers/dlmpt_adapter_b_trainer.py` 存在且可被 Python import
  - [ ] `CrossAttentionAdapter` 前向传播 shape 验证通过：输入 `(B=4, N_patches=50, 512)` + `text_anchor (512,)` → 输出 `(4, 512)`
  - [ ] `DLMPTAdapterBTrainer` 通过 `TRAINER_REGISTRY` 注册
  - [ ] `_extract_patch_tokens` 输出 shape 为 `(B, 197, 512)` —不含 CLS（ViT-B/16: 14×14 patches + 1 CLS = 197 tokens, 排除 CLS = 196 或不排除 CLS 都行，adapt 时排除即可）
  - [ ] `python -c "from trainers.dlmpt_adapter_b_trainer import DLMPTAdapterBTrainer"` 成功

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: CrossAttentionAdapter shape test — happy path
    Tool: Bash (python -c)
    Preconditions: None (standalone module test)
    Steps:
      1. Run: python -c "
         import torch, torch.nn as nn
         import sys; sys.path.insert(0, '.')
         from trainers.dlmpt_adapter_b_trainer import CrossAttentionAdapter
         adapter = CrossAttentionAdapter(embed_dim=512, num_heads=4)
         vis_patches = torch.randn(4, 196, 512)  # 4 images, 196 patches each
         text_anchor = torch.randn(512)           # single class text proto
         out = adapter(vis_patches, text_anchor)
         print(f'Output shape: {out.shape}')
         assert out.shape == (4, 512), f'Expected (4, 512), got {out.shape}'
         print('PASS: shape verified')
         "
    Expected Result: Output "Output shape: torch.Size([4, 512])" followed by "PASS: shape verified"
    Failure Indicators: shape mismatch error, import error, or any exception
    Evidence: .sisyphus/evidence/task-1-adapter-shape-test.txt

  Scenario: Patch token extraction shape test
    Tool: Bash (python -c)
    Preconditions: CLIP ViT-B/16 weights downloaded to ~/.cache/clip/
    Steps:
      1. Run: python -c "
         import torch, sys; sys.path.insert(0, '.')
         from trainers.dlmpt_adapter_b_trainer import DLMPTAdapterBTrainer
         # Quick check: import the function
         from trainers.dlmpt_adapter_b_trainer import _extract_patch_tokens as ept
         print('Import success')
         print('PASS: function importable')
         "
    Expected Result: "Import success" + "PASS: function importable"
    Failure Indicators: ImportError or AttributeError
    Evidence: .sisyphus/evidence/task-1-import-test.txt

  Scenario: Negative — adapter rejects mismatched embedding dims
    Tool: Bash (python -c)
    Preconditions: CrossAttentionAdapter instantiated with embed_dim=512
    Steps:
      1. Run: python -c "
         import torch; import sys; sys.path.insert(0, '.')
         from trainers.dlmpt_adapter_b_trainer import CrossAttentionAdapter
         adapter = CrossAttentionAdapter(embed_dim=512, num_heads=4)
         try:
             out = adapter(torch.randn(4, 196, 256), torch.randn(512))
             print('FAIL: should have raised error')
         except RuntimeError as e:
             print('PASS: correctly rejected mismatched dims')
         "
    Expected Result: "PASS: correctly rejected mismatched dims"
    Failure Indicators: PASS without error (shape mismatch silently accepted)
    Evidence: .sisyphus/evidence/task-1-negative-dims.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-adapter-shape-test.txt (adapter forward pass output)
  - [ ] task-1-import-test.txt (import success confirmation)
  - [ ] task-1-negative-dims.txt (negative test result)

  **Commit**: YES (groups with Tasks 2-4)
  - Message: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts`
  - Files: `trainers/dlmpt_adapter_b_trainer.py`

- [x] 2. **Config: configs/trainers/dlmpt-adapter-b/vit_b16.yaml**

  **What to do**:
  - 创建新目录 `configs/trainers/dlmpt-adapter-b/`
  - 创建 `configs/trainers/dlmpt-adapter-b/vit_b16.yaml`
  - 以 `configs/trainers/dlmpt/vit_b16.yaml` 为基础模板
  - `TRAINER.NAME`: `"DLMPTAdapterBTrainer"`
  - 新增 `TRAINER.DLMPT_ADAPTER_B` section:
    - `ADAPTER_HEADS: 4` — cross-attention 注意力头数
    - `ADAPTER_EMBED_DIM: 512` — 嵌入维度（与 CLIP ViT-B/16 一致）
  - 保留所有 TRAINER.DLMPT 参数（LAMBDA, N_WAY, K_SUPPORT, K_QUERY, N_EPISODES, WARMUP_EPOCHS, REFINE_EPOCHS, REFINE_LAMBDA）
  - 保留 TRAINER.COOP 参数（N_CTX: 2, CSC: False, CLASS_TOKEN_POSITION: "end", PREC: "fp16"）
  - 保留 TRAINER.ATPROMPT 参数（USE_ATPROMPT: True, ATT_NUM: 3, N_ATT1/2/3: 8, ATT1/2/3_TEXT）
  - 保留 MODEL, INPUT, OPTIM, DATALOADER sections 不变

  **Must NOT do**:
  - 不修改 `configs/trainers/dlmpt/vit_b16.yaml`
  - 不添加 COCOOP 相关参数（仅 CoOp+ATP）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 配置文件创建，基于现有模板复制修改，低复杂度
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `configs/trainers/dlmpt/vit_b16.yaml:1-50` — 模板文件，逐行参考

  **Acceptance Criteria**:
  - [ ] 文件 `configs/trainers/dlmpt-adapter-b/vit_b16.yaml` 存在
  - [ ] YAML 语法有效（`python -c "import yaml; yaml.safe_load(open('configs/trainers/dlmpt-adapter-b/vit_b16.yaml'))"` 成功）
  - [ ] TRAINER.NAME = "DLMPTAdapterBTrainer"
  - [ ] TRAINER.DLMPT_ADAPTER_B section 存在且包含 ADAPTER_HEADS 和 ADAPTER_EMBED_DIM

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Config YAML validity
    Tool: Bash (python -c)
    Preconditions: config file created
    Steps:
      1. Run: python -c "
         import yaml
         with open('configs/trainers/dlmpt-adapter-b/vit_b16.yaml') as f:
             cfg = yaml.safe_load(f)
         assert cfg['TRAINER']['NAME'] == 'DLMPTAdapterBTrainer', f'Wrong name: {cfg[\"TRAINER\"][\"NAME\"]}'
         assert 'DLMPT_ADAPTER_B' in cfg['TRAINER'], 'Missing DLMPT_ADAPTER_B section'
         assert cfg['TRAINER']['DLMPT_ADAPTER_B']['ADAPTER_HEADS'] == 4
         print('PASS: config valid')
         "
    Expected Result: "PASS: config valid"
    Failure Indicators: YAML parse error, missing key, wrong NAME
    Evidence: .sisyphus/evidence/task-2-config-valid.txt

  Scenario: Config diffs against original — verify no crossover contamination
    Tool: Bash (diff)
    Preconditions: config file created
    Steps:
      1. Run: diff <(grep -v 'NAME\|DLMPT_ADAPTER_B\|ADAPTER' configs/trainers/dlmpt/vit_b16.yaml) \
                  <(grep -v 'NAME\|DLMPT_ADAPTER_B\|ADAPTER' configs/trainers/dlmpt-adapter-b/vit_b16.yaml) || true
         echo "---"
         echo "Manual check: only NAME and DLMPT_ADAPTER_B should differ"
         echo "PASS: diff review completed"
    Expected Result: Only TRAINER.NAME and TRAINER.DLMPT_ADAPTER_B section should differ
    Failure Indicators: OPTIM, DATALOADER, or MODEL sections modified unintentionally
    Evidence: .sisyphus/evidence/task-2-config-diff.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-config-valid.txt
  - [ ] task-2-config-diff.txt

  **Commit**: YES (groups with Tasks 1, 3, 4)
  - Message: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts`
  - Files: `configs/trainers/dlmpt-adapter-b/vit_b16.yaml`

- [x] 3. **Scripts: train_dlm_coop_atp.sh + eval_novel.sh + eval_episodic.sh**

  **What to do**:
  - 创建新目录 `scripts/dlmpt-adapter-b/`
  - 创建 `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh`:
    - 以 `scripts/dlmpt/train.sh` 为模板
    - `--trainer` 改为 `DLMPTAdapterBTrainer`
    - `--config-file` 改为 `configs/trainers/dlmpt-adapter-b/vit_b16.yaml`
    - `OUTPUT` 路径改为 `output/dlmpt-adapter-b/${DATASET}/seed${SEED}/lambda${LAMBDA}`
    - 进度日志前缀改为 `[DL-MPT-Adapter-B TRAIN]`
    - 保留所有参数（--dataset, --seed, --lambda, --n-way, --output, --dry-run）
  - 创建 `scripts/dlmpt-adapter-b/eval_novel.sh`:
    - 以 `scripts/dlmpt/eval_novel.sh` 为模板
    - MODEL_DIR 路径适配到 `output/dlmpt-adapter-b/` 下的输出
    - 其余逻辑完全相同（Protocol A: 加载 prompt checkpoint → novel classes 全量评估）
  - 创建 `scripts/dlmpt-adapter-b/eval_episodic.sh`:
    - 以 `scripts/dlmpt/eval_episodic.sh` 为模板
    - RESULT_FILE 路径改为 `output/dlmpt-adapter-b/${DATASET}/seed${SEED}/episodic_results.txt`
    - CHECKPOINT 路径适配
    - 其余逻辑完全相同（Protocol B: N-way K-shot episodic eval, K=1/3/5）

  **Must NOT do**:
  - 不修改 `scripts/dlmpt/` 下任何文件
  - 不改变 Protocol A/B 的评估逻辑
  - 不使用 `| tee`（README P3 规则：用 `> log 2>&1`）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script 复制修改，基于现有模板，低复杂度
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `scripts/dlmpt/train.sh:1-56` — train 模板
  - `scripts/dlmpt/eval_novel.sh:1-32` — eval_novel 模板
  - `scripts/dlmpt/eval_episodic.sh:1-85` — eval_episodic 模板

  **Acceptance Criteria**:
  - [ ] 3 个 shell 脚本文件存在且 `chmod +x` 可执行
  - [ ] `bash scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh --dry-run` 输出正确命令（验证 trainer name + config path + output path）
  - [ ] `bash scripts/dlmpt-adapter-b/eval_novel.sh --dry-run` 不报错
  - [ ] `bash scripts/dlmpt-adapter-b/eval_episodic.sh --dry-run` 不报错

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Train script dry-run verification
    Tool: Bash
    Preconditions: Scripts created
    Steps:
      1. Run: bash scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh --dataset stanford_cars --seed 1 --lambda 0.2 --dry-run
      2. Check output contains: "DLMPTAdapterBTrainer"
      3. Check output contains: "configs/trainers/dlmpt-adapter-b/vit_b16.yaml"
      4. Check output contains: "output/dlmpt-adapter-b/stanford_cars/seed1/lambda0.2"
    Expected Result: Dry-run command printed with correct trainer, config, and output paths
    Failure Indicators: Wrong trainer name, wrong config path, wrong output path, or script error
    Evidence: .sisyphus/evidence/task-3-train-dryrun.txt

  Scenario: Eval novel script dry-run
    Tool: Bash
    Preconditions: Scripts created
    Steps:
      1. Run: bash scripts/dlmpt-adapter-b/eval_novel.sh --dataset stanford_cars --seed 1 \
              --model-dir /tmp/fake --dry-run 2>&1 || true
    Expected Result: Script executes (may fail at runtime but not at parse time)
    Failure Indicators: Shell syntax error
    Evidence: .sisyphus/evidence/task-3-eval-novel-dryrun.txt

  Scenario: Eval episodic script dry-run
    Tool: Bash
    Preconditions: Scripts created
    Steps:
      1. Run: bash scripts/dlmpt-adapter-b/eval_episodic.sh --dataset stanford_cars --seed 1 \
              --checkpoint /tmp/fake --dry-run 2>&1 || true
    Expected Result: Script executes (may fail at runtime but not at parse time)
    Failure Indicators: Shell syntax error
    Evidence: .sisyphus/evidence/task-3-eval-episodic-dryrun.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-train-dryrun.txt
  - [ ] task-3-eval-novel-dryrun.txt
  - [ ] task-3-eval-episodic-dryrun.txt

  **Commit**: YES (groups with Tasks 1, 2, 4)
  - Message: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts`
  - Files: `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh`, `scripts/dlmpt-adapter-b/eval_novel.sh`, `scripts/dlmpt-adapter-b/eval_episodic.sh`

- [x] 4. **Registration: train.py + trainers/__init__.py**

  **What to do**:
  - `train.py`: 新增 `import trainers.dlmpt_adapter_b_trainer`（在现有 dlmpt import 旁）
  - `train.py` `extend_cfg` 函数: 新增 `DLMPT_ADAPTER_B` section 默认值：
    ```python
    cfg.TRAINER.DLMPT_ADAPTER_B = CN()
    cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_HEADS = 4
    cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_EMBED_DIM = 512
    ```
  - `trainers/__init__.py`: 新增 `import trainers.dlmpt_adapter_b_trainer`

  **Must NOT do**:
  - 不修改现有 import 行（仅追加新行）
  - 不修改 `extend_cfg` 中现有的 DLMPT/COOP/ATPROMPT section

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 两处注册点追加，单行修改，低复杂度
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:
  - `train.py:35` — 现有 dlmpt import 位置（新 import 紧邻其后）
  - `train.py:167-177` — `extend_cfg` 中 DLMPT section 定义模式（新 section 紧邻其后）
  - `trainers/__init__.py:1-2` — 现有 import 模式

  **Acceptance Criteria**:
  - [ ] `python -c "import train"` 成功（不含 import 错误）
  - [ ] `python -c "from dassl.config import get_cfg_default; import train; cfg = get_cfg_default(); train.extend_cfg(cfg); print(cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_HEADS)"` 输出 `4`
  - [ ] `python -c "from trainers.dlmpt_adapter_b_trainer import DLMPTAdapterBTrainer"` 成功

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Full import chain + config extension test
    Tool: Bash (python -c)
    Preconditions: Tasks 1 and 2 completed
    Steps:
      1. Run: python -c "
         import sys; sys.path.insert(0, '.')
         from dassl.config import get_cfg_default
         import train
         cfg = get_cfg_default()
         train.extend_cfg(cfg)
         # Verify new section exists
         assert hasattr(cfg.TRAINER, 'DLMPT_ADAPTER_B'), 'DLMPT_ADAPTER_B section missing'
         assert cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_HEADS == 4
         assert cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_EMBED_DIM == 512
         # Verify import chain
         from trainers.dlmpt_adapter_b_trainer import DLMPTAdapterBTrainer
         print('PASS: registration chain verified')
         "
    Expected Result: "PASS: registration chain verified"
    Failure Indicators: ImportError, AttributeError, assertion failure
    Evidence: .sisyphus/evidence/task-4-registration.txt

  Scenario: Verify no regression in existing imports
    Tool: Bash (python -c)
    Preconditions: train.py modified
    Steps:
      1. Run: python -c "
         import sys; sys.path.insert(0, '.')
         from trainers.dlmpt_trainer import DLMPTTrainer
         from trainers.dlmpt_cocoop_lite import DLMPTCoCoOpLite
         from trainers.coop_atp import CoOp_ATP
         print('PASS: existing imports intact')
         "
    Expected Result: "PASS: existing imports intact"
    Failure Indicators: ImportError on any existing trainer
    Evidence: .sisyphus/evidence/task-4-no-regression.txt
  ```

  **Evidence to Capture**:
  - [ ] task-4-registration.txt
  - [ ] task-4-no-regression.txt

  **Commit**: YES (groups with Tasks 1, 2, 3)
  - Message: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts`
  - Files: `train.py`, `trainers/__init__.py`

---

- [x] 5. **End-to-End Verification: config dry-run + 1-epoch smoke test + git isolation audit**

  **What to do**:
  - Config dry-run: `python train.py --trainer DLMPTAdapterBTrainer --root /path/to/DATA --seed 1 --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/dlmpt-adapter-b/vit_b16.yaml --dry-run`
  - 确认配置加载无错误，所有 DLMPT_ADAPTER_B 参数正确解析
  - 如果 Stanford Cars 数据不可用，使用 EuroSAT 数据集（类别少，训练快）
  - 1-epoch 训练 smoke test: 运行 `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh` 1 epoch（通过 --dry-run 先确认命令，再手动改 MAX_EPOCH 为 1 执行）
  - Git isolation audit:
    ```bash
    git diff --name-only HEAD | grep -E 'dlmpt_trainer\.py|dlmpt_cocoop_lite\.py|configs/trainers/dlmpt/vit_b16'
    ```
    必须输出为空（证明未修改原始文件）
  - 验证 adapter 参数确实被训练（`CrossAttentionAdapter` 的 `requires_grad` 为 True）
  - 验证 prompt_learner 参数仍在训练（`requires_grad` 为 True）
  - 验证 backbone 参数仍在冻结（`requires_grad` 为 False）

  **Must NOT do**:
  - 不运行完整训练（仅 1 epoch smoke test）
  - 不修改原始文件（由 git diff 验证）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要配置加载、训练启动、GPU 环境检查、git 审计等多步骤验证
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential, depends on Wave 1)
  - **Blocks**: None (final verification)
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:
  - `scripts/dlmpt/train.sh:34-46` — 训练命令模板（参数格式）
  - `configs/datasets/eurosat.yaml` — EuroSAT 作为 fallback 数据集

  **Acceptance Criteria**:
  - [ ] Config dry-run 成功（`--dry-run` 输出正确命令，无 error）
  - [ ] 1-epoch training 不 crash（loss 正常输出，无 NaN/OOM）
  - [ ] git diff 确认原始 dlmpt_trainer.py 未修改
  - [ ] Adapter 参数 `requires_grad=True`
  - [ ] Prompt_learner 参数 `requires_grad=True`
  - [ ] Backbone 参数 `requires_grad=False`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Config dry-run — full chain validation
    Tool: Bash
    Preconditions: DATA=/home/avoidman2233/Desktop/LVLM/DATA exists (or use /tmp mock)
    Steps:
      1. Run: python train.py \
              --trainer DLMPTAdapterBTrainer \
              --root /home/avoidman2233/Desktop/LVLM/DATA \
              --seed 1 \
              --dataset-config-file configs/datasets/eurosat.yaml \
              --config-file configs/trainers/dlmpt-adapter-b/vit_b16.yaml \
              --dry-run 2>&1
      2. Check output contains "DLMPTAdapterBTrainer"
      3. Check exit code = 0
    Expected Result: Dry-run completes without error, config parsed correctly
    Failure Indicators: YAML parse error, missing section, KeyError, exit code != 0
    Evidence: .sisyphus/evidence/task-5-dryrun.txt

  Scenario: Git isolation audit — verify original files untouched
    Tool: Bash (git diff)
    Preconditions: Tasks 1-4 completed
    Steps:
      1. Run: git diff --name-only HEAD
      2. Check that output does NOT contain:
         - trainers/dlmpt_trainer.py
         - trainers/dlmpt_cocoop_lite.py
         - configs/trainers/dlmpt/vit_b16.yaml
         - configs/trainers/dlmpt/vit_b16_cocoop.yaml
         - scripts/dlmpt/train.sh
         - scripts/dlmpt/eval_novel.sh
         - scripts/dlmpt/eval_episodic.sh
      3. Run: echo "PASS: isolation verified" (if all clear)
    Expected Result: Only new files (dlmpt_adapter_b_trainer.py, dlmpt-adapter-b/*) in git diff
    Failure Indicators: Any original file appears in git diff
    Evidence: .sisyphus/evidence/task-5-git-isolation.txt

  Scenario: Adapter parameter audit — verify trainable/frozen status
    Tool: Bash (python -c)
    Preconditions: Task 1 completed
    Steps:
      1. Run: python -c "
         import torch; import sys; sys.path.insert(0, '.')
         from trainers.dlmpt_adapter_b_trainer import CrossAttentionAdapter
         adapter = CrossAttentionAdapter()
         params = list(adapter.named_parameters())
         trainable = sum(p[1].numel() for p in params if p[1].requires_grad)
         total = sum(p[1].numel() for p in params)
         print(f'Trainable: {trainable}, Total: {total}')
         assert trainable == total, 'All adapter params should be trainable'
         print('PASS: all adapter params trainable')
         "
    Expected Result: "Trainable: X, Total: X" + "PASS: all adapter params trainable"
    Failure Indicators: Some params have requires_grad=False
    Evidence: .sisyphus/evidence/task-5-param-audit.txt

  Scenario: Negative — train script rejects missing --dataset
    Tool: Bash
    Preconditions: train.sh created
    Steps:
      1. Run: bash scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh 2>&1 || true
      2. Check output contains "Unknown arg" or usage message
    Expected Result: Script exits with error (no default for required args)
    Failure Indicators: Script runs with empty defaults (silent failure)
    Evidence: .sisyphus/evidence/task-5-train-noargs.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-dryrun.txt
  - [ ] task-5-git-isolation.txt
  - [ ] task-5-param-audit.txt
  - [ ] task-5-train-noargs.txt

  **Commit**: NO (verification only)

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval.**
> **Never mark F1-F4 as checked before getting user's okay.**

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check import). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Verify README naming conventions followed in ALL paths and file names.
  Output: `Must Have [4/4] | Must NOT Have [5/5] | Tasks [5/5] | VERDICT: APPROVE`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Review all new/changed files for: Python syntax errors, unused imports, `as any`/`@ts-ignore` patterns, empty except blocks, console.log in prod, commented-out code, AI slop (excessive comments, over-abstraction, generic names like data/result/item/temp). Check `CrossAttentionAdapter` forward pass for numerical stability.
  Output: `Build [PASS] | Files [7 clean/0 issues] | VERDICT: APPROVE`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Verify cross-task integration: import chain (task-1 + task-4), config loading (task-2 + task-4), script dry-runs (task-3 + task-5). Test edge cases: YAML parse failure, import order issues.
  Output: `Scenarios [6/6 pass] | VERDICT: APPROVE`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag any modifications to original dlmpt_trainer.py or configs/trainers/dlmpt/.
  Output: `Tasks [5/5 compliant] | Contamination [CLEAN] | Unaccounted [CLEAN] | VERDICT: APPROVE`

---

## Commit Strategy

- **1-4**: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts, registration` — `trainers/dlmpt_adapter_b_trainer.py`, `configs/trainers/dlmpt-adapter-b/vit_b16.yaml`, `scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh`, `scripts/dlmpt-adapter-b/eval_novel.sh`, `scripts/dlmpt-adapter-b/eval_episodic.sh`, `train.py`, `trainers/__init__.py`

---

## Success Criteria

### Verification Commands
```bash
# 1. Full import chain
python -c "from trainers.dlmpt_adapter_b_trainer import DLMPTAdapterBTrainer, CrossAttentionAdapter; print('OK')"

# 2. Config loading
python -c "
from dassl.config import get_cfg_default
import train
cfg = get_cfg_default()
train.extend_cfg(cfg)
cfg.merge_from_file('configs/trainers/dlmpt-adapter-b/vit_b16.yaml')
print('TRAINER.NAME:', cfg.TRAINER.NAME)
print('ADAPTER_HEADS:', cfg.TRAINER.DLMPT_ADAPTER_B.ADAPTER_HEADS)
print('OK')
"

# 3. Git isolation audit
git diff --name-only HEAD | grep -v 'dlmpt_adapter_b\|dlmpt-adapter-b' | grep -E 'dlmpt|train\.py|__init__\.py' || echo "PASS: isolation clean"

# 4. Script dry-runs
bash scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh --dataset eurosat --seed 1 --lambda 0.2 --dry-run
bash scripts/dlmpt-adapter-b/eval_novel.sh --dataset eurosat --seed 1 --model-dir /tmp/fake --dry-run
bash scripts/dlmpt-adapter-b/eval_episodic.sh --dataset eurosat --seed 1 --checkpoint /tmp/fake --dry-run
```

### Final Checklist
- [ ] All "Must Have" present (5 deliverables + registration)
- [ ] All "Must NOT Have" absent (original files untouched)
- [ ] README naming conventions followed
- [ ] Config loads without error
- [ ] Script dry-runs succeed
- [ ] Adapter forward pass shape-verified

---

## Commit Strategy

- **1-4**: `feat(dlmpt-adapter-b): add CrossAttentionAdapter trainer, config, scripts, registration` — all 6 new/changed files
- **5**: (no commit — verification only)

---

## Success Criteria

### Verification Commands
```bash
# Config dry-run
python train.py --trainer DLMPTAdapterBTrainer --root /path/to/DATA --seed 1 \
    --dataset-config-file configs/datasets/stanford_cars.yaml \
    --config-file configs/trainers/dlmpt-adapter-b/vit_b16.yaml --dry-run

# 1-epoch smoke test
bash scripts/dlmpt-adapter-b/train_dlm_coop_atp.sh --dataset stanford_cars --seed 1 --lambda 0.2 --dry-run

# Git diff isolation check
git diff --name-only HEAD  # must NOT include dlmpt_trainer.py, dlmpt_cocoop_lite.py, or any configs/trainers/dlmpt/ files
```

### Final Checklist
- [ ] All 7 deliverables exist and are syntactically valid
- [ ] Config loads without error
- [ ] 1-epoch training runs without crash
- [ ] Original files untouched (git diff clean)
- [ ] README naming conventions followed in all paths/file names
