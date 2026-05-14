# Meta-Pretraining Framework for ATPrompt

## TL;DR

> **Quick Summary**: Add a MAML-style meta-pretraining phase to ATPrompt that trains prompt_learner parameters to rapidly adapt to few-shot tasks, then fine-tunes normally on target datasets. Works with all prompt learning methods (CoOp/CoCoOp/MaPLe/DePT, with/without ATP).
> 
> **Deliverables**:
> - EpisodicSampler (Dassl layer) for N-way K-shot episode construction
> - MetaPretrainer class with full second-order MAML inner/outer loop
> - Factory `build_model()` supporting all 8 prompt learning method variants
> - Config system (YACS META namespace) + shell scripts
> - Complete experimental design (baseline / meta-only / full + ablations)
> 
> **Estimated Effort**: Medium-Large (~13 implementation tasks + 4 verification)
> **Parallel Execution**: YES — 3 waves + final verification
> **Critical Path**: Task 1→6→8→12→F1

---

## Context

### Original Request
为 ATPrompt 项目添加元任务微调（MAML）的训练/预训练步骤，期望在小样本学习能力上有所提高。最终目标是构建"基于元任务微调与属性辅助推理的 LVLM 小样本学习框架"——可移植、可验证的学术研究项目。

### Interview Summary
**Key Discussions**:
- MAML 方案（方案 A）: 完整二阶梯度 MAML，在 base 类上构造 episode，训练 prompt_learner 快速适应能力
- 数据范围: 同时支持跨数据集 meta-training（ImageNet base → 下游）和单数据集内部 meta-training
- 可移植性: 单一 MetaPretrainer 通过 config 驱动 factory pattern，不做独立变体
- 向后兼容: 不修改现有 trainer，meta checkpoint 格式与标准 trainer 兼容
- 实验设计: 完整对比矩阵 + ablation study

**Research Findings**:
- Dassl 框架: TrainerBase → SimpleTrainer → TrainerX 继承链，`run_epoch()` 遍历 `train_loader_x`
- 现有 sampler: RandomClassSampler（N class × K instances = batch）可改造为 episode
- ATP trainers: PromptLearner 学习 ctx + ctx_att 向量，CustomCLIP 做 image-text matching
- 仅 prompt_learner 参数可训练（CLIP backbone 冻结），MAML 内循环代价低
- Base-to-new 协议: `DATASET.SUBSAMPLE_CLASSES base/new` + `--eval-only`

### Test Infrastructure Assessment
- **Infrastructure exists**: NO — 学术研究项目，无 pytest/unittest 框架
- **Automated tests**: NONE — 验证通过实验评估（base/new accuracy）
- **Agent-Executed QA**: MANDATORY — 每个 task 具有训练运行验证 + checkpoint 加载验证

---

## Work Objectives

### Core Objective
构建一个通用的 MAML 元预训练框架，使 prompt_learner 参数通过 episode-based 训练获得快速适应能力，然后加载到标准 trainer 进行 fine-tune，提升 few-shot 分类性能。

### Concrete Deliverables
- `Dassl.pytorch/dassl/data/episodic_sampler.py` — EpisodicSampler 类
- `trainers/meta_pretrainer.py` — MetaPretrainer 类（~400 lines）
- `configs/trainers/meta/vit_b16.yaml` — meta 训练配置
- `scripts/meta/meta_pretrain.sh` — 跨数据集预训练脚本
- `scripts/meta/meta_finetune.sh` — 微调脚本
- `scripts/meta/meta_eval.sh` — 评估脚本
- `docs/EXPERIMENT_DESIGN.md` — 实验设计矩阵

### Definition of Done
- [ ] `python train.py --trainer MetaPretrainer --config-file configs/trainers/meta/vit_b16.yaml` 成功运行 meta-pretraining（不崩溃）
- [ ] Meta checkpoint 可被标准 `CoOp_ATP.load_model()` 加载
- [ ] Meta-finetuned 模型的 base/new accuracy 可通过 `scripts/meta/meta_eval.sh` 复现
- [ ] 实验设计文档完整包含 baseline/meta-only/full 对比 + ablation 矩阵

### Must Have
- EpisodicSampler: N-way × (K_support + K_query) 每 episode，shuffle，可复现
- MetaPretrainer: 完整二阶 MAML（retain_graph=True + create_graph=True）
- Factory build_model: 支持 CoOp/CoCoOp/MaPLe/DePT × ATP/非ATP = 8 种组合
- Checkpoint 兼容: 仅保存 prompt_learner state_dict + epoch + optimizer
- Config 驱动: 所有 meta 参数通过 YACS META namespace 控制

### Must NOT Have (Guardrails)
- **不得修改** `trainers/coop_atp.py`, `cocoop_atp.py`, `maple_atp.py`, `dept_atp.py`（向后兼容）
- **不得修改** `Dassl.pytorch/dassl/engine/trainer.py` 中的 TrainerX/SimpleTrainer
- **不得修改** CLIP backbone（始终冻结）
- **不得引入** 新的深度学习框架依赖（仅使用已有 PyTorch + Dassl）
- **不得引入** 硬编码路径或 API key
- **避免** over-engineering：MetaPretrainer 是单一文件，不拆分层

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: NONE
- **Agent-Executed QA**: MANDATORY for ALL tasks

### QA Policy
- **Training tasks**: run the training script, verify no crash, verify loss decreases over epochs
- **API/Module tasks**: import in Python REPL, verify class instantiation, method signatures
- **Checkpoint tasks**: save → load → verify state_dict keys match
- **Script tasks**: run with `--help` dry-run, verify correct arg parsing

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — ALL parallel, 5 tasks):
├── Task 1: EpisodicSampler + Dassl integration [quick]
├── Task 2: Config system: extend_cfg META namespace [quick]
├── Task 3: MetaPretrainer skeleton + check_cfg + __init__ [quick]
├── Task 4: Model factory: build_model() for all 8 methods [deep]
└── Task 5: MAML inner_loop: clone+adapt utility [unspecified-high]

Wave 2 (After Wave 1 — core logic, 3 tasks):
├── Task 6: MetaPretrainer.run_epoch() — complete training loop [deep]
├── Task 7: Checkpoint save/load compatibility [quick]
└── Task 8: Trainer registration + imports [quick]

Wave 3 (After Wave 2 — integration, 3 tasks):
├── Task 9: Config YAML files [quick]
├── Task 10: Shell scripts (pretrain, finetune, eval) [quick]
└── Task 11: Experiment design document [writing]

Wave 4 (After Wave 3 — verification, 2 tasks):
├── Task 12: Single-dataset end-to-end integration test [deep]
└── Task 13: Cross-dataset end-to-end integration test [deep]

Wave FINAL (After ALL — 4 parallel reviews):
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: Real manual QA [unspecified-high]
└── Task F4: Scope fidelity check [deep]

Critical Path: Task 1 → Task 6 → Task 8 → Task 12 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 5 (Wave 1) + 5 (Wave 1) + 3 (Wave 2) + 3 (Wave 3) + 2 (Wave 4) + 4 (FINAL)
```

### Agent Dispatch Summary
- **Wave 1**: 5 — T1→quick, T2→quick, T3→quick, T4→deep, T5→unspecified-high
- **Wave 2**: 3 — T6→deep, T7→quick, T8→quick
- **Wave 3**: 3 — T9→quick, T10→quick, T11→writing
- **Wave 4**: 2 — T12→deep, T13→deep
- **FINAL**: 4 — F1→oracle, F2→unspecified-high, F3→unspecified-high, F4→deep

---

## TODOs

- [ ] 1. EpisodicSampler — N-way K-shot episode construction

  **What to do**:
  - Create NEW file `Dassl.pytorch/dassl/data/episodic_sampler.py`
  - Implement `EpisodicSampler(Sampler)` class:
    - `__init__(data_source, n_way, k_support, k_query, n_episodes)` — group data by label using `defaultdict(list)`, validate constraints (n_way ≤ num_classes, k_support + k_query ≤ min samples per class)
    - `__iter__()` — yield `(support_indices, query_indices)` tuples for each episode. Per episode: randomly sample `n_way` classes from `self.labels`, then for each class sample `k_support + k_query` indices from `self.index_dic[label]` (without replacement between support/query)
    - `__len__()` — return `n_episodes`
  - Register in `Dassl.pytorch/dassl/data/samplers.py` `build_sampler()`: add `"EpisodicSampler"` case that calls `EpisodicSampler(data_source, n_way, k_support, k_query, n_episodes)` — extract these from cfg or pass as kwargs
  - Export in `Dassl.pytorch/dassl/data/__init__.py`: add `from .episodic_sampler import EpisodicSampler`
  - For the `build_sampler` integration, use a 2-step approach: first check if sampler_type starts with "Episodic", then extract params from cfg. The `build_data_loader` function signature needs a way to pass n_way/k_support/k_query — add optional kwargs or read from cfg.

  **Must NOT do**:
  - Do NOT modify `build_data_loader()` function signature in a breaking way
  - Do NOT import CLIP or PyTorch modules in the sampler (keep it lightweight)
  - Do NOT hardcode episode parameters (read from config or constructor args)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file implementation with clear spec, no deep ML logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Task 6 (MetaPretrainer meta data loading)
  - **Blocked By**: None

  **References** (CRITICAL):
  - `Dassl.pytorch/dassl/data/samplers.py:117-178` — RandomClassSampler implementation pattern (label-indexed dict, __iter__ logic, build_sampler factory)
  - `Dassl.pytorch/dassl/data/samplers.py:181-205` — build_sampler() function (how to add a new sampler type)
  - `Dassl.pytorch/dassl/data/data_manager.py:14-48` — build_data_loader() signature (sampler_type, batch_size, n_domain, n_ins patterns to extend)
  - `Dassl.pytorch/dassl/data/datasets/base_dataset.py:12-46` — Datum class attributes (label, impath, domain, classname)
  - `Dassl.pytorch/dassl/data/__init__.py` — exports pattern to follow for new module

  **Acceptance Criteria**:
  - [ ] `from dassl.data import EpisodicSampler` works without error
  - [ ] `sampler = EpisodicSampler(data, n_way=5, k_support=1, k_query=3, n_episodes=10)` succeeds
  - [ ] `len(list(sampler)) == 10` — correct number of episodes
  - [ ] Each episode: `len(support)` == `n_way * k_support`, `len(query)` == `n_way * k_query`
  - [ ] Support and query indices are disjoint per episode
  - [ ] `build_sampler("EpisodicSampler", data_source=data, ...)` resolves correctly

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Basic episode construction
    Tool: Bash (python REPL)
    Preconditions: Create mock Datum list with 5 classes × 10 images each
    Steps:
      1. from dassl.data.datasets.base_dataset import Datum
      2. Create 50 Datum objects (label 0-4, 10 each, impath="/tmp/test.jpg")
      3. sampler = EpisodicSampler(data, n_way=3, k_support=2, k_query=4, n_episodes=5)
      4. episodes = list(sampler)
      5. assert len(episodes) == 5
      6. For episode 0: assert len(episodes[0][0]) == 6 (3×2), len(episodes[0][1]) == 12 (3×4)
      7. assert set(episodes[0][0]).isdisjoint(set(episodes[0][1]))
    Expected Result: All assertions pass, no exceptions
    Evidence: .sisyphus/evidence/task-1-basic-episode.txt

  Scenario: Edge case — fewer samples than k_support + k_query
    Tool: Bash (python REPL)
    Preconditions: Create mock Datum list with 1 class × 3 images
    Steps:
      1. Create 3 Datum objects (label=0, impath="/tmp/test.jpg")
      2. sampler = EpisodicSampler(data, n_way=1, k_support=2, k_query=3, n_episodes=1)
      3. episodes = list(sampler)  # Should raise or handle gracefully
    Expected Result: ValueError raised with clear message ("not enough samples per class")
    Evidence: .sisyphus/evidence/task-1-edge-case.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-1-basic-episode.txt` — REPL output with assertions
  - [ ] `task-1-edge-case.txt` — ValueError traceback

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(maml): add EpisodicSampler for N-way K-shot episode construction`
  - Files: `Dassl.pytorch/dassl/data/episodic_sampler.py`, `Dassl.pytorch/dassl/data/samplers.py`, `Dassl.pytorch/dassl/data/__init__.py`

- [ ] 2. Config system — extend_cfg for META namespace

  **What to do**:
  - Edit `train.py` in `extend_cfg()`: add `cfg.TRAINER.META` namespace using `CfgNode`
    ```python
    cfg.TRAINER.META = CN()
    cfg.TRAINER.META.ENABLED = False       # toggle meta-pretraining mode
    cfg.TRAINER.META.METHOD = "CoOp_ATP"   # base prompt method
    cfg.TRAINER.META.N_WAY = 5             # N-way classification per episode
    cfg.TRAINER.META.K_SUPPORT = 1         # support shots per class
    cfg.TRAINER.META.K_QUERY = 3           # query shots per class
    cfg.TRAINER.META.INNER_LR = 0.01       # inner loop learning rate
    cfg.TRAINER.META.INNER_STEPS = 3       # inner loop SGD steps
    cfg.TRAINER.META.N_EPISODES = 100      # episodes per epoch
    cfg.TRAINER.META.SECOND_ORDER = True   # use second-order gradients
    cfg.TRAINER.META.CROSS_DATASET = False # cross-dataset meta-training
    cfg.TRAINER.META.DATASETS = []         # list of datasets for cross-dataset
    ```
  - Edit `train_select_attribute.py`: add the same META namespace (or import from train.py)
  - Add META namespace to config resolution in `reset_cfg()` (optional — META is trainer-internal)

  **Must NOT do**:
  - Do NOT remove or rename existing config namespaces
  - Do NOT add META to default config (it's trainer-specific)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple YACS config addition, well-documented pattern in existing code
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: All tasks that read META config (Tasks 3-13)
  - **Blocked By**: None

  **References** (CRITICAL):
  - `train.py:82-164` — extend_cfg() pattern: how existing TRAINER.COOP config is defined
  - `train.py:50-80` — reset_cfg() pattern: how args override config
  - `trainers/coop_atp.py:119-120` — how cfg.TRAINER.ATPROMPT.USE_ATPROMPT is read at runtime
  - `configs/trainers/CoOp/vit_b16.yaml` — existing YAML config structure

  **Acceptance Criteria**:
  - [ ] `cfg.TRAINER.META.ENABLED` is accessible after parsing any config file
  - [ ] All META keys have correct types: bool, int, float, list[str]
  - [ ] Running `python train.py --help` doesn't break

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Config parsing works
    Tool: Bash (python REPL)
    Preconditions: project venv activated
    Steps:
      1. from dassl.config import get_cfg_default
      2. cfg = get_cfg_default()
      3. from train import extend_cfg; extend_cfg(cfg)
      4. assert cfg.TRAINER.META.N_WAY == 5
      5. assert cfg.TRAINER.META.SECOND_ORDER == True
      6. assert cfg.TRAINER.META.DATASETS == []
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/task-2-config-parse.txt

  Scenario: Command-line override works
    Tool: Bash
    Preconditions: project venv activated
    Steps:
      1. python train.py --trainer CoOp --dataset-config-file configs/datasets/caltech101.yaml --config-file configs/trainers/CoOp/vit_b16.yaml --output-dir /tmp/test_meta TRAINER.META.N_WAY 10 2>&1 | head -20
      2. Verify no "unrecognized key" error
    Expected Result: Training starts (or fails gracefully on data, not on config parse)
    Evidence: .sisyphus/evidence/task-2-cmd-override.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-2-config-parse.txt`
  - [ ] `task-2-cmd-override.txt`

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(maml): add TRAINER.META config namespace`
  - Files: `train.py`, `train_select_attribute.py`

- [ ] 3. MetaPretrainer skeleton — class definition + check_cfg + __init__

  **What to do**:
  - Create NEW file `trainers/meta_pretrainer.py`
  - Implement class skeleton:
    ```python
    from dassl.engine import TRAINER_REGISTRY, TrainerX
    from dassl.metrics import compute_accuracy

    @TRAINER_REGISTRY.register()
    class MetaPretrainer(TrainerX):
        """MAML-style meta-pretraining for prompt learning methods."""
        
        def check_cfg(self, cfg):
            assert cfg.TRAINER.META.ENABLED, "META.ENABLED must be True"
            assert cfg.TRAINER.META.N_WAY > 0
            assert cfg.TRAINER.META.K_SUPPORT > 0
            assert cfg.TRAINER.META.K_QUERY > 0
            assert cfg.TRAINER.META.INNER_STEPS > 0
            assert cfg.TRAINER.META.N_EPISODES > 0
        
        def build_data_loader(self):
            """Build episodic data loader for meta-training."""
            # Stub — implemented in Task 5
            self.train_loader_x = None
            self.test_loader = None  # reuse standard test loader
        
        def build_model(self):
            """Factory: create CustomCLIP based on META.METHOD."""
            # Stub — implemented in Task 4
            pass
        
        def parse_batch_train(self, batch):
            """Parse episodic batch into support/query tensors."""
            # Stub — implemented in Task 6
            pass
        
        def forward_backward(self, batch):
            """MAML inner/outer loop for one episode."""
            # Stub — implemented in Task 6
            pass
    ```
  - Ensure class inherits from `TrainerX` (not `TrainerBase` directly) — this gives us `test()`, `after_epoch()`, `parse_batch_test()` for free
  - Add module docstring explaining the 3-phase pipeline (meta-pretrain → fine-tune → eval)

  **Must NOT do**:
  - Do NOT implement run_epoch() yet — use TrainerX default until Task 6
  - Do NOT import specific trainer modules (coop_atp, cocoop_atp, etc.) — handled in Task 4

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Class skeleton with stubs, no deep logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Tasks 4, 6, 7, 8
  - **Blocked By**: Task 2 (META config)

  **References** (CRITICAL):
  - `trainers/coop_atp.py:314-360` — CoOp_ATP class structure: check_cfg(), build_model(), forward_backward() pattern
  - `Dassl.pytorch/dassl/engine/trainer.py:595-649` — TrainerX class: run_epoch(), parse_batch_train(), forward_backward() hook
  - `Dassl.pytorch/dassl/engine/trainer.py:313-393` — SimpleTrainer: __init__(), build_data_loader(), build_model(), train(), after_epoch(), test()
  - `Dassl.pytorch/dassl/engine/build.py:1-11` — TRAINER_REGISTRY and build_trainer()
  - `trainers/__init__.py` — existing trainer imports pattern

  **Acceptance Criteria**:
  - [ ] `from trainers.meta_pretrainer import MetaPretrainer` works without error
  - [ ] Class is registered: `TRAINER_REGISTRY.registered_names()` includes "MetaPretrainer"
  - [ ] `isinstance(MetaPretrainer, type)` and `issubclass(MetaPretrainer, TrainerX)`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Class import and registration
    Tool: Bash (python REPL)
    Preconditions: project venv activated, in project root
    Steps:
      1. import sys; sys.path.insert(0, '.')
      2. import train  # triggers all imports and registration
      3. from dassl.engine.build import TRAINER_REGISTRY
      4. assert "MetaPretrainer" in TRAINER_REGISTRY.registered_names()
      5. from trainers.meta_pretrainer import MetaPretrainer
      6. assert issubclass(MetaPretrainer, object)
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/task-3-import.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-3-import.txt` — REPL output

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(maml): add MetaPretrainer skeleton class`
  - Files: `trainers/meta_pretrainer.py`

- [ ] 4. Model factory — build_model() for all prompt learning methods

  **What to do**:
  - Edit `trainers/meta_pretrainer.py` — implement `MetaPretrainer.build_model()`
  - Factory pattern based on `cfg.TRAINER.META.METHOD`:
    ```python
    METHOD_BUILDERS = {
        "CoOp":        ("trainers.coop",        "CustomCLIP"),       # vanilla CoOp
        "CoOp_ATP":    ("trainers.coop_atp",    "CustomCLIP"),       # CoOp with attributes
        "CoCoOp":      ("trainers.cocoop",      "CustomCLIP"),       # conditional CoOp
        "CoCoOp_ATP":  ("trainers.cocoop_atp",  "CustomCLIP"),
        "MaPLe":       ("trainers.maple",       "CustomCLIP"),       # multi-modal prompt
        "MaPLe_ATP":   ("trainers.maple_atp",   "CustomCLIP"),
        "DePT":        ("trainers.dept",        "CustomCLIP"),       # decomposed prompt
        "DePT_ATP":    ("trainers.dept_atp",    "CustomCLIP"),
    }
    ```
  - For each method: dynamically import the module, call `load_clip_to_cpu(cfg)` (from that module), instantiate `CustomCLIP(cfg, classnames, clip_model)`
  - Freeze all parameters EXCEPT `prompt_learner` (same pattern as `coop_atp.py:339-341`)
  - Build optimizer/scheduler ONLY for `prompt_learner` params
  - Register model as `"prompt_learner"` (same key as standard trainers)
  - Handle `GradScaler` for AMP precision if configured
  - **Critical**: the `load_clip_to_cpu` function varies between methods (different `design_details` dict) — import the specific one from each trainer module

  **Must NOT do**:
  - Do NOT copy-paste entire build_model from each trainer — use dynamic import
  - Do NOT create separate trainer classes per method — one factory handles all
  - Do NOT train CLIP backbone (always `.requires_grad_(False)`)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex cross-module dynamic import, needs to handle 8 different trainer architectures correctly
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 6 (run_epoch needs model)
  - **Blocked By**: Task 3 (MetaPretrainer skeleton exists)

  **References** (CRITICAL):
  - `trainers/coop_atp.py:324-359` — CoOp_ATP.build_model(): full pattern (load CLIP, create CustomCLIP, freeze backbone, register)
  - `trainers/coop_atp.py:39-58` — load_clip_to_cpu() with design_details dict
  - `trainers/cocoop_atp.py` — different CustomCLIP: accepts image features as condition (read full file for build_model)
  - `trainers/maple_atp.py` — MaPLe: vision + language prompt (read full file for build_model)
  - `trainers/dept_atp.py` — DePT: decomposed prompt structure (read full file for build_model)
  - `trainers/attributecompute.py:49-67` — Another load_clip_to_cpu variant (for comparison)
  - `Dassl.pytorch/dassl/engine/trainer.py:367-390` — SimpleTrainer.build_model() default (optimizer, scheduler, register pattern)

  **Acceptance Criteria**:
  - [ ] `MetaPretrainer(cfg).build_model()` succeeds for METHOD="CoOp_ATP"
  - [ ] `self.model.prompt_learner` exists and has `requires_grad=True` params
  - [ ] CLIP backbone params have `requires_grad=False`
  - [ ] `self.optim` and `self.sched` are built
  - [ ] Test with at least 3 methods: CoOp_ATP, CoCoOp_ATP, MaPLe_ATP
  - [ ] Switch method via config: change `cfg.TRAINER.META.METHOD` → different CustomCLIP created

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Factory creates CoOp_ATP model correctly
    Tool: Bash (python REPL)
    Preconditions: config with META.METHOD="CoOp_ATP", valid dataset config
    Steps:
      1. Build cfg from configs/trainers/CoOp/vit_b16.yaml + extend_cfg
      2. Set cfg.TRAINER.META.ENABLED = True; cfg.TRAINER.META.METHOD = "CoOp_ATP"
      3. Set cfg.DATASET.ROOT = "/tmp/test_data"
      4. trainer = MetaPretrainer(cfg)  # this triggers build_data_loader + build_model
      5. assert hasattr(trainer.model, 'prompt_learner')
      6. assert hasattr(trainer.model, 'image_encoder')
      7. assert hasattr(trainer.model, 'text_encoder')
      8. prompt_params = sum(p.numel() for p in trainer.model.prompt_learner.parameters() if p.requires_grad)
      9. print(f"Trainable prompt params: {prompt_params}")
    Expected Result: Model created, prompt_learner has trainable params, backbone frozen
    Evidence: .sisyphus/evidence/task-4-factory-coop-atp.txt

  Scenario: Factory method switching works
    Tool: Bash (python REPL)
    Preconditions: Same as above
    Steps:
      1. Test METHOD="CoCoOp_ATP" → verify CustomCLIP created (check for conditional forward)
      2. Test METHOD="MaPLe_ATP" → verify CustomCLIP created with vision prompts
      3. Test METHOD="DePT_ATP" → verify CustomCLIP created with decomposed prompts
    Expected Result: Each method creates its specific CustomCLIP variant without crash
    Evidence: .sisyphus/evidence/task-4-factory-switching.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-4-factory-coop-atp.txt`
  - [ ] `task-4-factory-switching.txt`

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(maml): implement model factory supporting all 8 prompt methods`
  - Files: `trainers/meta_pretrainer.py`

- [ ] 5. MAML inner_loop — clone + adapt utility function

  **What to do**:
  - Edit `trainers/meta_pretrainer.py` — implement static method `MetaPretrainer.maml_inner_loop()`
  - Function signature:
    ```python
    @staticmethod
    def maml_inner_loop(model, support_img, support_label, inner_lr, inner_steps, 
                        first_order=False):
        """Run MAML inner loop adaptation.
        
        Args:
            model: CustomCLIP instance with prompt_learner
            support_img, support_label: tensors (support set)
            inner_lr: float, inner loop learning rate
            inner_steps: int, number of SGD steps
            first_order: bool, if True use FOMAML (detach grads)
            
        Returns:
            fast_weights: OrderedDict of adapted prompt_learner parameters
        """
    ```
  - Implementation:
    - Clone `model.prompt_learner.named_parameters()` into an `OrderedDict` → `fast_weights`
    - For each step in `range(inner_steps)`:
      - Temporarily replace prompt_learner weights with `fast_weights` (use a context manager or manual swap)
      - Forward pass: `logits = model(support_img)` → `loss = F.cross_entropy(logits, support_label)`
      - Compute gradients w.r.t. `fast_weights.values()` — use `torch.autograd.grad(..., create_graph=not first_order)`
      - Update: `fast_weights[name] = fast_weights[name] - inner_lr * grad`
    - Return `fast_weights` (with computation graph preserved for second-order MAML)
  - **Context manager approach**: Create a helper that temporarily replaces `prompt_learner` parameters using `param.data = fast_weight.data` pattern, or use `torch.nn.utils.vector_to_parameters` / `parameters_to_vector`

  **Must NOT do**:
  - Do NOT modify model.state_dict() directly (use OrderedDict clone)
  - Do NOT assume PromptLearner has a fixed set of parameter names — iterate `named_parameters()`
  - Do NOT create new nn.Module instances inside inner loop

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires careful autograd handling, second-order gradients, context manager design
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 6 (run_epoch calls this)
  - **Blocked By**: Task 4 (model factory — need to know prompt_learner structure)

  **References** (CRITICAL):
  - `trainers/coop_atp.py:361-386` — forward_backward() pattern: model(image) → CE loss → backward
  - `trainers/coop_atp.py:339-341` — how prompt_learner params are identified: `"prompt_learner" not in name` → freeze
  - `trainers/attributecompute.py:141` — att_weight as a trainable parameter (non-standard param to handle)
  - PyTorch MAML reference: `torch.autograd.grad(loss, params, create_graph=True)` — the critical second-order API
  - PyTorch docs: `collections.OrderedDict` for parameter cloning pattern

  **Acceptance Criteria**:
  - [ ] Inner loop runs for `inner_steps=3` without NaN
  - [ ] `fast_weights` values differ from original prompt_learner params after inner loop
  - [ ] `create_graph=True` preserves computation graph (verify via `fast_weight.grad_fn is not None`)
  - [ ] Support loss decreases across inner steps (printed or verified)
  - [ ] First-order mode (`first_order=True`) doesn't crash and uses less memory

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Inner loop adapts parameters correctly
    Tool: Bash (python REPL)
    Preconditions: MetaPretrainer instantiated with CoOp_ATP model, dummy support batch
    Steps:
      1. Create dummy support: img=torch.randn(15,3,224,224), label=torch.randint(0,5,(15,))
      2. original_params = {n: p.clone() for n,p in model.prompt_learner.named_parameters()}
      3. fast_weights = MetaPretrainer.maml_inner_loop(model, img, label, inner_lr=0.01, inner_steps=3)
      4. for name in fast_weights:
           assert not torch.equal(fast_weights[name], original_params[name])
      5. print("Inner loop adapted all parameters successfully")
    Expected Result: fast_weights differ from originals, no NaN
    Evidence: .sisyphus/evidence/task-5-inner-loop.txt

  Scenario: Second-order gradient flow preserved
    Tool: Bash (python REPL)
    Preconditions: Same setup
    Steps:
      1. fast_weights = MetaPretrainer.maml_inner_loop(model, img, label, inner_lr=0.01, inner_steps=2, first_order=False)
      2. # Check that fast_weights retain grad function
      3. for name, param in fast_weights.items():
           print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")
      4. # Compute meta-loss using fast_weights (simulated)
      5. # Verify backward through fast_weights doesn't raise
    Expected Result: grad_fn is not None for adapted params
    Evidence: .sisyphus/evidence/task-5-second-order.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-5-inner-loop.txt`
  - [ ] `task-5-second-order.txt`

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(maml): implement MAML inner loop with second-order gradient support`
  - Files: `trainers/meta_pretrainer.py`

- [ ] 6. MetaPretrainer.run_epoch() — complete MAML training loop

  **What to do**:
  - Edit `trainers/meta_pretrainer.py` — implement `run_epoch()` override:
    ```python
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        
        end = time.time()
        for self.batch_idx, (support_batch, query_batch) in enumerate(self.meta_loader):
            # --- MAML Episode ---
            support_img, support_label = self.parse_batch_train(support_batch)
            query_img, query_label = self.parse_batch_train(query_batch)
            
            # Inner loop: adapt on support
            fast_weights = self.maml_inner_loop(
                self.model, support_img, support_label,
                inner_lr=self.cfg.TRAINER.META.INNER_LR,
                inner_steps=self.cfg.TRAINER.META.INNER_STEPS,
                first_order=not self.cfg.TRAINER.META.SECOND_ORDER
            )
            
            # Outer loop: compute meta-loss on query using adapted weights
            with self.use_fast_weights(fast_weights):
                query_logits = self.model(query_img)
            meta_loss = F.cross_entropy(query_logits, query_label)
            
            # Meta-update
            self.optim.zero_grad()
            meta_loss.backward()
            self.optim.step()
            
            # Logging
            loss_summary = {
                "meta_loss": meta_loss.item(),
                "query_acc": compute_accuracy(query_logits, query_label)[0].item(),
            }
            losses.update(loss_summary)
            batch_time.update(time.time() - end)
            
            # Print progress
            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                print(f"epoch [{self.epoch+1}/{self.max_epoch}] "
                      f"episode [{self.batch_idx+1}/{len(self.meta_loader)}] "
                      f"{losses} lr {self.get_current_lr():.4e}")
            
            end = time.time()
        
        self.update_lr()
    ```
  - Implement `use_fast_weights()` context manager:
    ```python
    @contextmanager
    def use_fast_weights(self, fast_weights):
        """Temporarily replace prompt_learner params with fast_weights."""
        original_params = OrderedDict(
            (name, param.clone()) 
            for name, param in self.model.prompt_learner.named_parameters()
        )
        # Load fast_weights into prompt_learner
        for name, param in self.model.prompt_learner.named_parameters():
            param.data.copy_(fast_weights[name].data)
        try:
            yield
        finally:
            # Restore original params
            for name, param in self.model.prompt_learner.named_parameters():
                param.data.copy_(original_params[name].data)
    ```
  - Implement `parse_batch_train()` for episodic batch format:
    - The batch from EpisodicSampler contains both support and query DataLoaders
    - Need to handle the nested structure: `(support_batch_dict, query_batch_dict)`
    - Extract `batch["img"]`, `batch["label"]` from each

  **Must NOT do**:
  - Do NOT call `self.model_backward_and_update()` — use manual `optim.zero_grad()` + `loss.backward()` + `optim.step()` for MAML
  - Do NOT use AMP autocast inside inner loop (can work, but skip for v1)
  - Do NOT modify the model permanently after inner loop — restore original weights

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core training loop, MAML gradient flow, context manager, tensor device management
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on ALL Wave 1 tasks)
  - **Parallel Group**: Wave 2 (with Tasks 7, 8 in parallel)
  - **Blocks**: Task 12, 13 (integration tests)
  - **Blocked By**: Tasks 1, 4, 5 (EpisodicSampler, model, inner_loop)

  **References** (CRITICAL):
  - `Dassl.pytorch/dassl/engine/trainer.py:598-638` — TrainerX.run_epoch(): standard loop pattern, MetricMeter, batch_time, logging format
  - `trainers/coop_atp.py:361-386` — forward_backward(): loss computation + backward + optimizer step
  - `trainers/coop_atp.py:388-393` — parse_batch_train(): batch dict → tensor extraction
  - `Dassl.pytorch/dassl/engine/trainer.py:279-283` — parse_batch_train default: (img, label, domain)
  - `Dassl.pytorch/dassl/engine/trainer.py:109-118` — model_backward_and_update(): reference for manual backward pattern

  **Acceptance Criteria**:
  - [ ] `run_epoch()` completes one epoch without crash
  - [ ] Meta-loss decreases across batches (logged via MetricMeter)
  - [ ] Query accuracy is computed and logged
  - [ ] `use_fast_weights` restores original params after exit
  - [ ] Learning rate updates at epoch end (via `update_lr()`)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Single epoch runs without crash
    Tool: Bash (python runtime)
    Preconditions: MetaPretrainer fully built (model + meta_loader), 1 epoch configured
    Steps:
      1. Set cfg.OPTIM.MAX_EPOCH = 1
      2. Set cfg.TRAINER.META.N_EPISODES = 5 (small for test)
      3. Initialize trainer = MetaPretrainer(cfg)
      4. Run trainer.train()  # single epoch
      5. Verify: no RuntimeError, no NaN in loss
      6. Print: "Meta-training epoch completed successfully"
    Expected Result: Training completes, losses printed, no crash
    Evidence: .sisyphus/evidence/task-6-epoch-run.txt

  Scenario: Context manager restores original params
    Tool: Bash (python REPL)
    Preconditions: Model with prompt_learner params memorized
    Steps:
      1. Before: save copy of prompt_learner "ctx" parameter
      2. Create fake fast_weights with different values
      3. with trainer.use_fast_weights(fast_weights): pass
      4. After: verify "ctx" parameter equals saved copy (within 1e-7)
    Expected Result: Parameters restored exactly
    Evidence: .sisyphus/evidence/task-6-ctx-manager.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-6-epoch-run.txt`
  - [ ] `task-6-ctx-manager.txt`

  **Commit**: YES (groups with Wave 2)
  - Message: `feat(maml): implement MAML run_epoch with inner/outer loop`
  - Files: `trainers/meta_pretrainer.py`

- [ ] 7. Checkpoint save/load compatibility

  **What to do**:
  - Edit `trainers/meta_pretrainer.py` — ensure saved checkpoint is loadable by standard trainers
  - In `MetaPretrainer`: override `save_model()` if needed, or rely on `TrainerBase.save_model()` default
  - Key requirement: the saved state_dict for `prompt_learner` must have the SAME keys as what standard trainer `load_model()` expects
  - Verify: after meta-training, the checkpoint at `output_dir/prompt_learner/model-best.pth.tar` should contain:
    ```python
    {
        "state_dict": {...},  # prompt_learner state_dict (NOT wrapped in DataParallel)
        "epoch": N,
        "optimizer": {...},
        "scheduler": {...},
        "val_result": float
    }
    ```
  - Implement `load_meta_checkpoint()` static method for downstream fine-tuning:
    ```python
    @staticmethod
    def load_meta_checkpoint(standard_trainer, meta_checkpoint_path):
        """Load meta-pretrained weights into a standard trainer's prompt_learner."""
        checkpoint = torch.load(meta_checkpoint_path)
        # Handle DataParallel wrapper
        state_dict = checkpoint["state_dict"]
        # Strip "module." prefix if exists
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        standard_trainer.model.prompt_learner.load_state_dict(state_dict, strict=False)
        print(f"Loaded meta checkpoint from {meta_checkpoint_path}")
    ```
  - Handle the `attributecompute.py` edge case: it has `att_weight` as extra param — use `strict=False`

  **Must NOT do**:
  - Do NOT save CLIP backbone weights in meta checkpoint (they're frozen)
  - Do NOT break existing `model.pth.tar-N` naming convention

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: State dict manipulation, straightforward checkpoint logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8)
  - **Blocks**: Task 12 (needs checkpoint loading)
  - **Blocked By**: Task 3 (MetaPretrainer skeleton)

  **References** (CRITICAL):
  - `Dassl.pytorch/dassl/engine/trainer.py:120-148` — save_model(): checkpoint structure, state_dict extraction
  - `trainers/coop_atp.py:395-427` — CoOp_ATP.load_model(): how standard trainer loads checkpoint
  - `Dassl.pytorch/dassl/engine/trainer.py:174-203` — TrainerBase.load_model(): base loading logic
  - `Dassl.pytorch/dassl/utils/__init__.py` — save_checkpoint, load_checkpoint utilities

  **Acceptance Criteria**:
  - [ ] MetaPretrainer saves checkpoint at `output_dir/prompt_learner/model-best.pth.tar`
  - [ ] Checkpoint `state_dict` keys match prompt_learner `named_parameters()` keys
  - [ ] `load_meta_checkpoint(standard_trainer, meta_ckpt_path)` succeeds with `strict=False`
  - [ ] Standard trainer can fine-tune after loading meta checkpoint

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Save and cross-load checkpoint
    Tool: Bash (python REPL)
    Preconditions: MetaPretrainer after 1 epoch training
    Steps:
      1. Check that output_dir/prompt_learner/model-best.pth.tar exists
      2. Load checkpoint: ckpt = torch.load("output_dir/prompt_learner/model-best.pth.tar")
      3. Print ckpt.keys() — must include "state_dict", "epoch", "optimizer"
      4. Print list(ckpt["state_dict"].keys())[:10] — must match prompt_learner params
      5. Create standard CoOp_ATP trainer and call load_meta_checkpoint(trainer, ckpt_path)
      6. Verify standard trainer's prompt_learner weights match checkpoint
    Expected Result: Checkpoint loads, all keys match, no strict=False errors
    Evidence: .sisyphus/evidence/task-7-checkpoint.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-7-checkpoint.txt`

  **Commit**: YES (groups with Wave 2)
  - Message: `feat(maml): implement checkpoint save/load compatibility`
  - Files: `trainers/meta_pretrainer.py`

- [ ] 8. Trainer registration — imports and wiring

  **What to do**:
  - Edit `trainers/__init__.py`: add `import trainers.meta_pretrainer` line
  - Edit `train.py`: add `import trainers.meta_pretrainer` in the custom imports section (alongside existing `import trainers.coop_atp` etc.)
  - Edit `train_select_attribute.py`: add the same import
  - Verify: running `python train.py --trainer MetaPretrainer` resolves correctly
  - Add `MetaPretrainer` to the `reset_cfg` function in `train.py` so `args.trainer` can trigger META.ENABLED:
    ```python
    if args.trainer == "MetaPretrainer":
        cfg.TRAINER.META.ENABLED = True
    ```

  **Must NOT do**:
  - Do NOT modify the `build_trainer()` logic in Dassl (handled by TRAINER_REGISTRY)
  - Do NOT import meta_pretrainer before all dependencies are available
  - Do NOT create circular imports

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple import wiring, no new logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7)
  - **Blocks**: Task 12, 13 (integration tests need registration)
  - **Blocked By**: Task 3 (MetaPretrainer class must exist)

  **References** (CRITICAL):
  - `train.py:26-34` — existing trainer imports pattern
  - `train.py:72-73` — reset_cfg args.trainer handling
  - `trainers/__init__.py` — existing imports

  **Acceptance Criteria**:
  - [ ] `python -c "import train; print('MetaPretrainer' in TRAINER_REGISTRY.registered_names())"` prints True
  - [ ] `python train.py --trainer MetaPretrainer --help` shows no import errors
  - [ ] `train_select_attribute.py` also imports correctly

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Trainer resolves via registry
    Tool: Bash
    Preconditions: All files in place
    Steps:
      1. python -c "
         import sys; sys.path.insert(0, '.')
         import train
         from dassl.engine.build import TRAINER_REGISTRY
         names = TRAINER_REGISTRY.registered_names()
         assert 'MetaPretrainer' in names, f'Not found in {names}'
         print('MetaPretrainer registered successfully')
         "
    Expected Result: Assertion passes, no import errors
    Evidence: .sisyphus/evidence/task-8-registration.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-8-registration.txt`

  **Commit**: YES (groups with Wave 2)
  - Message: `feat(maml): register MetaPretrainer in trainer registry`
  - Files: `trainers/__init__.py`, `train.py`, `train_select_attribute.py`

- [ ] 9. Config YAML files — meta training configs

  **What to do**:
  - Create directory `configs/trainers/meta/`
  - Create `configs/trainers/meta/vit_b16.yaml`:
    ```yaml
    TRAINER:
      NAME: "MetaPretrainer"
      META:
        ENABLED: True
        METHOD: "CoOp_ATP"
        N_WAY: 5
        K_SUPPORT: 1
        K_QUERY: 3
        INNER_LR: 0.01
        INNER_STEPS: 5
        N_EPISODES: 100
        SECOND_ORDER: True
        CROSS_DATASET: False
        DATASETS: []
      COOP:
        N_CTX: 16
        CSC: False
        CTX_INIT: False
        CLASS_TOKEN_POSITION: "end"
        PREC: "fp16"
      ATPROMPT:
        USE_ATPROMPT: True
        ATT_NUM: 3
        N_ATT1: 8
        N_ATT2: 8
        N_ATT3: 8
        ATT1_TEXT: "shape"
        ATT2_TEXT: "color"
        ATT3_TEXT: "texture"
    OPTIM:
      NAME: "sgd"
      LR: 0.002
      MAX_EPOCH: 50
      LR_SCHEDULER: "cosine"
      WARMUP_EPOCH: 1
    DATALOADER:
      TRAIN_X:
        SAMPLER: "EpisodicSampler"
        BATCH_SIZE: 1  # one episode = one batch
    ```
  - Create `configs/trainers/meta/vit_b16_cross_dataset.yaml`:
    - Same as above but `META.CROSS_DATASET: True` and `META.DATASETS: ["imagenet", "caltech101", "..."]`
  - Optionally create `vit_b16_fomaml.yaml` with `META.SECOND_ORDER: False` for ablation
  - Optionally create `vit_b16_5way5shot.yaml` with N_WAY=5, K_SUPPORT=5 for bigger episodes

  **Must NOT do**:
  - Do NOT remove or modify existing config files in configs/trainers/CoOp/ etc.
  - Do NOT include hardcoded paths

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML file creation with well-defined schema
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 11)
  - **Blocks**: Task 12, 13 (integration tests use configs)
  - **Blocked By**: Task 2 (META config namespace defined)

  **References** (CRITICAL):
  - `configs/trainers/CoOp/vit_b16.yaml` — existing config structure, OPTIM, DATALOADER patterns
  - `configs/datasets/caltech101.yaml` — dataset config format
  - `train.py:95-135` — COOP config defaults (N_CTX, CSC, etc.)

  **Acceptance Criteria**:
  - [ ] `vit_b16.yaml` parses without error via YACS
  - [ ] All META.* keys exist and have correct types
  - [ ] Config can be loaded via `--config-file configs/trainers/meta/vit_b16.yaml`
  - [ ] Cross-dataset config has DATASETS list properly formatted

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Config parses correctly
    Tool: Bash (python REPL)
    Preconditions: Config file exists
    Steps:
      1. from dassl.config import get_cfg_default
      2. cfg = get_cfg_default()
      3. cfg.merge_from_file("configs/trainers/meta/vit_b16.yaml")
      4. import train; train.extend_cfg(cfg)
      5. assert cfg.TRAINER.NAME == "MetaPretrainer"
      6. assert cfg.TRAINER.META.N_WAY == 5
      7. assert cfg.TRAINER.META.SECOND_ORDER == True
      8. print("Config parsed successfully")
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/task-9-config-parse.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-9-config-parse.txt`

  **Commit**: YES (groups with Wave 3)
  - Message: `feat(maml): add meta-training YAML configs`
  - Files: `configs/trainers/meta/*.yaml`

- [ ] 10. Shell scripts — pretrain, finetune, eval

  **What to do**:
  - Create directory `scripts/meta/`
  - Create `scripts/meta/meta_pretrain.sh`:
    ```bash
    #!/bin/bash
    # Meta-pretraining on base classes
    # Usage: bash scripts/meta/meta_pretrain.sh <dataset> [cross]
    DATA=/root/prompt_dataset
    TRAINER=MetaPretrainer
    CFG=vit_b16
    SHOTS=16
    DATASET=$1  # caltech101, oxford_pets, etc.
    
    for SEED in 1 2 3
    do
        DIR=output/meta_pretrain/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
        CUDA_VISIBLE_DEVICES=0 python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/meta/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base \
            TRAINER.META.CROSS_DATASET False \
            OPTIM.MAX_EPOCH 20
    done
    ```
  - Create `scripts/meta/meta_pretrain_cross.sh`:
    - Uses multiple datasets' base classes for cross-dataset meta-training
    - `TRAINER.META.CROSS_DATASET True` and `TRAINER.META.DATASETS` list
  - Create `scripts/meta/meta_finetune.sh`:
    - Standard fine-tuning using CoOp_ATP trainer
    - Load meta checkpoint via `--model-dir` and `--load-epoch` flags
    - Same pattern as `scripts/coop/atp_base2new_train.sh` but with `MODEL.INIT_WEIGHTS` pointing to meta checkpoint
  - Create `scripts/meta/meta_eval.sh`:
    - Same as `scripts/coop/atp_base2new_test.sh` (unchanged evaluation logic)
  - Make all scripts executable with `chmod +x`

  **Must NOT do**:
  - Do NOT hardcode absolute paths (use `DATA` variable)
  - Do NOT hardcode GPU device numbers
  - Do NOT duplicate logic from existing scripts — reference them

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script creation following existing patterns
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 11)
  - **Blocks**: Task 12, 13
  - **Blocked By**: Tasks 8, 9 (trainer registered, configs exist)

  **References** (CRITICAL):
  - `scripts/coop/atp_base2new_train.sh` — standard training script pattern (all arguments, for loop)
  - `scripts/coop/atp_base2new_test.sh` — evaluation script pattern (--eval-only, --model-dir, --load-epoch)
  - `scripts/coop/vanila_base2new_train.sh` — vanilla CoOp training (no ATP) for comparison

  **Acceptance Criteria**:
  - [ ] `bash scripts/meta/meta_pretrain.sh caltech101` starts training (dry-run or actual)
  - [ ] `bash scripts/meta/meta_finetune.sh caltech101` references correct paths
  - [ ] `bash scripts/meta/meta_eval.sh caltech101` follows the standard eval pattern
  - [ ] All scripts are executable

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Script syntax validation
    Tool: Bash
    Preconditions: Scripts created
    Steps:
      1. bash -n scripts/meta/meta_pretrain.sh && echo "OK" || echo "SYNTAX ERROR"
      2. bash -n scripts/meta/meta_finetune.sh && echo "OK" || echo "SYNTAX ERROR"
      3. bash -n scripts/meta/meta_eval.sh && echo "OK" || echo "SYNTAX ERROR"
      4. ls -la scripts/meta/*.sh | grep "^-rwx"
    Expected Result: All "OK", all scripts executable
    Evidence: .sisyphus/evidence/task-10-syntax.txt

  Scenario: Dry-run shows correct args
    Tool: Bash
    Preconditions: Scripts syntax-valid
    Steps:
      1. Extract the python command from meta_pretrain.sh
      2. Run with --help appended (or check arg names are valid)
      3. Verify --trainer MetaPretrainer, --config-file points to existing file
    Expected Result: No "unrecognized arguments" error
    Evidence: .sisyphus/evidence/task-10-dry-run.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-10-syntax.txt`
  - [ ] `task-10-dry-run.txt`

  **Commit**: YES (groups with Wave 3)
  - Message: `feat(maml): add meta-training shell scripts`
  - Files: `scripts/meta/*.sh`

- [ ] 11. Experiment design document

  **What to do**:
  - Create `docs/EXPERIMENT_DESIGN.md`
  - Document the complete experimental protocol:
    
    **Phase 1: Meta-Pretraining**
    - Single-dataset: meta-train on dataset X base classes (5-way 1-shot episodes, 50 epochs)
    - Cross-dataset: meta-train on [ImageNet + Caltech101 + ...] base classes (5-way 1-shot, 100 epochs)
    
    **Phase 2: Fine-tuning**
    - Load meta checkpoint into CoOp_ATP
    - Fine-tune on target dataset base classes (16-shot, 100 epochs)
    
    **Phase 3: Evaluation**
    - Test on target dataset new classes (zero-shot from fine-tuned prompt)
    - Metrics: base accuracy, new accuracy, harmonic mean (H = 2*b*n/(b+n))
    
    **Baselines**
    - B1: CoOp_ATP without meta (existing)
    - B2: CoOp without ATP, without meta
    - B3: Zero-shot CLIP
    
    **Methods**
    - M1: Meta-only (meta-pretrained, skip fine-tune, test directly)
    - M2: Meta + Fine-tune (full pipeline)
    - M3: M2 with FOMAML (first-order ablation)
    
    **Ablation Study**
    - A1: Inner steps (1, 3, 5, 10)
    - A2: N-way (3, 5, 10)
    - A3: K-support (1, 3, 5)
    - A4: Meta epochs (10, 20, 50, 100)
    - A5: Second-order vs first-order
    - A6: With ATP vs without ATP
    
    **Result Table Template** (markdown table):
    | Method | Dataset | Base Acc | New Acc | H-Mean |
    |--------|---------|----------|---------|--------|
    
    **Reproducibility**
    - Seeds: 1, 2, 3 (report mean ± std)
    - Hardware: single GPU (specify model)
    - Environment: PyTorch version, CUDA version

  **Must NOT do**:
  - Do NOT include dummy/placeholder numbers — leave tables empty for filling
  - Do NOT prescribe exact hyperparameter values as "optimal" — frame as "recommended starting point"

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation with structured tables, research methodology
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10)
  - **Blocks**: None (standalone document)
  - **Blocked By**: None

  **References** (CRITICAL):
  - `docs/ATPrompt.md` — existing project documentation format
  - `docs/CoOp.md` — existing method description format
  - `scripts/coop/atp_base2new_train.sh` — existing experimental config (SHOTS=16, EPO=100, SEED=1-5)

  **Acceptance Criteria**:
  - [ ] Document includes all 4 sections (pipeline, baselines, methods, ablation)
  - [ ] Result table template has correct columns
  - [ ] Reproducibility section is complete
  - [ ] No placeholder numbers in methodology (leave tables empty)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Document structure validation
    Tool: Bash
    Preconditions: docs/EXPERIMENT_DESIGN.md exists
    Steps:
      1. grep -c "## Phase" docs/EXPERIMENT_DESIGN.md  # expect >= 3
      2. grep -c "## Baseline" docs/EXPERIMENT_DESIGN.md  # expect >= 1
      3. grep -c "## Ablation" docs/EXPERIMENT_DESIGN.md  # expect >= 1
      4. grep -c "## Reproducibility" docs/EXPERIMENT_DESIGN.md  # expect 1
      5. grep -c "| Method | Dataset | Base Acc | New Acc | H-Mean |" docs/EXPERIMENT_DESIGN.md  # expect 1
    Expected Result: All grep counts > 0
    Evidence: .sisyphus/evidence/task-11-doc-structure.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-11-doc-structure.txt`

  **Commit**: YES (groups with Wave 3)
  - Message: `docs(maml): add experiment design and ablation plan`
  - Files: `docs/EXPERIMENT_DESIGN.md`

- [ ] 12. Single-dataset end-to-end integration test

  **What to do**:
  - Run the complete 3-phase pipeline on a small, fast dataset (EuroSAT or DTD recommended — small image count, fast CLIP encoding):
    - Phase 1: Meta-pretrain `MetaPretrainer` on base classes with reduced config (5 epochs, 20 episodes/epoch, 3-way 1-shot)
    - Phase 2: Load meta checkpoint, fine-tune with `CoOp_ATP` on same base classes (10 epochs, 16-shot)
    - Phase 3: Evaluate on new classes
  - Verify:
    - Meta-pretraining loss decreases (not NaN)
    - Meta checkpoint loads correctly into CoOp_ATP
    - Fine-tuned model achieves > random baseline accuracy on test
    - Compare meta-finetuned accuracy vs baseline CoOp_ATP (no meta) accuracy
  - Log all metrics to `.sisyphus/evidence/task-12-e2e-metrics.txt`
  - If training is not possible (no GPU/data), do a dry-run with `--eval-only` on existing checkpoint to verify compatibility

  **Must NOT do**:
  - Do NOT train on large datasets (ImageNet) for integration test — use small/fast ones
  - Do NOT run full epochs — use reduced config for speed

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: End-to-end execution, debugging, metric comparison
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 13)
  - **Blocks**: None
  - **Blocked By**: Tasks 9, 10 (configs + scripts)

  **References** (CRITICAL):
  - `scripts/meta/meta_pretrain.sh` — pretraining script (created in Task 10)
  - `scripts/meta/meta_finetune.sh` — fine-tuning script (created in Task 10)
  - `scripts/meta/meta_eval.sh` — evaluation script (created in Task 10)
  - `configs/datasets/eurosat.yaml` — fast dataset for testing
  - `output/eurosat/` — existing training outputs for baseline comparison

  **Acceptance Criteria**:
  - [ ] Meta-pretraining phase completes without crash
  - [ ] Checkpoint file exists at expected path
  - [ ] Fine-tuning phase starts and loads meta checkpoint (print confirmation)
  - [ ] Evaluation produces numeric accuracy values
  - [ ] Evidence file contains all 3 phase outputs

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Full 3-phase pipeline on EuroSAT
    Tool: Bash (training scripts)
    Preconditions: EuroSAT dataset available at DATA path
    Steps:
      1. Run meta_pretrain.sh eurosat with reduced config (EPO=3)
      2. Check that output/meta_pretrain/eurosat/.../prompt_learner/model-best.pth.tar exists
      3. Run meta_finetune.sh eurosat (reduced EPO=5)
      4. Check that fine-tuning output dir exists
      5. Run meta_eval.sh eurosat
      6. Record base_acc and new_acc from eval output
    Expected Result: All 3 phases complete, eval produces valid accuracy numbers
    Failure Indicators: Crash, NaN loss, checkpoint missing
    Evidence: .sisyphus/evidence/task-12-e2e-eurosat.txt

  Scenario: Dry-run compatibility test (if GPU unavailable)
    Tool: Bash
    Preconditions: Config files exist
    Steps:
      1. python train.py --trainer MetaPretrainer --config-file configs/trainers/meta/vit_b16.yaml --dataset-config-file configs/datasets/eurosat.yaml --output-dir /tmp/test_meta DATASET.NUM_SHOTS 16 OPTIM.MAX_EPOCH 1 2>&1 | head -50
      2. Check output for "Building model" and "Loading CLIP"
      3. Check for any ImportError or ConfigError
    Expected Result: Training starts (may fail on data loading, but not on code errors)
    Evidence: .sisyphus/evidence/task-12-dry-run.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-12-e2e-eurosat.txt` OR `task-12-dry-run.txt`

  **Commit**: NO (verification task, no code changes)

- [ ] 13. Cross-dataset end-to-end integration test

  **What to do**:
  - Test the cross-dataset META pipeline (if feasible):
    - Phase 1: Meta-pretrain with `META.CROSS_DATASET True` on 2-3 small datasets' base classes combined
    - Phase 2: Fine-tune on a held-out dataset
    - Phase 3: Evaluate
  - If cross-dataset data loading is complex, do a unit test instead:
    - Verify `MetaPretrainer.build_data_loader()` handles cross-dataset mode
    - Verify the EpisodicSampler can accept multi-dataset data sources
    - Verify config parsing for `META.DATASETS` list
  - Document any limitations or TODO items for cross-dataset support

  **Must NOT do**:
  - Do NOT skip this task entirely — at minimum, verify the cross-dataset config path works
  - Do NOT attempt to load massive cross-dataset combinations for integration test

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Multi-dataset data pipeline, cross-dataset integration
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: None
  - **Blocked By**: Tasks 9, 10

  **References** (CRITICAL):
  - `scripts/meta/meta_pretrain_cross.sh` — cross-dataset script
  - `configs/trainers/meta/vit_b16_cross_dataset.yaml` — cross-dataset config
  - `Dassl.pytorch/dassl/data/data_manager.py:51-86` — DataManager combining datasets

  **Acceptance Criteria**:
  - [ ] Cross-dataset config parses without error
  - [ ] If feasible: meta-pretraining on multi-dataset starts
  - [ ] If not feasible: unit tests pass for cross-dataset code paths
  - [ ] Documented limitations (if any) are clear

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Cross-dataset config + data loading test
    Tool: Bash (python REPL)
    Preconditions: Config file exists
    Steps:
      1. Load cross_dataset config
      2. Verify cfg.TRAINER.META.CROSS_DATASET == True
      3. Verify len(cfg.TRAINER.META.DATASETS) > 0
      4. Try to instantiate MetaPretrainer and check data_loader construction
      5. If cross-dataset loader works: print episode counts
      6. If not: document the issue clearly
    Expected Result: Config parses, issues documented
    Evidence: .sisyphus/evidence/task-13-cross-dataset.txt
  ```

  **Evidence to Capture**:
  - [ ] `task-13-cross-dataset.txt`

  **Commit**: NO (verification task)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep pattern). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -c "import trainers.meta_pretrainer"` to verify import works. Review all changed files for: `as any`, empty catches, `print()` debug left in, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Import [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration: can EpisodicSampler be imported → can MetaPretrainer be instantiated → can checkpoint be saved/loaded → does config parse correctly. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(maml): add EpisodicSampler and config system` — samplers.py, episodic_sampler.py, train.py
- **Wave 2**: `feat(maml): implement MetaPretrainer with MAML inner/outer loop` — meta_pretrainer.py, __init__.py
- **Wave 3**: `feat(maml): add configs, scripts, experiment design` — configs/, scripts/, docs/
- **Wave 4**: `test(maml): integration tests and verification` — evidence/

---

## Success Criteria

### Verification Commands
```bash
# Meta-pretraining runs successfully
python train.py --trainer MetaPretrainer --config-file configs/trainers/meta/vit_b16.yaml --dataset-config-file configs/datasets/caltech101.yaml DATASET.NUM_SHOTS 16
# Expected: epoch loss decreasing, no NaN/crash

# Checkpoint is loadable by standard trainer
python -c "
from trainers.coop_atp import CoOp_ATP
# load meta checkpoint into prompt_learner
print('Checkpoint compatible: YES')
"

# Evaluation produces expected metrics
bash scripts/meta/meta_eval.sh caltech101
# Expected: base acc > 70%, new acc > 60%
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] EpisodicSampler produces correct episode shapes
- [ ] MetaPretrainer supports all 8 method variants
- [ ] Checkpoint forward-compatible with standard trainers
- [ ] Shell scripts runnable with standard args
- [ ] Experiment design document complete
