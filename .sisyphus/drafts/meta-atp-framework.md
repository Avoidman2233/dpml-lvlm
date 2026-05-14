# Draft: Meta-ATP Framework - 元任务微调训练框架

## Requirements (confirmed)
- [approach]: MAML-style meta-task fine-tuning (Approach A)
- [goal]: 构建可移植、可验证的 "基于元任务微调与属性辅助推理的LVLM小样本学习" 框架
- [quality]: 学术级代码，而非一次性实验代码

## Technical Decisions
- [meta-learning]: MAML style (inner-outer loop over prompt_learner parameters)
- [factor]: First-order MAML (FOMAML) for simplicity and speed
- [scope]: Support all ATP variants (CoOp_ATP, CoCoOp_ATP, MaPLe_ATP, DePT_ATP)
- [evaluation]: base-to-new generalization + cross-dataset + domain generalization
- [backward-compat]: Existing non-meta trainers remain unchanged

## Scope Boundaries
- INCLUDE: Meta-training framework (EpisodicSampler + MetaTrainer abstract class + concrete Meta*_ATP trainers)
- INCLUDE: Configuration system for meta-training hyperparameters
- INCLUDE: Training scripts for all ATP variants with meta pretraining
- INCLUDE: Evaluation scripts (base-to-new, cross-dataset, domain generalization)
- EXCLUDE: Modifying existing non-ATP trainers
- EXCLUDE: Modifying CLIP backbone
- EXCLUDE: Changing gpt_query.py (attribute generation pipeline)

## Research Findings
- [Dassl framework]: TrainerBase → SimpleTrainer → TrainerX; run_epoch() iterates train_loader_x, calls forward_backward()
- [DataManager]: Builds DatasetBase, creates torch DataLoader with samplers (RandomSampler, RandomClassSampler, etc.)
- [ATP trainers]: Override build_model() + forward_backward(); only prompt_learner params optimized
- [Attribute pipeline]: GPT generates attributes → hardcoded into train.py extend_cfg → PromptLearner creates ctx_att vectors
- [No test infrastructure]: Academic codebase, no pytest/unittest

## Test Strategy
- [infrastructure]: NONE (to be verified via evaluation benchmarks, not unit tests)
- [Agent QA]: Every task includes Playwright/curl/bash-based verification scenarios
