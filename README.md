# ATPrompt 项目概览

## 选题

> **基于元任务微调与属性辅助推理的 LVLM 小样本学习**
> 在 CLIP ViT-B/16 (frozen backbone) 的 Prompt Learning 框架下，验证**元任务微调**与**属性辅助推理**能否产生协同效应，提升小样本图像分类性能。

---

## 核心方法：DL-MPT(CoOp+ATP)

### 思路

DL-MPT = **Dual-loop Meta Prompt Tuning** + CoOp + ATP

```
L_total = L_base + λ · L_meta
L_base:   CoOp+ATP 标准分类（prompt → text_encoder → cos_sim(image) → CE）
L_meta:   元任务原型分类（support → proto → query_classify → CE）
λ 调度:  Warmup(ep1-5, λ=0) → Joint(ep6-20, λ=0.2) → Refine(ep21-25, λ=0.5)
```

**本质**: L_meta 不是真正的 meta-learning（无 inner-loop），而是 **episodic regularization**——每 batch 随机采样 N-way K-shot episode，提供不同方向的梯度，防止 prompt 过拟合 base 类分布。

### 核心发现

- 8 数据集 3-seed 验证：**7/8 数据集 Novel 正向提升，平均 ΔNovel +2.09%**

- 最大提升：Stanford Cars +5.38%

- 仅 25 epochs（vs Paper CoOp+ATP 100 epochs）

- 零额外参数
  
  ### 关键基线（3-seed mean）
  
  | 数据集            | DL-MPT Base | DL-MPT Novel | ΔNovel vs CoOp+ATP |
  | -------------- |:-----------:|:------------:|:------------------:|
  | Stanford Cars  | 74.56       | 71.93        | +5.38              |
  | Oxford Pets    | 95.38       | 97.58        | +0.99              |
  | Oxford Flowers | 97.28       | 69.10        | +1.58              |
  | Food101        | 89.96       | 90.46        | +3.02              |
  | FGVC Aircraft  | 36.53       | 31.77        | +4.55              |
  | UCF101         | 83.07       | 65.44        | +0.48              |
  | DTD            | 79.32       | 49.60        | +4.11              |
  | EuroSAT        | 84.90       | 56.35        | -3.44              |

---

## 设计原则

### P0: 命名规范

基线方法是 CoOp+ATP 和 CoCoOp+ATP，DL-MPT 注入其中。
| 正确命名 | 错误命名 |
|---------|---------|
| `DL-MPT(CoOp+ATP)` | ~~DL-MPT CoOp~~ |
| `DL-MPT(CoCoOp+ATP)` | ~~DL-MPT CoCoOp~~ |
| paper `CoOp+ATP` | ~~CoOp~~ |

### P1: 脚本即真理

所有实验流程固化为 `.sh` 脚本。禁止临时 `python -c`。每次实验的输出路径、超参数、随机种子固定。

### P2: 目录规范

```
训练脚本:    scripts/{method}/train_{variant}.sh
评估脚本:    scripts/{method}/eval_{protocol}.sh
Sweep 脚本:  scripts/{method}/sweep.sh
配置文件:    configs/trainers/{method}/vit_b16.yaml
输出目录:    output/{method}/{dataset}/seed{S}/lambda{L}/
结果文件:    output/{method}/FINAL_RESULTS.md
```

### P3: 输出重定向

训练 stdout/stderr 必须用 `> log 2>&1`，**禁止 `| tee`**（管道缓冲导致 GPU 显存堆积）。

### P4: 进度监控

固定格式:

```
[DL-MPT] epoch=01/25 batch=020/100 L_base=1.324 L_meta=2.863 acc_meta=75.0% λ=0.20 lr=1.88e-03
```

### P5: 协议分离

- **Protocol A**: Base 类训练 → Novel 类 zero-shot 评估
- **Protocol B**: N-way K-shot episodic evaluation on novel classes

---

## 关键路径

### 代码

| 文件                                    | 说明                           |
| ------------------------------------- | ---------------------------- |
| `trainers/dlmpt_trainer.py`           | DL-MPT(CoOp+ATP) 训练器（328行）   |
| `trainers/dlmpt_cocoop_lite.py`       | DL-MPT(CoCoOp+ATP) 训练器（190行） |
| `trainers/coop_atp.py`                | CoOp+ATP baseline            |
| `trainers/cocoop_atp.py`              | CoCoOp+ATP baseline          |
| `train.py`                            | 主训练入口                        |
| `configs/trainers/dlmpt/vit_b16.yaml` | DL-MPT 配置                    |

### 脚本

| 脚本                               | 用途                    |
| -------------------------------- | --------------------- |
| `scripts/dlmpt/train.sh`         | DL-MPT(CoOp+ATP) 训练   |
| `scripts/dlmpt/train_cocoop.sh`  | DL-MPT(CoCoOp+ATP) 训练 |
| `scripts/dlmpt/eval_novel.sh`    | Protocol A 评估         |
| `scripts/dlmpt/eval_episodic.sh` | Protocol B 评估         |
| `scripts/dlmpt/sweep_cocoop.sh`  | CoCoOp 全数据集 sweep     |

### 文档

| 文件                                    | 内容                          |
| ------------------------------------- | --------------------------- |
| `docs/DL-MPT(CoOp+ATP)_rates.md`      | DL-MPT 最终实验报告（8 数据集 3-seed） |
| `docs/基础Atprompt方法的准确率.md`            | Paper baseline 数据           |
| `docs/失败的dlmpt变体1.md`                 | 8 个失败变体记录                   |
| `docs/archive/failed-variants.tar.gz` | 失败变体完整归档                    |
| `docs/归档文件/`                          | 历史文档                        |

### 数据

| 路径                                                                          | 内容          |
| --------------------------------------------------------------------------- | ----------- |
| `/home/avoidman2233/Desktop/LVLM/DATA/`                                     | 所有数据集       |
| `stanford_cars` / `eurosat` / `oxford_pets` / `oxford_flowers`              | 主要实验数据集     |
| `ucf101` / `dtd` / `food101` / `fgvc_aircraft`                              | 主要实验数据集     |
| `imagenet` / `imagenetv2` / `imagenet_a` / `imagenet_r` / `imagenet_sketch` | ImageNet 变体 |
| `caltech101` / `sun397`                                                     | 额外数据集       |

### 环境

| 项目     | 值                                                        |
| ------ | -------------------------------------------------------- |
| Python | `/home/avoidman2233/miniconda3/envs/atprompt/bin/python` |
| 数据目录   | `/home/avoidman2233/Desktop/LVLM/DATA`                   |
| 项目目录   | `/home/avoidman2233/Desktop/LVLM/ATPrompt`               |

---

## 训练命令

```bash
# DL-MPT(CoOp+ATP)
bash scripts/dlmpt/train.sh --dataset stanford_cars --seed 1 --lambda 0.2
# Novel 评估
python train.py --root /home/avoidman2233/Desktop/LVLM/DATA --seed 1 --trainer CoOp_ATP \
    --dataset-config-file configs/datasets/stanford_cars.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --model-dir output/dlmpt/stanford_cars/seed1/lambda0.2 --load-epoch 25 --eval-only \
    DATASET.SUBSAMPLE_CLASSES new TRAINER.ATPROMPT.USE_ATPROMPT True TRAINER.COOP.N_CTX 2
```
