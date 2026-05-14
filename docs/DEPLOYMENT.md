# ATPrompt + DL-MPT 云计算部署指南

## 一、系统要求

### 最低（CoOp_ATP / DL-MPT CoOp 模式）

| 组件 | 规格 |
|------|------|
| GPU | ≥12GB VRAM（16GB 推荐） |
| CPU | 4 vCPU |
| RAM | 16GB |
| 磁盘 | 100GB SSD |
| OS | Ubuntu 20.04/22.04 LTS |
| CUDA | **11.8** |
| Python | **3.10.x** |

### 推荐（全方法 + CoCoOp 模式）

| 组件 | 规格 |
|------|------|
| GPU | **A6000 48GB** 或 A100 40GB |
| CPU | 16 vCPU |
| RAM | 64GB |
| 磁盘 | 500GB NVMe SSD |
| OS | Ubuntu 22.04 LTS |

### 显存需求矩阵

| 方法 | Stanford Cars | EuroSAT |
|------|:---:|:---:|
| CoOp_ATP | 3GB | 1.5GB |
| DL-MPT CoOp | 5GB | 3GB |
| DL-MPT CoCoOp | **22GB** | 15GB |
| ProtoATP | 5GB | 3GB |
| MAML+ATP | 5GB | 3GB |

---

## 二、PyTorch 版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | **3.10.x** | 不支持 3.11+ |
| PyTorch | **2.1.x** | CUDA 11.8 预编译版 |
| torchvision | 0.16.x | 匹配 PyTorch |
| CUDA Toolkit | 11.8 | nvcc --version |
| cuDNN | 8.x | 随 PyTorch 安装 |

---

## 三、部署步骤

### 1. 创建环境

```bash
conda create -n atprompt python=3.10 -y
conda activate atprompt
```

### 2. 安装 PyTorch

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. 验证 GPU

```bash
python -c "
import torch
print(f'version: {torch.__version__}, cuda: {torch.version.cuda}')
print(f'gpu: {torch.cuda.get_device_name(0)}')
print(f'vram: {torch.cuda.get_device_properties(0).total_mem//1024**3}GB')
"
```

### 4. 安装依赖

```bash
cd ATPrompt
pip install -r requirements.txt
pip install ftfy regex  # CLIP tokenizer 依赖
```

### 5. 准备数据集

```
DATA/
├── stanford_cars/   (split_zhou_StanfordCars.json + cars_train/ + cars_test/)
├── eurosat/         (split_zhou_EuroSAT.json + ...)
├── oxford_pets/     (split_zhou_OxfordPets.json + ...)
├── dtd/             (split_zhou_DescribableTextures.json + ...)
└── oxford_flowers/  (split_zhou_OxfordFlowers.json + ...)
```

### 6. 预下载 CLIP 模型

```bash
mkdir -p ~/.cache/clip
wget -O ~/.cache/clip/ViT-B-16.pt \
    https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

---

## 四、运行测试

### Test 1: 导入检查

```bash
python -c "
from trainers.dlmpt_trainer import DLMPTTrainer
from trainers.proto_trainer import ProtoTrainer
from trainers.coop_atp import load_clip_to_cpu
print('All imports OK')
"
```

### Test 2: CoOp_ATP 快速训练（3 分钟）

```bash
python train.py \
    --root DATA --seed 1 --trainer CoOp_ATP \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir /tmp/test_coop \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ATPROMPT.USE_ATPROMPT True TRAINER.COOP.N_CTX 2 \
    OPTIM.MAX_EPOCH 3 DATALOADER.NUM_WORKERS 4
```

### Test 3: DL-MPT CoOp 快速训练（5 分钟）

```bash
bash scripts/dlmpt/train.sh --dataset eurosat --seed 1 --lambda 0.2
```

### Test 4: DL-MPT CoCoOp（仅 ≥24GB GPU）

```bash
bash scripts/dlmpt/train_cocoop.sh --dataset eurosat --seed 1 --lambda 0.2
```

---

## 五、完整实验

```bash
# 1. Baseline
for DS in eurosat oxford_pets stanford_cars; do
    python train.py --root DATA --seed 1 --trainer CoOp_ATP \
        --dataset-config-file configs/datasets/${DS}.yaml \
        --config-file configs/trainers/CoOp/vit_b16.yaml \
        --output-dir output/baseline/${DS} \
        DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.ATPROMPT.USE_ATPROMPT True OPTIM.MAX_EPOCH 20 \
        DATALOADER.NUM_WORKERS 4
done

# 2. DL-MPT CoOp
for DS in stanford_cars oxford_pets eurosat; do
    bash scripts/dlmpt/train.sh --dataset ${DS} --seed 1 --lambda 0.2
done

# 3. DL-MPT CoCoOp（A6000）
bash scripts/dlmpt/train_cocoop.sh --dataset stanford_cars --seed 1 --lambda 0.2

# 4. 评估
bash scripts/dlmpt/eval_novel.sh \
    --dataset stanford_cars --seed 1 \
    --model-dir output/dlmpt/stanford_cars/seed1/lambda0.2

python scripts/dlmpt/eval_episodic.py \
    --checkpoint output/dlmpt/stanford_cars/seed1/lambda0.2 \
    --dataset stanford_cars
```

---

## 六、云平台推荐

| 平台 | GPU | 时价 | 适用 |
|------|-----|------|------|
| AutoDL | A6000 48GB | ¥3-5/h | 全方法 |
| 恒源云 | A100 40GB | ¥5-8/h | 高性能 |
| Lambda Labs | A6000 | $1.1/h | 性价比 |
| AWS g5 | A10G 24GB | $1.5/h | CoOp only |
