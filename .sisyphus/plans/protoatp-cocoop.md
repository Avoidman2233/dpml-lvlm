# ProtoATP CoCoOp 扩充计划

## 目标
将 ProtoATP 从 CoOp-based 扩展到 CoCoOp-based (instance-conditional prompt)，对比两种方法的 few-shot 性能。

## CoCoOp vs CoOp 关键差异

| 组件 | CoOp | CoCoOp |
|------|------|--------|
| Prompt 构造 | ctx: (n_cls, n_ctx, dim) 固定 | ctx = base_ctx + meta_net(im_features) 实例条件 |
| meta_net | 无 | Linear→ReLU→Linear (vis/16 → ctx_dim) |
| PromptLearner.forward() | 无参数, 返回全部类prompts | 接收 im_features, 返回per-image prompts |
| 文本原型构造 | 一个类一个文本原型 | K个support图像 → K个文本特征 → 取平均 |

## 实现方案

### 1. ProtoTrainer 扩展
在 `trainers/proto_trainer.py` 中, 将 `build_model()` 改为支持 `PROTO_METHOD`:
- `"coop"`: 现有 CoOp-based (默认)
- `"cocoop"`: CoCoOp-based, 加载 cocoop_atp 的 load_clip_to_cpu + CustomCLIP

### 2. PromptLearner 交互变化
```python
# CoOp mode (现有):
prompts = pl()  # (n_cls, n_tokens, dim)
text_features = text_encoder(prompts, tok)

# CoCoOp mode (新增):
im_features = image_encoder(support_img)  # (B, dim)
prompts = pl(im_features)  # (B, n_tokens, dim) — per-image!
text_features = text_encoder(prompts, tok)  # (B, dim)
# → reshape by class → average → class text prototype
```

### 3. 评估适配
Eval script 中同样需适配 per-image prompt 构造。

## 实验矩阵

| # | 方法 | Prompt类型 | 属性 | Protocol |
|---|------|:---:|:---:|------|
| E1 | ProtoATP-CoOp | CoOp | ✓ | Few-shot episodic |
| E2 | ProtoATP-CoCoOp | CoCoOp | ✓ | Few-shot episodic |
| E3 | Proto-CoCoOp | CoCoOp | ✗ | Few-shot episodic |
| E4 | CoCoOp+ATP | CoCoOp | ✓ | Base-to-New zero-shot |
| E5 | CoCoOp | CoCoOp | ✗ | Base-to-New zero-shot |

**对比分析:**
- E2 vs E1: CoCoOp是否优于CoOp? (instance-conditional > fixed prompt?)
- E2 vs E3: 属性词在CoCoOp中是否增强原型? (RQ4延伸)
- E4 vs E5: CoCoOp+ATP的base-to-new表现?

## 数据集
Stanford Cars (优先), Oxford Pets (验证)

## 执行顺序

1. 扩展 ProtoTrainer 支持 `PROTO_METHOD=cocoop`
2. Stanford Cars CoCoOp meta-training (20ep)
3. Stanford Cars CoCoOp few-shot eval (E1 vs E2 vs E3)
4. Stanford Cars Base-to-New eval (E4 vs E5)
5. Oxford Pets 验证
6. 汇总对比

## 预期
- E2 > E1: CoCoOp's instance-conditional mechanism should work better with few-shot prototype
- E2 > E3: 属性词增强应延续到CoCoOp
- CoCoOp 的 meta_net 参数量大 (~32k), 但 overall 仍在可控范围
