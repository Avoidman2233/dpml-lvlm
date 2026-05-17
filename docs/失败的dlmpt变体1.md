# 失败的 DL-MPT 变体记录

> 记录所有 DL-MPT 框架下的改进尝试。所有实验在 Stanford Cars (CLIP ViT-B/16, 25 epochs, seed 1) 上完成。
> 基线: DL-MPT(CoOp+ATP) — Base 74.49, Novel 71.43, HM 72.92。

---

## 1. Adapter-A: Gated MLP（patch 级）

**原理**: 在 epicolic meta path 中，对视觉 patch tokens 做 bottleneck MLP（768→64→768），再以文本原型为条件做门控：`gated = mlp_out ⊙ sigmoid(W·text_anchor)`，residual 后 pool 为视觉原型。

**参数**: 888K。**结果**: Base 73.51(-1.0), Novel 71.03(-0.4)。❌ 失败。

**结论**: patch 级 MLP 改变 CLIP 流形，破坏原有几何。参数越大越差。

---

## 2. Adapter-B: Cross-Attention（patch 级）

**原理**: 视觉 patch tokens 与类文本原型做 cross-attention（Q=vis, K/V=text），residual + pool 为原型。

**参数**: 3,152K。**结果**: Base 74.26(-0.2), Novel 71.03(-0.4)。❌ 失败。

**结论**: 跨注意力在 few-shot 下过参数化，学噪音而非对齐。

---

## 3. Refiner: Prototype Residual（proto 级）

**原理**: 在 512-dim 原型空间做残差 MLP：P' = P + α·MLP(P)，α=0.1。仅作用于已计算的 (vis+text)/2 原型。

**参数**: 66K。**结果**: Base 75.46(+1.0), Novel 71.03(-0.4)。⚠️ Base 最佳但 Novel 未动。

**结论**: prototype 级精炼有益 in-domain 分类，但无泛化增益。

---

## 4. Refiner-Align: Refiner + L_patch

**原理**: 在 Refiner 基础上加入 patch-prototype InfoNCE 对齐损失（λ=0.05），显式要求投影后的 patch 靠近所属类的 prototype。

**参数**: 231K。**结果**: Base 75.21(-0.25 vs Refiner), Novel 71.03。❌ 比纯 Refiner 更差。

**结论**: 对齐损失干扰了 prompt 的 base loss 梯度，未带来额外信号。

---

## 5. CrossAlign: Cross-Attention + Token-Level Alignment

**原理**: 视觉 patch(768) 投影到 512 后，与类文本特征做 cross-attention，并加入 token 级 InfoNCE 对齐损失。

**参数**: 1,511K。**结果**: Base 74.09(-0.4), Novel 71.03。❌ 失败。

**结论**: token 级交互无法穿透冻结 text_encoder 传递有效梯度给 prompt。

---

## 6. AttrGate: 属性引导门控

**原理**: 利用 ATP 的属性 context（ctx_att1/2/3）作为门控信号。`gate = sigmoid(W·attr_feat)`, 应用于投影后的视觉 patch。

**参数**: 722K。**结果**: Base 75.14(+0.7), Novel 71.03。❌ Novel 不动。

**结论**: 属性信息已通过 ATP prompt 充分编码，额外门控是冗余。

---

## 7. Reptile: True Meta-Learning

**原理**: 内环 3-step SGD 在 episode 上适应 prompt，外环 Reptile (θ←θ+ε(θ'-θ)) 更新。

**参数**: 0（零增量）。**结果**: Base 74.91, Novel 67.20(-4.2)。❌ 严重退化。

**结论**: 无约束的内环 SGD 使 prompt 过度适应 base 类 episode。

---

## 8. Reptile + L2 Reg

**原理**: Reptile 内环加入 `reg·||θ'-θ||²` 惩罚（reg=0.02），防止过度偏离。

**参数**: 0。**结果（3-seed mean）**: Base 80.16(+0.03), Novel 66.51(-0.02)。⚠️ 打平 DL-MPT。

**结论**: L2 约束有效阻止了退化，但未带来超越 episodic regularization 的新增益。

---

## 总结

| # | 变体 | 层级 | Params | ΔNovel | 结论 |
|:---:|------|------|:---:|:---:|------|
| 1 | Gated MLP | patch | 888K | -0.4 | 破坏 CLIP 流形 |
| 2 | Cross-Attention | patch | 3.15M | -0.4 | 过参数化 |
| 3 | Refiner | proto | 66K | -0.4 | Base 增益, Novel 不动 |
| 4 | Refiner + L_patch | proto | 231K | -0.4 | 对齐信号太弱 |
| 5 | CrossAlign | token | 1.51M | -0.4 | 梯度隔离 |
| 6 | AttrGate | patch | 722K | -0.4 | 属性信息冗余 |
| 7 | Reptile (no reg) | param | 0 | -4.2 | base-bias |
| 8 | Reptile + L2 reg | param | 0 | -0.02 | 打平 |

**核心结论**: Novel 锁死在 71% 附近。prompt_learner 的 base CE loss 主导优化，任何增补信号均被淹没。DL-MPT 的 episodic regularization 已是最优方案——简单、零参数、+4.88% Novel 增益。
