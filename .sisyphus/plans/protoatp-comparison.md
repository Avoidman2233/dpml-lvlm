# ProtoATP 可对比实验方案

## 核心问题
ProtoATP 的 5-way episodic 协议与论文的 full-class zero-shot 协议不可直接对比。需要统一协议的多层次验证。

---

## 三协议并行设计

```
Protocol A: 论文协议 (Base→Full Novel Zero-Shot)
  ├── 训练: base classes 上标准FT
  ├── 评估: 全量 novel classes zero-shot 分类
  └── 目的: 与论文 CoOp+ATP (66.55%) 直接对比

Protocol B: 小样本协议 (Episodic Few-Shot)
  ├── 训练: base classes episodic meta-training
  ├── 评估: novel classes N-way K-shot episodic
  └── 目的: 内部消融 + 验证方法论

Protocol C: 跨数据集协议 (Cross-Dataset Transfer)
  ├── 训练: Stanford Cars base proto-meta
  ├── 评估: 其他数据集 novel classes episodic
  └── 目的: 验证度量空间的泛化性
```

---

## 实验方法矩阵

| # | 方法 | Protocol | 训练 | 评估 |
|---|------|----------|------|------|
| P1 | CoOp (vanilla) | A | 标准FT 100ep | novel zero-shot |
| P2 | CoOp_ATP | A | 标准FT 100ep | novel zero-shot |
| P3 | **ProtoATP→FT100** | A | Proto meta + FT 100ep | novel zero-shot |
| P4 | **ProtoATP→FT20** | A | Proto meta + FT 20ep | novel zero-shot |
| P5 | Proto (vanilla) | B | Proto meta on base | novel 5-way K=1/3/5 |
| P6 | **ProtoATP** | B | Proto meta on base | novel 5-way K=1/3/5 |
| P7 | CoOp_ATP few-shot | B | novel上K-shot训练 | novel全量test |
| P8 | MAML+ATP | B | MAML meta on base | novel inner-loop | 
| P9 | ProtoATP (cross) | C | Cars proto-meta | other datasets B |

### 要回答的问题

| 对比 | 问题 | 期望 |
|------|------|------|
| P3 vs P2 | ProtoATP初始化是否优于随机初始化？ | P3 > P2 |
| P4 vs P2 | 短FT是否保留proto初始化优势？ | P4 ≈ P2 (或略优) |
| P6 vs P5 | 属性词是否增强原型？ | P6 > P5 (已验证 ✅) |
| P6 vs P8 | Metric-based是否优于MAML？ | P6 >> P8 (已验证 ✅) |
| P3 vs P6 | ProtoATP哪种用法更好？ | 取决于任务 |

---

## 数据集

| 数据集 | 理由 | Protocol |
|--------|------|----------|
| Stanford Cars | 大类数(98), MAML正例, ProtoATP已验证 | A+B |
| Oxford Pets | 中类数(18), ATP强 | A+B |
| DTD | N_CTX不兼容(4 vs 2), 测试鲁棒性 | A |
| EuroSAT | 小类数(5), 测试极限 | B |
| Oxford Flowers | 中类数(51), 细粒度 | A |

---

## 执行计划 (聚焦 P1-P6)

### Step 1: 已完成的复用
- ✅ ProtoATP meta-trained on Stanford Cars, Oxford Pets
- ✅ Proto + CoOp_ATP episodic baseline on Cars
- ✅ ProtoATP > Proto (+9.1%) validated
- ✅ ProtoATP >> MAML (+26.2%) validated

### Step 2: 论文协议实验 (核心新增)
- [ ] P1: CoOp standard 100ep FT on Cars → novel zero-shot
- [ ] P2: CoOp_ATP 100ep FT on Cars → novel zero-shot (已有 62.5%)
- [ ] P3: ProtoATP→FT100 on Cars → novel zero-shot
- [ ] P4: ProtoATP→FT20 on Cars → novel zero-shot

### Step 3: 扩展到更多数据集
- [ ] P2 on Oxford Pets, Flowers, DTD
- [ ] P3 on Oxford Pets, Flowers

### Step 4: 汇总
- [ ] 统一对比表 (P1-P8 × datasets)
- [ ] 分析报告

---

## 成功标准

```
Paper-level comparability:
  P3 (ProtoATP→FT100) 与 P2 (CoOp_ATP) 在 Stanford Cars 上直接对比
  → 如果 P3 > P2: ProtoATP 作为初始化的价值被验证
  → 如果 P3 ≈ P2: 与 MAML 类似，100ep FT 覆盖了初始化优势
  → 如果 P4 > P3: 短FT保留了ProtoATP的优势

Methodology validation (已完成):
  P6 > P5 (属性增强原型) ✅
  P6 >> P8 (metric > optimization) ✅
```
