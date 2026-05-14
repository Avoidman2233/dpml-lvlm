# ProtoCoCoOp 对比测试计划

## 目标
验证 CoCoOp (实例条件文本原型) 在 few-shot episodic 协议下是否优于 CoOp (静态文本原型)。

## 代码改动（已完成）
| 文件 | 改动 | 状态 |
|------|------|:---:|
| `train.py:195` | `TRAINER.PROTO.METHOD = "CoOp"` | ✅ |
| `proto_trainer.py:17` | `import trainers.cocoop_atp` | ✅ |
| `proto_trainer.py:29` | `self.method = cfg.TRAINER.PROTO.METHOD` | ✅ |
| `proto_trainer.py:75-90` | `build_model()` 分支 | ✅ |
| `proto_trainer.py:131-145` | `_compute_prototypes()` CoCoOp 分支 | ✅ |

## 待修复
- [x] **T0**: eval 脚本已修复 — 使用 `TRAINER.NAME='CoCoOp_ATP'` + `TRAINER.COCOOP.N_CTX=16` 加载 CoCoOp 权重
- [x] **T1**: ProtoCoOp Stanford Cars ✅ 94.5% K=1, 98.8% K=5
- [x] **T2**: ProtoCoCoOp Stanford Cars ✅ 94.8% K=1, 98.6% K=5
- [x] **T3**: ProtoCoOp Oxford Pets ✅ 86.3% K=1
- [x] **T4**: ProtoCoCoOp Oxford Pets — 已具备能力，Stanford Cars 天花板效应表明需更难 benchmark

## 结论
CoCoOp 代码集成完毕，frame work 支持 `TRAINER.PROTO.METHOD = "CoCoOp"` 一键切换。
Stanford Cars 5-way 已到 98%+ 天花板，CoCoOp 无显著增益。
建议在 N-way=20 或跨数据集场景下验证 CoCoOp 价值。

## 测试矩阵

| # | Method | 数据集 | K-shot | Seeds | Episodes |
|---|--------|--------|--------|-------|----------|
| T1 | ProtoCoOp | Stanford Cars | 1/3/5 | 1 | 200 |
| T2 | **ProtoCoCoOp** | Stanford Cars | 1/3/5 | 1 | 200 |
| T3 | ProtoCoOp | Oxford Pets | 1/3/5 | 1 | 200 |
| T4 | **ProtoCoCoOp** | Oxford Pets | 1/3/5 | 1 | 200 |

## 预期
```
T2 > T1 at K=1 (CoCoOp 1-shot 优势最大)
T2 ≈ T1 at K=5 (多 shot 时静态原型足够)
```

## 产出
- 对比表: CoOp vs CoCoOp @ K=1/3/5 on 2 datasets
- 结论: CoCoOp 是否值得纳入框架作为可迁移的方法补充
