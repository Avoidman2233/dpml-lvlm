# Draft: ProtoATP 实验方案设计

## 核心命题
> 基于元任务微调与属性辅助推理的小样本学习

## 方法论反思
- **MAML失败根因**: CLIP prompt learning 参数空间极小(1024 params)，优化landscape近乎凸平滑，random init已收敛到最优。MAML的"更好初始化"无增量价值。
- **Metric-based 契合原因**: CLIP本身就是cosine-similarity度量模型，原型网络是其自然延伸。属性词作为语义锚点引导prototype在度量空间中的位置，与metric-based框架天然融合。
- **协同机制**: Meta-training学习最优的prompt embedding(含ctx+ctx_att)，使得class prototype具有最大判别力；属性词增强text encoder输出，使不同类的prototype在度量空间中分离度增大

## 实验设计原则
1. **完整消融矩阵**: 分离attribute贡献、metric-learning贡献、组合贡献
2. **统一评估协议**: 所有方法使用相同的base-to-new split和episodic evaluation
3. **多数据集验证**: 覆盖不同规模和领域的数据集
4. **统计显著性**: 3 seeds + std报告
5. **正负对照**: 包含已知有效的baseline和已知失败的MAML作为参照

## 消融矩阵
| ID | 方法 | 属性推理 | 元任务学习 | 推理方式 |
|----|------|---------|-----------|---------|
| A | CoOp (vanilla) | ✗ | ✗ | 标准FT→zero-shot |
| B | CoOp_ATP | ✓ | ✗ | 标准FT→zero-shot |
| C | Proto (vanilla) | ✗ | ✓(metric) | 原型推断 |
| D | ProtoATP | ✓ | ✓(metric) | 原型推断+属性 |
| E | MAML+ATP | ✓ | ✓(opt) | inner-loop SGD |

## 研究问题
| RQ | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | 属性辅助推理是否提升few-shot分类？ | B > A |
| RQ2 | Metric-based元学习是否优于标准FT？ | C > A (few-shot protocol) |
| RQ3 | ProtoATP组合是否优于单独使用属性？ | D > B |
| RQ4 | 属性词是否增强原型判别力？ | D > C |
| RQ5 | Metric-based是否优于Optimization-based？ | D >> E |
| RQ6 | 组合效应的数据集依赖性？ | D vs B across datasets |

## 成功标准
- RQ1-RQ5 至少4/5在≥2个数据集上成立
- RQ6 提供明确的适用条件(如: ≥N base classes, domain similarity阈值)
- ProtoATP在至少1个数据集上达到state-of-the-art水平

## 实施方案
1. 实现ProtoTrainer (episode-based原型网络训练)
2. 实现ProtoEvaluator (few-shot原型推断评估)
3. 运行完整消融矩阵(5 methods × 5 datasets × 3 seeds)
4. 统计分析+报告

## 关键风险
- 原型网络在1-shot时就是直接使用单样本embedding，方差大
- 跨数据集时CLIP backbone的domain gap可能影响prototype质量
- 属性词对不同数据集的有效性不均衡
