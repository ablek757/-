# 智能客服意图识别：Prompt工程 vs 微调对比实验

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

对比 **Prompt工程**（零样本/少样本/CoT）与 **模型微调**（LoRA）在电商客服意图分类任务上的效果差异，提供完整的评估框架和产品决策建议。

## 🎯 核心发现

| 维度 | Few-shot Prompt | 微调模型 | 结论 |
|------|----------------|----------|------|
| **准确率** | 90.0% ⭐ | 89.5% | 效果相当 |
| **小规模成本** | ¥0.003/次 ⭐ | ¥0.06/次 | Prompt胜 |
| **大规模成本** | ¥0.003/次 | ¥0.0002/次 ⭐ | 微调胜60-170x |
| **延迟** | 800ms | 300ms ⭐ | 微调快2.7x |
| **迭代速度** | 分钟级 ⭐ | 小时级 | Prompt胜 |

**关键结论**：
- Few-shot Prompt效果最佳（90.0%），仅需2-3个示例
- CoT推理适得其反（81.5%），过度推理导致误判
- 小规模用Prompt，大规模用微调，混合策略最优

## 📊 实验设置

- **任务**: 15类电商客服意图分类
- **数据**: 1000条标注样本（800训练 + 200测试）
- **Prompt实验**: Qwen-Plus (DashScope)
- **微调实验**: Qwen3-4B-Instruct (百炼平台 LoRA)

### 15类意图标签

物流查询、退款申请、退换货、商品咨询、价格优惠、订单修改、支付问题、账户问题、投诉建议、售后维修、发票相关、会员权益、配送服务、活动规则、闲聊其他

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整评估

```bash
# 1. 运行综合分析（生成指标、混淆矩阵、成本、延迟数据）
python scripts/comprehensive_analysis.py

# 2. 生成产品决策报告
python scripts/generate_final_report.py

# 3. 查看报告
cat results/final_decision_report.md
```

## 📁 项目结构

```
├── README.md                           # 本文件
├── LICENSE                             # MIT许可证
├── requirements.txt                    # Python依赖
├── intent_taxonomy.md                  # 15类意图定义
├── 交付清单.md                         # 交付物说明
├── data/
│   ├── train.json                      # 训练集 (800条)
│   ├── test.json                       # 测试集 (200条)
│   ├── dataset_full.json               # 完整数据集
│   ├── bailian_train.jsonl             # 百炼格式训练集
│   └── bailian_test.jsonl              # 百炼格式测试集
├── prompts/
│   ├── v1_zero_shot.md                 # 零样本Prompt
│   ├── v2_definitions.md               # 带定义Prompt
│   ├── v3_few_shot.md                  # 少样本Prompt ⭐
│   └── v4_chain_of_thought.md          # CoT Prompt
├── scripts/
│   ├── generate_dataset.py             # 数据集生成
│   ├── convert_to_bailian.py           # 转换为百炼格式
│   ├── prompt_experiment.py            # Prompt实验
│   ├── eval_finetuned.py               # 微调模型评估
│   ├── compare_results.py              # 基础对比
│   ├── comprehensive_analysis.py       # 完整评估框架 ⭐
│   └── generate_final_report.py        # 报告生成器 ⭐
└── results/
    ├── v1~v4_results.json              # Prompt实验结果
    ├── v1~v4_report.md                 # Prompt评估报告
    ├── finetuned_predictions.jsonl     # 微调模型预测
    ├── comparison_report.md            # 基础对比报告
    ├── comprehensive_analysis.json     # 完整分析数据 ⭐
    └── final_decision_report.md        # 产品决策报告 ⭐
```

## 📈 实验结果

### Prompt工程对比

| 版本 | 策略 | 准确率 | Macro-F1 |
|------|------|--------|----------|
| v1 | 零样本直接分类 | 89.5% | 0.890 |
| v2 | 意图定义+边界说明 | 88.0% | 0.872 |
| **v3** | **Few-shot示例** | **90.0%** | **0.892** ⭐ |
| v4 | Chain-of-thought | 81.5% | 0.810 |

### 微调 vs Prompt最佳

| 指标 | 微调模型 | Prompt最佳(v3) | 差异 |
|------|---------|---------------|------|
| 准确率 | 89.50% | 90.00% | -0.50% |
| Macro-F1 | 0.8900 | 0.8920 | -0.20% |

## 🔍 深度分析

完整报告包含：

1. **准确率与F1对比** - 5种方法 × 15个意图的详细指标
2. **错误分析** - 易混淆意图对识别与原因分析
3. **成本分析** - 不同规模下的成本对比（100/1K/10K/100K请求/天）
4. **延迟分析** - 平均延迟、P50、P95、P99对比
5. **产品决策建议** - 什么场景用Prompt，什么场景用微调

查看完整报告：`results/final_decision_report.md`

## 💡 产品决策建议

### 推荐使用Prompt的场景

✅ MVP快速验证
✅ 意图体系频繁变化
✅ 小规模流量（<1000次/天）
✅ 多租户SaaS
✅ 对延迟不敏感（800ms可接受）
✅ 无GPU资源

### 推荐使用微调的场景

✅ 生产环境稳定运行
✅ 大规模流量（>10000次/天）
✅ 延迟敏感（<500ms）
✅ 私有化部署
✅ 特定领域术语多
✅ 长期运营成本优化

### 混合策略（推荐）

- **主路径**: 微调模型处理常见意图（90%流量）
- **兜底路径**: Prompt处理长尾/新增意图
- **置信度阈值**: 微调置信度<0.8时回退到Prompt
- **持续学习**: 收集Prompt处理的case定期重新微调

## 🛠️ 技术栈

- **Prompt实验**: 阿里云DashScope - Qwen-Plus
- **微调实验**: 阿里云百炼平台 - Qwen3-4B-Instruct (LoRA)
- **评估框架**: scikit-learn, numpy

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🤝 贡献

欢迎提Issue和PR！

## 📮 联系

如有问题或建议，欢迎提Issue讨论。

---

⭐ 如果这个项目对你有帮助，欢迎Star支持！
