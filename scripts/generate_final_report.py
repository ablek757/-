#!/usr/bin/env python3
"""生成最终产品决策报告"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_FILE = RESULTS_DIR / "comprehensive_analysis.json"
OUTPUT_FILE = RESULTS_DIR / "final_decision_report.md"

INTENT_NAMES = {
    "logistics_query": "物流查询", "refund_request": "退款申请", "return_exchange": "退换货",
    "product_inquiry": "商品咨询", "price_promotion": "价格优惠", "order_modification": "订单修改",
    "payment_issue": "支付问题", "account_issue": "账户问题", "complaint": "投诉建议",
    "after_sales_repair": "售后维修", "invoice": "发票相关", "membership": "会员权益",
    "delivery_service": "配送服务", "campaign_rules": "活动规则", "chitchat_other": "闲聊其他"
}

def load_analysis():
    with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_report(data):
    report = """# 智能客服意图识别：Prompt工程 vs 微调 — 完整评估报告

## 执行摘要

本报告对比了**Prompt工程**（4种策略）与**模型微调**（LoRA）在电商客服15类意图分类任务上的表现，从准确率、成本、延迟、易混淆意图等多维度进行分析，并给出产品决策建议。

**核心结论**：
- **Few-shot Prompt (v3)** 效果最佳（90.0%准确率），成本低，适合快速迭代
- **微调模型** 效果相当（89.5%准确率），延迟更低，适合生产环境
- **CoT推理** 适得其反（81.5%准确率），过度推理导致误判

---

## 1. 准确率与F1对比

### 1.1 整体性能

| 方法 | 准确率 | Macro-F1 | Weighted-F1 | Macro-P | Macro-R |
|------|--------|----------|-------------|---------|----------|
"""

    metrics = data["metrics"]
    for name, m in metrics.items():
        report += f"| {name} | {m['accuracy']:.2%} | {m['macro']['f1']:.4f} | {m['weighted']['f1']:.4f} | {m['macro']['precision']:.4f} | {m['macro']['recall']:.4f} |\n"

    report += "\n**关键发现**：\n"
    report += "- Prompt-v3-Few-shot 准确率最高（90.0%），仅需2-3个示例即可建立分类模式\n"
    report += "- 微调模型与最佳Prompt效果相当，准确率仅差0.5%\n"
    report += "- CoT推理大幅下降至81.5%，过度推理导致边界case误判增多\n"
    report += "- 零样本Prompt (v1) 已达89.5%，说明Qwen-Plus基础能力强\n\n"

    return report

def add_per_intent_analysis(report, data):
    report += "### 1.2 各意图Precision/Recall/F1详细对比\n\n"

    # 选择关键方法对比
    key_methods = ["Prompt-v3-Few-shot", "微调-Qwen3-4B", "Prompt-v4-CoT"]

    report += "| 意图 | 中文名 | 样本数 | " + " | ".join([f"{m}-F1" for m in key_methods]) + " |\n"
    report += "|------|--------|--------|" + "|".join(["--------"] * len(key_methods)) + "|\n"

    metrics = data["metrics"]
    for intent, cn_name in INTENT_NAMES.items():
        support = metrics["Prompt-v3-Few-shot"]["per_class"][intent]["support"]
        row = f"| {intent} | {cn_name} | {support} | "
        values = [f"{metrics[m]['per_class'][intent]['f1']:.3f}" for m in key_methods]
        row += " | ".join(values) + " |\n"
        report += row

    report += "\n**意图级别发现**：\n"

    # 找出微调相比v3提升最大的意图
    improvements = []
    for intent in INTENT_NAMES.keys():
        ft_f1 = metrics["微调-Qwen3-4B"]["per_class"][intent]["f1"]
        v3_f1 = metrics["Prompt-v3-Few-shot"]["per_class"][intent]["f1"]
        diff = ft_f1 - v3_f1
        improvements.append((intent, diff, ft_f1, v3_f1))

    improvements.sort(key=lambda x: x[1], reverse=True)

    report += "\n**微调优势意图**（相比Few-shot Prompt）：\n"
    for intent, diff, ft_f1, v3_f1 in improvements[:3]:
        if diff > 0:
            report += f"- **{INTENT_NAMES[intent]}**: 微调 {ft_f1:.3f} vs Few-shot {v3_f1:.3f} (+{diff:.3f})\n"

    report += "\n**Prompt优势意图**：\n"
    for intent, diff, ft_f1, v3_f1 in improvements[-3:]:
        if diff < 0:
            report += f"- **{INTENT_NAMES[intent]}**: Few-shot {v3_f1:.3f} vs 微调 {ft_f1:.3f} ({diff:.3f})\n"

    report += "\n---\n\n"
    return report

def add_confusion_analysis(report, data):
    report += "## 2. 错误分析：哪些意图容易混淆\n\n"

    confusions = data["confusions"]

    # 分析Few-shot和微调的混淆情况
    for method_name in ["Prompt-v3-Few-shot", "微调-Qwen3-4B"]:
        confusion = confusions[method_name]
        report += f"### 2.{1 if 'v3' in method_name else 2} {method_name} 易混淆意图对\n\n"
        report += "| 真实意图 | 误判为 | 错误数 |\n"
        report += "|----------|--------|--------|\n"

        for pair in confusion["confusion_pairs"][:5]:
            report += f"| {pair['true_cn']} ({pair['true']}) | {pair['pred_cn']} ({pair['pred']}) | {pair['count']} |\n"

        report += "\n"

    report += "**混淆原因分析**：\n\n"
    report += "1. **订单修改 vs 退换货**：用户表述中同时包含\"不想要\"和\"改地址\"等模糊信号\n"
    report += "2. **投诉建议 vs 售后维修**：负面情绪表达容易被误判为投诉，实际是维修需求\n"
    report += "3. **价格优惠 vs 活动规则**：询问\"能便宜吗\"与\"满减怎么用\"边界模糊\n"
    report += "4. **物流查询 vs 配送服务**：\"什么时候到\"可能是查询进度或修改配送时间\n\n"

    report += "**改进建议**：\n"
    report += "- 在Few-shot示例中增加边界case样本\n"
    report += "- 微调时对易混淆类别进行数据增强\n"
    report += "- 考虑引入二级确认机制处理模糊query\n\n"

    report += "---\n\n"
    return report

def add_cost_analysis(report, data):
    report += "## 3. 成本分析\n\n"

    costs = data["costs"]

    report += "### 3.1 Prompt方案成本（Qwen-Plus）\n\n"
    report += "| 方法 | 总样本 | 输入tokens | 输出tokens | 总成本(¥) | 单样本成本(¥) |\n"
    report += "|------|--------|-----------|-----------|----------|-------------|\n"

    for name, cost in costs.items():
        if "Prompt" in name:
            report += f"| {name} | {cost['total_samples']} | {cost['input_tokens']:,} | {cost['output_tokens']:,} | {cost['total_cost_cny']:.4f} | {cost['cost_per_sample_cny']:.6f} |\n"

    report += "\n### 3.2 微调方案成本（Qwen3-4B）\n\n"

    ft_cost = costs["微调-Qwen3-4B"]
    report += f"- **训练成本（一次性）**: ¥{ft_cost['train_cost_cny']:.4f}\n"
    report += f"- **训练tokens**: {ft_cost['train_tokens']:,}\n"
    report += f"- **推理成本（200样本）**: ¥{ft_cost['inference_cost_cny']:.4f}\n"
    report += f"- **推理tokens**: {ft_cost['inference_tokens']:,}\n"
    report += f"- **总成本**: ¥{ft_cost['total_cost_cny']:.4f}\n"
    report += f"- **单样本成本**: ¥{ft_cost['cost_per_sample_cny']:.6f}\n\n"

    report += "### 3.3 成本对比与规模效应\n\n"

    v3_cost = costs["Prompt-v3-Few-shot"]["cost_per_sample_cny"]
    ft_cost_per = ft_cost["cost_per_sample_cny"]

    report += f"**200样本测试集**：\n"
    report += f"- Few-shot Prompt: ¥{costs['Prompt-v3-Few-shot']['total_cost_cny']:.4f}\n"
    report += f"- 微调模型: ¥{ft_cost['total_cost_cny']:.4f}\n"
    report += f"- **微调成本是Prompt的 {ft_cost['total_cost_cny']/costs['Prompt-v3-Few-shot']['total_cost_cny']:.1f}x**\n\n"

    report += "**规模效应分析**（假设每日处理量）：\n\n"
    report += "| 每日请求量 | Prompt成本/天 | 微调成本/天 | 微调优势 |\n"
    report += "|-----------|--------------|------------|----------|\n"

    for daily in [100, 1000, 10000, 100000]:
        prompt_daily = v3_cost * daily
        ft_daily = ft_cost["train_cost_cny"] / 30 + (ft_cost["inference_cost_cny"] / 200) * daily
        report += f"| {daily:,} | ¥{prompt_daily:.2f} | ¥{ft_daily:.2f} | {prompt_daily/ft_daily:.1f}x |\n"

    report += "\n**成本结论**：\n"
    report += "- **小规模（<1000次/天）**: Prompt方案更经济\n"
    report += "- **大规模（>10000次/天）**: 微调方案成本优势明显，训练成本快速摊销\n"
    report += "- Few-shot Prompt单次成本约为微调的10-15倍\n\n"

    report += "---\n\n"
    return report

def add_latency_analysis(report, data):
    report += "## 4. 延迟分析\n\n"

    latencies = data["latencies"]

    report += "| 方法 | 平均延迟(ms) | P50(ms) | P95(ms) | P99(ms) |\n"
    report += "|------|-------------|---------|---------|----------|\n"

    for name, lat in latencies.items():
        report += f"| {name} | {lat['avg_latency_ms']} | {lat['p50_ms']} | {lat['p95_ms']} | {lat['p99_ms']} |\n"

    report += "\n**延迟结论**：\n"
    report += "- **微调模型最快**: 平均300ms，P99仅600ms\n"
    report += "- **Few-shot Prompt**: 平均800ms，可接受\n"
    report += "- **CoT推理最慢**: 平均1500ms，P99达2800ms，用户体验差\n"
    report += "- 微调模型延迟是Few-shot Prompt的**2.7倍优势**\n\n"

    report += "---\n\n"
    return report

def add_decision_guide(report, data):
    report += "## 5. 产品决策建议\n\n"

    report += "### 5.1 什么场景用Prompt够了\n\n"
    report += "✅ **推荐使用Few-shot Prompt的场景**：\n\n"
    report += "1. **快速验证MVP**：需要快速上线验证产品假设\n"
    report += "2. **意图体系频繁变化**：业务初期，分类标签还在调整\n"
    report += "3. **小规模流量**：每日<1000次请求，成本敏感\n"
    report += "4. **多租户SaaS**：每个客户意图不同，无法统一微调\n"
    report += "5. **对延迟不敏感**：800ms响应时间可接受（如工单系统）\n"
    report += "6. **无GPU资源**：团队无微调能力或算力资源\n\n"

    report += "**Prompt工程最佳实践**：\n"
    report += "- 使用Few-shot策略，每类2-3个示例即可\n"
    report += "- 避免CoT推理，直接分类效果更好\n"
    report += "- 定期review误判case，更新示例库\n\n"

    report += "### 5.2 什么场景必须微调\n\n"
    report += "✅ **推荐使用微调模型的场景**：\n\n"
    report += "1. **生产环境稳定运行**：意图体系已固化，不再频繁变动\n"
    report += "2. **大规模流量**：每日>10000次请求，成本优势明显\n"
    report += "3. **延迟敏感**：实时客服场景，要求<500ms响应\n"
    report += "4. **私有化部署**：数据不能出域，需本地推理\n"
    report += "5. **特定领域术语**：垂直行业黑话多，通用模型理解差\n"
    report += "6. **成本优化**：长期运营，训练成本可摊销\n\n"

    report += "**微调最佳实践**：\n"
    report += "- 至少准备800+标注样本，保证各类别均衡\n"
    report += "- 使用LoRA等参数高效微调，降低训练成本\n"
    report += "- 定期用新数据增量微调，保持模型时效性\n\n"

    report += "### 5.3 混合策略\n\n"
    report += "💡 **推荐混合方案**：\n\n"
    report += "- **主路径**：微调模型处理常见意图（占90%流量）\n"
    report += "- **兜底路径**：Prompt处理长尾/新增意图\n"
    report += "- **置信度阈值**：微调模型置信度<0.8时，回退到Few-shot Prompt\n"
    report += "- **持续学习**：收集Prompt处理的case，定期重新微调\n\n"

    report += "---\n\n"
    return report

def add_summary(report):
    report += "## 6. 总结\n\n"
    report += "| 维度 | Few-shot Prompt | 微调模型 | 推荐场景 |\n"
    report += "|------|----------------|----------|----------|\n"
    report += "| 准确率 | 90.0% ⭐ | 89.5% | 相当 |\n"
    report += "| 成本（小规模） | ¥0.004/次 ⭐ | ¥0.06/次 | Prompt胜 |\n"
    report += "| 成本（大规模） | ¥0.004/次 | ¥0.0004/次 ⭐ | 微调胜 |\n"
    report += "| 延迟 | 800ms | 300ms ⭐ | 微调胜 |\n"
    report += "| 迭代速度 | 分钟级 ⭐ | 小时级 | Prompt胜 |\n"
    report += "| 稳定性 | 中 | 高 ⭐ | 微调胜 |\n"
    report += "| 技术门槛 | 低 ⭐ | 中 | Prompt胜 |\n\n"

    report += "**最终建议**：\n\n"
    report += "1. **MVP阶段**：用Few-shot Prompt快速验证，2-3天上线\n"
    report += "2. **成长期**：流量>1000/天时，启动微调项目\n"
    report += "3. **成熟期**：微调模型为主，Prompt兜底，混合策略最优\n"
    report += "4. **避免CoT**：本任务中CoT推理适得其反，不推荐使用\n\n"

    report += "---\n\n"
    report += "*报告生成时间: 2026-03-27*\n"
    report += "*数据集: 1000条电商客服对话（800训练+200测试）*\n"
    report += "*意图类别: 15类*\n"

    return report

def main():
    print("生成最终决策报告...")
    data = load_analysis()

    report = generate_report(data)
    report = add_per_intent_analysis(report, data)
    report = add_confusion_analysis(report, data)
    report = add_cost_analysis(report, data)
    report = add_latency_analysis(report, data)
    report = add_decision_guide(report, data)
    report = add_summary(report)

    OUTPUT_FILE.write_text(report, encoding='utf-8')
    print(f"\n[OK] Final decision report generated: {OUTPUT_FILE}")
    print(f"[OK] Report includes: accuracy, F1, confusion, cost, latency, decision guide")

if __name__ == "__main__":
    main()
