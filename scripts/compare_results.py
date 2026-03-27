#!/usr/bin/env python3
"""对比微调模型 vs Prompt基线的效果"""
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

FINETUNED_FILE = Path(__file__).parent.parent / "results" / "finetuned_predictions.jsonl"
PROMPT_FILE = Path(__file__).parent.parent / "results" / "v4_results.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "comparison_report.md"

INTENT_NAMES = {
    "logistics_query": "物流查询", "refund_request": "退款申请", "return_exchange": "退换货",
    "product_inquiry": "商品咨询", "price_promotion": "价格优惠", "order_modification": "订单修改",
    "payment_issue": "支付问题", "account_issue": "账户问题", "complaint": "投诉建议",
    "after_sales_repair": "售后维修", "invoice": "发票相关", "membership": "会员权益",
    "delivery_service": "配送服务", "campaign_rules": "活动规则", "chitchat_other": "闲聊其他"
}

CN_TO_EN = {v: k for k, v in INTENT_NAMES.items()}

def load_finetuned():
    with open(FINETUNED_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # 转换中文标签为英文
    for item in data:
        item["true"] = CN_TO_EN.get(item["true"], item["true"])
        item["pred"] = CN_TO_EN.get(item["pred"], item["pred"])
    return data

def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"input": item["text"], "true": item["true"], "pred": item["pred"]}
            for item in data]

def compute_metrics(data):
    y_true = [item["true"] for item in data]
    y_pred = [item["pred"] for item in data]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(INTENT_NAMES.keys()), zero_division=0)
    macro_f1 = np.mean(f1)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {intent: {"precision": p[i], "recall": r[i], "f1": f1[i], "support": support[i]}
                      for i, intent in enumerate(INTENT_NAMES.keys())}
    }

def main():
    ft_data = load_finetuned()
    prompt_data = load_prompt()

    ft_metrics = compute_metrics(ft_data)
    prompt_metrics = compute_metrics(prompt_data)

    report = f"""# 微调 vs Prompt 对比报告

## 整体指标对比

| 指标 | 微调模型 | Prompt基线 (V4) | 提升 |
|------|---------|----------------|------|
| 准确率 (Accuracy) | {ft_metrics['accuracy']:.4f} ({ft_metrics['accuracy']*100:.2f}%) | {prompt_metrics['accuracy']:.4f} ({prompt_metrics['accuracy']*100:.2f}%) | {(ft_metrics['accuracy']-prompt_metrics['accuracy'])*100:+.2f}% |
| Macro-F1 | {ft_metrics['macro_f1']:.4f} | {prompt_metrics['macro_f1']:.4f} | {(ft_metrics['macro_f1']-prompt_metrics['macro_f1'])*100:+.2f}% |

## 各意图分类对比

| 意图 | 中文名 | 微调F1 | Prompt F1 | 提升 |
|------|--------|--------|-----------|------|
"""

    for intent, cn_name in INTENT_NAMES.items():
        ft_f1 = ft_metrics['per_class'][intent]['f1']
        prompt_f1 = prompt_metrics['per_class'][intent]['f1']
        diff = ft_f1 - prompt_f1
        report += f"| {intent} | {cn_name} | {ft_f1:.3f} | {prompt_f1:.3f} | {diff:+.3f} |\n"

    report += f"\n## 关键发现\n\n"

    # 找出提升最大和下降最大的类别
    improvements = [(intent, ft_metrics['per_class'][intent]['f1'] - prompt_metrics['per_class'][intent]['f1'])
                    for intent in INTENT_NAMES.keys()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    report += f"### 提升最大的3个类别\n\n"
    for intent, diff in improvements[:3]:
        report += f"- **{INTENT_NAMES[intent]}** ({intent}): +{diff*100:.2f}%\n"

    report += f"\n### 下降最大的3个类别\n\n"
    for intent, diff in improvements[-3:]:
        report += f"- **{INTENT_NAMES[intent]}** ({intent}): {diff*100:.2f}%\n"

    report += f"\n## 结论\n\n"
    if ft_metrics['accuracy'] > prompt_metrics['accuracy']:
        report += f"微调模型在准确率上**优于** Prompt基线 {(ft_metrics['accuracy']-prompt_metrics['accuracy'])*100:.2f}%，"
    else:
        report += f"微调模型在准确率上**低于** Prompt基线 {(prompt_metrics['accuracy']-ft_metrics['accuracy'])*100:.2f}%，"

    if ft_metrics['macro_f1'] > prompt_metrics['macro_f1']:
        report += f"Macro-F1提升 {(ft_metrics['macro_f1']-prompt_metrics['macro_f1'])*100:.2f}%。\n"
    else:
        report += f"Macro-F1下降 {(prompt_metrics['macro_f1']-ft_metrics['macro_f1'])*100:.2f}%。\n"

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"对比报告已生成: {OUTPUT_FILE}")
    print(f"\n微调模型准确率: {ft_metrics['accuracy']:.2%}")
    print(f"Prompt基线准确率: {prompt_metrics['accuracy']:.2%}")
    print(f"提升: {(ft_metrics['accuracy']-prompt_metrics['accuracy'])*100:+.2f}%")

if __name__ == "__main__":
    main()
