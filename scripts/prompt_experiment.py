"""
智能客服意图识别 — Prompt工程迭代实验 (并发版)
v1: 零样本直接分类
v2: 加意图定义和边界说明
v3: Few-shot (每个意图2-3个示例)
v4: Chain-of-thought (先分析再分类)

使用 DashScope OpenAI-compatible API + asyncio并发
"""

import asyncio
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

# ─── 配置 ───
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-61dbded3216b47c3b9812871d402f27a"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-plus"
CONCURRENCY = 20  # 同时发20个请求

INTENT_LABELS = [
    "logistics_query", "refund_request", "return_exchange",
    "product_inquiry", "price_promotion", "order_modification",
    "payment_issue", "account_issue", "complaint",
    "after_sales_repair", "invoice", "membership",
    "delivery_service", "campaign_rules", "chitchat_other",
]

INTENT_CN = {
    "logistics_query": "物流查询", "refund_request": "退款申请",
    "return_exchange": "退换货", "product_inquiry": "商品咨询",
    "price_promotion": "价格优惠", "order_modification": "订单修改",
    "payment_issue": "支付问题", "account_issue": "账户问题",
    "complaint": "投诉建议", "after_sales_repair": "售后维修",
    "invoice": "发票相关", "membership": "会员权益",
    "delivery_service": "配送服务", "campaign_rules": "活动规则",
    "chitchat_other": "闲聊其他",
}

INTENT_DEFS = {
    "logistics_query": "用户询问订单的物流状态、配送进度、快递公司等",
    "refund_request": "用户要求退还已支付的款项",
    "return_exchange": "用户要求退货或换货（不含纯退款）",
    "product_inquiry": "用户对商品的规格、功能、材质等进行询问",
    "price_promotion": "用户咨询价格、优惠券、满减、打折等",
    "order_modification": "用户要求修改已下单的信息（地址、数量、规格等）",
    "payment_issue": "用户遇到支付失败、扣款异常、支付方式等问题",
    "account_issue": "用户遇到登录、注册、密码、账户安全等问题",
    "complaint": "用户对服务、商品质量、商家态度等表达不满或投诉",
    "after_sales_repair": "用户咨询保修、维修、质量问题处理",
    "invoice": "用户咨询开发票、发票类型、发票信息修改",
    "membership": "用户咨询会员等级、积分、会员专属权益",
    "delivery_service": "用户咨询配送方式、配送范围、预约送达等（非物流跟踪）",
    "campaign_rules": "用户咨询促销活动的规则、参与方式、活动时间",
    "chitchat_other": "打招呼、感谢、无明确意图、无法归入以上类别",
}

FEW_SHOT_EXAMPLES = {
    "logistics_query": ["我的快递到哪了", "物流怎么这么慢啊", "什么时候到货呢"],
    "refund_request": ["我想申请退款", "退款能快一点吗", "钱什么时候退到账"],
    "return_exchange": ["这件衣服尺码不合适，能换一件吗", "退货原因选什么好", "我选错尺码了帮我换"],
    "product_inquiry": ["这个手机支持5G吗", "电池容量多大", "有没有黑色的"],
    "price_promotion": ["有没有什么优惠券可以用", "这已经是最低价了吗", "凑单满减怎么算"],
    "order_modification": ["我想改一下收货地址", "要减少一件数量", "发货前帮我修改一下可以吗"],
    "payment_issue": ["付款的时候一直失败怎么办", "微信支付扫码没反应", "银行卡限额了付不了"],
    "account_issue": ["我忘记密码了怎么办", "有人在异地登录我的账号"],
    "complaint": ["你们服务态度太差了，我要投诉", "我非常不满意你们的服务"],
    "after_sales_repair": ["买了三个月屏幕就坏了，怎么保修", "产品坏了怎么申请维修"],
    "invoice": ["能开增值税专用发票吗", "开发票要加税点吗"],
    "membership": ["我是金牌会员有什么优惠", "开通会员送什么"],
    "delivery_service": ["你们支持送货上门安装吗", "乡镇地区送不送"],
    "campaign_rules": ["双十一的满减规则是怎样的", "618活动力度大吗"],
    "chitchat_other": ["你好", "谢谢", "拜拜"],
}


# ─── 数据加载 ───
def load_test_set():
    samples = []
    with open(DATA_DIR / "test.csv", "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            samples.append({"id": int(row["id"]), "text": row["text"], "intent": row["intent"]})
    return samples


# ─── 4 版 Prompt 构造 ───
def prompt_v1(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "你是一个电商客服意图分类器。"},
        {"role": "user", "content": (
            f"请将以下用户消息分类到一个意图标签中。\n"
            f"可选标签: {', '.join(INTENT_LABELS)}\n\n"
            f"用户消息: {text}\n\n请只输出标签名称，不要输出任何其他内容。"
        )},
    ]


def prompt_v2(text: str) -> list[dict]:
    defs = "\n".join(f"- {l}: {INTENT_DEFS[l]}" for l in INTENT_LABELS)
    return [
        {"role": "system", "content": "你是一个电商客服意图分类器。根据用户消息的字面意思判断意图，取最主要的意图。每条消息只属于一个意图类别。"},
        {"role": "user", "content": (
            f"请将以下用户消息分类到一个意图标签中。\n\n"
            f"## 意图定义\n{defs}\n\n"
            f"## 边界说明\n"
            f"- 「退换货」指退货或换货行为，「退款申请」指纯退钱\n"
            f"- 「物流查询」指追踪已有订单的物流，「配送服务」指咨询配送方式/范围\n"
            f"- 「价格优惠」指一般性价格/优惠咨询，「活动规则」指特定促销活动的规则\n"
            f"- 「售后维修」指保修/维修问题，「投诉建议」指表达不满/投诉\n"
            f"- 无法归入以上类别的归为「chitchat_other」\n\n"
            f"用户消息: {text}\n\n请只输出标签名称（英文标识），不要输出任何其他内容。"
        )},
    ]


def prompt_v3(text: str) -> list[dict]:
    examples = "\n\n".join(
        f"用户消息: {t}\n意图: {label}"
        for label, texts in FEW_SHOT_EXAMPLES.items() for t in texts
    )
    return [
        {"role": "system", "content": "你是一个电商客服意图分类器。根据以下示例的模式，将用户消息分类到一个意图标签中。"},
        {"role": "user", "content": (
            f"以下是各意图的分类示例：\n\n{examples}\n\n---\n\n"
            f"可选标签: {', '.join(INTENT_LABELS)}\n\n"
            f"现在请分类这条消息：\n用户消息: {text}\n\n请只输出标签名称，不要输出任何其他内容。"
        )},
    ]


def prompt_v4(text: str) -> list[dict]:
    defs = "\n".join(f"- {l}: {INTENT_DEFS[l]}" for l in INTENT_LABELS)
    return [
        {"role": "system", "content": "你是一个电商客服意图分类器。你需要逐步分析用户消息，然后给出分类结果。"},
        {"role": "user", "content": (
            f"请对以下用户消息进行意图分类。\n\n"
            f"## 意图定义\n{defs}\n\n"
            f"## 边界说明\n"
            f"- 「退换货」指退货或换货行为，「退款申请」指纯退钱\n"
            f"- 「物流查询」指追踪已有订单的物流，「配送服务」指咨询配送方式/范围\n"
            f"- 「价格优惠」指一般性价格/优惠咨询，「活动规则」指特定促销活动的规则\n"
            f"- 「售后维修」指保修/维修问题，「投诉建议」指表达不满/投诉\n\n"
            f"用户消息: {text}\n\n"
            f"请按以下格式回答：\n分析：<先分析用户消息的关键词和意图指向>\n结论：<输出一个英文标签名称>"
        )},
    ]


# ─── 标签提取 ───
def extract_label(raw: str, version: str) -> str:
    text = raw.strip()
    if version == "v4":
        m = re.search(r"结论[：:]\s*(.+)", text)
        if m:
            text = m.group(1).strip()
    text = text.strip("\"'`。.，, \n")
    if text in INTENT_LABELS:
        return text
    for label in INTENT_LABELS:
        if label in text:
            return label
    cn_to_en = {v: k for k, v in INTENT_CN.items()}
    for cn, en in cn_to_en.items():
        if cn in text:
            return en
    return f"UNKNOWN:{text[:50]}"


# ─── 并发 API 调用 ───
async def classify_one(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                       messages: list[dict], idx: int) -> tuple[int, str]:
    """单个分类请求，带信号量控制并发"""
    async with sem:
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 200,
        }
        for attempt in range(3):
            try:
                resp = await client.post(API_URL, json=payload)
                if resp.status_code == 429:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return idx, data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    print(f"  API失败 idx={idx}: {e}")
                    return idx, "API_ERROR"
    return idx, "API_ERROR"


async def run_version_async(version: str, prompt_fn, test_data: list[dict]) -> list[dict]:
    """并发运行单个版本的全部测试"""
    sem = asyncio.Semaphore(CONCURRENCY)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        tasks = []
        for i, sample in enumerate(test_data):
            msgs = prompt_fn(sample["text"])
            tasks.append(classify_one(client, sem, msgs, i))

        raw_results = await asyncio.gather(*tasks)

    # 按原始顺序组装结果
    idx_to_raw = {idx: raw for idx, raw in raw_results}
    results = []
    for i, sample in enumerate(test_data):
        raw = idx_to_raw[i]
        pred = extract_label(raw, version)
        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "true": sample["intent"],
            "pred": pred,
            "raw_output": raw,
            "correct": sample["intent"] == pred,
        })
    return results


# ─── 评估指标 ───
def compute_metrics(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    metrics = {}
    for label in INTENT_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": sum(1 for t in y_true if t == label)}

    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics.values()) / len(y_true)

    confusion = {t: Counter() for t in INTENT_LABELS}
    for t, p in zip(y_true, y_pred):
        confusion[t][p if p in INTENT_LABELS else "OTHER"] += 1

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "per_class": metrics, "confusion": confusion}


# ─── 报告生成 ───
def generate_report(version, results, metrics):
    L = []
    L.append(f"# Prompt {version.upper()} 评估报告\n")
    L.append(f"- **模型**: {MODEL}")
    L.append(f"- **测试集大小**: {len(results)} 条")
    L.append(f"- **准确率 (Accuracy)**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    L.append(f"- **Macro-F1**: {metrics['macro_f1']:.4f}")
    L.append(f"- **Weighted-F1**: {metrics['weighted_f1']:.4f}\n")

    L.append("## 各意图分类指标\n")
    L.append("| 意图 | 中文名 | Precision | Recall | F1 | Support |")
    L.append("|------|--------|-----------|--------|-----|---------|")
    for label in INTENT_LABELS:
        m = metrics["per_class"][label]
        L.append(f"| {label} | {INTENT_CN[label]} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")

    L.append("\n## 混淆矩阵\n")
    pred_labels = INTENT_LABELS[:]
    has_other = any("OTHER" in metrics["confusion"][t] for t in INTENT_LABELS)
    if has_other:
        pred_labels.append("OTHER")
    short = {**{l: INTENT_CN[l] for l in INTENT_LABELS}, "OTHER": "其他"}
    L.append("| 真实\\预测 | " + " | ".join(short[l] for l in pred_labels) + " |")
    L.append("|" + "|".join(["---"] * (len(pred_labels) + 1)) + "|")
    for tl in INTENT_LABELS:
        row = f"| {short[tl]} |" + "".join(f" {metrics['confusion'][tl].get(pl, 0)} |" for pl in pred_labels)
        L.append(row)

    errors = [r for r in results if r["true"] != r["pred"]]
    L.append(f"\n## 错误案例分析 ({len(errors)} 条错误)\n")
    if errors:
        error_types = Counter((e["true"], e["pred"]) for e in errors)
        L.append("### 错误类型分布\n")
        L.append("| 真实意图 | 预测意图 | 次数 |")
        L.append("|---------|---------|------|")
        for (t, p), c in error_types.most_common():
            L.append(f"| {INTENT_CN.get(t,t)} | {INTENT_CN.get(p,p)} | {c} |")
        L.append("\n### 具体错误案例\n")
        for e in errors[:30]:
            L.append(f"- **文本**: {e['text']}")
            L.append(f"  - 真实: {INTENT_CN.get(e['true'],e['true'])} ({e['true']})")
            L.append(f"  - 预测: {INTENT_CN.get(e['pred'],e['pred'])} ({e['pred']})")
            if e.get("raw_output") and version == "v4":
                L.append(f"  - 模型输出: {e['raw_output'][:200]}")
            L.append("")
    return "\n".join(L)


def generate_summary(all_metrics, all_results):
    L = []
    L.append("# Prompt 迭代实验总结报告\n")
    L.append(f"- **模型**: {MODEL}")
    L.append(f"- **测试集**: 200 条\n")

    L.append("## 各版本总体指标对比\n")
    L.append("| 版本 | 策略 | Accuracy | Macro-F1 | Weighted-F1 |")
    L.append("|------|------|----------|----------|-------------|")
    vn = {"v1": "零样本直接分类", "v2": "意图定义+边界说明", "v3": "Few-shot 示例", "v4": "Chain-of-thought"}
    for v in ["v1", "v2", "v3", "v4"]:
        m = all_metrics[v]
        L.append(f"| {v} | {vn[v]} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")

    L.append("\n## 各意图 F1 跨版本对比\n")
    L.append("| 意图 | 中文名 | v1 | v2 | v3 | v4 |")
    L.append("|------|--------|-----|-----|-----|-----|")
    for label in INTENT_LABELS:
        f1s = " | ".join(f"{all_metrics[v]['per_class'][label]['f1']:.3f}" for v in ["v1","v2","v3","v4"])
        L.append(f"| {label} | {INTENT_CN[label]} | {f1s} |")

    L.append("\n## 从 v1 到 v4 提升最大的意图\n")
    imps = sorted(
        [(l, all_metrics["v1"]["per_class"][l]["f1"], all_metrics["v4"]["per_class"][l]["f1"]) for l in INTENT_LABELS],
        key=lambda x: -(x[2]-x[1])
    )
    L.append("| 意图 | v1 F1 | v4 F1 | 提升 |")
    L.append("|------|-------|-------|------|")
    for l, f1, f4 in imps[:5]:
        L.append(f"| {INTENT_CN[l]} | {f1:.3f} | {f4:.3f} | {f4-f1:+.3f} |")

    L.append("\n## Prompt 能力瓶颈分析\n")
    weak = sorted([(l, all_metrics["v4"]["per_class"][l]["f1"]) for l in INTENT_LABELS if all_metrics["v4"]["per_class"][l]["f1"] < 0.85], key=lambda x: x[1])
    if weak:
        L.append("### 即使 v4 (CoT) 仍表现较弱的意图\n")
        for label, f1 in weak:
            L.append(f"- **{INTENT_CN[label]}** ({label}): F1 = {f1:.3f}")
            errs = [r for r in all_results["v4"] if r["true"] == label and r["pred"] != label]
            if errs:
                tops = Counter(r["pred"] for r in errs).most_common(3)
                L.append(f"  - 常被误判为: {', '.join(f'{INTENT_CN.get(p,p)}({c}次)' for p,c in tops)}")
                L.append(f"  - 示例: 「{errs[0]['text']}」")

    L.append("\n### 跨版本持续出错的样本\n")
    ids = sorted(set(r["id"] for r in all_results["v1"]))
    persistent = []
    for tid in ids:
        wrong = sum(1 for v in ["v1","v2","v3","v4"] for r in all_results[v] if r["id"]==tid and not r["correct"])
        if wrong >= 3:
            s = next(r for r in all_results["v4"] if r["id"]==tid)
            preds = {v: next(r["pred"] for r in all_results[v] if r["id"]==tid) for v in ["v1","v2","v3","v4"]}
            persistent.append({"id": tid, "text": s["text"], "true": s["true"], "wrong": wrong, "preds": preds})
    if persistent:
        L.append(f"共 {len(persistent)} 条样本在3个或以上版本中被错误分类：\n")
        for pe in persistent[:15]:
            L.append(f"- **ID {pe['id']}**: 「{pe['text']}」")
            L.append(f"  - 真实: {INTENT_CN.get(pe['true'],pe['true'])}")
            L.append(f"  - 预测: {' → '.join(f'{v}:{INTENT_CN.get(p,p)}' for v,p in pe['preds'].items())}")
            L.append(f"  - 错误版本数: {pe['wrong']}/4\n")
    else:
        L.append("无跨版本持续出错的样本。\n")

    L.append("## 结论与建议\n")
    best_v = max(all_metrics, key=lambda v: all_metrics[v]["accuracy"])
    L.append(f"1. **最佳版本**: {best_v} (Accuracy={all_metrics[best_v]['accuracy']:.1%})")
    L.append(f"2. **从 v1 到 v4 准确率提升**: {all_metrics['v1']['accuracy']:.1%} → {all_metrics['v4']['accuracy']:.1%} ({all_metrics['v4']['accuracy']-all_metrics['v1']['accuracy']:+.1%})")
    if weak:
        L.append(f"3. **Prompt 瓶颈意图**: {'、'.join(INTENT_CN[l] for l,_ in weak[:3])}，建议通过微调解决")
    else:
        L.append("3. **Prompt 表现良好**: 所有意图 F1 >= 0.85")
    return "\n".join(L)


# ─── 主函数 ───
async def main():
    test_data = load_test_set()
    print(f"已加载测试集: {len(test_data)} 条")

    versions = [
        ("v1", "零样本直接分类", prompt_v1),
        ("v2", "意图定义+边界说明", prompt_v2),
        ("v3", "Few-shot 示例", prompt_v3),
        ("v4", "Chain-of-thought", prompt_v4),
    ]

    all_results = {}
    all_metrics = {}

    # 4个版本串行（每版内部200条并发）
    for ver, desc, fn in versions:
        print(f"\n>>> {ver}: {desc} — 并发{CONCURRENCY}发送中...")
        t0 = time.time()
        results = await run_version_async(ver, fn, test_data)
        elapsed = time.time() - t0

        metrics = compute_metrics([r["true"] for r in results], [r["pred"] for r in results])
        all_results[ver] = results
        all_metrics[ver] = metrics

        acc = metrics["accuracy"]
        print(f"    完成 ({elapsed:.1f}s) | Accuracy={acc:.1%} | Macro-F1={metrics['macro_f1']:.3f}")

        with open(RESULTS_DIR / f"{ver}_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        report = generate_report(ver, results, metrics)
        with open(RESULTS_DIR / f"{ver}_report.md", "w", encoding="utf-8") as f:
            f.write(report)

    summary = generate_summary(all_metrics, all_results)
    with open(RESULTS_DIR / "summary_report.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{'='*50}")
    print("全部完成！结果在 results/ 目录下")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
