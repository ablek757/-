"""
将现有数据集转换为百炼平台（DashScope）微调训练所需的 JSONL 格式。

百炼平台格式要求：
- 文件格式：.jsonl（每行一个 JSON 对象）
- 每个 JSON 对象包含一个 messages 数组
- messages 数组中包含 system / user / assistant 三种角色的消息
- system: 设定模型角色和任务说明
- user: 用户输入
- assistant: 期望的模型输出（即标签）

输出文件：
- data/bailian_train.jsonl  （训练集，800条）
- data/bailian_test.jsonl   （验证集，200条）
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 15个意图标签的完整列表
INTENT_LABELS = [
    "物流查询", "退款申请", "退换货", "商品咨询", "价格优惠",
    "订单修改", "支付问题", "账户问题", "投诉建议", "售后维修",
    "发票相关", "会员权益", "配送服务", "活动规则", "闲聊其他",
]

SYSTEM_PROMPT = (
    "你是一个电商客服意图分类器。"
    "根据用户的输入，判断其意图属于以下15个类别之一，只输出类别名称：\n"
    + "、".join(INTENT_LABELS)
)


def convert_to_bailian_format(input_json_path: str, output_jsonl_path: str) -> int:
    """将 train.json / test.json 转换为百炼 JSONL 格式。"""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["data"]
    count = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in samples:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["text"]},
                    {"role": "assistant", "content": item["intent_cn"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    splits = [
        ("train.json", "bailian_train.jsonl"),
        ("test.json", "bailian_test.jsonl"),
    ]

    for src, dst in splits:
        src_path = os.path.join(DATA_DIR, src)
        dst_path = os.path.join(DATA_DIR, dst)
        n = convert_to_bailian_format(src_path, dst_path)
        print(f"{src} -> {dst}: {n} 条")

    # 打印一条样例
    sample_path = os.path.join(DATA_DIR, "bailian_train.jsonl")
    with open(sample_path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("\n样例（第1条）:")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
