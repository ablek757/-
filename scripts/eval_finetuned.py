#!/usr/bin/env python3
"""评估百炼微调模型"""
import json
from pathlib import Path
import dashscope
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor, as_completed

dashscope.api_key = "sk-61dbded3216b47c3b9812871d402f27a"
MODEL_ID = "qwen3-4b-instruct-2507-d33dbfcdcb21"
MAX_WORKERS = 10

TEST_FILE = Path(__file__).parent.parent / "data" / "bailian_test.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "finetuned_predictions.jsonl"

def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def predict_one(item):
    messages = item["messages"]
    true_label = messages[-1]["content"]
    user_input = messages[-2]["content"]

    try:
        response = dashscope.Generation.call(
            model=MODEL_ID,
            messages=messages[:-1],
            result_format="message",
            enable_thinking=False
        )
        if response.status_code == HTTPStatus.OK:
            pred_label = response.output.choices[0].message.content.strip()
        else:
            pred_label = f"ERROR: {response.code}"

        return {
            "input": user_input,
            "true": true_label,
            "pred": pred_label,
            "correct": pred_label == true_label
        }
    except Exception as e:
        return {"input": user_input, "true": true_label, "pred": f"ERROR: {e}", "correct": False}

def main():
    test_data = load_test_data()
    total = len(test_data)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_one, item): i for i, item in enumerate(test_data)}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if len(results) % 20 == 0:
                correct = sum(1 for r in results if r["correct"])
                print(f"Progress: {len(results)}/{total}, Accuracy: {correct/len(results):.2%}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r["correct"])
    print(f"\nFinal: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()
