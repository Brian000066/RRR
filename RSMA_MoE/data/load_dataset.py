import json
import random
from datasets import load_dataset
from collections import defaultdict

def prepare_balanced_dataset(output_file="balanced_mmlu.json", samples_per_domain=10):
    # 1. 一次載入全部測試資料 (MMLU-Pro 測試集約 12k 題)
    print("正在載入原始資料集...")
    full_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split='test')

    # 2. 按 Domain (category) 分類
    buckets = defaultdict(list)
    for data in full_dataset:
        buckets[data['category']].append(data)

    # 3. 每個領域精選 N 題
    final_data = []
    for domain, questions in buckets.items():
        # 如果該領域題目不夠，就全取；夠的話就隨機抽 samples_per_domain 題
        num_to_sample = min(len(questions), samples_per_domain)
        selected = random.sample(questions, num_to_sample)
        final_data.extend(selected)
        print(f"領域 [{domain}]: 已挑選 {num_to_sample} 題")

    # 4. 存成本地 JSON，以後實驗直接讀這個檔案
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 預處理完成！共 {len(final_data)} 題已存入 {output_file}")

# 執行一次即可
prepare_balanced_dataset(samples_per_domain=30)