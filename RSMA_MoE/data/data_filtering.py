'''
python3 -m venv .venv
--> -m(連接詞): 這是 module(模組)的縮寫。意思是告訴 Python :「請幫我啟動一個你內建的工具模組」。
--> venv(動詞/工具名稱）： 這是 virtual environment(虛擬環境）的縮寫，也就是那個負責「蓋無塵室」的專屬工具
--> .venv(受詞/名字）： 這是你要幫這間無塵室「取的名字」。
'''

'''
source .venv/bin/activate
--> 
'''
import pandas as pd
import networkx as nx
import random
from datasets import load_dataset

# # 載入資料
# dataset = load_dataset("TIGER-Lab/MMLU-Pro", split='test[:5]')
# iot_devices = ['IoT_A', 'IoT_B', 'IoT_C']

# # 準備兩個獨立的 List 來存放不同層級的資料
# questions_data = []
# iot_distribution_data = []

# # --- 階段一：記錄 Test Question 與 IoT 分佈  ---
# for i, data in enumerate(dataset):
#     q_id = f"Q_{i+1:03d}"  # 產生像是 Q_001, Q_002 的 ID (這就是 Link)
    
#     # 1. 記錄 Test Question (不包含選項)
#     questions_data.append({
#         "Question_ID": q_id,
#         "Question_Text": data['question'][:30] + "...",
#         "Answer": data['answer']
#     })
    
#     # 2. 記錄 Options 並分散到 IoT
#     for option_idx, option_text in enumerate(data['options']):
#         iot_distribution_data.append({
#             "Question_ID": q_id,  # 用這個 ID 串接回題目
#             "Option": chr(65 + option_idx),
#             "Stored_in_IoT": random.choice(iot_devices)
#         })

# df_questions = pd.DataFrame(questions_data)
# df_iot = pd.DataFrame(iot_distribution_data)

# # --- 階段二：將 Tests 組合成有向圖的 Task (你的最後一步) ---
# task_id = "Task_Alpha"
# G = nx.DiGraph(name=task_id)

# # 將 Q_001 到 Q_005 加入圖中，這時候 Graph 裡只有 ID，非常乾淨
# G.add_nodes_from([f"Q_{i+1:03d}" for i in range(5)])
# G.add_edges_from([("Q_001", "Q_002"), ("Q_001", "Q_003")])

# print("--- 1. 題庫表 (Test Data) ---")
# print(df_questions.head(2).to_string(index=False))

# print("\n--- 2. IoT 分佈表 (Options -> IoT) ---")
# print(df_iot.head(4).to_string(index=False))

# print(f"\n--- 3. 任務 Graph ({task_id}) ---")
# print(f"Edges: {G.edges()}")