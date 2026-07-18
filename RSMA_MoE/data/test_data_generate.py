import torch
from transformers import pipeline
import pandas as pd
import networkx as nx
import random
from datasets import load_dataset
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def load_mmlu_json(file_path):
    # 使用 'r' (read) 模式開啟檔案
    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load 會把檔案內容直接變成 Python 的 list 或 dict
        data = json.load(f)
    return data

def build_task_graphs(test_data, num_tasks):
    task_graphs = []
    
    
    for i in range(num_tasks):

        # 1. 隨機決定這張圖要用多少個點 (例如 5 ~ 10 個)
        num_nodes = random.randint(5, 20)
        sampled_data_indices = [random.randint(0, len(test_data)-1) for _ in range(num_nodes)]
        G = nx.DiGraph()

        # 2. 建立節點
        for i, data_idx in enumerate(sampled_data_indices):
            node_id = i  # 這是唯一的「任務流水號」，絕對不會重複
            content = test_data[data_idx] # 這是抽到的「題目內容」
            
            G.add_node(
                node_id, 
                source_id=data_idx,  # 紀錄這題原本是 JSON 裡的第幾題
                q_id=f"Q_{data_idx:04d}_instance_{i}", # 標示這是第幾次用到這題
                **content
            )
        
        # 3. 建立連線 (讓它變成真正的 DAG)
        for u in range(num_nodes):
            # 為了保證無環，我們只讓編號小的點指向編號大的點 (u < v)
            # 隨機決定要連出 1 到 2 條邊
            num_edges = random.randint(1, 2)
            
            # 潛在的目標必須是編號比自己大的節點
            possible_targets = list(range(u + 1, num_nodes))
            
            if possible_targets:
                # 隨機抽取目標
                targets = random.sample(possible_targets, min(len(possible_targets), num_edges))
                for v in targets:
                    G.add_edge(u, v)

        # 4. (選用) 存入你的 task_graphs 字典中
        task_graphs.append(G)

    return task_graphs

def distribute_graph_fragments(subgraphs, iot_devices):
# 計算每張圖分配給多少台設備 (例如 90 / 5 = 18)
    # num_devs_per_graph = len(iot_devices) // len(subgraphs)
    for g_idx, graph in enumerate(subgraphs):
        # # 1. 決定負責這張「任務圖」的 IoT 群組
        # start_idx = g_idx * num_devs_per_graph
        # end_idx = (g_idx + 1) * num_devs_per_graph
        # target_group = iot_devices[start_idx:end_idx]
        
        # 2. 遍歷這張子圖裡的每一道題目 (Node)
        for node_id in graph.nodes():
            test_data = graph.nodes[node_id]
            q_id = test_data['q_id']
            
            # 3. 【核心邏輯】將這題的 A, B, C, D 選項隨機打散給這群 IoT
            options = test_data.get('options', [])
            for opt_idx, opt_text in enumerate(options):
                letter = chr(65 + opt_idx) # A, B, C, D
                
                # 從這 18 台設備中隨機選一台來存放這個碎片
                chosen_device = random.choice(iot_devices)
                
                # 建立碎片字典
                fragment = {
                    "q_id": q_id,
                    "question_text": test_data.get('question', 'unknown'),
                    "domain": test_data.get('category', 'unknown'),
                    "option": letter,
                    "text": opt_text, 
                }
                
                # 存入設備的背包 (假設 Device 類別有 my_fragments 列表)
                chosen_device.my_fragments.append(fragment)
                # 同步更新該設備關注的題目集合，方便後續 RSMA 計算
                chosen_device.locations.add(q_id)

# def draw_one_subgraph(subgraph):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(6, 6))
#     nx.draw(subgraph, with_labels=True, node_color='lightgreen', edge_color='gray')
#     plt.show()

'''測試

class IoT_Device:
    def __init__(self, iot_device_id, association_map_entry, my_fragments, args):
        """
        iot_device_id: 設備 ID (0, 1, 2...)
        association_map_entry: 該設備對應的物理層資訊 (字典)
        my_fragments: 初始化的碎片列表 (通常傳入空列表 [])
        args: 外部參數 (如系統設定)
        """
        # --- 基礎身分 ---
        self.id = iot_device_id
        self.args = args

        # --- 物理層屬性 (收訊品質) ---
        # 從傳入的單一設備 map 中提取 channel_gains
        # self.channel_gain = association_map_entry['channel_gains']
        # # 計算通道增益的絕對值 (訊號強度)，這在計算傳輸速率時非常重要
        # self.channel_gain_abs = np.abs(self.channel_gain)

        # --- 應用層屬性 (任務與資料) ---
        # 這裡就是存放 distribute_graph_fragments 塞進來的碎片
        self.my_fragments = my_fragments 
        
        # 用 set 存 Q_ID，方便快速計算與其他設備的「共同題目」
        self.locations = set() 
        
        # 如果初始化時就有傳入碎片，則更新 locations
        if my_fragments:
            self.locations = set(f['q_id'] for f in my_fragments)

    def __repr__(self):
        """讓 print(device) 時可以看到有意義的資訊"""
        return f"<IoT_Device ID:{self.id} | Fragments:{len(self.my_fragments)} | Tasks:{len(self.locations)}>"
all_test_datas = load_mmlu_json("balanced_mmlu.json")
num_tasks = 5

iot_devices = []
for i in range(90):
    dev = IoT_Device(
        iot_device_id=i, 
        association_map_entry = None,
        args=None,
        my_fragments=[]
    )
    iot_devices.append(dev)

# 3. 邏輯層：建構任務圖 (Task Logic)
task_graphs = build_task_graphs(all_test_datas, num_tasks)

# 4. 融合層：分發碎片 (Data Distribution)
distribute_graph_fragments(task_graphs, iot_devices)

# --- 驗證結果 ---
sample_dev = iot_devices[0]
print(f"設備 0 負責的題目數: {len(sample_dev.locations)}")
print(f"設備 0 擁有的選項碎片數: {len(sample_dev.my_fragments)}")
print(len(task_graphs))
for i, graph in enumerate(task_graphs):
    # 這裡假設你的 graph 物件有 nodes 和 edges 屬性
    # 或是你可以印出 vars(graph) 來看所有屬性
    print(f"--- Subtask Graph {i} ---")
    print(f"節點數: {len(graph.nodes)}")
    print(f"邊的數量: {len(graph.edges)}")
    print(f"內容詳情: {graph}")

for node_id, data in task_graphs[0].nodes(data=True):
    print(f"--- 節點 ID: {node_id} ---")
    print(f"原始題目編號 (source_id): {data.get('source_id')}")
    print(f"題目 ID (q_id): {data.get('q_id')}")
    print(f"📂 分類 (Category): {data.get('category', '未分類')}")
    question_text = data.get('question') or data.get('text') or "（無題目文字）"
    print(f"\n📝 題目敘述：\n   {question_text}")
    # 讀取 options 列表
    options = data.get('options', []) # 如果沒找到 options，預設給個空列表 []
    
    if options:
        print("   - 選項內容：")
        for i, opt in enumerate(options):
            # chr(65+i) 會把 0, 1, 2 轉成 A, B, C
            print(f"     ({chr(65+i)}) {opt}")
    else:
        print("   - (此題目沒有選項資料)")

for node_id, data in task_graphs[0].nodes(data=True):

    print(f"\n--- Node {node_id} Raw Data ---")

    pprint(data) # 這會把所有 key-value 全部印出來，非常適合 Debug



# 驗證代碼示例
total_options_in_graphs = sum(len(g.nodes[n].get('options', [])) for g in task_graphs for n in g.nodes())
total_fragments_in_devices = sum(len(dev.my_fragments) for dev in iot_devices)

print(f"原始總選項數: {total_options_in_graphs}")
print(f"設備總碎片數: {total_fragments_in_devices}")

if total_options_in_graphs == total_fragments_in_devices:
    print("✅ 數量完全吻合，沒有碎片遺失！")
else:
    print("❌ 數量不對，請檢查邏輯。")


dev = iot_devices[0]

print(f"=== 設備 {dev.id} 的詳細內容 ===")
print(f"所在位置 (Locations): {dev.locations}")
print(f"擁有碎片數量: {len(dev.my_fragments)}")

# 逐一印出碎片內容
for i, frag in enumerate(dev.my_fragments):
    print(f"  碎片 {i+1}: {frag}")




'''