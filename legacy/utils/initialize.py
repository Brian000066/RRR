# utils 資料夾中的 edge_server.py 檔案，匯入 EdgeServer 類別(class)
from utils.edge_server import EdgeServer 

#從 utils.multi_cell 匯入多個跟建立網路環境相關的函式
from utils.multi_cell import init_multi_cell_system, get_device_edge_association_map, compute_channel_gain, compute_spatial_correlation_table
import random
import networkx as nx
from utils.IoT_device import IoT_Device

#torch (PyTorch，通常用於深度學習或機器學習) 與 numpy (用於高階數學與陣列運算)
import torch
import numpy as np

import json


#args：一個包含各種系統設定和超參數的物件（例如細胞數量、學習率等）
#association_map：一個記錄著「哪個設備連接到哪個伺服器」的對應表（通常是一個包含字典的列表）。
#intialize_edge_servers--> 接手
def initialize_system(num_devices, num_edge_servers, cell_radius, num_antennas):
    device_position, edge_server_position, distances = init_multi_cell_system(
                                                    num_devices = num_devices,
                                                    num_edge_servers = num_edge_servers,
                                                    area_size = 1000,
                                                    cell_radius = cell_radius)


    association_map = get_device_edge_association_map(device_positions = device_position,
                                                distances = distances,
                                                cell_radius = cell_radius)
    

    association_map = compute_channel_gain(association_map = association_map,
                                            num_antennas = num_antennas)
    
    correlation_tables = compute_spatial_correlation_table(association_map = association_map,
                                                            num_devices = num_devices,
                                                            num_edge_servers = num_edge_servers)


    return association_map, correlation_tables # [{'device_id': 0, 'connected_edge_server_ids': [0], 'distances': [100.0], 'channel_gains': [0.0001]},
                            # {'device_id': 1, 'connected_edge_server_ids': [0], 'distances': [100.0], 'channel_gains': [0.0001]},
                            # ...]
   
def initialize_edge_servers(num_edge_servers, args, association_map):

    # Create a dictionary to store devices under each edge server's coverage
    # 為每一台 Server 準備一個點名簿
    edge_server_coverage = {}
    for i in range(args.num_cells):

        #在 edge_server_coverage 字典中，為第 i 號伺服器建立一個空的清單 []，準備用來裝入設備 ID
        edge_server_coverage[i] = []
        for j in range(len(association_map)):

            # 如果第 j 個設備的「連線清單」裡有我這台 Server 的 ID
            if i in association_map[j]['connected_edge_server_ids']:
                edge_server_coverage[i].append(association_map[j]['device_id'])

    return [EdgeServer(server_id = i, args = args, coverage = edge_server_coverage[i]) for i in range(num_edge_servers)], edge_server_coverage

def intialize_fill_servers_with_experts(edge_servers, expert_dict):

    all_expert_ids = list(expert_dict.keys())
    random.shuffle(all_expert_ids) # 增加隨機性
    
    num_servers = len(edge_servers)
    
    # 第一階段：保底分配 (確保每個領域的專家都有地方待)
    for i, exp_id in enumerate(all_expert_ids):
        target_server = edge_servers[i % num_servers]
        exp_info = expert_dict[exp_id]
        
        if target_server.add_expert_label(exp_id, exp_info['mem']):
            target_server.existing_experts[exp_id].update({
                'domain': exp_info['domain'],
                'accuracy': exp_info['accuracy']
            })

    # 第二階段：填滿剩餘空間 (貪婪做法)
    for server in edge_servers:
        attempts = 0
        while attempts < 30:
            rand_exp_id = random.choice(all_expert_ids)
            if rand_exp_id in server.existing_experts:
                attempts += 1
                continue
                
            exp_info = expert_dict[rand_exp_id]
            if server.add_expert_label(rand_exp_id, exp_info['mem']):
                server.existing_experts[rand_exp_id].update({
                    'domain': exp_info['domain'],
                    'accuracy': exp_info['accuracy']
                })
                attempts = 0 # 成功就重置
            else:
                attempts += 1

def intialize_task_graphs(test_data, num_tasks):
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

def initialize_devices(num_devices, association_map, args):
    iot_devices = []

    for i in range(num_devices):
        # 1. 準備碎片資料 (my_fragments)
        # 這裡假設你有一個全域或傳入的 task_map 來篩選屬於設備 i 的碎片
        my_fragments = [] 
        # (這裡應加入你原本註解掉的碎片過濾邏輯，確保每個設備領到自己的題目)

        # 2. 建立設備實體
        # 注意：參數順序必須跟類別定義的一模一樣！
        dev = IoT_Device(
            iot_device_id=i, 
            association_map=association_map[i], # 這裡要從 map 抓出第 i 個設備的資訊
            args=args,
            my_fragments=my_fragments
        )
        iot_devices.append(dev)

    return iot_devices

def intialize_data_to_iot(subgraphs, iot_devices):
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


'''
def initialize_global_class_proportions(dataset_train):
    # Calculate class distribution in the training dataset
    total_samples = len(dataset_train)
    class_counts = torch.zeros(len(dataset_train.classes))  # Get number of classes from dataset
    for _, label in dataset_train:
        class_counts[label] += 1
    class_proportions = class_counts / total_samples

    return class_proportions



# # Check if each edge server can meet data class distribution constraints with available devices
# print("\nChecking data class distribution feasibility for each edge server:")
# for i, edge_server in enumerate(edge_servers):
#     print(f"\nEdge Server {i}:")
    
#     # Get all connectable devices that haven't been selected yet
#     available_devices = [device for device in devices if device.selected == -1 and i in device.connected_edge_server_ids]
#     print(f"Available device IDs: {[device.id for device in available_devices]}")
    
#     if not available_devices:
#         print("No available devices to connect!")
#         continue
    
#     # Add all available devices to the edge server
#     for device in available_devices:
#         edge_server.add_device(device)
#         device.selected = i
    
#     # Calculate total data and class distribution after adding devices
#     total_data = edge_server.get_total_data_quantity()
#     data_dist = edge_server.get_data_class_list(args)
    
#     print("\nClass Distribution after adding devices:")
#     for class_idx, (count, target) in enumerate(zip(data_dist, class_proportions)):
#         proportion = count / total_data if total_data > 0 else 0
#         print(f"Class {class_idx}: {count} samples ({proportion:.2%})")
        
#         if abs(proportion - target) > args.data_class_threshold:
#             print(f"  ⚠️ Warning: Class {class_idx} proportion ({proportion:.2%}) is over threshold ({args.data_class_threshold:.2%})")

# # Check if total data across all devices equals dataset size
# def verify_data_distribution(devices, dataset_train):
#     total_device_data = sum(device.data_quantity for device in devices)
#     total_dataset_size = len(dataset_train)

#     print(f"\nVerifying total data distribution:")
#     print(f"Total data points across all devices: {total_device_data}")
#     print(f"Total dataset size: {total_dataset_size}")

#     if total_device_data != total_dataset_size:
#         print(f"⚠️ Warning: Data distribution mismatch!")
#         print(f"  {total_dataset_size - total_device_data} data points are unaccounted for")
#     else:
#         print("✓ All dataset points are properly distributed across devices")

#     # Verify no data point is assigned to multiple devices
#     all_data_points = []
#     for device in devices:
#         all_data_points.extend(device.data_idxs)
        
#     unique_data_points = set(all_data_points)
#     if len(all_data_points) != len(unique_data_points):
#         print("\n⚠️ Warning: Some data points are assigned to multiple devices!")
#         print(f"  Total assignments: {len(all_data_points)}")
#         print(f"  Unique assignments: {len(unique_data_points)}")
#         print(devices[0].data_class_list)
#     else:
#         print("\n✓ Each data point is assigned to exactly one device")

'''