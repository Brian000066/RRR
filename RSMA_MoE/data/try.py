import random
import numpy as np
import torch
from utils.edge_server import EdgeServer
from utils.expert import EXPERT_SET

# class EdgeServer:
#     def __init__(self, server_id, args, coverage, gpu_mem):
#         self.id = server_id
#         self.devices = []
#         self.args = args
#         self.coverage = coverage
#         self.group_set = []
        
#         # 存在server都算
#         self.existing_expert_ = []
        
#         # 啟用的才算
#         self.activated_expert = []
#         self.gpu_memory = gpu_mem
#         self.gpu_memory_used = 0


def intialize_fill_servers_with_experts(edge_servers, expert_dict):

    all_expert_ids = list(expert_dict.keys())
    random.shuffle(all_expert_ids) # 增加隨機性
    
    num_servers = len(edge_servers)
    
    # 第一階段：保底分配 (確保每個領域的專家都有地方待)
    for i, exp_id in enumerate(all_expert_ids):
        target_server = edge_servers[i % num_servers]
        exp_info = expert_dict[exp_id]
        
        if target_server.add_expert_label(exp_id, exp_info['mem']):
            target_server.experts[exp_id].update({
                'domain': exp_info['domain'],
                'accuracy': exp_info['accuracy']
            })
    
    # 第二階段：填滿剩餘空間 (貪婪做法)
    for server in edge_servers:
        attempts = 0
        while attempts < 30:
            rand_exp_id = random.choice(all_expert_ids)
            if rand_exp_id in server.experts:
                attempts += 1
                continue
                
            exp_info = expert_dict[rand_exp_id]
            if server.add_expert_label(rand_exp_id, exp_info['mem']):
                server.experts[rand_exp_id].update({
                    'domain': exp_info['domain'],
                    'accuracy': exp_info['accuracy']
                })
                attempts = 0 # 成功就重置
            else:
                attempts += 1


edge_servers = []
for i in range(4):
    ser = EdgeServer(
        server_id=i, 
        coverage=None,
        args=None,
        gpu_mem=1000
    )
    edge_servers.append(ser)

intialize_fill_servers_with_experts(edge_servers, EXPERT_SET)


# 2. 開始檢查
print(f"--- 🚀 邊緣伺服器專家分配檢查 ---")
for server in edge_servers:
    print(f"\n[Server {server.id}]")
    print(f"  - 剩餘/總記憶體: {server.gpu_memory - server.gpu_memory_used}/{server.gpu_memory} MB")
    print(f"  - 專家數量: {len(server.experts)}")
    
    # 印出這台 Server 拿到了哪些領域的專家
    domains = [info['domain'] for info in server.experts.values()]
    print(f"  - 涵蓋領域: {set(domains)}")
    
    # 印出具體的專家 ID 清單
    print(f"  - 專家名單: {list(server.experts.keys())}")

# 3. 總體檢查：是否有專家被遺漏了？
all_assigned_experts = set()
for s in edge_servers:
    all_assigned_experts.update(s.experts.keys())

missing = set(EXPERT_SET.keys()) - all_assigned_experts
if not missing:
    print("\n✅ 完美！所有專家都至少被分配到一台伺服器上。")
else:
    print(f"\n⚠️ 警告！有 {len(missing)} 個專家沒被分配到：{missing}")