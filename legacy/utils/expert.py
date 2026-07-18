



'''

biology
psychology
physics
philosophy
other
math
law
history
health
engineering
economics
computer science
chemistry
business
14 type * 2 (1 types * 2)
'''

# 這裡定義每一個具體的專家個體
EXPERT_SET = {
# 生物 (Biology)
    "expert1": {"domain": "biology", "mem": 500, "accuracy": 0.92},
    "expert2": {"domain": "biology", "mem": 350, "accuracy": 0.85},
    # 心理 (Psychology)
    "expert3": {"domain": "psychology", "mem": 600, "accuracy": 0.94},
    "expert4": {"domain": "psychology", "mem": 500, "accuracy": 0.89},
    # 物理 (Physics)
    "expert5": {"domain": "physics", "mem": 200, "accuracy": 0.78},
    "expert6": {"domain": "physics", "mem": 500, "accuracy": 0.91},
    # 哲學 (Philosophy)
    "expert7": {"domain": "philosophy", "mem": 350, "accuracy": 0.82},
    "expert8": {"domain": "philosophy", "mem": 600, "accuracy": 0.95},
    # 其他 (Other)
    "expert9": {"domain": "other", "mem": 500, "accuracy": 0.88},
    "expert10": {"domain": "other", "mem": 200, "accuracy": 0.75},
    # 數學 (Math)
    "expert11": {"domain": "math", "mem": 500, "accuracy": 0.93},
    "expert12": {"domain": "math", "mem": 350, "accuracy": 0.86},
    # 法律 (Law)
    "expert13": {"domain": "law", "mem": 600, "accuracy": 0.96},
    "expert14": {"domain": "law", "mem": 500, "accuracy": 0.90},
    # 歷史 (History)
    "expert15": {"domain": "history", "mem": 200, "accuracy": 0.77},
    "expert16": {"domain": "history", "mem": 500, "accuracy": 0.89},
    # 健康 (Health)
    "expert17": {"domain": "health", "mem": 350, "accuracy": 0.84},
    "expert18": {"domain": "health", "mem": 600, "accuracy": 0.95},
    # 工程 (Engineering)
    "expert19": {"domain": "engineering", "mem": 500, "accuracy": 0.92},
    "expert20": {"domain": "engineering", "mem": 200, "accuracy": 0.79},
    # 經濟 (Economics)
    "expert21": {"domain": "economics", "mem": 500, "accuracy": 0.90},
    "expert22": {"domain": "economics", "mem": 350, "accuracy": 0.83},
    # 電腦科學 (Computer Science)
    "expert23": {"domain": "computer science", "mem": 600, "accuracy": 0.98},
    "expert24": {"domain": "computer science", "mem": 500, "accuracy": 0.91},
    # 化學 (Chemistry)
    "expert25": {"domain": "chemistry", "mem": 200, "accuracy": 0.80},
    "expert26": {"domain": "chemistry", "mem": 500, "accuracy": 0.92},
    # 商業 (Business)
    "expert27": {"domain": "business", "mem": 350, "accuracy": 0.81},
    "expert28": {"domain": "business", "mem": 600, "accuracy": 0.94},
}


def fill_servers_with_experts(edge_servers, expert_dict):
    """
    1. 確保 28 個專家至少各出現一次。
    2. 盡可能填滿每個 Server 的 GPU 記憶體。
    """
    all_expert_ids = list(expert_dict.keys())
    
    # --- 第一階段：全員保底分配 ---
    # 先洗牌，讓每次實驗的初始分配不同
    random.shuffle(all_expert_ids)
    
    # 輪流分配給 Server
    server_idx = 0
    num_servers = len(edge_servers)
    
    for exp_id in all_expert_ids:
        target_server = edge_servers[server_idx % num_servers]
        exp_info = expert_dict[exp_id]
        
        # 嘗試加入（add_expert_label 內建了 GPU 空間檢查）
        success = target_server.add_expert_label(exp_id, exp_info['mem'])
        if success:
            # 補上 domain 和 accuracy 資訊
            target_server.experts[exp_id]['domain'] = exp_info['domain']
            target_server.experts[exp_id]['accuracy'] = exp_info['accuracy']
        
        server_idx += 1

    # --- 第二階段：填滿剩餘空間 ---
    print("第一階段完成，開始填滿剩餘 GPU 空間...")
    
    for server in edge_servers:
        # 只要還有空間，就隨機抽專家來塞
        # 為了避免無窮迴圈（所有專家都塞不下時），我們設定一個嘗試上限
        attempts = 0
        while attempts < 50: 
            # 隨機挑一個專家（這次不限沒出現過的）
            random_exp_id = random.choice(all_expert_ids)
            
            # 如果這台 Server 已經有這個專家了，就跳過（避免重複部署同一個專家到同一台 Server）
            if random_exp_id in server.experts:
                attempts += 1
                continue
                
            exp_info = expert_dict[random_exp_id]
            
            # 嘗試加入
            success = server.add_expert_label(random_exp_id, exp_info['mem'])
            if success:
                server.experts[random_exp_id]['domain'] = exp_info['domain']
                server.experts[random_exp_id]['accuracy'] = exp_info['accuracy']
                attempts = 0 # 成功塞入一個，重置嘗試次數
            else:
                # 塞不下了，代表這台 Server 滿了
                attempts += 1 

    # --- 最後檢查 ---
    for s in edge_servers:
        print(f"Server {s.id}: 已用 {s.gpu_mem_used}/{s.gpu_mem_capacity} MB, 專家數: {len(s.experts)}")

