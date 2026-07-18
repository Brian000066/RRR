"""
测试论文算法的专家分配
"""

import random
from legacy.utils.expert_allocation_paper import ExpertAllocator

def create_test_data():
    """创建测试数据"""
    
    # 专家信息
    experts = {
        "expert1": {"domain": "biology", "mem": 500, "accuracy": 0.92},
        "expert2": {"domain": "biology", "mem": 350, "accuracy": 0.85},
        "expert5": {"domain": "physics", "mem": 200, "accuracy": 0.78},
        "expert6": {"domain": "physics", "mem": 500, "accuracy": 0.91},
        "expert11": {"domain": "math", "mem": 500, "accuracy": 0.93},
        "expert12": {"domain": "math", "mem": 350, "accuracy": 0.86},
    }
    
    # 服务器信息
    servers = [
        {"server_id": "server_0", "gpu_memory": 1000},
        {"server_id": "server_1", "gpu_memory": 1000},
        {"server_id": "server_2", "gpu_memory": 1000},
    ]
    
    # 请求（子任务）信息
    requests = [
        {"id": "request_0", "domain": "biology"},
        {"id": "request_1", "domain": "physics"},
        {"id": "request_2", "domain": "math"},
        {"id": "request_3", "domain": "biology"},
        {"id": "request_4", "domain": "physics"},
    ]
    
    # 生成需求矩阵 Cp (K x N)
    # Cp[k][n] = 请求k对专家n的需求程度 (0-1)
    K = len(requests)  # 5个请求
    N = len(experts)   # 6个专家
    
    Cp = [[0.0] * N for _ in range(K)]
    
    expert_list = list(experts.keys())
    domains = [experts[e]['domain'] for e in expert_list]
    
    # 根据领域匹配设置需求
    for k in range(K):
        req_domain = requests[k]['domain']
        for n in range(N):
            expert_domain = domains[n]
            
            if expert_domain == req_domain:
                # 同领域的专家需求高
                Cp[k][n] = random.uniform(0.7, 1.0)
            else:
                # 不同领域的专家需求低
                Cp[k][n] = random.uniform(0.0, 0.3)
    
    return experts, servers, requests, Cp

def main():
    print("=" * 60)
    print("论文算法测试：专家分配")
    print("=" * 60)
    
    # 创建测试数据
    experts, servers, requests, Cp = create_test_data()
    
    print(f"\n输入数据:")
    print(f"- 专家数量 (N): {len(experts)}")
    print(f"- 服务器数量 (E): {len(servers)}")
    print(f"- 请求数量 (K): {len(requests)}")
    
    print("\n需求矩阵 Cp (请求 x 专家) 样本:")
    print("(" + "代表需求值从0-1)")
    
    expert_list = list(experts.keys())
    
    # 显示前3个请求
    for k in range(min(3, len(requests))):
        print(f"\n  Request {requests[k]['id']} ({requests[k]['domain']}):")
        for n in range(len(experts)):
            val = Cp[k][n]
            bar = "█" * int(val * 10)
            print(f"    {expert_list[n]}: {val:.2f} {bar}")
    
    # 创建分配器
    allocator = ExpertAllocator(
        n_experts=len(experts),
        n_servers=len(servers),
        n_requests=len(requests),
        experts=experts,
        servers=servers,
        requests=requests,
        alpha_e=0.33,   # 专家-专家亲和力权重
        beta_e=0.33,    # 请求-专家亲和力权重
        gamma_e=0.34,   # 负载均衡权重
        ft_steps=50     # 本地搜索迭代次数
    )
    
    # 运行分配算法
    print("\n\n运行专家分配算法...")
    result = allocator.allocate(Cp)
    
    # 显示结果
    print(allocator.get_allocation_summary(result))
    
    # 详细分析
    print("\n=== 详细分析 ===\n")
    
    print("专家负载分布:")
    for e, expert_id in enumerate(expert_list):
        print(f"  {expert_id}: {result.expert_loads[e]:.2f}")
    
    print("\n服务器中的专家组合:")
    for s in range(len(servers)):
        experts_in_server = [expert_list[e] 
                            for e in range(len(experts))
                            if result.p_matrix_ep[e][s] == 1]
        
        if experts_in_server:
            # 获取这些专家的领域
            domains = [experts[e]['domain'] for e in experts_in_server]
            print(f"\n  {servers[s]['server_id']}:")
            for exp_id in experts_in_server:
                domain = experts[exp_id]['domain']
                load = result.expert_loads[expert_list.index(exp_id)]
                print(f"    - {exp_id} ({domain}), load={load:.2f}")
            
            # 检查领域覆盖
            unique_domains = set(domains)
            print(f"    -> 覆盖领域: {unique_domains}")
    
    # 验证约束
    print("\n\n=== 约束检查 ===\n")
    
    # 检查1：每个专家只分配到一个服务器
    check1 = True
    for e in range(len(experts)):
        allocation_count = sum(result.p_matrix_ep[e])
        if allocation_count != 1:
            print(f"❌ 专家 {expert_list[e]} 分配数量错误: {allocation_count}")
            check1 = False
    
    if check1:
        print("✅ 每个专家恰好分配到一个服务器")
    
    # 检查2：负载均衡
    server_loads = result.server_loads
    avg_load = sum(server_loads) / len(server_loads)
    max_load = max(server_loads)
    min_load = min(server_loads)
    imbalance_ratio = max_load / min_load if min_load > 0 else float('inf')
    
    print(f"\n负载均衡:")
    print(f"  平均负载: {avg_load:.2f}")
    print(f"  最大负载: {max_load:.2f}")
    print(f"  最小负载: {min_load:.2f}")
    print(f"  不平衡比率: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 2.0:
        print(f"  ✅ 负载相对均衡 (比率 < 2.0)")
    else:
        print(f"  ⚠️ 负载不平衡 (比率 >= 2.0)")

if __name__ == "__main__":
    main()