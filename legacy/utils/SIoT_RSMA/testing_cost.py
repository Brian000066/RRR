import numpy as np
from phase2 import find_longest_path
from phase3 import phase3, compute_common_message_ratio, distributed_beamforming_gain, compute_private_rate_group
from config import rate_mapping


def main():
    rsma_groups, collaborative_groups=phase3()
    print("Final RSMA groups")
    for i, g in enumerate(rsma_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        cost = compute_RSMA_cost(g, bandwidth_cost_unit=1, mec_cost_unit=1)
        print(f"RSMA Group {i+1} Cost: {cost}\n")

    print("Final Collaborative groups:")
    for i, g in enumerate(collaborative_groups):
        print(f"Group {i+1}: {[siot.index for siot in g]}")
        cost = compute_collaborative_cost(g, bandwidth_cost_unit=1, eta=1)
        print(f"Collaborative Group {i+1} Cost: {cost}\n")

def compute_RSMA_cost(rsma_group, bandwidth_cost_unit, mec_cost_unit):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not rsma_group:
        return 0  # 空群組直接返回 0

    # 1. Bandwidth cost C_c^bw
    C_bw = bandwidth_cost_unit * compute_rsma_required_bandwidth(rsma_group, T_threshold=5)
    

    # 2. Computation cost C_c^comp at MEC server
    C_comp = mec_cost_unit * len(set.union(*(set(siot.locations) for siot in rsma_group)))


    # 總成本
    total_cost = C_bw + C_comp
    print(f"Total:{total_cost}, Bandwith: {C_bw}, Computation:{C_comp}")
    return total_cost

def compute_collaborative_cost(collaborative_group, bandwidth_cost_unit, eta):
    """
    計算 collaborative group c 的總成本 (Bandwidth cost + Communication cost + Computation cost)
    collaborative_group: 目前的 collaborative group
    siots: 所有 SIoTs 資訊
    bandwidth_cost_unit: c_bw, bandwidth unit cost
    eta: collaborative parameter
    """
    if not collaborative_group:
        return 0  # 空群組直接返回 0

    # 1. Bandwidth cost C_c^bw
    C_bw = bandwidth_cost_unit * compute_collaborative_required_bandwidth(collaborative_group, T_threshold=5)
    
    # 2. Communication cost C_c^com
    num_siot = len(collaborative_group)  # 群組內 SIoT 數量
    if num_siot == 1:
        C_com = 0  # 只有一個設備，無需通信
    else:
        C_com = eta * (num_siot - 1)  # 可選用 |V|-1 (確保連通) 或 Edge 數

    # 3. Computation cost C_c^comp
    C_comp = eta *(sum(abs(len(siot.locations)) for siot in collaborative_group) ) # Covered Locations 絕對值計算

    # 總成本
    total_cost = C_bw + C_com + C_comp
    print(f"Total:{total_cost}, Bandwith: {C_bw}, Communication:{C_com}, Computation:{C_comp}")
    return total_cost

def compute_rsma_required_bandwidth(rsma_group, T_threshold):
    """
    計算該 rsma group 需要的 Bandwidth.
    """   
    if not rsma_group:
        return 0
    if len(rsma_group) == 1:
        B_g = compute_single_required_bandwidth(rsma_group, T_threshold)  # 獨立計算單個 SIoT 的 bandwidth
        print(f"Single SIoT Bandwidth: {B_g}")  # 確保值是合理的
        return B_g
    # **1. 計算計算時間 (Computation Time)**
    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    mec_cpu = 1  # 總 CPU 頻率
    T_comp = total_locations / mec_cpu if mec_cpu > 0 else float('inf')
    #print(f"T_comp:{T_comp}")
        
    #剩餘時間 給傳輸時間
    T_rest=(len(rsma_group) * T_threshold) - T_comp
    #print(f"T_rest:{T_rest}")  
    common_message_ratio, common_locations= compute_common_message_ratio(rsma_group)
  
    #Common SINR
    common_numerator = sum(abs(siot.gain)**2 * (common_message_ratio*siot.power) for siot in rsma_group)
    common_denominator = sum(abs(siot.gain)**2 * ((1-common_message_ratio)*siot.power) for siot in rsma_group) + 1e-18
    common_sinr= common_numerator / common_denominator if common_denominator > 0 else float('inf') # 避免分母為 0
    beamforming_gain=distributed_beamforming_gain(rsma_group,common_message_ratio)
    #Common data rate
    #common_rate= np.log2(1 + beamforming_gain * common_sinr)
    common_rate=rate_mapping(beamforming_gain * common_sinr)

    #Private SINR data rate
    private_rate_dict=compute_private_rate_group(rsma_group, common_message_ratio)

    common_message_data_size=common_message_ratio*total_locations
    private_message_data_size = {siot.index: len(set(siot.locations) - common_locations) for siot in rsma_group }  # 計算每個 SIoT 的私有訊息
        
    # 計算公用訊息的延遲
    common_uplink_time = common_message_data_size / common_rate if common_rate > 0 else float('inf')
    print(f"common_uplink_time:{common_uplink_time},  common rate:{common_rate}, common sinr:{common_sinr}")

    # 計算私有訊息的延遲 (對所有 SIoT 取最大值)
    private_uplink_time = [private_message_data_size[siot.index] / private_rate_dict[siot.index] if private_rate_dict.get(siot.index, 0) > 0 else float('inf')for siot in rsma_group]
    print(f"private_uplink_time: {private_uplink_time}")

    # 取最大值作為最終的上行延遲
    total_uplink_delay = max(common_uplink_time, max(private_uplink_time)) if private_uplink_time else common_uplink_time
    B_g = total_uplink_delay/T_rest
     
    print(f"Bandwidth:{B_g}")
    return B_g

def compute_single_required_bandwidth(rsma_group, T_threshold):
    total_locations = len(set.union(*(set(siot.locations) for siot in rsma_group)))
    mec_cpu = 2  # 總 CPU 頻率
    T_comp = total_locations / mec_cpu if mec_cpu > 0 else float('inf')
    print(f"T_comp:{T_comp}")
    T_rest=(len(rsma_group) * T_threshold) - T_comp
    print(f"T_rest:{T_rest}")

    #SIoT data rate
    siot_rate = total_locations / T_rest
    gamma_siot = sum((abs(siot.gain) ** 2 * siot.power) / 1e-18 for siot in rsma_group)
    data_rate=rate_mapping(gamma_siot)
    B_g=siot_rate/ data_rate
    return B_g

def compute_collaborative_required_bandwidth(collaborative_group, T_threshold):
    """
    計算該 collaborative group 需要的 Bandwidth.
    
    Parameters:
    - collaborative_group: 該 group 的 SIoT 設備集合
    - T_threshold: 時間門檻
    - SNR_d2d: D2D 通訊的 SNR
    - SNR_u: 上行傳輸的 SNR
    - eta: 加權參數
    """
    
    if not collaborative_group:
        return 0
 
    representative_siot = max(collaborative_group, key=lambda s: s.gain)  # 挑最高 channel gain
    

    # **1. 計算計算時間 (Computation Time)**
    T_comp = max(len(siot.locations) / siot.cpu_frequency for siot in collaborative_group)
    
    # **2. 計算 D2D 通訊時間**
    longest_path_length, critical_siot = find_longest_path(representative_siot,collaborative_group)

    T_d2d = ((len(critical_siot.locations) * 0.4) / critical_siot.d2d_rate) * longest_path_length
    
    # **3. 計算上行傳輸時間**
    
    #剩餘時間 (也扣過整合時間)
    T_rest=(len(collaborative_group) * T_threshold) - T_comp - T_d2d - ((sum(len(siot.locations) * 0.2 for siot in collaborative_group) / representative_siot.cpu_frequency))
    
    #Representative SIoT data rate
    R_representative_siot = (len(set.union(*[set(siot.locations) for siot in collaborative_group])) * 0.2) / T_rest
    gamma_representative_siot= (abs(representative_siot.gain) ** 2 * representative_siot.power) / 1e-18
    
    B_u_c=R_representative_siot/ np.log2(1 + gamma_representative_siot)
    
    #print(f"Bandwidth:{B_u_c}")
    return B_u_c



if __name__ == '__main__':
    main()