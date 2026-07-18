import numpy as np
import matplotlib.pyplot as plt
import pprint
import math

# build egde server
# area_size 長寬都一樣1000
# cell_radius : edge server 訊號覆蓋範圍
def init_multi_cell_system(num_devices, num_edge_servers, area_size=1000, cell_radius=300):

    if num_devices <= 0 or num_edge_servers <= 0:
        raise ValueError("Number of devices and edge servers must be positive")
    if area_size <= 0 or cell_radius <= 0:
        raise ValueError("Area size and cell radius must be positive")
    if cell_radius > area_size:
        raise ValueError("Cell radius cannot be larger than area size")

    '''
    cell_size 是決定網格的大小，（用num_edge_servers 決定），因為我想要把 servers 均勻分散在系統上

    step : 網格每一格要多大

    edge_positions --> (x, y) : edge servers 的位置
    '''
    edge_positions = []
    
    # 用num_edge_servers推到需要多少的 網格 才能均勻地放下這些edge servers
    cell_size = int(np.ceil(np.sqrt(num_edge_servers)))

    # 網格每一格要多大
    step = area_size // cell_size

    # 用edge_position record the position of each server
    count = 0
    for i in range(cell_size):
        for j in range(cell_size):
            if count < num_edge_servers:
                x = i * step + step // 2
                y = j * step + step // 2
                edge_positions.append((x, y))
                count += 1

    edge_positions = np.array(edge_positions)
    '''
    =========================================================================================================================
    '''

    '''
    這裡是分配 IoT_devices 
    dists = np.linalg.norm(edge_positions - np.array([x, y]), axis=1) --> 1個iot對到所有edge servers的距離

    distances row 是 IoT_devices, col 是 edge_servers

    ex. 
    devices_positions[0] = (x, y)
    ditances[0] = (the distance of server1, the distance of server2, ........) 

    []     [server1][server2][server3][server4]
    [iot_1]   距離
    [iot_2]
    '''
    # 分配和紀錄device 的position 
    device_positions = []
    distances = []

    # 要丟到每一個device都有訊號為止
    while len(device_positions) < num_devices:
        x, y = np.random.randint(0, area_size), np.random.randint(0, area_size)

        # dists 是一個 NumPy Array（陣列）
        # axis 請你一行一行（row by row）』分開算, if axis = 0 --> 會把所有的 x 加起來和 y 加起來才做畢氏定理 
        dists = np.linalg.norm(edge_positions - np.array([x, y]), axis=1)

        # dists 是一個 NumPy Array（超級清單）：它有一種叫做**「廣播 (Broadcasting)」的超能力。當你寫 dists <= 300 時，它不需要迴圈，它會瞬間**把裡面的每一個數字都拿去跟 300 比較，然後立刻變身成一張「是非題考卷」：
        if np.any(dists <= cell_radius):
            device_positions.append((x, y))
            distances.append(dists)
            
    # device_positions[0] --> 地0個 device的位置
    device_positions = np.array(device_positions)
    # distances[0] --> 第0 個 device 跟所有 edge servers 的distance
    distances = np.array(distances)

    return device_positions, edge_positions, distances




'''
association_map[0]--> iot_0
association_map[0] {
    IoT = 0;
    感應到的servers ID
    感應到的servers distance
}
'''
# 取得設備與基地台的連線對應表
def get_device_edge_association_map(device_positions, distances, cell_radius):
    association_map = []

    # zip 就像一個拉線把兩個東西組了在一起
    # for dev_id, (dev_pos, dist_list) in enumerate(zip(device_positions, distances)):--> 先看zip --> (0"enumerate會自動產生0, 1, 2..." to dev_id), (device_postions to dev_pos), (dist_list to distances) 
    for dev_id, (dev_pos, dist_list) in enumerate(zip(device_positions, distances)):
        connected_edges = []
        connected_distances = []

        # 再次先看in -->對dist_list 給編號0,1,2 .... 標號代表的是edge_server，裡面放的是距離（）
        # for 得到 一個 iot 可以連到哪寫 edge_server
        # cell_radius 是 edge_server的覆蓋半徑
        for edge_id, dist in enumerate(dist_list): 
            if dist <= cell_radius:

                # edge_id 對應的是 接到的edge_server編號
                connected_edges.append(edge_id)
                connected_distances.append(round(dist, 2))

        association_map.append({
            "device_id": dev_id,
            "connected_edge_server_ids": connected_edges,
            "distances": connected_distances
        })

    return association_map
'''

'''
def compute_channel_gain(association_map, num_antennas):

    for info in association_map:
        device_gains = {}     
        for idx_edge, edge_id in enumerate(info['connected_edge_server_ids']):
            # small-scale fading
            shadowing = 10 ** (np.random.normal(0, 1) / 10)
            # Path loss
            PL = 10 ** ((50 + 25 * math.log10(info['distances'][idx_edge])) / 10)
            loss = np.sqrt(PL * shadowing)  # Total loss caused by path loss and shadowing

            # channel gain --> 有能量有角度
            g_k = np.random.normal(0, 1, num_antennas) + 1j * np.random.normal(0, 1, num_antennas)  # Small-scale fading
            h_k = g_k / loss  # Channel gain
            device_gains[edge_id] = h_k
        
        # h_k = association_map[0]['channel_gains'][5] --> print(f"設備 0 對基地台 5 的頻道增益向量為: {h_k}")
        info['channel_gains'] = device_gains

    return association_map



'''
在無線通訊（特別是 RSMA）中，如果兩台設備的訊號向量（Channel Gains）長得很像，它們就會產生強烈的干擾。
這段 code 會為每一座基地台生成一個 $N \times N$ 的矩陣（Correlation Table）

'''
# 為「每一座基地台」製作一張「用戶訊號互相干擾的關係表」。
def compute_spatial_correlation_table(association_map, num_devices, num_edge_servers):
    """
    Compute spatial correlation tables for each edge server
    Args:
        association_map: Map containing device associations and channel gains
        num_antennas: Number of antennas at each edge server
    Returns:
        List of correlation tables, one per edge server
    """
    # Get number of edge servers by finding max edge server ID
    
    # Initialize correlation tables for each cell
    correlation_tables = []

    for e in range(num_edge_servers):

        # Get devices connected to this edge server
        """
        connected_devices = []  # 準備一個空名單
        for info in association_map:  # 把總名冊裡的履歷表一張一張拿出來看
            if e in info['connected_edge_server_ids']:  # 檢查條件
                connected_devices.append(info)  # 條件符合，就把整張履歷表收進名單裡
        """
        connected_devices = [info for info in association_map if e in info['connected_edge_server_ids']]
        
        # Create correlation table for this cell
        corr_table = np.zeros((num_devices, num_devices))
        
        # Fill correlation table
        # _ --> 當作垃圾
        for _, dev1 in enumerate(connected_devices):
            for _, dev2 in enumerate(connected_devices):

                # Get channel gains for this edge server
                gain1 = dev1['channel_gains'][e]
                gain2 = dev2['channel_gains'][e]
                
                idx1 = dev1['device_id']
                idx2 = dev2['device_id']

                # Compute correlation
                corr_table[idx1, idx2] = compute_spatial_correlation(gain1, gain2)

        correlation_tables.append(corr_table)
    
    return correlation_tables


def compute_spatial_correlation(channel_gain_i, channel_gain_j):
    """
    Compute spatial correlation between two channel gains
    Args:
        channel_gain_i: Complex channel gain vector for device i
        channel_gain_j: Complex channel gain vector for device j
    Returns:
        Spatial correlation value between 0 and 1
    """
    numerator = np.abs(np.dot(channel_gain_i.conj(), channel_gain_j))**2
    denominator = np.linalg.norm(channel_gain_i)**2 * np.linalg.norm(channel_gain_j)**2
    
    if denominator == 0:
        return 0
        
    correlation = numerator / denominator
    
    # # Ensure correlation is between 0 and 1
    # correlation = np.clip(correlation, 0, 1)
    
    return correlation

if __name__ == '__main__':
    # Test parameters
    num_devices = 90
    num_edge_servers = 9
    area_size = 1000
    cell_radius = 300
    num_antennas = 2

    # Run the initialization
    device_positions, edge_positions, distances = init_multi_cell_system(
        num_devices=num_devices, 
        num_edge_servers=num_edge_servers, 
        area_size=area_size, 
        cell_radius=cell_radius
    )

    # Get device-edge association map
    association_map = get_device_edge_association_map(device_positions, distances, cell_radius)

    # Compute channel gains
    association_map_with_gains = compute_channel_gain(association_map, num_antennas)
    pprint.pprint(association_map)

    # Print correlation tables
    print("\nSpatial Correlation Tables:")
    correlation_tables = compute_spatial_correlation_table(association_map_with_gains, num_devices, num_edge_servers)
    for edge_idx, corr_table in enumerate(correlation_tables):
        print(f"\nEdge Server {edge_idx} Correlation Table:")
        print(corr_table)
        print()

    # Plot the system layout
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot devices and add device IDs as labels
    for device_id, (x, y) in enumerate(device_positions):
        ax.scatter(x, y, c='blue')
        ax.annotate(f'D{device_id}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot edge servers
    ax.scatter(*edge_positions.T, c='red', marker='^', s=100, label='Edge Servers')

    # Add coverage circles
    for (x, y) in edge_positions:
        circle = plt.Circle((x, y), cell_radius, color='red', alpha=0.1)
        ax.add_patch(circle)

    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_title('Multi-Cell System with Coverage Verification')
    
    # Update legend to include both devices and edge servers
    ax.scatter([], [], c='blue', label='Devices')  # Add devices to legend
    ax.legend()
    ax.grid(True)
    plt.show()
