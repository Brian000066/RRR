from utils.IoT_device import IoT_Device
# from utils.dropout import update_group_cmn_pvt_part


class EdgeServer:
    def __init__(self, server_id, args, coverage):
        self.id = server_id
        self.devices = []
        self.args = args
        self.coverage = coverage
        self.group_set = []
        
        #    # 存在server都算
        self.existing_experts = {}
        self.total_memory = 4000
        self.total_memory_used = 0
        # 啟用的才算
        self.activated_expert = []
        self.gpu_memory = 1400
        
        self.gpu_memory_used = 0

        # --- 核心修正 2：新增分配專家的方法 ---
    def add_expert_label(self, expert_id, mem_cost):
        """檢查記憶體是否足夠，若足夠則登記該專家"""
        if self.total_memory_used + mem_cost <= self.total_memory:
            if expert_id not in self.existing_experts:
                self.existing_experts[expert_id] = {} # 開發一個空位給它
                self.total_memory_used += mem_cost
                return True
        return False

    
    def activate_expert(self, expert_id, mem_cost) :
        if self.gpu_memory_used + mem_cost <= self.gpu_memory:
            self.activated_expert.append(expert_id)
            self.gpu_memory_used += mem_cost

            print(f"✅ Expert {expert_id} 已載入 Server {self.id}")
            return True
        else:
            print(f"❌ Server {self.id} GPU 記憶體不足！")
            return False

    # 看看這台 Server 上有哪些專家
    def get_server_expert_list(self) :
        return list(self.expert.keys)

        '''
        因為多了一個人，原本的「公共訊息」和「私有訊息」的分配比例必須重新計算，以確保通訊效率。
        '''
    # def add_device(self, device: IoT_Device, group_idx: int = -1):
    #     self.devices.append(device)
    #     device.selected = self.id
    #     if group_idx != -1:
    #         self.group_set[group_idx].append(device)
    #     device.group_idx = group_idx
    #     if group_idx != -1:
    #         # update common/private parts
    #         # update_group_cmn_pvt_part(self.group_set[group_idx])

    # def delete_device(self, device: Device):
    #     try:
    #         self.devices.remove(device)
    #         # Remove device from its group if it belongs to one
    #         if device.group_idx != -1:
    #             group_idx = device.group_idx
    #             self.group_set[group_idx].remove(device)
    #             # Remove empty group if no devices left
    #             if not self.group_set[group_idx]:
    #                 self.group_set.pop(group_idx)
    #                 # Update group indices for remaining devices
    #                 for d in self.devices:
    #                     if d.group_idx > group_idx:
    #                         d.group_idx -= 1
    #             # update common/private parts
    #             else:
    #                 update_group_cmn_pvt_part(self.group_set[group_idx])

    #             device.group_idx = -1
    #             device.selected = -1
    #         return True
    #     except ValueError:
    #         return False


    '''
don't need
'''
    def get_total_data_quantity(self): # get the total data quantity
        return sum(d.data_quantity for d in self.devices)
    
    def get_data_class_list(self, args): # get class distribution
        # Initialize list based on device's data_class_list length        
        
        data_class_list = [0] * args.num_classes

        for d in self.devices:
            for label, count in enumerate(d.data_class_list):
                data_class_list[label] += count
        
        return data_class_list


