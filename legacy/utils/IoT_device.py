import numpy as np

class IoT_Device:
    def __init__(self, iot_device_id, association_map, args, my_fragments):
        '''
        my_fragments: 從 global_task_map 領回來的屬於我的題目碎片
        '''

        self.id = iot_device_id

        '''
        self.channel_gain: 設備與伺服器之間的「通道增益」（Channel Gain），簡單來說就是訊號的好壞程度。
        白話文： 「我跟伺服器之間的訊號有多好/多壞？」
        專業解釋：  這是通訊裡極度重要的「通道增益」。電磁波在空氣中傳輸會遇到牆壁反彈、衰減，這個變數通常是一個複數 (Complex Number)，包含了訊號振幅的衰減與相位的偏移。它是由外部傳進來的 association_map（配對地圖）賦予的。
        
        self.channel_gain_abs
        因為上面的 channel_gain 通常是帶有虛數的複數，在計算傳輸速率時，我們往往只需要它的絕對值 (Absolute Value)，也就是純粹的強度大小。這裡先放 None 當初始值，稍後程式就會把 channel_gain 取絕對值後填入這裡
        '''
        self.channel_gain = association_map['channel_gains']
        self.channel_gain_abs = None

# --- 2. 資料層 (從影像分類改為 MMLU 碎片) ---
        # 存入領到的碎片清單：[{'q_id': 'Q1', 'option': 'A', 'text': '...'}, ...]
        self.my_fragments = my_fragments 
        
        # 這裡是關鍵！將碎片轉化為「我關注的題目 ID 集合」
        # 這樣 RSMA 才能秒算：誰跟我在算同一題？
        self.locations = set(item['q_id'] for item in my_fragments)
        
        # 紀錄這台設備擁有的資料量
        self.data_quantity = len(my_fragments)

        # --- 3. 領域分佈 (取代原本的 data_class_list) ---
        # 算出這台設備包含哪些領域 (Domain)，用來跟 Server 的專家匹配
        self.domains = set(item['domain'] for item in my_fragments)
        
        
        '''
        self.selected是指有沒有被選到, -1 --> 只初始化：沒有被選 
        self.group_idx, 被分到哪一組
        '''
        self.selected = -1
        self.group_idx = -1

   
        self.connected_edge_server_ids = association_map['connected_edge_server_ids']
        

        '''
        self.pvt_pow, self.cmn_pow: Private Power (私有功率) 與 Common Power (公共功率)。這非常明顯是 RSMA (Rate-Splitting Multiple Access) 通訊技術的特徵，表示傳輸時將訊號拆分成公共與私有兩部分
        這是在講啥： 當這台 IoT 設備要用天線把 AI 模型數據發送給邊緣伺服器時，它擁有的總電力（發射功率）是有限的。在 RSMA 技術中，設備會把總功率拆分成兩份：一份給「公共訊號 (Common)」，一份給「私有訊號 (Private)」。
        '''
        self.pvt_pow = 0
        self.cmn_pow = 0

        '''
        這是在講啥： 現代通訊設備通常有多根天線（MIMO 技術）。「預編碼（Precoding）」就是透過數學矩陣去調整每根天線發射訊號的相位和振幅，讓電磁波能在空間中「集中」朝著目標伺服器的方向發射，並減少對其他設備的干擾（這稱為波束成形 Beamforming）。
        '''
        self.pvt_precoder = None
        self.cmn_precoder = None

        '''
        這是在講啥： 現代通訊設備通常有多根天線（MIMO 技術）。「預編碼（Precoding）」就是透過數學矩陣去調整每根天線發射訊號的相位和振幅，讓電磁波能在空間中「集中」朝著目標伺服器的方向發射，並減少對其他設備的干擾（這稱為波束成形 Beamforming）。
        '''
        self.pvt_rate = 0
        self.cmn_rate = 0
        # self.model = model --> no
        # self.comp_capability = np.random.randint(1, 11) * 1e12 # Computation capability (TFlops/s)
        
        '''
        self.cmn_part = self.drop_mask (公共部分攜帶的模型資料)
        self.pvt_part = self.drop_mask (私有部分攜帶的模型資料)
        '''
        # self.cmn_part = self.drop_mask
        # self.pvt_part = self.drop_mask


    def compute_class_list(self, dataset, args):
        # Initialize a list to store count of samples for each class
        class_distribution = [0] * args.num_classes
        
        # Count samples for each class
        for _, label in dataset:
            class_distribution[label] += 1
            
        return class_distribution
    '''
    =================================================
    '''  

    def _collect_my_options(self, all_questions):
        """
        核心邏輯：從總字典中過濾出屬於『我』的選項碎片。
        回傳格式範例：[('Q_0001', 'A'), ('Q_0001', 'C'), ('Q_0005', 'B')]
        """
        my_stuff = []
        for q_id, q_obj in all_questions.items():
            for option_letter, info in q_obj.options_info.items():
                if info['location'] == self.id:
                    my_stuff.append((q_id, option_letter))
        return my_stuff

    def compute_domain_list(self, all_questions, args):
        """
        統計這台設備擁有的選項分別屬於哪些專業領域 (Domain)
        """
        # 初始化統計字典，例如 {'physics': 0, 'history': 0, ...}
        domain_counts = {d: 0 for d in args.all_domains} 
        
        for q_id, option_letter in self.my_fragments:
            # 透過 q_id 去總表查這題是什麼領域
            domain = all_questions[q_id].domain
            if domain in domain_counts:
                domain_counts[domain] += 1
                
        return domain_counts

    def get_option_text(self, all_questions, q_id, option_letter):
        """當伺服器需要讀取資料時，設備提供它擁有的文字"""
        return all_questions[q_id].options_info[option_letter]['text']
    




