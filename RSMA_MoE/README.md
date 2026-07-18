# RSMA_MoE

本專案目前用來模擬多伺服器環境下的 MoE expert selection、IoT feature 傳輸、RSMA grouping、backhaul forwarding 與 task DAG scheduling。

目前預設跑四個比較方法：Top-K + Distance、Top-K + GSSGD DBG、WDMoE + Distance、WDMoE + GSSGD DBG。

執行方式：

```bash
python main.py
```

所有主要參數都集中在 `config.py` 的 `ExperimentConfig`。

## 1. Input Values

### 1.1 Reproducibility / Output

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `auto_random_seed` | `True` | 每次執行自動換 DAG seed 與 network seed。 |
| `num_runs` | `1000` | 每個啟用的方法會跑 100 次，最後輸出平均成本與平均 violations。 |
| `num_dags` | `10` | 產生 10 個 task graph。 |
| `random_seed` | `42` | 若 `auto_random_seed=False`，DAG 使用此 seed。 |
| `network_random_seed` | `1` | 若 `auto_random_seed=False`，network 使用此 seed。 |
| `output_dir` | `generated_dags/` | DAG JSON / GraphML / PNG 輸出位置。 |
| `results_dir` | `results/` | 演算法結果 JSON 輸出位置。 |
| `image_dpi` | `180` | DAG 圖片解析度。 |

### 1.2 DAG / Task Graph

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `num_nodes` | `12` | 每個 task graph 有 12 個 node/subtask。 |
| `num_branches` | `4` | DAG 分支數。 |
| `branch_length_range` | `(1, 5)` | 每個分支長度可不同。 |
| `depth` | `5` | DAG 最大深度參考值。 |
| `max_width` | `5` | 同一 layer 最大寬度參考值。 |
| `edge_probability` | `0.1` | 額外 edge 機率。 |
| `allow_skip_edges` | `False` | 不允許跳層邊。 |
| `allow_early_branch_end` | `True` | 分支可以提早結束。 |
| `single_source` | `True` | 單一 source node。 |
| `single_sink` | `False` | 不強制單一 sink node。 |

### 1.3 Expert / Task Metadata

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `num_experts` | `15` | 系統總 expert 數量。 |
| `task_deadline_seconds_range` | `(60, 120)` | 每個 task deadline，單位秒。 |
| `num_iot_features` | `60` | 全域 IoT feature index 數量。 |
| `iot_features_per_node_range` | `(5, 10)` | 每個 DAG node 需要的 IoT feature 數量範圍。 |

每個 node 會產生：`prompt`、`required_data.upstream_outputs`、`required_data.iot_features`、`gating_weights`、`expert_confidence`、`reconstruction_loss`、`calibration_losses`。

### 1.4 Performance Loss

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `loss_threshold` | `6.0` | 固定 q_hat。若不是 `None`，就不用 calibration quantile。 |
| `lambda_reconstruction` | `0.5` | reconstruction loss 權重。 |
| `calibration_alpha` | `0.1` | 若使用 calibration quantile，違反風險 alpha。 |
| `reconstruction_sigma` | `1.0` | reconstruction loss sigma。 |
| `reconstruction_error_range` | `(0.8, 1.5)` | 產生 node reconstruction loss 的誤差範圍。現在同一 node 內每個 feature loss 視為相同。 |
| `num_calibration_samples` | `20` | calibration loss 樣本數。 |
| `calibration_loss_range` | `(1.0, 4.0)` | synthetic calibration loss 範圍。 |

目前因為 `loss_threshold = 6.0`，所以 `q_hat_j^i = 6.0`。

### 1.5 Method Switches

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `run_rsma_scheduler` | `False` | 不跑舊 RSMA scheduler。 |
| `run_hybrid_topk_gssgd_backhaul` | `True` | 跑 Top-K + GSSGD DBG hybrid。 |
| `run_hybrid_topk_distance_backhaul` | `True` | 跑 Top-K + Distance Grouping hybrid。 |
| `run_wdmoe_gssgd` | `True` | 跑 WDMoE expert selection + GSSGD DBG grouping。 |
| `run_wdmoe_distance` | `True` | 跑 WDMoE expert selection + Distance/K-means grouping。 |

### 1.6 Hybrid / Baseline Parameters

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `topk_k` | `2` | 每個 subtask 初始選 Top-2 experts。 |
| `topk_rank_by` | `gating` | 目前以 gating weight 排序選 expert。 |
| `distance_grouping_clusters_per_server` | `None` | K-means K 由候選 IoT 數量與 group size 推出。 |
| `distance_grouping_kmeans_iterations` | `20` | K-means 迭代次數。 |

注意：目前 Top-K 排序用 `gating_weights`，但 performance probability 仍然用 `gating_weights * expert_confidence` 計算。

### 1.7 RSMA / Edge Network

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `num_edge_servers` | `9` | edge server 數量。 |
| `num_iot_devices` | `100` | IoT device 數量。 |
| `area_size` | `1000.0` | 網路區域大小；IoT 會在此區域內產生連續 `(x, y)` 座標。 |
| `cell_radius` | `300.0` | server 覆蓋半徑。 |
| `num_antennas` | `4` | server 端天線數。 |
| `phase_angle` | derived | 每個 `(IoT n, server s)` 的幾何角度，以 MEC/server 水平軸為 0，逆時針量到 IoT 位置：`atan2(y_n-y_s, x_n-x_s)` 並正規化到 `[0, 2pi)`。 |
| `distance` | derived | 每個 `(IoT n, server s)` 的歐氏距離，由 IoT/server 座標計算。 |
| `channel_phase` | derived | complex channel vector 合成後的通道相位，只保留在輸出資料中，不再拿來當幾何角度。 |
| `beamforming_correlation_weight` | `0.0` | 歷史參數；SIoT 公式沒有這個額外 private channel 修正，所以目前關閉。 |
| `connected_servers_per_device` | `None` | IoT 可連到覆蓋範圍內所有 server；若為數字則限制數量。 |
| `iot_features_per_device_range` | `(2, 6)` | 每個 IoT device 持有的 feature 數量。 |
| `experts_per_server` | `4` | 每個 server 最多儲存 4 個 experts；目前全域不重複，每個 expert 只會存在一台 server。 |
| `server_gpu_memory` | `1024.0` | server GPU memory 基準值；實際每台 server 會依比例產生不同 memory。 |
| `default_wired_rate` | `1e9` | server-server wired/backhaul rate fallback；若某條 link 沒有指定 rate，才使用這個值。 |
| `wired_rate_range` | `(5e8, 1.5e9)` | 每條 server-server wired/backhaul link 會在這個範圍內用 random seed 產生不同 rate。 |
| `uplink_time_budget_seconds` | `None` | 若指定，所有 group 使用固定 uplink budget。 |
| `bandwidth_time_fraction` | `1.0` | 備援值：只有當 group 找不到任何 dependent subtask 時，才用 `min(deadline) * 1.0` 當 uplink budget；正常 hybrid 會用該 group 服務的子任務最緊剩餘時間。 |
| `min_bandwidth` | `0.0` | bandwidth 下限。 |
| `wavelength` | `0.125` | SIoT DB phase term 使用的波長。 |
| `default_feature_bits` | `12000.0` | SIoT `data_quantity_constraint=12000`，每個 feature 的資料量。 |
| `noise_power` | `1e-18` | SIoT 公式中 SINR denominator 使用的 noise power。 |
| `common_power_ratio` | `0.6` | SIoT `cmn_ratio=0.6`；private stream 使用剩下 `0.4`。 |
| `max_device_power` | `1.2589e-3` | IoT device power，對齊 SIoT 參考碼中的 `power=1.2589e-3`。 |
| `max_group_size` | `4` | SIoT `count_limit=4`，每個 RSMA group 最多 4 個 IoT。 |

### 1.8 Objective Unit Cost

| 參數 | 目前值 | 意義 |
|---|---:|---|
| `c_bw` | `1e-1` | bandwidth unit cost。 |
| `c_act` | `60.0` | active server-expert unit cost。 |
| `c_fwd` | `0.3` | forwarding event unit cost。 |

目前 total cost 會同時看 activation、bandwidth、forwarding；forwarding 仍是按次數算，但 `c_fwd=0.3`。

## 2. Symbols

| 符號 | 程式概念 |
|---|---|
| `i` | task index。 |
| `j` | subtask/node index。 |
| `s` | edge server。 |
| `p` | expert。 |
| `n` | IoT device。 |
| `g` | IoT RSMA group。 |
| `v_j^i` | task i 的 subtask j。 |
| `G_p(v_j^i)` | gating weight。 |
| `pi(p | v_j^i)` | expert confidence。 |
| `P_sel` | selected expert set。 |
| `F_j^i` | subtask 需要的 IoT feature set。 |
| `F_hat_j^i` | 已由 selected groups 傳到的 feature set。 |

## 3. Current Hybrid Algorithm

目前兩個比較用 hybrid 方法都放在 `compared_method/`：`hybrid_topk_distance_backhaul.py` 與 `hybrid_topk_gssgd_backhaul.py`。`our_alg/` 保留給你自己的演算法草稿，不放比較方法。

### Step 1: Top-K Expert Selection

對每個 subtask：

1. 依 `gating_weights` 排序 expert。
2. 選前 `topk_k = 2` 個 experts。
3. 對每個 expert，找一個存有該 expert 且 GPU memory 可行的 server。
4. 若該 server 尚未 active 該 expert，會計入 activation。

目前選 server 的 heuristic：

```text
score = activation_penalty + expert_latency + used_memory / server_gpu_memory
```

選 score 最小的 server。

### WDMoE-Based DAG-Aware Expert Selection

目前 WDMoE 已改成 **每個 node/subtask 自己計算 WLR**，不是用整個 graph 的平均 WLR 來決定所有 node。

對每個 task graph `G_i`，仍然依 DAG topological order 處理 node，但每個 node `v_j^i` 的判斷獨立進行：

1. 對目前 node 先用普通 Top-K 建立 baseline selection。
2. 對同一個 node 建立完整 expert latency vector：

```text
latency_p(v_j^i) = predecessor forwarding time + expert_p inference time
```

3. 計算該 node 的 baseline WLR：

```text
WLR_base(i,j) = average_{p in TopK} G_p(v_j^i) / latency_p(v_j^i)
```

4. 設 `theta = alpha`，目前 `alpha = 0.5`。
5. 對該 node 計算完整 gating weight vector 與 latency vector 的 cosine similarity：

```text
S_j^i = cosine(G(v_j^i), latency(v_j^i))
```

6. 若 `S_j^i <= theta`，WDMoE 會從 Top-K 裡移除 gating score 最低的 expert，因此最後變成 `K-1` 個 expert。
7. 針對該 node 計算 candidate WLR：

```text
WLR_current(i,j) = average_{p in selected} G_p(v_j^i) / latency_p(v_j^i)
```

8. 若：

```text
WLR_current(i,j) / WLR_base(i,j) > gamma
```

則該 node 停止 threshold search，保留目前 selection；否則 `theta = theta + Delta_theta` 後只重跑該 node 的 selection。

最後，對每個 `(s,p)`：

```text
Y_(s,p) = 1, if any X_(i,j)^(s,p) = 1
Y_(s,p) = 0, otherwise
```

輸出的 `results/wdmoe_gssgd_result.json` 與 `results/wdmoe_distance_result.json` 會在 `network_model.wdmoe_parameters.task_metrics` 中記錄每個 graph 的平均 `baseline_wlr`、`selected_wlr`、`wlr_ratio`，以及 `nodes` 裡每個 node 自己的 WLR、similarity、final_threshold、met_gamma、選擇 expert 數量。
### Step 2: Candidate Distance Grouping

先建立候選 IoT groups，不代表全部啟用。

對每個 server：

1. 找該 server 底下的 local IoT devices。
2. 取 server 可能需要的 feature 與 local IoT features 的交集。
3. 候選 IoT 必須至少擁有一個 local required feature。
4. 以 server 為原點建立座標：

```text
x_n,s = d(n,s) cos(theta_geo(n,s))
y_n,s = d(n,s) sin(theta_geo(n,s))
```

5. 決定 K：

```text
K = ceil(num_candidate_iot / max_group_size)
```

6. 對候選 IoT 跑 K-means。
7. 若 group size 超過 `max_group_size`，再拆成小組。

### Step 3: Performance Loss Repair by IoT Group

對每個 subtask：

1. 先用 Top-K experts 算 selected probability。
2. 若 performance loss 已滿足，該 subtask 不一定啟用 IoT group。
3. 若 loss 不滿足，從候選 IoT groups 中選 group。
4. 優先選「加入後可以滿足 loss 且 incremental cost 最小」的 group。
5. 若沒有單一 group 可以滿足，就選 coverage/cost 最好的 group 繼續補。
6. 如果 group 都補完還是不滿足，才 fallback 補 expert。

目前 group incremental cost：

```text
DeltaCost(g) = DeltaBandwidthCost(g) + DeltaForwardingCost(g)
DeltaBandwidthCost(g) = c_bw * required_bandwidth(g)
```

若 group 所在 server 與 target server 不同：

```text
DeltaForwardingCost += c_fwd
```

### Step 4: Data Requirement

每個 subtask 最後真正要傳的 features 來自被選中的 IoT groups：

```text
selected_features(i,j) = union of selected group features intersect required_features(i,j)
```

注意：現在不是每個 subtask 都一定傳 IoT data。若 Top-K experts 已滿足 loss，該 subtask 的 selected features 可能為空。

### Step 5: Backhaul

如果 target server 需要某 feature，但該 feature 的 group 在另一台 server，則加入 backhaul edge：

```text
(source_server, group_id, target_server)
```

目前 backhaul 是以 group 為單位；一個 backhaul edge 會把該 group 傳到 target server。

### Step 6: Bandwidth Derivation

每個 active group 只計算一次 bandwidth：

```text
BandwidthCost = c_bw * sum_g B_g
```

其中 `B_g` 根據該 group 的資料量與時間 budget 推回來。

## 4. Performance Loss Formula

目前 constraint：

```text
L(P_sel, v_j^i) <= q_hat_j^i
```

目前因 `loss_threshold = 6.0`：

```text
q_hat_j^i = 6.0
```

### 4.1 Selected Expert Probability

```text
P(P_sel | v_j^i)
= sum_{p in P_sel} G_p(v_j^i) * pi(p | v_j^i)
```

### 4.2 Reconstruction Loss

目前假設：每個 feature 的 reconstruction loss 貢獻相同。

```text
missing_count = |F_j^i \ F_hat_j^i|
L_rec = L_rec_full * missing_count / max(|F_j^i|, 1)
```

若所有 required features 都被傳到：

```text
L_rec = 0
```

若完全沒傳：

```text
L_rec = L_rec_full
```

### 4.3 Performance Loss

```text
L(P_sel, v_j^i)
= 1 / P(P_sel | v_j^i) + lambda_reconstruction * L_rec
```

目前：

```text
lambda_reconstruction = 0.5
```

### 4.4 Calibration Threshold

若 `loss_threshold = None`，則使用：

```text
q_hat = sorted(calibration_losses)[ceil((|D_cal| + 1)(1 - alpha)) - 1]
```

目前因為 `loss_threshold = 6.0`，這段沒有生效。

## 5. RSMA Rate / Bandwidth Formula

### 5.1 Common Stream SINR

```text
h_n,s is generated with SIoT channel coefficients:
PL_dB(d_n,s) = 50 + 15 log10(d_n,s)
loss_n,s = 10^(PL_dB / 10)
shadow_n,s = 10^(N(0,1) / 10)
h_vector,n,s = complex Gaussian / (loss_n,s * shadow_n,s)
h_n,s = ||h_vector,n,s||

common_signal = sum_{n in g} |h_n,s|^2 * P_common,n
private_interference = sum_{n in g} |h_n,s|^2 * P_private,n
SINR_common = DB_gain(g) * common_signal / (private_interference + noise_power)
R_common_eff = log2(1 + SINR_common)
```

目前有加入 distributed beamforming gain，其中 `theta_geo(n,s)` 是相對 MEC 水平軸逆時針量測的幾何角度，不使用隨機 channel phase：

```text
DB_gain(g) = |sum weighted phase terms| / sum magnitudes
```

### 5.2 Private Stream SINR

```text
signal_n = |h_n,s|^2 * P_private,n
interference_n = sum_{m in g, m != n} |h_m,s|^2 * P_private,m * corr(n,m,s)
SINR_private,n = signal_n / (interference_n + noise_power)
R_private_eff,n = log2(1 + SINR_private,n)
```

其中：

```text
private stream 目前直接使用 h_n,s = channel_gain(n,s)
```

### 5.3 Bandwidth from Time Budget

```text
B_common = common_volume / (budget * R_common_eff)
B_private,n = private_volume,n / (budget * R_private_eff,n)
B_g = max(min_bandwidth, B_common, max_n B_private,n)
```

budget：

```text
若 uplink_time_budget_seconds 不為 None：
    budget = uplink_time_budget_seconds
否則：
    budget = min(dependent subtask remaining time)
fallback = min(task deadlines) * bandwidth_time_fraction，且目前 hybrid 主要使用 group 所服務 subtask 的剩餘 deadline
```

目前：

```text
bandwidth_time_fraction = 1.0
min_bandwidth = 0.0
```

## 6. Timing Formula

### 6.1 Uplink Time

```text
T_uplink(g) = max(common_volume / common_rate,
                  max_n private_volume,n / private_rate,n)
```

### 6.2 Feature Ready Time

```text
T_feature_ready(i,j,s)
= max over selected required features f of earliest arrival time of f at s
```

若 feature 在 local group：

```text
arrival = T_uplink(group)
```

若 feature 需要 backhaul：

```text
arrival = T_uplink(group) + group_feature_volume / wired_rate(source_server, target_server)
```

### 6.3 Predecessor Ready Time

```text
T_pred_ready(i,j,s)
= max over predecessors and source servers
  [finish_time(pred, source_server) + output_bits / wired_rate(source, s)]
```

### 6.4 Start / Finish Time

```text
T_start(i,j,s) = max(T_feature_ready(i,j,s), T_pred_ready(i,j,s))
T_finish(i,j,s) = T_start(i,j,s) + computation_time(i,j,s)
T_task_finish(i) = max_j,s T_finish(i,j,s)
```

同 layer / independent subtasks 可平行，因為 start time 只受 feature ready 與 predecessor ready 限制。

## 7. Objective Function

```text
TotalCost = ActivationCost + BandwidthCost + ForwardingCost
```

### 7.1 Activation Cost

```text
ActivationCost = c_act * number of unique (server, expert) activated
```

目前 `c_act = 60.0`。

### 7.2 Bandwidth Cost

```text
BandwidthCost = c_bw * sum_g B_g
```

每個 active group 的 bandwidth 只算一次。目前 `c_bw = 1e-2`。

### 7.3 Forwarding Cost

Forwarding cost 包含兩種來源。

第一種：DAG dependency output forwarding。

若 parent subtask 在 server `s1`，child subtask 在 server `s2`，且 `s1 != s2`：

```text
ForwardingCost += c_fwd
```

第二種：IoT group backhaul。

若 group 在 source server，但 target server 需要它：

```text
ForwardingCost += c_fwd
```

目前 `c_fwd = 0.3`。Forwarding 目前按次數算，不乘資料量、不乘時間。

## 8. Constraints Currently Checked

### C1 Deadline Constraint

```text
T_task_finish(i) <= deadline_i
```

### C2 GPU Memory Constraint

```text
sum active expert memory on server s <= server_gpu_memory_s
```

### C3 Performance Loss Constraint

```text
L(P_sel, v_j^i) <= q_hat_j^i
```

### C4 RSMA Group Size Constraint

```text
|group.devices| <= max_group_size
```

### C5 IoT Grouping Constraint

同一個 IoT device 不可出現在多個 active groups。

### C6 Feature / Dependency Execution Order

```text
T_start(i,j,s) >= T_feature_ready(i,j,s)
T_start(i,j,s) >= T_pred_ready(i,j,s)
```

若 feature 或 predecessor 無法到達，會產生 violation。

### C7 Minimum Rate Constraint

```text
common_rate(g) >= min_rate   若 group 有 common volume
private_rate(g,n) >= min_rate
```

## 9. Current Important Behavior To Check

這些是目前程式可能需要你確認是否符合研究設定的地方。

### 9.1 Top-K 可能造成 subtask 分散到多台 server

目前每個 expert 各自找可行 server，所以同一個 subtask 可能被放到多台 server。

這會讓 DAG predecessor output forwarding 次數變多。若 parent 與 child 都分散到多台 server，forwarding 會接近：

```text
number_parent_servers * number_child_servers
```

所以目前 total cost 可能被 forwarding 主導。

### 9.2 Subtask 不一定會傳 IoT data

如果 Top-K experts 已經讓 performance loss 滿足：

```text
L <= q_hat
```

則該 subtask 可能不選任何 IoT group，也就是 selected features 為空。

若你的模型要求每個 subtask 至少必須接收一個 IoT group，需要額外加 constraint。

### 9.3 Loss Repair 現在以 Group 為單位

現在不是單一 feature repair，而是：

```text
Top-K 不滿足 loss -> 找 IoT group -> group union features 降低 reconstruction loss
```

### 9.4 Forwarding Cost 目前不考慮資料量

目前 forwarding cost 是次數成本：

```text
cost = c_fwd * forwarding_event_count
```

沒有乘 output size、group size 或 backhaul time。

### 9.5 Bandwidth Cost 可能偏小

目前 bandwidth cost 是：

```text
c_bw * sum active group bandwidth
```

若 active groups 很少，bandwidth cost 就會很小。

若希望 IoT data 更常被傳，可能需要：

- 降低 `loss_threshold`
- 降低 `topk_k`
- 限制 selected experts 的 server 數
- 要求每個 subtask 至少選一個 IoT group

## 10. Output Files

每次執行會輸出：

- `generated_dags/dag_N.json`
- `generated_dags/dag_N.graphml`
- `generated_dags/dag_N.png`
- `results/hybrid_topk_distance_backhaul_result.json`
- `results/hybrid_topk_gssgd_backhaul_result.json`
- `results/wdmoe_gssgd_result.json`
- `results/wdmoe_distance_result.json`

結果 JSON 中重要欄位：

| 欄位 | 意義 |
|---|---|
| `objective` | total / activation / bandwidth / forwarding cost。 |
| `cost_breakdown` | usage、unit cost、cost。 |
| `expert_placement` | 每個 server activated experts。 |
| `subtask_assignment` | 每個 subtask 被分配到哪些 `(server, expert)`。 |
| `subtask_data_requirements` | 每個 subtask/server 最後需要哪些 features。 |
| `selected_probability` | 每個 subtask 的 selected expert probability。 |
| `required_probability` | 在目前 selected features 下滿足 loss 所需的最低 probability。 |
| `reconstruction_loss` | 每個 subtask 最終 reconstruction loss。 |
| `performance_loss` | 每個 subtask 最終 performance loss。 |
| `rsma_groups` | active IoT groups。 |
| `rsma_group_bandwidths` | 每個 active group 的 bandwidth。 |
| `backhaul_plan` | group 從 source server backhaul 到 target server 的計畫。 |
| `violations` | constraint violations。 |

## 11. Main Files

| 檔案 | 功能 |
|---|---|
| `main.py` | 執行入口。 |
| `config.py` | 所有 input values。 |
| `generate_dags.py` | 產生 DAG。 |
| `task_metadata.py` | 產生 node metadata。 |
| `rsma_integration.py` | 建立 tasks、experts、servers、IoT devices、channels。 |
| `utils/formulation.py` | formulas、objective、timing、constraints。 |
| `compared_method/hybrid_topk_distance_backhaul.py` | Top-K + Distance Grouping hybrid。 |
| `compared_method/hybrid_topk_gssgd_backhaul.py` | Top-K + GSSGD DBG hybrid。 |
| `compared_method/WDMoE.py` | WDMoE Algorithm 1 expert selection，搭配 Distance 或 GSSGD grouping。 |
| `our_alg/` | 你自己的演算法區，目前不放比較方法。 |













