# RSMA_MoE

這個專案用來產生 DAG 型 task graph，並比較多種「Expert selection + IoT grouping + backhaul」方法在 RSMA-MoE 環境下的成本與限制式違反情況。

目前主流程會比較四種方法：

1. `Hybrid TopK+GSSGD`
2. `Hybrid TopK+Distance`
3. `WDMoE+GSSGD`
4. `WDMoE+Distance`

執行方式：

```bash
python main.py
```

主要參數集中在 `config.py` 的 `ExperimentConfig`。

## 1. 專案結構

| 檔案 / 資料夾 | 作用 |
|---|---|
| `main.py` | 程式入口，呼叫 `run_experiment(ExperimentConfig())`。 |
| `config.py` | 所有主要實驗參數。 |
| `experiment_runner.py` | 產生 DAG、建立 network、執行比較方法、統計平均結果。 |
| `generate_dags.py` | 產生 DAG task graph。 |
| `task_metadata.py` | 為每個 DAG node 產生 prompt、required features、gating weights、expert confidence、loss metadata。 |
| `rsma_integration.py` | 建立 experts、servers、IoT devices、channel、tasks。 |
| `utils/formulation.py` | 主要公式、rate、bandwidth、timing、objective、constraints。 |
| `compared_method/hybrid_topk_distance_backhaul.py` | Top-K + K-means distance grouping + backhaul。 |
| `compared_method/hybrid_topk_gssgd_backhaul.py` | Top-K + GSSGD DBG grouping + backhaul。 |
| `compared_method/WDMoE.py` | WDMoE expert selection，接 Distance 或 GSSGD grouping。 |
| `compared_method/GSSGD/` | GSSGD DBG 的 phase 1-3。 |
| `SIoT_RSMA/` | SIoT-RSMA 參考程式碼，保留作為對照來源。 |
| `generated_dags/` | DAG JSON / GraphML / PNG 輸出。 |
| `results/` | 各方法結果 JSON。 |

## 2. 目前 Input 設定

以下數值對應目前 `config.py`。

### 2.1 Reproducibility / Output

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `auto_random_seed` | `True` | 每次 run 自動產生 DAG seed 與 network seed。 |
| `num_runs` | `100` | 重複實驗次數，最後輸出平均結果。 |
| `num_dags` | `3` | 每次 run 生成 3 個 Task graph。若 activation cost 再度飽和，可以降到 `1` 來觀察單一 graph 的 expert 啟用差異。 |
| `random_seed` | `42` | `auto_random_seed=False` 時使用的 DAG seed 起點。 |
| `network_random_seed` | `1` | `auto_random_seed=False` 時使用的 network seed 起點。 |
| `output_dir` | `generated_dags/` | DAG artifact 輸出位置。 |
| `results_dir` | `results/` | 結果 JSON 輸出位置。 |
| `image_dpi` | `180` | DAG 圖片解析度。 |

### 2.2 DAG / Task Graph

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_nodes` | `6` | 每個 task graph 的 node/subtask 數量。 |
| `num_branches` | `4` | 分支數量。 |
| `branch_length_range` | `(1, 5)` | 每個分支長度範圍，允許分支深度不同。 |
| `depth` | `5` | DAG 最大深度參考值。 |
| `max_width` | `5` | 同一 layer 最大寬度。 |
| `edge_probability` | `0.1` | 額外 edge 產生機率。 |
| `allow_skip_edges` | `False` | 不允許跨層 skip edge。 |
| `allow_early_branch_end` | `True` | 允許分支提早結束。 |
| `single_source` | `True` | 使用單一 source node。 |
| `single_sink` | `False` | 不強制單一 sink node。 |

### 2.3 Expert / Node Metadata

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_experts` | `36` | 全域 expert 數量；目前等於 `num_edge_servers * experts_per_server`，讓 activation 不會因 expert pool 太小而容易全部啟用。 |
| `expert_memory_range` | `(128.0, 256.0)` | 每個 expert 的 model size 隨機範圍。 |
| `task_deadline_seconds_range` | `(60, 120)` | 每個 task 的 deadline 秒數範圍。 |
| `num_iot_features` | `100` | 全域 IoT feature index 數量。 |
| `iot_features_per_node_range` | `(1, 3)` | 每個 node 需要的 IoT feature 數量。 |
| `gating_peak_count_range` | `(2, 4)` | 每個 subtask 會有 2 到 4 個 gating 高峰 experts。 |
| `gating_peak_mass_range` | `(0.65, 0.85)` | 高峰 experts 合計拿走 65% 到 85% 的 gating mass，其餘 experts 分享 tail mass。 |

Gating network 目前不是平均亂數，而是 peaked distribution：每個 subtask 會抽出少數高峰 experts，讓它們合計取得大部分 gating mass。`gating_weights` 仍會正規化且總和等於 1。

每個 node 會包含：

- `prompt`
- `required_data.upstream_outputs`
- `required_data.iot_features`
- `gating_weights`
- `expert_confidence`
- `reconstruction_loss`
- `calibration_losses`

### 2.4 Performance Loss

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `loss_threshold` | `5.0` | 每個 node 的 loss bound `q_hat`。數值越小代表 Performance loss constraint 越嚴格；若為 `None`，改用 calibration quantile。 |
| `lambda_reconstruction` | `0.8` | Reconstruction loss 權重。 |
| `calibration_alpha` | `0.1` | Calibration tolerated violation risk。 |
| `reconstruction_sigma` | `1.0` | Reconstruction loss 中的 sigma。 |
| `reconstruction_error_range` | `(0.8, 1.5)` | 產生 synthetic reconstruction loss 的範圍。 |
| `num_calibration_samples` | `20` | Calibration samples 數量。 |
| `calibration_loss_range` | `(1.0, 4.0)` | Synthetic calibration loss 範圍。 |

### 2.5 Method Switches

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `run_rsma_scheduler` | `False` | 舊的 JRGEP/RSMA scheduler baseline，預設不跑。 |
| `run_hybrid_topk_gssgd_backhaul` | `True` | 跑 Top-K + GSSGD DBG + backhaul。 |
| `run_hybrid_topk_distance_backhaul` | `True` | 跑 Top-K + Distance/K-means + backhaul。 |
| `run_wdmoe_gssgd` | `True` | 跑 WDMoE + GSSGD DBG + backhaul。 |
| `run_wdmoe_distance` | `True` | 跑 WDMoE + Distance/K-means + backhaul。 |

### 2.6 Expert Selection / Grouping

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `topk_k` | `4` | Top-K baseline selects Top-4 experts per subtask. WDMoE can prune at most one expert per subtask in each theta trial. |
| `topk_rank_by` | `gating` | Expert ranking 使用 gating weight。Performance probability 仍使用 `gating_weight * expert_confidence`。 |
| `distance_grouping_clusters_per_server` | `None` | 若為 `None`，K-means 的 K 由 candidate IoT 數量與 `max_group_size` 推出。 |
| `distance_grouping_kmeans_iterations` | `20` | K-means iteration 次數。 |
| `gssgd_beamforming_gain_threshold` | `0.0` | GSSGD 的硬性 DB gain threshold 目前關閉；RSMA rate 公式仍吃 DB gain。 |

### 2.7 WDMoE

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `wdmoe_initial_threshold` | `0.8` | WDMoE cosine similarity threshold。 |
| `wdmoe_threshold_step` | `0.05` | Theta increment used by the paper-style iterative search. |
| `wdmoe_max_threshold` | `1.0` | Maximum theta value for the iterative search. |
| `wdmoe_wlr_target_ratio` | `1.05` | Task/graph-level WLR ratio gamma. |

WDMoE now follows the paper-style task/graph-level iterative theta search. Each theta trial reruns all subtasks from the same pre-task state.

### 2.8 RSMA / Edge Network

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_edge_servers` | `12` | Edge servers 數量。 |
| `num_iot_devices` | `100` | IoT devices 數量。 |
| `area_size` | `1000.0` | IoT/server 位置平面大小。 |
| `cell_radius` | `220.0` | Server 接收覆蓋半徑。IoT 可連到所有覆蓋自己的 servers。 |
| `num_antennas` | `4` | Server 端天線數。Channel gain 目前是 scalar SIoT-like channel，不是每根天線一個 gain。 |
| `iot_features_per_device_range` | `(4, 9)` | 每個 IoT 儲存的 feature 數量範圍。 |
| `iot_server_feature_overlap_ratio` | `0.25` | 鄰近 server feature pool 的重疊比例。 |
| `iot_global_random_feature_fraction` | `0.15` | IoT feature 中保留的全域隨機比例，使 feature 與位置正相關但不完全相關。 |
| `experts_per_server` | `3` | 每台 server 最多儲存的 experts 數量。因為 expert 不重複配置，若 `num_experts < num_edge_servers * experts_per_server`，部分 slot 會是空的。 |
| `server_gpu_memory_range` | `(768.0, 1024.0)` | 每台 server GPU capacity 隨機範圍。目前下限設為 `3 * max_expert_memory`，避免 GPU 容量不足干擾 performance loss constraint 測試。 |
| `wired_rate_range` | `(5e8, 1.5e9)` | 每條 server-server wired/backhaul link 隨機 rate 範圍。 |
| `bandwidth_time_fraction` | `1.0` | Fallback bandwidth budget 使用最小 task deadline 的比例；有 dependent subtask 時使用更嚴格的 group-specific remaining time。 |
| `wavelength` | `0.125` | DB phase term 使用的波長。 |
| `default_feature_bits` | `1200000.0` | Feature size 的 fallback。 |
| `feature_bits_range` | `(800000.0, 2400000.0)` | 每個 feature 的資料量隨機範圍，實際 group payload 會把 group 內 IoT 所有 features 的 bits 做 union 後加總。 |
| `noise_power` | `1e-18` | SINR denominator 的 noise power。 |
| `common_power_ratio` | `0.6` | IoT power 中 common stream 的比例；private stream 為 `0.4`。 |
| `max_device_power` | `1.2589e-3` | IoT device power，對齊 SIoT 設定。 |
| `max_group_size` | `4` | 每個 RSMA IoT group 最多 4 個 devices。 |
| `min_rate` | `1.0` | C7 minimum transmission rate `R_min`。 |

### 2.9 Objective Unit Cost

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `c_bw` | `1e-1` | Bandwidth unit cost。 |
| `c_act` | `60.0` | Server-expert activation unit cost。 |
| `c_fwd` | `0.3` | Forwarding event unit cost。Forwarding 只算次數，不乘時間。 |

## 3. 四種比較方法

### 3.1 Hybrid TopK+Distance

流程：

1. 每個 subtask 根據 gating weight 選 Top-K experts。
2. 為每個 expert 選一台有儲存該 expert 且 GPU memory 可行的 server。
3. 根據 server 上被分配到的 subtasks，整理該 server 需要的 IoT features。
4. 對每個 server，只拿該 server 底下且擁有所需 features 的 IoT 做 K-means。
5. K-means 使用相對 server 的幾何座標：

```text
x(n,s) = d(n,s) cos(theta(n,s))
y(n,s) = d(n,s) sin(theta(n,s))
K = ceil(number_of_candidate_IoT / max_group_size)
```

6. Candidate groups are generated first; only groups selected by loss repair transmit.
7. 如果 target server 需要某個 feature，但 feature 只在其他 server 的 group 中，則建立 backhaul edge。
8. 計算 bandwidth、timing、cost、violations。

### 3.2 Hybrid TopK+GSSGD

流程：

1. Expert selection 同 Top-K。
2. 每個 server 整理自己需要的 local required features。
3. GSSGD DBG phase 1 用 CCM/SPCCM 選 IoT。
4. GSSGD DBG phase 2 用 SPCI 做 DBG grouping。
5. GSSGD DBG phase 3 做 SER elimination/replacement refinement。
6. GSSGD 選出的 groups 一開始就傳輸。
7. 接 backhaul 與共同 evaluator 算 cost/constraints。

注意：

- 只保留 DBG，DCG 已移除。
- `gssgd_beamforming_gain_threshold=0.0`，所以硬性 `gamma >= 0.8` constraint 目前關閉。
- RSMA rate 公式仍使用 distributed beamforming gain。

### 3.3 WDMoE+Distance / WDMoE+GSSGD

WDMoE only replaces expert selection. The following IoT grouping stage is still either Distance/K-means or GSSGD DBG.

Current WDMoE logic follows the paper-style Algorithm 1 theta search:

1. For each DAG/task, run ordinary Top-K over all subtasks to compute the baseline WLR.

```text
WLR_base(i) = average_{v_j in G_i} average_{p in TopK(v_j)} G_p(v_j^i) / latency_p(v_j^i)
```

2. Set `theta = wdmoe_initial_threshold`.
3. Each theta trial clears the temporary selection for the current task and reruns the whole DAG from the same pre-task activation/memory state.
4. For each node, build the full latency vector:

```text
latency_p(v_j^i) = predecessor_forwarding_time + expert_p_inference_time
```

5. Compute cosine similarity between the full gating vector and full latency vector:

```text
S_j^i = cosine(G(v_j^i), latency(v_j^i))
```

6. If `S_j^i <= theta`, remove the lowest-gating expert from the current Top-K set. Therefore one subtask can only move from K to K-1 within one theta trial.
7. After finishing the whole DAG, compute the current task WLR:

```text
WLR_current(i) = average_{v_j in G_i} average_{p in selected(v_j)} G_p(v_j^i) / latency_p(v_j^i)
```

8. Keep the current theta trial when the paper stopping condition is satisfied:

```text
WLR_current(i) / WLR_base(i) > wdmoe_wlr_target_ratio
```

9. Otherwise set `theta = theta + wdmoe_threshold_step` and rerun the whole DAG until the ratio passes or theta reaches `wdmoe_max_threshold`.
10. Because this project also has per-node performance-loss bounds, a K-1 pruning is reverted to Top-K if the node cannot satisfy its loss bound even after all required features are available.

## 4. Performance Loss

每個 node/subtask 都檢查：

```text
L(P_sel, v_j^i) <= q_hat_j^i
```

### 4.1 Selected Expert Probability

```text
P(P_sel | v_j^i)
= sum_{p in P_sel} G_p(v_j^i) * pi(p | v_j^i)
```

其中：

- `G_p(v_j^i)` 是 gating weight。
- `pi(p | v_j^i)` 是 expert confidence。
- Top-K 選擇不允許重複 expert。

### 4.2 Reconstruction Loss

目前每個 feature 的 reconstruction contribution 視為相同。若 subtask 需要的 features 中有一部分沒有被 selected groups 傳到，則：

```text
missing_count = |F_j^i \ F_hat_j^i|
L_rec = L_rec_full * missing_count / max(|F_j^i|, 1)
```

若 required features 全部滿足，`L_rec = 0`。

### 4.3 Total Performance Loss

```text
L(P_sel, v_j^i)
= 1 / P(P_sel | v_j^i) + lambda_reconstruction * L_rec
```

若 `loss_threshold` 不是 `None`：

```text
q_hat_j^i = loss_threshold
```

若 `loss_threshold = None`，使用 calibration losses 的 conformal quantile。

## 5. RSMA Rate / Bandwidth

### 5.1 IoT / Server Geometry

Server 位置以 grid 放在 `area_size x area_size` 平面。IoT 隨機落點後，只要與 server 距離小於 `cell_radius`，就可連線到該 server。

每個 IoT 對每個可連線 server 都有：

```text
d(n,s) = distance between IoT n and server s
theta(n,s) = atan2(y_n - y_s, x_n - x_s), measured counterclockwise from server horizontal axis
```

### 5.2 Channel Model

目前使用 SIoT-like scalar channel：

```text
PL_dB(d_n,s) = 50 + 15 log10(d_n,s)
loss_n,s = 10^(PL_dB / 10)
shadow_n,s = 10^(N(0,1) / 10)
h_n,s = |complex Gaussian| / (loss_n,s * shadow_n,s)
```

`num_antennas=4` 表示 server 端天線數設定，但目前 channel gain 是每個 `(IoT n, server s)` 一個 scalar value。

### 5.3 Distributed Beamforming Gain

```text
phase(n,s) = 2 pi d(n,s) cos(theta(n,s)) / wavelength
DB_gain(g) = |sum_{n in g} w_n exp(j phase(n,s))| / sum_{n in g} w_n
w_n = sqrt(P_common,n * common_message_ratio(g)) * h_n,s
```

### 5.4 Common / Private SINR

Common stream：

```text
common_signal = sum_{n in g} |h_n,s|^2 * P_common,n
private_interference = sum_{n in g} |h_n,s|^2 * P_private,n
SINR_common = DB_gain(g) * common_signal / (private_interference + noise_power)
```

Private stream：

```text
signal_n = |h_n,s|^2 * P_private,n
interference_n = sum_{m in g, m != n} |h_m,s|^2 * P_private,m * corr(n,m,s)
SINR_private,n = signal_n / (interference_n + noise_power)
```

### 5.5 SIoT Rate Mapping

`SINR_common` 和 `SINR_private` 會經過 `siot_rate_mapping()`，使用 SIoT 的離散 MCS/rate table，而不是單純 `log2(1+SINR)`。

```text
R_common_eff = siot_rate_mapping(SINR_common)
R_private_eff,n = siot_rate_mapping(SINR_private,n)
```

### 5.6 Bandwidth Derivation

每個 group 的 bandwidth 根據 group 依賴的 subtasks 中最嚴格的剩餘時間推回去：

```text
budget(g) = min_{(i,j) depends on g} [deadline_i - downstream_compute_time(i,j)]
```

若 group 沒有 dependent subtask，使用 fallback：

```text
fallback = min_task_deadline * bandwidth_time_fraction
```

Bandwidth：

```text
B_common = common_volume(g) / (budget(g) * R_common_eff)
B_private,n = private_volume(n,g) / (budget(g) * R_private_eff,n)
B_g = max(B_common, max_n B_private,n)
```

## 6. Timing

### 6.1 Uplink Time

```text
T_uplink(g) = max(common_volume(g) / common_rate(g),
                  max_n private_volume(n,g) / private_rate(n,g))
```

### 6.2 Feature Ready Time

```text
T_feature_ready(i,j,s)
= max over required selected features f of earliest arrival time of f at server s
```

Local group：

```text
arrival = T_uplink(g)
```

Backhaul group：

```text
arrival = T_uplink(g) + group_feature_volume(g) / wired_rate(source_server, target_server)
```

### 6.3 Predecessor Ready Time

```text
T_pred_ready(i,j,s)
= max over predecessor outputs
  [finish_time(pred, source_server) + output_bits / wired_rate(source_server, s)]
```

若 predecessor 與 child 在同一台 server，不需要 forwarding time。

### 6.4 Start / Finish Time

```text
T_start(i,j,s) = max(T_feature_ready(i,j,s), T_pred_ready(i,j,s))
T_finish(i,j,s) = T_start(i,j,s) + computation_time(i,j,s)
T_task_finish(i) = max_{j,s} T_finish(i,j,s)
```

同 layer 且沒有 dependency 阻塞的 subtasks 可以平行處理；不是把同 layer 全部串接。

## 7. Objective Function

```text
TotalCost = ActivationCost + BandwidthCost + ForwardingCost
```

Activation：

```text
ActivationCost = c_act * number_of_unique_active_server_expert_pairs
```

Bandwidth：

```text
BandwidthCost = c_bw * sum_g B_g
```

Forwarding：

```text
ForwardingCost = c_fwd * forwarding_event_count
```

Forwarding event 包含兩種：

1. DAG predecessor output 從一台 server 傳到另一台 server。
2. IoT group data 經 backhaul 傳到 target server。

Forwarding cost 只算次數，不乘時間。

## 8. Constraints

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
|g| <= max_group_size
```

### C5 IoT Grouping Constraint

```text
Each IoT device can appear in at most one active group.
```

### C6 Feature / Dependency Execution Order Constraint

```text
T_start(i,j,s) >= T_feature_ready(i,j,s)
T_start(i,j,s) >= T_pred_ready(i,j,s)
```

### C7 Minimum Transmission Rate Constraint

```text
R_cm(n,g^s) >= R_min, for each n in group g^s when common volume exists
R_priv(n,g^s) >= B(n,g^s) * R_min
```

目前實作中：

```text
R_min = min_rate
B(n,g)=1 if IoT n is in group g, otherwise 0
```

## 9. Output

每次執行會輸出：

- `generated_dags/dag_N.json`
- `generated_dags/dag_N.graphml`
- `generated_dags/dag_N.png`
- `results/hybrid_topk_gssgd_backhaul_result.json`
- `results/hybrid_topk_distance_backhaul_result.json`
- `results/wdmoe_gssgd_result.json`
- `results/wdmoe_distance_result.json`

每個 result JSON 主要包含：

| 欄位 | 說明 |
|---|---|
| `objective` | total / activation / bandwidth / forwarding cost。 |
| `cost_breakdown` | usage、unit cost、cost。 |
| `expert_placement` | 每台 server 啟用的 experts。 |
| `subtask_assignment` | 每個 subtask 分配到哪些 `(server, expert)`。 |
| `subtask_data_requirements` | 每個 subtask/server 需要的 features。 |
| `selected_probability` | 每個 subtask 的 selected expert probability。 |
| `required_probability` | 在目前 selected features 下滿足 loss 所需 probability。 |
| `reconstruction_loss` | 每個 subtask 的 reconstruction loss。 |
| `performance_loss` | 每個 subtask 的 total performance loss。 |
| `rsma_groups` | Active IoT groups。 |
| `rsma_group_bandwidths` | 每個 active group 的 bandwidth。 |
| `backhaul_plan` | 哪個 group 從 source server backhaul 到 target server。 |
| `violations` | C1-C7 constraint violations。 |
| `network_model` | 本次 network、topology、參數記錄。 |

## 10. 目前重要假設

1. Expert 不重複配置到多台 server，避免額外 server selection ambiguity。
2. IoT 只要在 server coverage radius 內就可連線。
3. IoT feature 與位置正相關，但保留全域隨機 feature，不完全由位置決定。
4. Group 傳輸單位是整個 IoT group；group 內 IoT 的所有 features 都形成 payload。
5. Distance grouping is a pure K-means baseline; candidate groups only transmit when selected by performance-loss repair.
6. GSSGD 只保留 DBG，不使用 DCG。
7. WDMoE uses task/graph-level iterative Algorithm 1; each subtask can prune at most one expert in a theta trial.
8. Bandwidth is charged once per active group; active groups are the groups selected by loss repair.
9. Forwarding cost 是 event count，不乘 forwarding time。
10. RSMA rate 使用 SIoT MCS table 與 DB gain。
