from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from generate_dags import DAGSpec


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    # Reproducibility / output.
    auto_random_seed: bool = True
    num_runs: int = 100
    num_dags: int = 10
    random_seed: int = 42
    network_random_seed: int = 1
    output_dir: Path = BASE_DIR / "generated_dags"
    results_dir: Path = BASE_DIR / "results"
    image_dpi: int = 180

    # DAG / task graph settings.
    common_dag_spec: DAGSpec = field(
        default_factory=lambda: DAGSpec(
            num_nodes=12,
            num_branches=4,
            branch_length_range=(1, 5),
            depth=5,
            max_width=5,
            edge_probability=0.1,
            allow_skip_edges=False,
            allow_early_branch_end=True,
            single_source=True,
            single_sink=False,
        )
    )

    # Expert / task metadata settings.
    num_experts: int = 12
    task_deadline_seconds_range: tuple[int, int] = (60, 120)
    num_iot_features: int = 60
    iot_features_per_node_range: tuple[int, int] = (5, 10)

    # Performance loss settings.
    loss_threshold: Optional[float] = 6.0
    lambda_reconstruction: float = 0.5
    calibration_alpha: float = 0.1
    reconstruction_sigma: float = 1.0
    reconstruction_error_range: tuple[float, float] = (0.8, 1.5)
    num_calibration_samples: int = 20
    calibration_loss_range: tuple[float, float] = (1.0, 4.0)

    # Which methods to run. The default workflow compares two hybrid algorithms.
    run_rsma_scheduler: bool = False



    run_hybrid_topk_gssgd_backhaul: bool = True
    run_hybrid_topk_distance_backhaul: bool = True
    run_wdmoe_gssgd: bool = True
    run_wdmoe_distance: bool = True

    # Shared Top-K and grouping settings.
    topk_k: int = 4
    topk_rank_by: str = "gating"
    distance_grouping_clusters_per_server: Optional[int] = None
    distance_grouping_kmeans_iterations: int = 20

    # WDMoE pruning settings. Per-node WDMoE uses alpha once: nodes with
    # similarity above alpha keep Top-K; nodes below alpha try K-1.
    wdmoe_initial_threshold: float = 0.8
    wdmoe_threshold_step: float = 0.05
    wdmoe_max_threshold: float = 0.8
    wdmoe_wlr_target_ratio: float = 1.05

    # RSMA / edge network settings. These follow the original SIoT-RSMA defaults
    # as closely as possible, with multiple edge servers and coverage limits.
    num_edge_servers: int = 9
    num_iot_devices: int = 90
    area_size: float = 1000.0
    cell_radius: float = 300.0
    num_antennas: int = 4
    beamforming_correlation_weight: float = 0.0
    connected_servers_per_device: Optional[int] = None
    iot_features_per_device_range: tuple[int, int] = (5, 12)
    experts_per_server: int = 3
    server_gpu_memory: float = 1024.0
    default_wired_rate: float = 1e9
    wired_rate_range: tuple[float, float] = (5e8, 1.5e9)
    uplink_time_budget_seconds: Optional[float] = None
    bandwidth_time_fraction: float = 1.0
    min_bandwidth: float = 0.0
    wavelength: float = 0.125
    default_feature_bits: float = 12000.0
    noise_power: float = 1e-18
    common_power_ratio: float = 0.6
    max_device_power: float = 1.2589e-3
    max_group_size: int = 5

    # Objective cost weights.
    # Calibrated so activation, bandwidth, and forwarding each contribute a
    # similar order of magnitude in a typical run.
    # 5,000 Hz bandwidth gap now contributes about 50 cost.
    c_bw: float = 1e-2
    c_act: float = 60.0
    c_fwd: float = 0.3















