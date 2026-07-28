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
    num_runs: int = 10
    num_dags: int = 5
    random_seed: int = 42
    network_random_seed: int = 1
    output_dir: Path = BASE_DIR / "generated_dags"
    results_dir: Path = BASE_DIR / "results"
    image_dpi: int = 180

    # DAG / task graph settings.
    common_dag_spec: DAGSpec = field(
        default_factory=lambda: DAGSpec(
            num_nodes=6,
            num_branches=(2,4),
            branch_length_range=(1, 5),
            depth=8,
            max_width=5,
            edge_probability=0.2,
            allow_skip_edges=False,
            allow_early_branch_end=True,
            single_source=True,
            single_sink=False,
        )
    )

    # Expert / task metadata settings.
    num_experts: int = 16
    expert_memory_range: tuple[float, float] = (128.0, 256.0)
    task_deadline_seconds_range: tuple[int, int] = (30, 120)
    num_iot_features: int = 100
    iot_features_per_node_range: tuple[int, int] = (2, 5)
    gating_peak_count_range: tuple[int, int] = (2, 4)
    gating_peak_mass_range: tuple[float, float] = (0.65, 0.85)

    # Performance loss settings.
    loss_threshold: Optional[float] = None
    lambda_reconstruction: float = 2.0
    calibration_alpha: float = 0.1
    reconstruction_sigma: float = 1.0
    reconstruction_error_range: tuple[float, float] = (0.8, 1.5)
    num_calibration_samples: int = 20
    calibration_loss_range: tuple[float, float] = (1.5, 3.3)

    # Which methods to run. The default workflow compares four baselines and JRGEP.
    run_hybrid_topk_gssgd_backhaul: bool = True
    run_hybrid_topk_location_aware_backhaul: bool = True
    run_wdmoe_gssgd: bool = True
    run_wdmoe_location_aware: bool = True
    run_our_similarity_aware_topk_backhaul: bool = True

    # Shared Top-K and grouping settings.
    topk_k: int = 5
    topk_rank_by: str = "gating"
    location_aware_grouping_clusters_per_server: Optional[int] = None
    location_aware_grouping_kmeans_iterations: int = 20

    # Our algorithm settings.
    similarity_weight: float = 0.05
    random_similarity_pair_count: Optional[int] = 4
    enable_offline_group_pruning: bool = False
    jrgep_server_spread_penalty_multiplier: float = 1.0
    jrgep_predecessor_penalty_multiplier: float = 1.0
    # WDMoE pruning settings. Paper-style WDMoE searches theta at task/DAG
    # scope: each theta trial reruns all subtasks and stops when graph WLR
    # ratio passes gamma.
    wdmoe_initial_threshold: float = 0.8
    wdmoe_threshold_step: float = 0.05
    wdmoe_max_threshold: float = 1.0
    wdmoe_wlr_target_ratio: float = 1.05

    # RSMA / edge network settings. These follow the original SIoT-RSMA defaults
    # as closely as possible, with multiple edge servers and coverage limits.
    num_edge_servers: int = 10
    num_iot_devices: int = 100
    area_size: float = 2000.0
    cell_radius: float = 350.0
    num_antennas: int = 4
    iot_features_per_device_range: tuple[int, int] = (3, 12)
    iot_server_feature_overlap_ratio: float = 0.20
    iot_global_random_feature_fraction: float = 0.2
    experts_per_server: int = 5
    server_gpu_memory_range: tuple[float, float] = (640.0, 896.0)
    wired_rate_range: tuple[float, float] = (5e8, 1.5e9)
    wired_extra_link_probability: float = 0.05
    wavelength: float = 0.125
    feature_bits_range: tuple[float, float] = (8000.0, 80000.0)
    noise_power: float = 1e-18
    common_power_ratio: float = 0.6
    max_device_power: float = 1.2589e-3
    max_group_size: int = 4
    min_rate: float = 1.0

    # Objective cost weights.
    # Calibrated so activation, bandwidth, and forwarding each contribute a
    # similar order of magnitude in a typical run.
    # 5,000 Hz bandwidth gap contributes 500 cost with c_bw=0.1.
    c_bw: float = 0.05
    c_act: float = 20.0
    c_fwd: float = 1.0














