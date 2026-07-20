"""Hybrid Top-K + GSSGD + Backhaul method.

Pipeline:
1. Top-K MoE selects experts and servers for each subtask.
2. GSSGD DBG phase 1-3 builds local IoT groups under each server.
3. GSSGD builds candidate groups; groups transmit only when needed to repair performance loss.
4. Backhaul routes selected group data when the assigned server needs it.
5. The shared formulation evaluator computes timing, constraints, and cost.

Only DBG is kept from the original GSSGD/SIoT baseline. DCG/collaborative
computing code is intentionally not used here.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Set

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.GSSGD.phase1 import phase1_dbg_selection  # noqa: E402
from compared_method.GSSGD.phase2 import phase2_dbg_grouping  # noqa: E402
from compared_method.GSSGD.phase3 import phase3_dbg_refinement  # noqa: E402
from compared_method.hybrid_topk_distance_backhaul import (  # noqa: E402
    ChainedComparisonPipeline,
    ChainedPipelineResult,
    result_to_jsonable as hybrid_result_to_jsonable,
)
from rsma_integration import (  # noqa: E402
    build_feature_bits,
    build_simple_devices,
    build_simple_experts,
    build_simple_servers,
    graphs_to_tasks,
)
from utils.formulation import FormulationConfig, GroupSpec, ServerId  # noqa: E402


class HybridTopKGSSGDBackhaulPipeline(ChainedComparisonPipeline):
    """Hybrid-shaped pipeline whose IoT grouping stage is GSSGD DBG."""
    activate_all_candidate_groups = False

    def _build_distance_groups(self, server_required_features):
        groups: list[GroupSpec] = []
        globally_required = set()
        for features in server_required_features.values():
            globally_required.update(features)
        for server_id in self.servers:
            local_devices = [device_id for device_id, device in self.devices.items() if device.home_server == server_id]
            local_required = globally_required & self._server_local_features(local_devices)
            if not local_required:
                continue

            selected, unselected = phase1_dbg_selection(
                devices=self.devices,
                evaluator=self.evaluator,
                server_id=server_id,
                local_device_ids=local_devices,
                required_features=local_required,
            )
            raw_groups = phase2_dbg_grouping(
                devices=self.devices,
                evaluator=self.evaluator,
                server_id=server_id,
                selected_device_ids=selected,
                required_features=local_required,
                max_group_size=self.config.max_group_size,
                beamforming_gain_threshold=self.config.gssgd_beamforming_gain_threshold,
            )
            refined_groups = phase3_dbg_refinement(
                devices=self.devices,
                evaluator=self.evaluator,
                server_id=server_id,
                groups=raw_groups,
                unselected_device_ids=unselected,
                required_features=local_required,
                beamforming_gain_threshold=self.config.gssgd_beamforming_gain_threshold,
            )

            for members in refined_groups:
                required_features = self._group_local_features(members, local_required)
                if not required_features:
                    continue
                groups.append(
                    self._make_group(
                        server_id=server_id,
                        group_id=f"g{len(groups)}",
                        members=tuple(members),
                        required_features=required_features,
                    )
                )

            covered = set().union(*(group.required_features for group in groups if group.server_id == server_id))
            missing = local_required - covered
            for feature in sorted(missing):
                self.scheduler_violations.append(
                    f"GSSGD DBG grouping: server {server_id} cannot cover local feature {feature}"
                )
        return groups
def result_to_jsonable(
    result: ChainedPipelineResult,
    network_context: Mapping[str, Any] | None = None,
    cost_units: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    payload = hybrid_result_to_jsonable(result, network_context, cost_units)
    payload["method"] = "hybrid_topk_gssgd_backhaul"
    if "network_model" in payload:
        payload["network_model"].setdefault("gssgd_parameters", {})
        payload["network_model"]["gssgd_parameters"].update(
            {
                "pipeline": "Top-K expert placement + GSSGD DBG IoT grouping + backhaul routing",
                "dbg_only": True,
                "dcg_removed": True,
                "grouping_rule": "phase-aware DBG grouping using common feature ratio and beamforming gain",
            }
        )
    return payload


def run_hybrid_topk_gssgd_backhaul(
    graphs: Sequence[Any],
    output_path: Path,
    num_experts: int,
    num_iot_features: int,
    expert_memory_range: tuple[float, float] | None = None,
    feature_bits_range: tuple[float, float] | None = None,
    top_k: int = 2,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (5, 12),
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
    server_gpu_memory_range: tuple[float, float] | None = None,
    wired_rate_range: tuple[float, float] | None = None,
    c_bw: float = 1e-3,
    c_act: float = 1.0,
    c_fwd: float = 1.0,
    bandwidth_time_fraction: float = 1.0,
    default_feature_bits: float = 12000.0,
    wavelength: float = 0.125,
    noise_power: float = 1e-18,
    common_power_ratio: float = 0.6,
    max_device_power: float = 1.2589e-3,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    loss_threshold: float | None = None,
    lambda_reconstruction: float = 0.1,
    calibration_alpha: float = 0.1,
    reconstruction_sigma: float = 1.0,
    random_seed: int = 42,
    max_group_size: int = 5,
    min_rate: float = 1.0,
    gssgd_beamforming_gain_threshold: float = 0.0,
    rank_by: str = "gating",
    clusters_per_server: Optional[int] = None,
    kmeans_iterations: int = 20,
) -> ChainedPipelineResult:
    del clusters_per_server, kmeans_iterations
    rng = random.Random(random_seed)
    tasks = graphs_to_tasks(graphs, loss_threshold)
    feature_bits_by_name = build_feature_bits(num_iot_features, default_feature_bits, feature_bits_range, rng)
    experts = build_simple_experts(num_experts, memory_range=expert_memory_range, rng=rng)
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        gpu_memory=server_gpu_memory,
        gpu_memory_range=server_gpu_memory_range,
        wired_rate_range=wired_rate_range,
        rng=rng,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=num_servers,
        features_per_device_range=features_per_device_range,
        server_feature_overlap_ratio=server_feature_overlap_ratio,
        global_random_feature_fraction=global_random_feature_fraction,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        rng=rng,
        max_power=max_device_power,
    )

    pipeline = HybridTopKGSSGDBackhaulPipeline(
        servers=servers,
        devices=devices,
        experts=experts,
        tasks=tasks,
        top_k=top_k,
        rank_by=rank_by,
        config=FormulationConfig(
            max_group_size=max_group_size,
            min_rate=min_rate,
            gssgd_beamforming_gain_threshold=gssgd_beamforming_gain_threshold,
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            derive_bandwidth=True,
            bandwidth_time_fraction=bandwidth_time_fraction,
            noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            default_feature_bits=default_feature_bits,
            feature_bits_by_name=feature_bits_by_name,
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
            reconstruction_sigma=reconstruction_sigma,
            wavelength=wavelength,
        ),
    )
    result = pipeline.run()
    network_context = {
        "topology": topology,
        "unit_cost": {
            "activation": c_act,
            "bandwidth": c_bw,
            "forwarding": c_fwd,
        },
        "gssgd_parameters": {
            "top_k": top_k,
            "rank_by": rank_by,
            "dbg_only": True,
            "dcg_removed": True,
            "max_group_size": max_group_size,
            "beamforming_gain_threshold": gssgd_beamforming_gain_threshold,
        },
        "rsma_parameters": {
            "num_servers": num_servers,
            "num_iot_devices": num_iot_devices,
            "features_per_device_range": features_per_device_range,
            "expert_memory_range": expert_memory_range,
            "experts_per_server": experts_per_server,
            "server_gpu_memory_range": server_gpu_memory_range,
            "wired_rate_range": wired_rate_range,
            "bandwidth_mode": "derived_by_group_slack",
            "bandwidth_time_fraction": bandwidth_time_fraction,
            "default_feature_bits": default_feature_bits,
            "feature_bits_range": feature_bits_range,
            "feature_bits_by_name": feature_bits_by_name,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "min_rate": min_rate,
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
            "reconstruction_sigma": reconstruction_sigma,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result_to_jsonable(
                result,
                network_context,
                cost_units={"activation": c_act, "bandwidth": c_bw, "forwarding": c_fwd},
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
    return result








