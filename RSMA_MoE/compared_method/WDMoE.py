"""WDMoE expert selection adapted to this project.

This module implements a DAG-aware version of WDMoE Algorithm 1 for the
current RSMA-MoE environment. The original paper routes tokens to wireless
experts; here each DAG node is treated as one subtask and each selected expert
must be placed on a feasible edge server that stores it.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_distance_backhaul import (  # noqa: E402
    ChainedComparisonPipeline,
    ChainedPipelineResult,
    result_to_jsonable as hybrid_result_to_jsonable,
)
from compared_method.hybrid_topk_gssgd_backhaul import HybridTopKGSSGDBackhaulPipeline  # noqa: E402
from rsma_integration import (  # noqa: E402
    build_simple_devices,
    build_simple_experts,
    build_simple_servers,
    graphs_to_tasks,
)
from utils.formulation import AssignmentKey, ExpertId, FormulationConfig, ServerId  # noqa: E402


class WDMoEExpertSelectionMixin:
    """DAG-aware WDMoE expert selection adapted to per-subtask loss bounds.

    Each DAG node first builds the ordinary Top-K baseline and a full expert
    latency vector. If cosine(G, latency) is below the WDMoE threshold, the
    lowest-gating selected expert is pruned once. The prune is kept only when
    the node-level WLR ratio remains above the configured target.
    """

    allow_expert_loss_repair: bool = False
    wdmoe_initial_threshold: float = 0.8
    wdmoe_threshold_step: float = 0.05
    wdmoe_max_threshold: float = 0.8
    wdmoe_wlr_target_ratio: float = 1.05

    def _assign_topk_experts(self):
        assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        used_memory = {
            sid: sum(self.experts[eid].memory for eid in eids if eid in self.experts)
            for sid, eids in activated.items()
        }
        self.wdmoe_task_metrics: Dict[str, Dict[str, float]] = {}

        for task in self.tasks.values():
            task_assignments, task_selected, activated, used_memory, metrics = self._wdmoe_assign_task_graph(
                task=task,
                committed_assignments=assignments,
                activated=activated,
                used_memory=used_memory,
            )
            assignments.update(task_assignments)
            selected_experts_by_key.update(task_selected)
            self.wdmoe_task_metrics[task.id] = metrics

            for key, selected_experts in task_selected.items():
                subtask = self._task_subtask_by_id(task, key[1])
                if subtask is None:
                    continue
                selected_probability[self._assignment_label(key)] = self.evaluator.selection_probability(
                    subtask,
                    selected_experts,
                )

        return assignments, selected_probability, selected_experts_by_key, activated, used_memory

    def _wdmoe_assign_task_graph(self, task, committed_assignments, activated, used_memory):
        task_assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        committed_activated = {sid: set(eids) for sid, eids in activated.items()}
        committed_memory = dict(used_memory)
        node_metrics: Dict[str, Dict[str, float]] = {}

        for subtask in self._topological_subtasks(task):
            key = (task.id, subtask.id)
            pairs, feasible_experts, metrics = self._wdmoe_assign_subtask(
                task=task,
                subtask=subtask,
                committed_assignments=committed_assignments,
                task_assignments=task_assignments,
                activated=committed_activated,
                used_memory=committed_memory,
            )
            task_assignments[key] = pairs
            selected_by_key[key] = feasible_experts
            node_metrics[subtask.id] = metrics

        baseline_values = [item["baseline_wlr"] for item in node_metrics.values()]
        selected_values = [item["selected_wlr"] for item in node_metrics.values()]
        baseline_avg = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
        selected_avg = sum(selected_values) / len(selected_values) if selected_values else 0.0
        metrics = {
            "wlr_scope": "per_subtask",
            "baseline_wlr": baseline_avg,
            "selected_wlr": selected_avg,
            "wlr_ratio": selected_avg / max(baseline_avg, 1e-12),
            "nodes": node_metrics,
        }
        return task_assignments, selected_by_key, committed_activated, committed_memory, metrics

    def _wdmoe_assign_subtask(
        self,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ):
        key = (task.id, subtask.id)
        latency_vector = self._wdmoe_latency_vector(
            task=task,
            subtask=subtask,
            committed_assignments=committed_assignments,
            task_assignments=task_assignments,
            activated=activated,
            used_memory=used_memory,
        )
        baseline_experts = set(self._top_k_experts(subtask)[: self.top_k])
        if not baseline_experts:
            baseline_experts = set(self._fallback_top_experts(subtask, count=self.top_k))

        baseline_wlr = self._wdmoe_subtask_wlr(subtask, baseline_experts, latency_vector)
        best_experts = set(baseline_experts)
        best_wlr = baseline_wlr
        best_theta = None
        met_gamma = False
        similarity = self._wdmoe_weight_latency_similarity(subtask, latency_vector)
        threshold = self.wdmoe_initial_threshold
        candidate_experts = set(baseline_experts)
        attempted_pruning = False

        if len(candidate_experts) > 1 and similarity <= threshold:
            dropped = min(
                candidate_experts,
                key=lambda expert_id: (self._expert_score(subtask, expert_id), expert_id),
            )
            candidate_experts.remove(dropped)
            attempted_pruning = True

        current_wlr = self._wdmoe_subtask_wlr(subtask, candidate_experts, latency_vector)
        current_ratio = current_wlr / max(baseline_wlr, 1e-12)
        met_gamma = current_ratio > self.wdmoe_wlr_target_ratio

        if attempted_pruning and not met_gamma:
            candidate_experts = set(baseline_experts)
            current_wlr = baseline_wlr
            current_ratio = 1.0

        best_experts = candidate_experts
        best_wlr = current_wlr
        best_theta = threshold

        pairs: list[tuple[ServerId, ExpertId]] = []
        feasible_experts: Set[ExpertId] = set()
        ordered_experts = sorted(best_experts, key=lambda eid: (-self._expert_score(subtask, eid), eid))
        for expert_id in ordered_experts:
            server_id = self._best_server_for_expert_with_predecessors(
                expert_id=expert_id,
                task=task,
                subtask=subtask,
                committed_assignments=committed_assignments,
                task_assignments=task_assignments,
                activated=activated,
                used_memory=used_memory,
            )
            if server_id is None:
                self.scheduler_violations.append(f"WDMoE placement: no feasible server stores {expert_id} for {key}")
                continue
            if not self._append_unique_assignment(pairs, server_id, expert_id):
                continue
            feasible_experts.add(expert_id)
            self._activate(server_id, expert_id, activated, used_memory)

        if not pairs:
            fallback = self._best_repair_candidate(subtask, set(), activated, used_memory)
            if fallback is not None:
                server_id, expert_id = fallback
                if self._append_unique_assignment(pairs, server_id, expert_id):
                    feasible_experts.add(expert_id)
                    self._activate(server_id, expert_id, activated, used_memory)

        metrics = {
            "baseline_wlr": baseline_wlr,
            "selected_wlr": best_wlr,
            "wlr_ratio": best_wlr / max(baseline_wlr, 1e-12),
            "similarity": similarity,
            "final_threshold": best_theta,
            "met_gamma": met_gamma,
            "attempted_pruning": attempted_pruning,
            "pruned": len(best_experts) < len(baseline_experts),
            "baseline_expert_count": len(baseline_experts),
            "selected_expert_count": len(feasible_experts),
        }
        return pairs, feasible_experts, metrics

    def _wdmoe_subtask_wlr(self, subtask, selected_experts: Set[ExpertId], latency_vector: Mapping[ExpertId, float]) -> float:
        values = [
            self._expert_weight(subtask, expert_id) / max(latency_vector.get(expert_id, 1e6), 1e-12)
            for expert_id in selected_experts
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)
    def _wdmoe_latency_vector(
        self,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ) -> Dict[ExpertId, float]:
        latencies: Dict[ExpertId, float] = {}
        for expert_id in self.experts:
            server_id = self._best_server_for_expert_with_predecessors(
                expert_id=expert_id,
                task=task,
                subtask=subtask,
                committed_assignments=committed_assignments,
                task_assignments=task_assignments,
                activated=activated,
                used_memory=used_memory,
            )
            if server_id is None:
                latencies[expert_id] = 1e6
                continue
            merged = dict(committed_assignments)
            merged.update(task_assignments)
            latencies[expert_id] = (
                self._wdmoe_predecessor_forwarding_time(task, subtask, server_id, merged, {})
                + self.experts[expert_id].latency
            )
        return latencies

    def _wdmoe_weight_latency_similarity(self, subtask, latency_vector) -> float:
        weights = []
        latencies = []
        for expert_id in self.experts:
            weights.append(self._expert_weight(subtask, expert_id))
            latencies.append(latency_vector.get(expert_id, 1e6))
        weight_norm = math.sqrt(sum(value * value for value in weights))
        latency_norm = math.sqrt(sum(value * value for value in latencies))
        if weight_norm <= 0.0 or latency_norm <= 0.0:
            return 1.0
        return sum(w * t for w, t in zip(weights, latencies)) / (weight_norm * latency_norm)

    def _best_server_for_expert_with_predecessors(
        self,
        expert_id,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ):
        expert = self.experts[expert_id]
        merged = dict(committed_assignments)
        merged.update(task_assignments)
        best = None
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            memory_after = used_memory.get(server_id, 0.0)
            if expert_id not in activated.get(server_id, set()):
                memory_after += expert.memory
            if memory_after > server.gpu_memory:
                continue
            pred_forward = self._wdmoe_predecessor_forwarding_time(task, subtask, server_id, merged, {})
            score = pred_forward + expert.latency
            if best is None or score < best[0]:
                best = (score, server_id)
        return None if best is None else best[1]

    def _wdmoe_predecessor_forwarding_time(self, task, subtask, target_server, assignments, finish_cache) -> float:
        ready = 0.0
        subtask_map = {item.id: item for item in task.subtasks}
        for pred_id in subtask.predecessors:
            pred = subtask_map.get(pred_id)
            if pred is None:
                continue
            pred_key = (task.id, pred_id)
            for src_server in self.evaluator.participating_servers(assignments, pred_key):
                candidate = finish_cache.get((pred_key, src_server), 0.0)
                if src_server != target_server:
                    pred_output_bits = self.evaluator.subtask_output_volume(pred_key, src_server, assignments)
                    candidate += pred_output_bits / self.evaluator.wired_rate(src_server, target_server)
                ready = max(ready, candidate)
        return ready

    def _expert_weight(self, subtask, expert_id: ExpertId) -> float:
        expert = self.experts[expert_id]
        if expert.index < 0 or expert.index >= len(subtask.gating_weights):
            return 0.0
        return float(subtask.gating_weights[expert.index])

    def _fallback_top_experts(self, subtask, count: int) -> list[ExpertId]:
        ranked = sorted(self.experts, key=lambda eid: self._expert_weight(subtask, eid), reverse=True)
        return ranked[:count]

    def _topological_subtasks(self, task):
        remaining = {subtask.id: subtask for subtask in task.subtasks}
        ordered = []
        while remaining:
            ready_ids = [
                subtask_id
                for subtask_id, subtask in remaining.items()
                if all(pred_id not in remaining for pred_id in subtask.predecessors)
            ]
            if not ready_ids:
                ordered.extend(remaining.values())
                break
            for subtask_id in sorted(ready_ids, key=self._id_sort_key):
                ordered.append(remaining.pop(subtask_id))
        return ordered

    @staticmethod
    def _task_subtask_by_id(task, subtask_id):
        return next((subtask for subtask in task.subtasks if subtask.id == subtask_id), None)


class WDMoEDistancePipeline(WDMoEExpertSelectionMixin, ChainedComparisonPipeline):
    """WDMoE expert selection + distance/K-means IoT grouping."""


class WDMoEGSSGDPipeline(WDMoEExpertSelectionMixin, HybridTopKGSSGDBackhaulPipeline):
    """WDMoE expert selection + GSSGD DBG IoT grouping."""


def _run_wdmoe_pipeline(
    pipeline_cls,
    method_name: str,
    grouping_name: str,
    graphs: Sequence[Any],
    output_path: Path,
    num_experts: int,
    num_iot_features: int,
    top_k: int = 2,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (5, 12),
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
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
    wdmoe_initial_threshold: float = 0.8,
    wdmoe_threshold_step: float = 0.05,
    wdmoe_max_threshold: float = 0.8,
    wdmoe_wlr_target_ratio: float = 1.05,
) -> ChainedPipelineResult:
    rng = random.Random(random_seed)
    tasks = graphs_to_tasks(graphs, loss_threshold)
    experts = build_simple_experts(num_experts)
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        gpu_memory=server_gpu_memory,
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
    pipeline = pipeline_cls(
        servers=servers,
        devices=devices,
        experts=experts,
        tasks=tasks,
        top_k=top_k,
        rank_by=rank_by,
        clusters_per_server=clusters_per_server,
        kmeans_iterations=kmeans_iterations,
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
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
            reconstruction_sigma=reconstruction_sigma,
            default_feature_bits=default_feature_bits,
            wavelength=wavelength,
        ),
    )
    pipeline.wdmoe_initial_threshold = wdmoe_initial_threshold
    pipeline.wdmoe_threshold_step = wdmoe_threshold_step
    pipeline.wdmoe_max_threshold = wdmoe_max_threshold
    pipeline.wdmoe_wlr_target_ratio = wdmoe_wlr_target_ratio
    result = pipeline.run()
    network_context = {
        "experts": experts,
        "servers": servers,
        "devices": devices,
        "topology": topology,
        "unit_cost": {
            "activation": c_act,
            "bandwidth": c_bw,
            "forwarding": c_fwd,
        },
        "wdmoe_parameters": {
            "algorithm": "WDMoE-Based DAG-Aware Expert Selection",
            "initial_top_k": top_k,
            "initial_threshold_alpha": pipeline.wdmoe_initial_threshold,
            "threshold_step_delta": pipeline.wdmoe_threshold_step,
            "max_threshold": pipeline.wdmoe_max_threshold,
            "wlr_target_ratio_gamma": pipeline.wdmoe_wlr_target_ratio,
            "wlr_scope": "per subtask/node",
            "latency_vector": "predecessor forwarding time + expert inference time",
            "cosine_similarity": "full gating weight vector versus full expert latency vector",
            "dropped_expert_rule": "per-node adaptation: keep Top-K when similarity > alpha; try K-1 only when similarity <= alpha, then fall back to Top-K if WLR ratio does not pass gamma",
            "activation_y_rule": "Y(s,p)=1 if any selected X(i,j,s,p)=1",
            "task_metrics": getattr(pipeline, "wdmoe_task_metrics", {}),
            "paper_preserving_note": "WDMoE keeps the paper-style pruning rule, does not add cost-aware activation penalties, and does not add extra experts during loss repair",
            "grouping": grouping_name,
        },
        "rsma_parameters": {
            "num_servers": num_servers,
            "num_iot_devices": num_iot_devices,
            "features_per_device_range": features_per_device_range,
            "experts_per_server": experts_per_server,
            "server_gpu_memory": server_gpu_memory,
            "wired_rate_range": wired_rate_range,
            "bandwidth_mode": "derived_by_group_slack",
            "bandwidth_time_fraction": bandwidth_time_fraction,
            "default_feature_bits": default_feature_bits,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "max_group_size": max_group_size,
            "min_rate": min_rate,
            "gssgd_beamforming_gain_threshold": gssgd_beamforming_gain_threshold,
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
            "reconstruction_sigma": reconstruction_sigma,
        },
    }
    payload = hybrid_result_to_jsonable(
        result,
        network_context,
        cost_units={"activation": c_act, "bandwidth": c_bw, "forwarding": c_fwd},
    )
    payload["method"] = method_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return result


def run_wdmoe_distance(
    graphs: Sequence[Any],
    output_path: Path,
    **kwargs: Any,
) -> ChainedPipelineResult:
    return _run_wdmoe_pipeline(
        WDMoEDistancePipeline,
        "wdmoe_distance",
        "Distance/K-means IoT grouping",
        graphs,
        output_path,
        **kwargs,
    )


def run_wdmoe_gssgd(
    graphs: Sequence[Any],
    output_path: Path,
    **kwargs: Any,
) -> ChainedPipelineResult:
    return _run_wdmoe_pipeline(
        WDMoEGSSGDPipeline,
        "wdmoe_gssgd",
        "GSSGD DBG IoT grouping",
        graphs,
        output_path,
        **kwargs,
    )







