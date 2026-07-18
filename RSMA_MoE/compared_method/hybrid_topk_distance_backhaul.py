"""Hybrid algorithm: Top-K expert placement -> distance grouping -> backhaul repair."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rsma_integration import build_simple_devices, build_simple_experts, build_simple_servers, graphs_to_tasks  # noqa: E402
from utils.formulation import (  # noqa: E402
    AssignmentKey,
    DeviceId,
    EvaluationResult,
    ExpertId,
    FormulationConfig,
    FormulationEvaluator,
    GroupId,
    GroupSpec,
    ServerId,
    SubtaskSpec,
    normalize_devices,
    normalize_experts,
    normalize_servers,
    normalize_tasks,
)

BackhaulEdge = Tuple[ServerId, GroupId, ServerId]
GroupKey = Tuple[ServerId, GroupId]


@dataclass
class ChainedBackhaulDecision:
    source_server: ServerId
    group_id: GroupId
    target_server: ServerId
    features: List[str]


@dataclass
class ChainedPipelineResult:
    expert_placement: Dict[ServerId, List[ExpertId]]
    subtask_assignment: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]]
    subtask_data_requirements: Dict[str, Dict[ServerId, List[str]]]
    server_required_features: Dict[ServerId, List[str]]
    rsma_groups: Dict[ServerId, List[List[DeviceId]]]
    group_required_features: Dict[str, List[str]]
    group_time_budgets: Dict[str, float]
    rsma_group_bandwidths: Dict[ServerId, List[float]]
    backhaul_plan: List[ChainedBackhaulDecision]
    selected_probability: Dict[str, float]
    required_probability: Dict[str, float]
    reconstruction_loss: Dict[str, float]
    performance_loss: Dict[str, float]
    total_cost: float
    activation_cost: float
    bandwidth_cost: float
    forwarding_cost: float
    backhaul: Set[BackhaulEdge] = field(default_factory=set)
    violations: List[str] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None


class ChainedComparisonPipeline:
    allow_expert_loss_repair: bool = True

    def __init__(
        self,
        servers: Iterable[Any],
        devices: Iterable[Any],
        experts: Mapping[str, Any],
        tasks: Iterable[Any],
        top_k: int = 2,
        rank_by: str = "gating",
        clusters_per_server: Optional[int] = None,
        kmeans_iterations: int = 20,
        config: Optional[FormulationConfig] = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if rank_by not in {"gating", "contribution"}:
            raise ValueError("rank_by must be either 'gating' or 'contribution'.")
        self.top_k = top_k
        self.rank_by = rank_by
        self.clusters_per_server = clusters_per_server
        self.kmeans_iterations = kmeans_iterations
        self.config = config or FormulationConfig()
        self.servers = normalize_servers(servers)
        self.devices = normalize_devices(devices)
        self.experts = normalize_experts(experts)
        self.tasks = normalize_tasks(tasks)
        self.evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, self.config)
        self.scheduler_violations: List[str] = []

    def run(self) -> ChainedPipelineResult:
        assignments, selected_prob, selected_experts, activated, used_memory = self._assign_topk_experts()
        candidate_features = self._candidate_server_features(assignments)
        candidate_groups = self._build_distance_groups(candidate_features, select_groups=False)
        active_groups, subtask_features, required_prob, rec_loss, perf_loss = self._repair_loss_with_groups(
            assignments,
            selected_prob,
            selected_experts,
            candidate_groups,
            activated,
            used_memory,
        )
        data_req = self._subtask_data_requirements(assignments, subtask_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(active_groups)
        backhaul, plan = self._derive_backhaul(data_req, feature_to_groups)
        groups, budgets = self._derive_group_bandwidths(active_groups, assignments, data_req, backhaul, feature_to_groups)
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        evaluation = evaluator.evaluate(assignments, groups, backhaul, subtask_features=subtask_features)
        violations = [*self.scheduler_violations, *evaluation.violations]
        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment=assignments,
            subtask_data_requirements=data_req,
            server_required_features={sid: sorted(v) for sid, v in server_features.items()},
            rsma_groups=self._group_map(groups),
            group_required_features={self._group_label(g): sorted(g.required_features) for g in groups},
            group_time_budgets={self._group_key_label(k): v for k, v in budgets.items()},
            rsma_group_bandwidths=self._group_bandwidth_map(groups),
            backhaul_plan=plan,
            selected_probability=selected_prob,
            required_probability=required_prob,
            reconstruction_loss=rec_loss,
            performance_loss=perf_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            backhaul=backhaul,
            violations=violations,
            evaluation=evaluation,
        )
    @staticmethod
    def _append_unique_assignment(
        pairs: List[Tuple[ServerId, ExpertId]],
        server_id: ServerId,
        expert_id: ExpertId,
    ) -> bool:
        if any(existing_expert == expert_id for _, existing_expert in pairs):
            return False
        pairs.append((server_id, expert_id))
        return True

    def _assign_topk_experts(self):
        assignments: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        used_memory = {
            sid: sum(self.experts[eid].memory for eid in eids if eid in self.experts)
            for sid, eids in activated.items()
        }
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                selected: List[Tuple[ServerId, ExpertId]] = []
                selected_experts: Set[ExpertId] = set()
                for expert_id in self._top_k_experts(subtask):
                    server_id = self._best_server_for_expert(expert_id, activated, used_memory)
                    if server_id is None:
                        self.scheduler_violations.append(f"TopK placement: no feasible server stores {expert_id} for {key}")
                        continue
                    if not self._append_unique_assignment(selected, server_id, expert_id):
                        continue
                    selected_experts.add(expert_id)
                    self._activate(server_id, expert_id, activated, used_memory)
                assignments[key] = selected
                selected_experts_by_key[key] = selected_experts
                selected_probability[label] = self.evaluator.selection_probability(subtask, selected_experts)
        return assignments, selected_probability, selected_experts_by_key, activated, used_memory

    def _repair_loss_with_groups(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        candidate_groups,
        activated,
        used_memory,
    ):
        active_groups: Dict[GroupKey, GroupSpec] = {}
        active_backhaul: Set[BackhaulEdge] = set()
        active_group_budgets: Dict[GroupKey, float] = {}
        subtask_features: Dict[AssignmentKey, Set[str]] = {}
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}

        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                target_servers = self.evaluator.participating_servers(assignments, key)
                selected_features: Set[str] = set()
                probability = selected_probability.get(label, 0.0)
                threshold = self.evaluator.conformal_loss_threshold(subtask)
                loss = self.evaluator.performance_loss_from_probability(subtask, probability, selected_features)

                while loss > threshold + 1e-12:
                    group = self._best_group_loss_repair(
                        subtask,
                        key,
                        selected_features,
                        probability,
                        threshold,
                        target_servers,
                        candidate_groups,
                        active_groups,
                        active_backhaul,
                        active_group_budgets,
                        assignments,
                    )
                    if group is None:
                        break
                    group_key = (group.server_id, group.id)
                    active_groups[group_key] = group
                    group_budget = self._subtask_bandwidth_budget(key, assignments)
                    active_group_budgets[group_key] = min(active_group_budgets.get(group_key, group_budget), group_budget)
                    for target_server in target_servers:
                        if group.server_id != target_server:
                            active_backhaul.add((group.server_id, group.id, target_server))
                    selected_features.update(group.required_features & subtask.required_features)
                    loss = self.evaluator.performance_loss_from_probability(subtask, probability, selected_features)

                while self.allow_expert_loss_repair and loss > threshold + 1e-12:
                    selected_experts = selected_experts_by_key.setdefault(key, set())
                    candidate = self._best_repair_candidate(subtask, selected_experts, activated, used_memory)
                    if candidate is None:
                        break
                    server_id, expert_id = candidate
                    pairs = assignments.setdefault(key, [])
                    if not self._append_unique_assignment(pairs, server_id, expert_id):
                        break
                    selected_experts.add(expert_id)
                    self._activate(server_id, expert_id, activated, used_memory)
                    probability += self.evaluator.expert_contribution(subtask, expert_id)
                    selected_probability[label] = probability
                    loss = self.evaluator.performance_loss_from_probability(subtask, probability, selected_features)

                if loss > threshold + 1e-12:
                    self.scheduler_violations.append(
                        f"Loss repair: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                subtask_features[key] = selected_features
                required_probability[label] = self.evaluator.required_selection_probability(subtask, selected_features)
                reconstruction_loss[label] = self.evaluator.reconstruction_loss_for_features(subtask, selected_features)
                performance_loss[label] = loss

        return list(active_groups.values()), subtask_features, required_probability, reconstruction_loss, performance_loss

    def _best_group_loss_repair(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        selected_features: Set[str],
        probability: float,
        threshold: float,
        target_servers: Sequence[ServerId],
        candidate_groups: Sequence[GroupSpec],
        active_groups: Mapping[GroupKey, GroupSpec],
        active_backhaul: Set[BackhaulEdge],
        active_group_budgets: Mapping[GroupKey, float],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> Optional[GroupSpec]:
        missing = set(subtask.required_features) - set(selected_features)
        if not missing:
            return None
        usable = [group for group in candidate_groups if group.required_features & missing]
        if not usable:
            return None

        satisfying = []
        improving = []
        for group in usable:
            new_features = selected_features | (group.required_features & subtask.required_features)
            new_loss = self.evaluator.performance_loss_from_probability(subtask, probability, new_features)
            cost = self._group_incremental_cost(
                group,
                key,
                target_servers,
                active_groups,
                active_backhaul,
                active_group_budgets,
                assignments,
            )
            coverage = len(group.required_features & missing)
            item = (cost, -coverage, group.server_id, group.id, group)
            if new_loss <= threshold + 1e-12:
                satisfying.append(item)
            else:
                improving.append((-(coverage / max(cost, 1e-12)), cost, group.server_id, group.id, group))

        if satisfying:
            return min(satisfying)[-1]
        if improving:
            return min(improving)[-1]
        return None

    def _group_incremental_cost(
        self,
        group: GroupSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        active_groups: Mapping[GroupKey, GroupSpec],
        active_backhaul: Set[BackhaulEdge],
        active_group_budgets: Mapping[GroupKey, float],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        cost = 0.0
        group_key = (group.server_id, group.id)
        new_budget = self._subtask_bandwidth_budget(key, assignments)
        if group_key in active_groups:
            old_budget = active_group_budgets.get(group_key, self.evaluator.bandwidth_time_budget())
            resolved_budget = min(old_budget, new_budget)
            if resolved_budget < old_budget - 1e-12:
                old_bw = self.evaluator.derive_group_bandwidth_for_budget(group, old_budget)
                new_bw = self.evaluator.derive_group_bandwidth_for_budget(group, resolved_budget)
                cost += self.config.c_bw * max(0.0, new_bw - old_bw)
        else:
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, new_budget)
            cost += self.config.c_bw * bandwidth
        for target_server in target_servers:
            edge = (group.server_id, group.id, target_server)
            if group.server_id != target_server and edge not in active_backhaul:
                cost += self.config.c_fwd
        return cost

    def _subtask_bandwidth_budget(
        self,
        key: AssignmentKey,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        task = self.tasks.get(key[0])
        if task is None or task.deadline == float("inf"):
            return self.evaluator.bandwidth_time_budget()
        downstream = self._downstream_compute_times(assignments)
        remaining = task.deadline - downstream.get(key, 0.0)
        return max(remaining, 1e-12)
    def _candidate_server_features(self, assignments):
        required = {server_id: set() for server_id in self.servers}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                for server_id in self.evaluator.participating_servers(assignments, key):
                    required.setdefault(server_id, set()).update(subtask.required_features)
        return required
    def _top_k_experts(self, subtask: SubtaskSpec) -> List[ExpertId]:
        ranked = sorted(self.experts, key=lambda eid: self._expert_score(subtask, eid), reverse=True)
        return [eid for eid in ranked if self._expert_score(subtask, eid) > 0.0][: min(self.top_k, len(ranked))]

    def _expert_score(self, subtask: SubtaskSpec, expert_id: ExpertId) -> float:
        expert = self.experts[expert_id]
        index = expert.index
        if index < 0 or index >= len(subtask.gating_weights):
            return 0.0
        gating = subtask.gating_weights[index]
        if self.rank_by == "gating":
            return gating
        if index >= len(subtask.expert_confidence):
            return 0.0
        return gating * subtask.expert_confidence[index]

    def _best_server_for_expert(self, expert_id, activated, used_memory):
        best = None
        expert = self.experts[expert_id]
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            memory_after = used_memory.get(server_id, 0.0)
            activation_penalty = 0.0
            if expert_id not in activated.get(server_id, set()):
                memory_after += expert.memory
                activation_penalty = self.config.c_act
            if memory_after > server.gpu_memory:
                continue
            score = activation_penalty + expert.latency + (memory_after / max(server.gpu_memory, 1e-12))
            if best is None or score < best[0]:
                best = (score, server_id)
        return None if best is None else best[1]

    def _best_repair_candidate(self, subtask, selected_experts, activated, used_memory):
        best = None
        for server_id, server in self.servers.items():
            for expert_id in server.stored_experts:
                if expert_id in selected_experts:
                    continue
                contribution = self.evaluator.expert_contribution(subtask, expert_id)
                if contribution <= 0.0:
                    continue
                expert = self.experts[expert_id]
                memory_after = used_memory.get(server_id, 0.0)
                activation_penalty = 0.0
                if expert_id not in activated.get(server_id, set()):
                    memory_after += expert.memory
                    activation_penalty = self.config.c_act
                if memory_after > server.gpu_memory:
                    continue
                score = -contribution + activation_penalty + expert.latency
                if best is None or score < best[0]:
                    best = (score, server_id, expert_id)
        return None if best is None else (best[1], best[2])

    def _activate(self, server_id, expert_id, activated, used_memory):
        if expert_id in activated.setdefault(server_id, set()):
            return
        activated[server_id].add(expert_id)
        used_memory[server_id] = used_memory.get(server_id, 0.0) + self.experts[expert_id].memory

    def _subtask_data_requirements(self, assignments, subtask_features):
        requirements: Dict[str, Dict[ServerId, List[str]]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                features = sorted(subtask_features.get(key, set()))
                server_map: Dict[ServerId, List[str]] = {}
                for server_id, _ in assignments.get(key, []):
                    server_map[server_id] = features
                requirements[self._assignment_label(key)] = server_map
        return requirements
    def _server_required_features(self, subtask_data_requirements):
        required = {sid: set() for sid in self.servers}
        for server_map in subtask_data_requirements.values():
            for server_id, features in server_map.items():
                required.setdefault(server_id, set()).update(features)
        return required

    def _build_distance_groups(self, server_required_features, select_groups=True):
        groups: List[GroupSpec] = []
        for server_id in self.servers:
            assigned_required = set(server_required_features.get(server_id, set()))
            local_devices = [did for did, dev in self.devices.items() if dev.home_server == server_id]
            local_required = assigned_required & self._server_local_features(local_devices)
            if not local_required:
                continue
            candidates = [did for did in local_devices if self.devices[did].features & local_required]
            if not candidates:
                self._record_local_missing_features(server_id, local_required, groups)
                continue

            candidate_groups: List[GroupSpec] = []
            clusters = self._kmeans_device_clusters(server_id, candidates, self._cluster_count(len(candidates)))
            for cluster in clusters:
                for chunk in self._chunk_by_distance(server_id, cluster):
                    required_features = self._group_local_features(chunk, local_required)
                    if required_features:
                        candidate_groups.append(
                            self._make_group(
                                server_id,
                                f"g{len(groups) + len(candidate_groups)}",
                                tuple(chunk),
                                required_features,
                            )
                        )
            if select_groups:
                groups.extend(self._select_groups_for_required_features(candidate_groups, local_required))
            else:
                groups.extend(candidate_groups)
            self._record_local_missing_features(server_id, local_required, groups)
        return groups

    def _server_local_features(self, local_devices):
        features = set()
        for device_id in local_devices:
            features.update(self.devices[device_id].features)
        return features
    def _group_local_features(self, device_ids, local_required):
        features = set()
        for device_id in device_ids:
            features.update(self.devices[device_id].features & local_required)
        return features

    def _select_groups_for_required_features(self, candidate_groups, required_features):
        selected = []
        missing = set(required_features)
        unused = list(candidate_groups)
        while missing and unused:
            best = max(
                unused,
                key=lambda group: (
                    len(group.required_features & missing),
                    -len(group.devices),
                    group.server_id,
                    group.id,
                ),
            )
            covered = best.required_features & missing
            if not covered:
                break
            selected.append(best)
            missing -= covered
            unused.remove(best)
        return selected

    def _record_local_missing_features(self, server_id, local_required, groups):
        available = set()
        for group in groups:
            if group.server_id == server_id:
                available.update(group.required_features)
        for feature in sorted(local_required - available):
            if not any(feature in device.features for device in self.devices.values()):
                self.scheduler_violations.append(
                    f"Data requirement: feature {feature} required by {server_id} is unavailable from all IoT groups"
                )

    def _cluster_count(self, num_devices):
        if self.clusters_per_server is not None:
            return min(self.clusters_per_server, num_devices)
        return max(1, math.ceil(num_devices / max(1, self.config.max_group_size)))

    def _kmeans_device_clusters(self, server_id, device_ids, cluster_count):
        ordered = sorted(device_ids, key=self._id_sort_key)
        centers = self._initial_centers(server_id, ordered, cluster_count)
        clusters = [[] for _ in range(cluster_count)]
        for _ in range(self.kmeans_iterations):
            next_clusters = [[] for _ in range(cluster_count)]
            for device_id in ordered:
                point = self._device_position(device_id, server_id)
                best_index = min(range(cluster_count), key=lambda index: self._squared_distance(point, centers[index]))
                next_clusters[best_index].append(device_id)
            for index, cluster in enumerate(next_clusters):
                if cluster:
                    centers[index] = self._centroid(server_id, cluster)
            if next_clusters == clusters:
                break
            clusters = next_clusters
        return [cluster for cluster in clusters if cluster]

    def _initial_centers(self, server_id, ordered, cluster_count):
        if cluster_count == 1:
            return [self._device_position(ordered[0], server_id)]
        centers = []
        for index in range(cluster_count):
            position = round(index * (len(ordered) - 1) / max(cluster_count - 1, 1))
            centers.append(self._device_position(ordered[position], server_id))
        return centers

    def _chunk_by_distance(self, server_id, cluster):
        centroid = self._centroid(server_id, cluster)
        ordered = sorted(
            cluster,
            key=lambda did: (self._squared_distance(self._device_position(did, server_id), centroid), self._id_sort_key(did)),
        )
        size = max(1, self.config.max_group_size)
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    def _device_position(self, device_id, server_id):
        device = self.devices[device_id]
        if server_id in device.distance:
            distance = float(device.distance[server_id])
            theta = float(device.phase_angle.get(server_id, 0.0))
            return distance * math.cos(theta), distance * math.sin(theta)
        return device.position

    def _centroid(self, server_id, device_ids):
        points = [self._device_position(did, server_id) for did in device_ids]
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

    def _make_group(self, server_id, group_id, members, required_features):
        common_power = {did: self.devices[did].max_power * self.config.common_power_ratio for did in members}
        private_power = {did: self.devices[did].max_power * (1.0 - self.config.common_power_ratio) for did in members}
        return GroupSpec(
            server_id=server_id,
            id=group_id,
            devices=members,
            bandwidth=0.0,
            required_features=set(required_features),
            common_power=common_power,
            private_power=private_power,
        )

    def _derive_backhaul(self, subtask_data_requirements, feature_to_groups):
        backhaul: Set[BackhaulEdge] = set()
        features_by_edge: Dict[BackhaulEdge, Set[str]] = {}
        for server_map in subtask_data_requirements.values():
            for target_server, features in server_map.items():
                for feature in features:
                    candidates = feature_to_groups.get(feature, [])
                    if any(group.server_id == target_server for group in candidates):
                        continue
                    source = self._best_source_group(target_server, candidates)
                    if source is None:
                        self.scheduler_violations.append(
                            f"Backhaul routing: feature {feature} is unavailable for server {target_server}"
                        )
                        continue
                    edge = (source.server_id, source.id, target_server)
                    backhaul.add(edge)
                    features_by_edge.setdefault(edge, set()).add(feature)
        plan = [
            ChainedBackhaulDecision(source_server=s, group_id=g, target_server=t, features=sorted(features_by_edge[(s, g, t)]))
            for s, g, t in sorted(backhaul)
        ]
        return backhaul, plan

    def _best_source_group(self, target_server, candidates):
        remote = [group for group in candidates if group.server_id != target_server]
        if not remote:
            return None
        return min(
            remote,
            key=lambda group: (
                self.evaluator.group_feature_volume(group) / self.evaluator.wired_rate(group.server_id, target_server),
                group.server_id,
                group.id,
            ),
        )

    def _derive_group_bandwidths(self, groups, assignments, data_req, backhaul, feature_to_groups):
        dependencies = self._group_dependencies(groups, assignments, data_req, backhaul, feature_to_groups)
        downstream = self._downstream_compute_times(assignments)
        budgets: Dict[GroupKey, float] = {}
        resolved: List[GroupSpec] = []
        for group in groups:
            group_key = (group.server_id, group.id)
            budget = self._strict_group_budget(group_key, dependencies, downstream)
            budgets[group_key] = budget
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, budget)
            resolved.append(replace(group, bandwidth=bandwidth))
        return resolved, budgets
    def _group_dependencies(self, groups, assignments, data_req, backhaul, feature_to_groups):
        dependencies: Dict[GroupKey, Set[AssignmentKey]] = {(g.server_id, g.id): set() for g in groups}
        for key in assignments:
            label = self._assignment_label(key)
            for target_server, features in data_req.get(label, {}).items():
                for feature in features:
                    group = self._serving_group_for_feature(feature, target_server, backhaul, feature_to_groups)
                    if group is not None:
                        dependencies.setdefault((group.server_id, group.id), set()).add(key)
        return dependencies

    def _serving_group_for_feature(self, feature, target_server, backhaul, feature_to_groups):
        candidates = list(feature_to_groups.get(feature, []))
        local = [group for group in candidates if group.server_id == target_server]
        if local:
            return sorted(local, key=lambda group: group.id)[0]
        for source_server, group_id, dst_server in sorted(backhaul):
            if dst_server != target_server:
                continue
            for group in candidates:
                if group.server_id == source_server and group.id == group_id:
                    return group
        return None

    def _downstream_compute_times(self, assignments):
        downstream: Dict[AssignmentKey, float] = {}
        for task in self.tasks.values():
            children = {sub.id: [] for sub in task.subtasks}
            for sub in task.subtasks:
                for pred_id in sub.predecessors:
                    if pred_id in children:
                        children[pred_id].append(sub)

            def visit(subtask):
                key = (task.id, subtask.id)
                if key in downstream:
                    return downstream[key]
                compute = self._subtask_compute_time(key, assignments)
                child_tail = max((visit(child) for child in children[subtask.id]), default=0.0)
                downstream[key] = compute + child_tail
                return downstream[key]

            for subtask in task.subtasks:
                visit(subtask)
        return downstream

    def _subtask_compute_time(self, key, assignments):
        return max((self.experts[eid].latency for _, eid in assignments.get(key, [])), default=0.0)

    def _strict_group_budget(self, group_key, dependencies, downstream):
        dependent_keys = dependencies.get(group_key, set())
        if not dependent_keys:
            return self.evaluator.bandwidth_time_budget()
        budgets = []
        for task_id, subtask_id in dependent_keys:
            task = self.tasks[task_id]
            if task.deadline == float("inf"):
                budgets.append(self.evaluator.bandwidth_time_budget())
            else:
                remaining = task.deadline - downstream.get((task_id, subtask_id), 0.0)
                budgets.append(max(remaining, 1e-12))
        return max(min(budgets), 1e-12)

    def _derive_bandwidth_for_budget(self, group, budget):
        if not group.devices or not group.required_features:
            return self.config.min_bandwidth
        return self.evaluator.derive_group_bandwidth_for_budget(group, budget)

    def _feature_to_groups(self, groups):
        feature_to_groups: Dict[str, List[GroupSpec]] = {}
        for group in groups:
            for device_id in group.devices:
                for feature in self.devices[device_id].features:
                    feature_to_groups.setdefault(feature, []).append(group)
        return feature_to_groups

    def _activated_expert_map(self, assignments):
        placement = {sid: set() for sid in self.servers}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                placement.setdefault(server_id, set()).add(expert_id)
        return {sid: sorted(experts) for sid, experts in placement.items()}

    def _group_map(self, groups):
        grouped = {sid: [] for sid in self.servers}
        for group in groups:
            grouped.setdefault(group.server_id, []).append(list(group.devices))
        return grouped

    def _group_bandwidth_map(self, groups):
        bandwidths = {sid: [] for sid in self.servers}
        for group in groups:
            bandwidths.setdefault(group.server_id, []).append(group.bandwidth)
        return bandwidths

    @staticmethod
    def _assignment_label(key):
        return f"{key[0]}:{key[1]}"

    @staticmethod
    def _group_label(group):
        return f"{group.server_id}:{group.id}"

    @staticmethod
    def _group_key_label(group_key):
        return f"{group_key[0]}:{group_key[1]}"

    @staticmethod
    def _id_sort_key(value):
        prefix, _, suffix = str(value).rpartition("_")
        try:
            return prefix, int(suffix), str(value)
        except ValueError:
            return str(value), 0, str(value)

    @staticmethod
    def _squared_distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _safe_usage(cost: float, unit_cost: float) -> float | int:
    if unit_cost == 0:
        return 0
    usage = cost / unit_cost
    if abs(usage - round(usage)) < 1e-9:
        return int(round(usage))
    return usage


def _cost_breakdown(result: ChainedPipelineResult, cost_units: Mapping[str, float]) -> Dict[str, Any]:
    c_act = float(cost_units.get("activation", 0.0))
    c_bw = float(cost_units.get("bandwidth", 0.0))
    c_fwd = float(cost_units.get("forwarding", 0.0))
    return {
        "activation": {
            "usage": _safe_usage(result.activation_cost, c_act),
            "usage_unit": "server-expert activations",
            "unit_cost": c_act,
            "cost": int(result.activation_cost + 0.5),
        },
        "bandwidth": {
            "usage": _safe_usage(result.bandwidth_cost, c_bw),
            "usage_unit": "Hz",
            "unit_cost": c_bw,
            "cost": int(result.bandwidth_cost + 0.5),
        },
        "forwarding": {
            "usage": _safe_usage(result.forwarding_cost, c_fwd),
            "usage_unit": "forwarding events",
            "unit_cost": c_fwd,
            "cost": int(result.forwarding_cost + 0.5),
        },
    }


def result_to_jsonable(
    result: ChainedPipelineResult,
    network_context: Mapping[str, Any] | None = None,
    cost_units: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    evaluation = result.evaluation
    cost_units = cost_units or {}
    payload: Dict[str, Any] = {
        "method": "hybrid_topk_distance_backhaul",
        "objective": {
            "total_cost": int(result.total_cost + 0.5),
            "activation_cost": int(result.activation_cost + 0.5),
            "bandwidth_cost": int(result.bandwidth_cost + 0.5),
            "forwarding_cost": int(result.forwarding_cost + 0.5),
        },
        "cost_breakdown": _cost_breakdown(result, cost_units),
        "expert_placement": result.expert_placement,
        "subtask_assignment": {f"{tid}:{sid}": pairs for (tid, sid), pairs in result.subtask_assignment.items()},
        "subtask_data_requirements": result.subtask_data_requirements,
        "server_required_features": result.server_required_features,
        "selected_probability": result.selected_probability,
        "required_probability": result.required_probability,
        "reconstruction_loss": result.reconstruction_loss,
        "performance_loss": result.performance_loss,
        "rsma_groups": result.rsma_groups,
        "group_required_features": result.group_required_features,
        "group_time_budgets": result.group_time_budgets,
        "rsma_group_bandwidths": result.rsma_group_bandwidths,
        "backhaul": [list(item) for item in sorted(result.backhaul)],
        "backhaul_plan": [decision.__dict__ for decision in result.backhaul_plan],
        "violations": result.violations,
        "task_finish_time": evaluation.timing.task_finish_time if evaluation else {},
    }
    if network_context is not None:
        payload["network_model"] = dict(network_context)
    return payload


def run_chained_pipeline(
    graphs: Sequence[Any],
    output_path: Path,
    num_experts: int,
    num_iot_features: int,
    top_k: int = 2,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (2, 6),
    connected_servers_per_device: int | None = None,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
    default_wired_rate: float = 1e9,
    wired_rate_range: tuple[float, float] | None = None,
    c_bw: float = 1e-3,
    c_act: float = 1.0,
    c_fwd: float = 1.0,
    uplink_time_budget: float | None = None,
    bandwidth_time_fraction: float = 1.0,
    min_bandwidth: float = 0.0,
    default_feature_bits: float = 12000.0,
    wavelength: float = 0.125,
    noise_power: float = 1e-18,
    common_power_ratio: float = 0.6,
    max_device_power: float = 1.2589e-3,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    beamforming_correlation_weight: float = 0.0,
    loss_threshold: float | None = None,
    lambda_reconstruction: float = 0.1,
    calibration_alpha: float = 0.1,
    reconstruction_sigma: float = 1.0,
    random_seed: int = 42,
    max_group_size: int = 4,
    rank_by: str = "gating",
    clusters_per_server: Optional[int] = None,
    kmeans_iterations: int = 20,
) -> ChainedPipelineResult:
    rng = random.Random(random_seed)
    tasks = graphs_to_tasks(graphs, loss_threshold)
    experts = build_simple_experts(num_experts)
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        gpu_memory=server_gpu_memory,
        default_wired_rate=default_wired_rate,
        wired_rate_range=wired_rate_range,
        rng=rng,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=num_servers,
        features_per_device_range=features_per_device_range,
        connected_servers_per_device=connected_servers_per_device,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        rng=rng,
        max_power=max_device_power,
    )
    pipeline = ChainedComparisonPipeline(
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
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            derive_bandwidth=True,
            uplink_time_budget=uplink_time_budget,
            bandwidth_time_fraction=bandwidth_time_fraction,
            min_bandwidth=min_bandwidth,
            default_wired_rate=default_wired_rate,
            noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            beamforming_correlation_weight=beamforming_correlation_weight,
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
            reconstruction_sigma=reconstruction_sigma,
            default_feature_bits=default_feature_bits,
            wavelength=wavelength,
        ),
    )
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
        "hybrid_algorithm_parameters": {
            "top_k": top_k,
            "rank_by": rank_by,
            "clusters_per_server": clusters_per_server,
            "kmeans_iterations": kmeans_iterations,
            "local_grouping_only": True,
            "bandwidth_budget": "strictest dependent subtask remaining time",
        },
        "rsma_parameters": {
            "num_servers": num_servers,
            "num_iot_devices": num_iot_devices,
            "features_per_device_range": features_per_device_range,
            "connected_servers_per_device": connected_servers_per_device,
            "experts_per_server": experts_per_server,
            "server_gpu_memory": server_gpu_memory,
            "default_wired_rate": default_wired_rate,
            "wired_rate_range": wired_rate_range,
            "bandwidth_mode": "derived_by_group_slack",
            "uplink_time_budget": uplink_time_budget,
            "bandwidth_time_fraction": bandwidth_time_fraction,
            "min_bandwidth": min_bandwidth,
            "default_feature_bits": default_feature_bits,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "beamforming_correlation_weight": beamforming_correlation_weight,
            "max_group_size": max_group_size,
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



















