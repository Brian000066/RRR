from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.formulation import (
    AssignmentKey,
    EvaluationResult,
    ExpertId,
    FormulationConfig,
    FormulationEvaluator,
    GroupSpec,
    ServerId,
    normalize_devices,
    normalize_experts,
    normalize_servers,
    normalize_tasks,
)


@dataclass
class JRGEPResult:
    expert_placement: Dict[ServerId, List[ExpertId]]
    subtask_assignment: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]]
    rsma_groups: Dict[ServerId, List[List[str]]]
    rsma_group_bandwidths: Dict[ServerId, List[float]]
    total_cost: float
    activation_cost: float
    bandwidth_cost: float
    forwarding_cost: float
    backhaul: Set[Tuple[ServerId, str, ServerId]] = field(default_factory=set)
    violations: List[str] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None


class JRGEPScheduler:
    """Greedy scheduler for the PPTX formulation.

    This class does not generate tasks, experts, or IoT devices. It only consumes
    dictionaries/objects created by the user's own code and produces the decision
    variables:

    x: subtask -> selected (server, expert)
    y: activated expert on server
    b: IoT/device membership in RSMA DB groups
    w: feature-group backhaul between servers
    """

    def __init__(
        self,
        servers: Iterable[Any],
        devices: Iterable[Any],
        experts: Mapping[str, Any],
        tasks: Iterable[Any],
        config: Optional[FormulationConfig] = None,
    ):
        self.config = config or FormulationConfig()
        self.servers = normalize_servers(servers)
        self.devices = normalize_devices(devices)
        self.experts = normalize_experts(experts)
        self.tasks = normalize_tasks(tasks)
        self.evaluator = FormulationEvaluator(
            self.servers,
            self.devices,
            self.experts,
            self.tasks,
            self.config,
        )

    def schedule(self) -> JRGEPResult:
        assignments = self._assign_subtasks()
        groups = self._build_rsma_groups()
        backhaul = self._derive_backhaul(assignments, groups)
        groups = self.evaluator.resolve_group_bandwidths(groups, assignments, backhaul)
        evaluation = self.evaluator.evaluate(assignments, groups, backhaul)
        placement = self._activated_expert_map(assignments)
        grouped = self._group_map(groups)
        group_bandwidths = self._group_bandwidth_map(groups)
        return JRGEPResult(
            expert_placement=placement,
            subtask_assignment=assignments,
            rsma_groups=grouped,
            rsma_group_bandwidths=group_bandwidths,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            backhaul=backhaul,
            violations=evaluation.violations,
            evaluation=evaluation,
        )

    def _build_rsma_groups(self) -> List[GroupSpec]:
        global_required = self._global_required_features()
        devices_by_server: Dict[ServerId, List[str]] = {server_id: [] for server_id in self.servers}
        fallback_server = next(iter(self.servers), "0")
        for device_id, device in self.devices.items():
            if global_required and not (device.features & global_required):
                continue
            server_id = device.home_server or fallback_server
            if server_id not in devices_by_server:
                devices_by_server[server_id] = []
            devices_by_server[server_id].append(device_id)

        groups: List[GroupSpec] = []
        for server_id, device_ids in devices_by_server.items():
            for index in range(0, len(device_ids), max(1, self.config.max_group_size)):
                members = tuple(device_ids[index : index + self.config.max_group_size])
                if not members:
                    continue
                required_features = set()
                for device_id in members:
                    required_features.update(self.devices[device_id].features & global_required)
                if not required_features:
                    continue
                group_id = f"g{len(groups)}"
                common_power = {
                    device_id: self.devices[device_id].max_power * self.config.common_power_ratio
                    for device_id in members
                }
                private_power = {
                    device_id: self.devices[device_id].max_power * (1.0 - self.config.common_power_ratio)
                    for device_id in members
                }
                groups.append(
                    GroupSpec(
                        server_id=server_id,
                        id=group_id,
                        devices=members,
                        bandwidth=0.0,
                        required_features=required_features,
                        common_power=common_power,
                        private_power=private_power,
                    )
                )
        return groups

    def _global_required_features(self) -> Set[str]:
        required: Set[str] = set()
        for task in self.tasks.values():
            for subtask in task.subtasks:
                required.update(subtask.required_features)
        return required

    def _assign_subtasks(self) -> Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]]:
        assignments: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]] = {}
        activated: Dict[ServerId, Set[ExpertId]] = {
            server_id: set(server.active_experts) for server_id, server in self.servers.items()
        }
        used_memory: Dict[ServerId, float] = {
            server_id: sum(self.experts[e].memory for e in expert_ids if e in self.experts)
            for server_id, expert_ids in activated.items()
        }

        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                selected: List[Tuple[ServerId, ExpertId]] = []
                selected_experts: Set[ExpertId] = set()
                performance = 0.0
                target = self.evaluator.required_selection_probability(subtask)

                while not selected or performance < target:
                    candidate = self._best_candidate(
                        subtask=subtask,
                        activated=activated,
                        used_memory=used_memory,
                        selected_experts=selected_experts,
                        target_remaining=max(target - performance, 0.0),
                    )
                    if candidate is None:
                        break
                    server_id, expert_id = candidate
                    selected.append((server_id, expert_id))
                    selected_experts.add(expert_id)
                    performance += self.evaluator.expert_contribution(subtask, expert_id)
                    if expert_id not in activated[server_id]:
                        activated[server_id].add(expert_id)
                        used_memory[server_id] += self.experts[expert_id].memory

                assignments[key] = selected
        return assignments


    def _best_candidate(
        self,
        subtask,
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
        selected_experts: Set[ExpertId],
        target_remaining: float,
    ) -> Optional[Tuple[ServerId, ExpertId]]:
        best: Optional[Tuple[float, ServerId, ExpertId]] = None
        for server_id, server in self.servers.items():
            deployable = server.stored_experts
            for expert_id in deployable:
                if expert_id in selected_experts:
                    continue
                expert = self.experts.get(expert_id)
                if expert is None:
                    continue
                contribution = self.evaluator.expert_contribution(subtask, expert_id)
                if contribution <= 0.0:
                    continue
                memory_after = used_memory.get(server_id, 0.0)
                activation_penalty = 0.0
                if expert_id not in activated.get(server_id, set()):
                    memory_after += expert.memory
                    activation_penalty = self.config.c_act
                if memory_after > server.gpu_memory:
                    continue
                uncovered_loss = max(0.0, target_remaining - contribution)
                score = activation_penalty + expert.latency + uncovered_loss
                if best is None or score < best[0]:
                    best = (score, server_id, expert_id)
        if best is None:
            return None
        return best[1], best[2]

    def _derive_backhaul(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
    ) -> Set[Tuple[ServerId, str, ServerId]]:
        feature_to_groups: Dict[str, List[GroupSpec]] = {}
        for group in groups:
            for device_id in group.devices:
                for feature in self.devices[device_id].features:
                    feature_to_groups.setdefault(feature, []).append(group)

        backhaul: Set[Tuple[ServerId, str, ServerId]] = set()
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                for target_server, _ in assignments.get(key, []):
                    for feature in subtask.required_features:
                        candidates = feature_to_groups.get(feature, [])
                        if any(group.server_id == target_server for group in candidates):
                            continue
                        if candidates:
                            source = candidates[0]
                            backhaul.add((source.server_id, source.id, target_server))
        return backhaul

    def _activated_expert_map(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> Dict[ServerId, List[ExpertId]]:
        placement: Dict[ServerId, Set[ExpertId]] = {server_id: set() for server_id in self.servers}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                placement.setdefault(server_id, set()).add(expert_id)
        return {server_id: sorted(experts) for server_id, experts in placement.items()}

    def _group_map(self, groups: Sequence[GroupSpec]) -> Dict[ServerId, List[List[str]]]:
        grouped: Dict[ServerId, List[List[str]]] = {server_id: [] for server_id in self.servers}
        for group in groups:
            grouped.setdefault(group.server_id, []).append(list(group.devices))
        return grouped

    def _group_bandwidth_map(self, groups: Sequence[GroupSpec]) -> Dict[ServerId, List[float]]:
        bandwidths: Dict[ServerId, List[float]] = {server_id: [] for server_id in self.servers}
        for group in groups:
            bandwidths.setdefault(group.server_id, []).append(group.bandwidth)
        return bandwidths
