from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.formulation import AssignmentKey, DeviceId, ExpertId, GroupSpec, ServerId, SubtaskSpec


class Phase1ServerExpertSelectionMixin:
    """Phase 1: cluster-aware server/expert selection."""

    def _assign_topk_experts(self):
        assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        used_memory = {
            sid: sum(self.experts[eid].memory for eid in eids if eid in self.experts)
            for sid, eids in activated.items()
        }

        selection_units = [
            ((task.id, subtask.id), subtask)
            for task in self.tasks.values()
            for subtask in task.subtasks
        ]
        unsatisfied_after_phase1 = 0
        total_pairs = 0
        top_k_limit = max(1, min(int(self.top_k), len(self.experts)))

        for key, subtask in selection_units:
            selected = self._select_single_server_expert_set(
                key=key,
                subtask=subtask,
                assignments=assignments,
                activated=activated,
                used_memory=used_memory,
                top_k_limit=top_k_limit,
            )
            selected_experts = {expert_id for _, expert_id in selected}
            threshold = self.evaluator.conformal_loss_threshold(subtask)

            if not selected:
                self.scheduler_violations.append(
                    f"Phase1 {key}: no feasible (server, expert) set"
                )
            elif self._loss_for_pairs(subtask, selected_experts, selected) > threshold + 1e-12:
                unsatisfied_after_phase1 += 1

            for server_id, expert_id in selected:
                self._activate(server_id, expert_id, activated, used_memory)

            assignments[key] = list(selected)
            selected_experts_by_key[key] = set(selected_experts)
            selected_probability[self._assignment_label(key)] = self.evaluator.selection_probability(
                subtask,
                selected_experts,
            )
            total_pairs += len(selected)

        self.phase1_report.update(
            {
                "selection_units": len(selection_units),
                "top_k_limit": top_k_limit,
                "unsatisfied_after_phase1": unsatisfied_after_phase1,
                "average_pairs_per_node": total_pairs / len(selection_units) if selection_units else 0.0,
                "mode": "single_server_expert_set_selection",
                "note": "r_sim is used for gating smoothing; Phase 1 assigns each node to one MEC server and selects all experts on that server.",
            }
        )
        return assignments, selected_probability, selected_experts_by_key, activated, used_memory

    def _select_single_server_expert_set(
        self,
        *,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
        top_k_limit: int,
    ) -> list[tuple[ServerId, ExpertId]]:
        threshold = self.evaluator.conformal_loss_threshold(subtask)
        best_feasible = None
        best_fallback = None

        for server_id in sorted(self.servers):
            candidate_experts = self._single_server_candidate_experts(
                server_id,
                subtask,
                activated,
                used_memory,
                top_k_limit,
            )
            if not candidate_experts:
                continue
            for size in range(1, len(candidate_experts) + 1):
                for combo in combinations(candidate_experts, size):
                    pairs = tuple((server_id, expert_id) for expert_id in combo)
                    if not self._pair_set_memory_feasible(pairs, activated, used_memory):
                        continue
                    experts = set(combo)
                    probability = self.evaluator.selection_probability(subtask, experts)
                    rec_loss = self._estimated_local_reconstruction_for_pairs(subtask, pairs)
                    loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, rec_loss)
                    cost = self._pair_set_incremental_cost(key, subtask, pairs, assignments, activated, used_memory)
                    item = (cost, len(pairs), rec_loss, -probability, server_id, pairs)
                    if loss <= threshold + 1e-12:
                        if best_feasible is None or item[:5] < best_feasible[:5]:
                            best_feasible = item
                    fallback_item = (loss, cost, len(pairs), rec_loss, -probability, server_id, pairs)
                    if best_fallback is None or fallback_item[:6] < best_fallback[:6]:
                        best_fallback = fallback_item

        if best_feasible is not None:
            return list(best_feasible[-1])
        if best_fallback is not None:
            return list(best_fallback[-1])
        return []

    def _single_server_candidate_experts(
        self,
        server_id: ServerId,
        subtask: SubtaskSpec,
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
        top_k_limit: int,
    ) -> list[ExpertId]:
        server = self.servers[server_id]
        ranked = [
            expert_id
            for expert_id in self._cluster_ranked_experts([((server_id, subtask.id), subtask)])
            if expert_id in server.stored_experts
        ]
        candidates: list[ExpertId] = []
        memory = used_memory.get(server_id, 0.0)
        for expert_id in ranked:
            extra_memory = 0.0
            if expert_id not in activated.get(server_id, set()):
                extra_memory = self.experts[expert_id].memory
            if memory + extra_memory > server.gpu_memory + 1e-12:
                continue
            candidates.append(expert_id)
            memory += extra_memory
            if len(candidates) >= top_k_limit:
                break
        return candidates

    def _select_bounded_expert_server_set(
        self,
        *,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
        top_k_limit: int,
    ) -> list[tuple[ServerId, ExpertId]]:
        candidates = self._bounded_pair_candidates(
            key,
            subtask,
            assignments,
            activated,
            used_memory,
            top_k_limit,
        )
        if not candidates:
            return []

        threshold = self.evaluator.conformal_loss_threshold(subtask)
        best_feasible = None
        best_fallback = None
        max_size = min(top_k_limit, len(candidates))
        for size in range(1, max_size + 1):
            for combo in combinations(candidates, size):
                pairs = tuple((server_id, expert_id) for server_id, expert_id, _ in combo)
                experts = {expert_id for _, expert_id in pairs}
                if len(experts) != len(pairs):
                    continue
                if not self._pair_set_memory_feasible(pairs, activated, used_memory):
                    continue
                probability = self.evaluator.selection_probability(subtask, experts)
                rec_loss = self._estimated_local_reconstruction_for_pairs(subtask, pairs)
                loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, rec_loss)
                cost = self._pair_set_incremental_cost(key, subtask, pairs, assignments, activated, used_memory)
                servers = len({server_id for server_id, _ in pairs})
                item = (cost, servers, len(pairs), rec_loss, -probability, pairs)
                if loss <= threshold + 1e-12:
                    if best_feasible is None or item[:5] < best_feasible[:5]:
                        best_feasible = item
                fallback_item = (loss, cost, servers, len(pairs), rec_loss, -probability, pairs)
                if best_fallback is None or fallback_item[:6] < best_fallback[:6]:
                    best_fallback = fallback_item
            if best_feasible is not None:
                break

        if best_feasible is not None:
            return list(best_feasible[-1])
        if best_fallback is not None:
            return list(best_fallback[-1])
        return []

    def _bounded_pair_candidates(
        self,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
        top_k_limit: int,
    ) -> list[tuple[ServerId, ExpertId, float]]:
        expert_limit = min(len(self.experts), top_k_limit + 2)
        ranked_experts = self._cluster_ranked_experts([(key, subtask)])[:expert_limit]
        candidates: list[tuple[ServerId, ExpertId, float]] = []
        for expert_id in ranked_experts:
            server_items = []
            for server_id in self._feasible_servers_for_cluster_expert(expert_id, activated, used_memory):
                cost = self._cluster_pair_incremental_cost(
                    cluster=[(key, subtask)],
                    server_id=server_id,
                    expert_id=expert_id,
                    selected_pairs=(),
                    assignments=assignments,
                    activated=activated,
                    used_memory=used_memory,
                )
                server_items.append((cost, server_id))
            server_items.sort()
            for cost, server_id in server_items[:2]:
                candidates.append((server_id, expert_id, cost))
        candidates.sort(key=lambda item: (item[2], item[0], item[1]))
        return candidates

    def _pair_set_memory_feasible(
        self,
        pairs: Sequence[tuple[ServerId, ExpertId]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> bool:
        trial_memory = dict(used_memory)
        trial_active = {server_id: set(experts) for server_id, experts in activated.items()}
        for server_id, expert_id in pairs:
            trial_active.setdefault(server_id, set())
            if expert_id in trial_active[server_id]:
                continue
            trial_memory[server_id] = trial_memory.get(server_id, 0.0) + self.experts[expert_id].memory
            trial_active[server_id].add(expert_id)
            if trial_memory[server_id] > self.servers[server_id].gpu_memory + 1e-12:
                return False
        return True

    def _pair_set_incremental_cost(
        self,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        pairs: Sequence[tuple[ServerId, ExpertId]],
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> float:
        activation_cost = 0.0
        seen_activation = set()
        for server_id, expert_id in pairs:
            pair = (server_id, expert_id)
            if pair in seen_activation:
                continue
            seen_activation.add(pair)
            if expert_id not in activated.get(server_id, set()):
                activation_cost += self.config.c_act * self.experts[expert_id].memory

        forwarding_cost = 0.0
        for server_id, _ in pairs:
            forwarding_cost += self._phase1_forwarding_cost(
                key,
                subtask,
                server_id,
                assignments,
                pairs,
            )
        data_cost = sum(
            self._phase1_estimated_local_data_cost([(key, subtask)], server_id)
            for server_id in {pair_server for pair_server, _ in pairs}
        )
        return activation_cost + forwarding_cost + data_cost + 1e-12
    def _node_clusters(self) -> list[list[tuple[AssignmentKey, SubtaskSpec]]]:
        subtask_by_key = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        parent = {key: key for key in subtask_by_key}

        def find(key: AssignmentKey) -> AssignmentKey:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(left: AssignmentKey, right: AssignmentKey) -> None:
            if left not in parent or right not in parent:
                return
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left, right in self.r_sim:
            union(left, right)

        grouped: Dict[AssignmentKey, list[tuple[AssignmentKey, SubtaskSpec]]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                grouped.setdefault(find(key), []).append((key, subtask))
        return list(grouped.values())

    def _cluster_can_meet_loss_bound(
        self,
        cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]],
        selected_experts: Set[ExpertId],
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
    ) -> bool:
        if not selected_experts or not selected_pairs:
            return False
        return all(
            self._loss_for_pairs(subtask, selected_experts, selected_pairs)
            <= self.evaluator.conformal_loss_threshold(subtask) + 1e-12
            for _, subtask in cluster
        )

    def _loss_for_pairs(
        self,
        subtask: SubtaskSpec,
        selected_experts: Set[ExpertId],
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
    ) -> float:
        if not selected_experts or not selected_pairs:
            return float("inf")
        probability = self.evaluator.selection_probability(subtask, selected_experts)
        local_reconstruction = self._estimated_local_reconstruction_for_pairs(subtask, selected_pairs)
        return self.evaluator.performance_loss_from_probability_and_reconstruction(
            probability,
            local_reconstruction,
        )

    def _best_cluster_pair(
        self,
        cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]],
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
        selected_experts: Set[ExpertId],
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> Optional[tuple[ServerId, ExpertId, float, float, float]]:
        best = None
        for expert_id in self._cluster_ranked_experts(cluster):
            if expert_id in selected_experts:
                continue
            improvement = self._cluster_expert_probability_improvement(
                cluster,
                selected_experts,
                expert_id,
                selected_pairs,
            )
            if improvement <= 0.0:
                continue

            best_server_item = None
            for server_id in self._feasible_servers_for_cluster_expert(expert_id, activated, used_memory):
                cost = self._cluster_pair_incremental_cost(
                    cluster=cluster,
                    server_id=server_id,
                    expert_id=expert_id,
                    selected_pairs=selected_pairs,
                    assignments=assignments,
                    activated=activated,
                    used_memory=used_memory,
                )
                item = (cost, server_id)
                if best_server_item is None or item < best_server_item:
                    best_server_item = item
            if best_server_item is None:
                continue

            cost, server_id = best_server_item
            score = improvement / max(cost, 1e-12)
            item = (score, improvement, -cost, server_id, expert_id)
            if best is None or item > best:
                best = item
        if best is None:
            return None
        return best[3], best[4], best[0], best[1], -best[2]

    def _cluster_expert_probability_improvement(
        self,
        cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]],
        selected_experts: Set[ExpertId],
        expert_id: ExpertId,
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
    ) -> float:
        trial_experts = set(selected_experts)
        trial_experts.add(expert_id)
        improvement = 0.0
        for _, subtask in cluster:
            current_rec = self._estimated_local_reconstruction_for_pairs(subtask, selected_pairs)
            current_probability = self.evaluator.selection_probability(subtask, selected_experts)
            trial_probability = self.evaluator.selection_probability(subtask, trial_experts)
            current_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                current_probability,
                current_rec,
            )
            trial_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                trial_probability,
                current_rec,
            )
            improvement += max(0.0, min(current_loss, 1e12) - min(trial_loss, 1e12))
        return improvement

    def _cluster_ranked_experts(self, cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]]) -> list[ExpertId]:
        def score(expert_id: ExpertId) -> float:
            return sum(self.evaluator.expert_contribution(subtask, expert_id) for _, subtask in cluster)
        return [expert_id for expert_id in sorted(self.experts, key=score, reverse=True) if score(expert_id) > 0.0]

    def _feasible_servers_for_cluster_expert(
        self,
        expert_id: ExpertId,
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> list[ServerId]:
        expert = self.experts[expert_id]
        feasible = []
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            memory_after = used_memory.get(server_id, 0.0)
            if expert_id not in activated.get(server_id, set()):
                memory_after += expert.memory
            if memory_after <= server.gpu_memory:
                feasible.append(server_id)
        return sorted(feasible)

    def _cluster_pair_incremental_cost(
        self,
        cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]],
        server_id: ServerId,
        expert_id: ExpertId,
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> float:
        expert = self.experts[expert_id]
        server = self.servers[server_id]
        memory_after = used_memory.get(server_id, 0.0)
        activation_cost = 0.0
        if expert_id not in activated.get(server_id, set()):
            memory_after += expert.memory
            activation_cost = self.config.c_act * expert.memory
        if memory_after > server.gpu_memory:
            return float("inf")

        memory_factor = self._memory_availability_factor(
            server_id,
            expert_id,
            activated,
            used_memory,
        )
        if memory_factor <= 0.0:
            return float("inf")

        assigned_edges = [
            (key, subtask)
            for key, subtask in cluster
            if self._has_assigned_predecessor(key, subtask, assignments)
        ]
        if assigned_edges:
            forwarding_cost = sum(
                self._phase1_forwarding_cost(
                    key,
                    subtask,
                    server_id,
                    assignments,
                    selected_pairs,
                )
                for key, subtask in assigned_edges
            ) / len(assigned_edges)
            data_cost = self._phase1_estimated_local_data_cost(cluster, server_id)
            denominator = activation_cost + forwarding_cost + data_cost + 1e-12
            return max(denominator / max(memory_factor, 1e-12), 1e-12)

        normalized_ratio = self._normalized_rate_hop_ratios().get(server_id, 1.0)
        data_cost = self._phase1_estimated_local_data_cost(cluster, server_id)
        return max(data_cost + (1.0 / max(memory_factor * normalized_ratio, 1e-12)), 1e-12)

    def _phase1_estimated_local_data_cost(
        self,
        cluster: Sequence[tuple[AssignmentKey, SubtaskSpec]],
        server_id: ServerId,
    ) -> float:
        total_cost = 0.0
        for _, subtask in cluster:
            required = set(subtask.required_features)
            if not required:
                continue
            selected_devices, covered = self._phase1_greedy_local_cover(server_id, required)
            if selected_devices:
                total_cost += self._phase1_estimated_uplink_cost(server_id, selected_devices, covered)
            missing = required - covered
            if missing:
                total_cost += self._phase1_missing_feature_backhaul_proxy(server_id, missing)
        return total_cost / max(len(cluster), 1)

    def _phase1_greedy_local_cover(
        self,
        server_id: ServerId,
        required: Set[str],
    ) -> tuple[list[DeviceId], Set[str]]:
        remaining = set(required)
        selected: list[DeviceId] = []
        covered: Set[str] = set()
        local_devices = [
            device_id
            for device_id, device in self.devices.items()
            if self._device_can_transmit_to_server(device_id, server_id)
            and device.features & required
        ]
        while remaining and local_devices:
            best = None
            for device_id in local_devices:
                device_features = self.devices[device_id].features
                gain = len(device_features & remaining)
                if gain <= 0:
                    continue
                payload = self.evaluator.device_feature_volume(device_id)
                item = (gain / max(payload, 1e-12), gain, -payload, device_id)
                if best is None or item > best:
                    best = item
            if best is None:
                break
            device_id = best[3]
            selected.append(device_id)
            newly_covered = self.devices[device_id].features & remaining
            covered.update(newly_covered)
            remaining -= newly_covered
            local_devices.remove(device_id)
        return selected, covered

    def _phase1_estimated_uplink_cost(
        self,
        server_id: ServerId,
        devices: Sequence[DeviceId],
        covered_features: Set[str],
    ) -> float:
        if not devices:
            return 0.0
        max_group_size = max(int(self.config.max_group_size), 1)
        total_bandwidth = 0.0
        for index in range(0, len(devices), max_group_size):
            group_devices = tuple(devices[index:index + max_group_size])
            group = GroupSpec(
                server_id=server_id,
                id=f"phase1_est_{server_id}_{index}",
                devices=group_devices,
                bandwidth=0.0,
                required_features=set(covered_features),
            )
            total_bandwidth += self.evaluator.derive_group_bandwidth(group)
        return self.config.c_bw * total_bandwidth

    def _phase1_missing_feature_backhaul_proxy(
        self,
        server_id: ServerId,
        missing_features: Set[str],
    ) -> float:
        proxy_cost = 0.0
        for feature in missing_features:
            best = None
            for source_server in self.servers:
                if source_server == server_id:
                    continue
                has_feature = any(
                    self._device_can_transmit_to_server(device_id, source_server)
                    and feature in device.features
                    for device_id, device in self.devices.items()
                )
                if not has_feature:
                    continue
                hop_cost = self.config.c_fwd * self.evaluator.wired_weight(source_server, server_id)
                transfer_time = self.evaluator.feature_bits(feature) / max(
                    self.evaluator.wired_rate(source_server, server_id),
                    1e-12,
                )
                item = (hop_cost, transfer_time)
                if best is None or item < best:
                    best = item
            if best is None:
                proxy_cost += self.config.c_fwd + self.config.c_bw * self.evaluator.feature_bits(feature)
            else:
                proxy_cost += best[0]
        return proxy_cost

    def _estimated_local_reconstruction_for_pairs(
        self,
        subtask: SubtaskSpec,
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
    ) -> float:
        required = set(subtask.required_features)
        if not required:
            return 0.0
        weighted_loss = 0.0
        total_experts = 0
        for server_id, _ in selected_pairs:
            local_devices = self._server_transmittable_devices(server_id)
            local_features = self._server_local_features(local_devices) & required
            weighted_loss += self.evaluator.reconstruction_loss_for_features(subtask, local_features)
            total_experts += 1
        if total_experts <= 0:
            return subtask.reconstruction_loss
        return weighted_loss / total_experts

    def _best_server_for_expert_with_dependency(
        self,
        expert_id: ExpertId,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        current_pairs: Sequence[tuple[ServerId, ExpertId]],
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> Optional[ServerId]:
        best = None
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            cost = self._cluster_pair_incremental_cost(
                cluster=[(key, subtask)],
                server_id=server_id,
                expert_id=expert_id,
                selected_pairs=current_pairs,
                assignments=assignments,
                activated=activated,
                used_memory=used_memory,
            )
            if best is None or cost < best[0]:
                best = (cost, server_id)
        return None if best is None else best[1]

    def _memory_availability_factor(
        self,
        server_id: ServerId,
        expert_id: ExpertId,
        activated: Mapping[ServerId, Set[ExpertId]],
        used_memory: Mapping[ServerId, float],
    ) -> float:
        if expert_id in activated.get(server_id, set()):
            return 1.0
        server = self.servers[server_id]
        expert = self.experts[expert_id]
        memory_after = used_memory.get(server_id, 0.0) + expert.memory
        if memory_after > server.gpu_memory:
            return 0.0
        return max((server.gpu_memory - memory_after) / max(server.gpu_memory, 1e-12), 1e-12)

    def _has_assigned_predecessor(
        self,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
    ) -> bool:
        for predecessor_id in subtask.predecessors:
            predecessor_key = (key[0], predecessor_id)
            if assignments.get(predecessor_key):
                return True
        return False

    def _phase1_forwarding_cost(
        self,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        server_id: ServerId,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        selected_pairs: Sequence[tuple[ServerId, ExpertId]],
    ) -> float:
        reference_servers: Set[ServerId] = set()
        for predecessor_id in subtask.predecessors:
            predecessor_key = (key[0], predecessor_id)
            reference_servers.update(
                source_server
                for source_server, _ in assignments.get(predecessor_key, [])
            )
        reference_servers.update(pair_server for pair_server, _ in selected_pairs)
        return sum(
            self._forwarding_cost_between(reference_server, server_id)
            for reference_server in reference_servers
        )

    def _normalized_rate_hop_ratios(self) -> Dict[ServerId, float]:
        ratios = {server_id: self._rate_hop_ratio(server_id) for server_id in self.servers}
        max_ratio = max(ratios.values()) if ratios else 1.0
        return {
            server_id: max(ratio / max(max_ratio, 1e-12), 1e-12)
            for server_id, ratio in ratios.items()
        }

    def _rate_hop_ratio(self, server_id: ServerId) -> float:
        values = []
        for target_server in self.servers:
            if target_server == server_id:
                continue
            rate = max(float(self.evaluator.wired_rate(server_id, target_server)), 1e-12)
            hop = max(float(self.evaluator.wired_weight(server_id, target_server)), 1e-12)
            values.append(rate / hop)
        if not values:
            return 1.0
        return sum(values) / len(values)

    def _predecessor_forwarding_penalty(
        self,
        key: AssignmentKey,
        subtask: SubtaskSpec,
        server_id: ServerId,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
    ) -> float:
        return self._phase1_forwarding_cost(key, subtask, server_id, assignments, ())

    def _forwarding_cost_between(self, source_server: ServerId, target_server: ServerId) -> float:
        if source_server == target_server:
            return 0.0
        return self.config.c_fwd * self.evaluator.wired_weight(source_server, target_server)

