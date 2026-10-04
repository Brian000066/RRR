from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import (
    ChainedBackhaulDecision,
    ChainedPipelineResult,
)
from utils.formulation import (
    AssignmentKey,
    DeviceId,
    FormulationEvaluator,
    GroupSpec,
    ServerId,
)


class Phase2GroupingBackhaulMixin:
    """Phase II: CFRG IoT selection, compatibility ordering, and grouping."""

    def _run_phase2_for_assignments(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        activated,
        used_memory,
    ) -> ChainedPipelineResult:
        del selected_experts_by_key, activated, used_memory
        assignments = {key: list(value) for key, value in assignments.items()}
        selected_probability = dict(selected_probability)
        local_violations: list[str] = []

        initial_deficits = self._cfrg_feature_deficits(assignments)
        selected_devices, ownership, remaining, selection_report = self._cfrg_select_iots(
            initial_deficits
        )
        for server_id, features in remaining.items():
            for feature in sorted(features):
                local_violations.append(
                    f"CFRG infeasible: feature {feature} required by {server_id} "
                    "has no selectable IoT source"
                )

        order, mst_report = self._cfrg_mst_order(selected_devices)
        self._cfrg_assignments = assignments
        self._cfrg_latest_start = self.evaluator.latest_subtask_start_times(assignments)
        groups, group_records, dp_report = self._cfrg_interval_groups(order, ownership)
        backhaul, backhaul_plan = self._cfrg_backhaul_plan(group_records)
        subtask_features = self._cfrg_subtask_features(assignments, ownership)

        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                servers = self.evaluator.participating_servers(assignments, key)
                server_id = servers[0] if servers else None
                selected = (
                    set()
                    if server_id is None
                    else set(subtask_features.get(self._server_feature_key(key, server_id), set()))
                )
                probability = selected_probability.get(label, 0.0)
                rec_loss = self.evaluator.reconstruction_loss_for_features(subtask, selected)
                loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                    probability,
                    rec_loss,
                )
                threshold = self.evaluator.conformal_loss_threshold(subtask)
                if loss > threshold + 1e-12:
                    local_violations.append(
                        f"CFRG loss: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                required_probability[label] = self.evaluator.required_selection_probability(
                    subtask,
                    selected,
                )
                reconstruction_loss[label] = rec_loss
                performance_loss[label] = loss

        data_req = self._subtask_data_requirements(assignments, subtask_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(groups)
        resolved_groups, budgets = self._derive_group_bandwidths(
            groups,
            assignments,
            data_req,
            backhaul,
            feature_to_groups,
        )
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(
            self.servers,
            self.devices,
            self.experts,
            self.tasks,
            eval_config,
        )
        evaluation = evaluator.evaluate(
            assignments,
            resolved_groups,
            backhaul,
            subtask_features=subtask_features,
        )
        violations = [*self.scheduler_violations, *local_violations, *evaluation.violations]

        self.phase2_report.update(
            {
                "mode": "CFRG",
                "initial_feature_deficits": {
                    server_id: sorted(features)
                    for server_id, features in initial_deficits.items()
                },
                "selected_iots": list(selected_devices),
                "iot_selection": selection_report,
                "feature_ownership": {
                    f"{server_id}:{feature}": device_id
                    for (server_id, feature), device_id in sorted(ownership.items())
                },
                "remaining_feature_deficits": {
                    server_id: sorted(features)
                    for server_id, features in remaining.items()
                    if features
                },
                "mst_order": order,
                "mst": mst_report,
                "dp": dp_report,
                "groups": group_records,
                "backhaul_plan": [decision.__dict__ for decision in backhaul_plan],
                "selection_index_formula": (
                    "Delta_n*|h[n,o_n]|^2 / "
                    "((1+sum_s w[o_n,s])*(1+delta_n))"
                ),
                "compatibility_formula": (
                    "min_s distance(n,m)*(2-cos(phase_angle[n,s]-phase_angle[m,s]))"
                ),
                "utility_formula": (
                    "c_bw*(sum singleton bandwidth-group bandwidth)"
                    "-c_fwd*sum destination weights"
                ),
            }
        )

        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment=assignments,
            subtask_data_requirements=data_req,
            server_required_features={
                server_id: sorted(features)
                for server_id, features in server_features.items()
            },
            rsma_groups=self._group_map(resolved_groups),
            group_required_features={
                self._group_label(group): sorted(group.required_features)
                for group in resolved_groups
            },
            group_time_budgets={
                self._group_key_label(key): value
                for key, value in budgets.items()
            },
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=backhaul_plan,
            selected_probability=selected_probability,
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=violations,
            evaluation=evaluation,
        )

    def _cfrg_feature_deficits(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, str]]],
    ) -> Dict[ServerId, Set[str]]:
        deficits: Dict[ServerId, Set[str]] = {
            server_id: set() for server_id in self.servers
        }
        subtask_by_key = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        for key, pairs in assignments.items():
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            for server_id in {server_id for server_id, _ in pairs}:
                deficits.setdefault(server_id, set()).update(subtask.required_features)
        return deficits

    def _cfrg_select_iots(
        self,
        deficits: Mapping[ServerId, Set[str]],
    ) -> tuple[
        list[DeviceId],
        Dict[Tuple[ServerId, str], DeviceId],
        Dict[ServerId, Set[str]],
        list[dict[str, Any]],
    ]:
        remaining = {
            server_id: set(features)
            for server_id, features in deficits.items()
        }
        selected: list[DeviceId] = []
        ownership: Dict[Tuple[ServerId, str], DeviceId] = {}
        report: list[dict[str, Any]] = []

        while any(remaining.values()):
            candidates = []
            for device_id in self.devices:
                if device_id in selected:
                    continue
                device = self.devices[device_id]
                covered_by_server = {
                    server_id: set(device.features) & features
                    for server_id, features in remaining.items()
                    if set(device.features) & features
                }
                coverage_gain = sum(len(features) for features in covered_by_server.values())
                if coverage_gain <= 0:
                    continue
                reachable = self._cfrg_reachable_servers(device_id)
                if not reachable:
                    continue
                origin = max(
                    reachable,
                    key=lambda server_id: (
                        self.evaluator.channel_gain(device_id, server_id) ** 2,
                        server_id,
                    ),
                )
                channel_quality = self.evaluator.channel_gain(device_id, origin) ** 2
                backhaul_weight = sum(
                    self.evaluator.wired_weight(origin, server_id)
                    for server_id in covered_by_server
                )
                incompatibility = self._cfrg_selected_incompatibility(device_id, selected)
                index = (
                    coverage_gain * channel_quality
                    / ((1.0 + backhaul_weight) * (1.0 + incompatibility))
                )
                candidates.append(
                    (
                        index,
                        coverage_gain,
                        channel_quality,
                        -backhaul_weight,
                        -incompatibility,
                        device_id,
                        origin,
                        covered_by_server,
                        backhaul_weight,
                        incompatibility,
                    )
                )

            if not candidates:
                break
            best = max(candidates, key=lambda item: item[:6])
            (
                index,
                coverage_gain,
                channel_quality,
                _,
                _,
                device_id,
                origin,
                covered_by_server,
                backhaul_weight,
                incompatibility,
            ) = best
            selected.append(device_id)
            newly_owned = []
            for server_id in sorted(covered_by_server):
                for feature in sorted(covered_by_server[server_id]):
                    pair = (server_id, feature)
                    if pair not in ownership:
                        ownership[pair] = device_id
                        remaining[server_id].discard(feature)
                        newly_owned.append([server_id, feature])
            report.append(
                {
                    "iteration": len(selected),
                    "device": device_id,
                    "origin_candidate": origin,
                    "coverage_gain": coverage_gain,
                    "channel_quality": channel_quality,
                    "backhaul_weight": backhaul_weight,
                    "incompatibility": incompatibility,
                    "selection_index": index,
                    "newly_owned_demands": newly_owned,
                }
            )
        return selected, ownership, remaining, report

    def _cfrg_reachable_servers(self, device_id: DeviceId) -> Set[ServerId]:
        device = self.devices[device_id]
        reachable = {
            server_id
            for server_id in self.servers
            if self._device_can_transmit_to_server(device_id, server_id)
        }
        if not reachable and device.home_server in self.servers:
            reachable.add(device.home_server)
        return reachable

    def _cfrg_selected_incompatibility(
        self,
        device_id: DeviceId,
        selected: Sequence[DeviceId],
    ) -> float:
        if not selected:
            return 0.0
        values = [
            value
            for other_id in selected
            for value in [self._cfrg_pair_dissimilarity(device_id, other_id)[0]]
            if value is not None
        ]
        return min(values) if values else 0.0

    def _cfrg_pair_dissimilarity(
        self,
        first: DeviceId,
        second: DeviceId,
    ) -> tuple[float | None, ServerId | None]:
        common_servers = self._cfrg_reachable_servers(first) & self._cfrg_reachable_servers(second)
        if not common_servers:
            return None, None
        distance = math.dist(self.devices[first].position, self.devices[second].position)
        candidates = []
        for server_id in common_servers:
            first_phase = float(self.devices[first].phase_angle.get(server_id, 0.0))
            second_phase = float(self.devices[second].phase_angle.get(server_id, 0.0))
            value = distance * (2.0 - math.cos(first_phase - second_phase))
            candidates.append((value, server_id))
        return min(candidates, key=lambda item: (item[0], item[1]))

    def _cfrg_mst_order(
        self,
        selected: Sequence[DeviceId],
    ) -> tuple[list[DeviceId], dict[str, Any]]:
        devices = sorted(set(selected), key=self._id_sort_key)
        pair_values: Dict[Tuple[DeviceId, DeviceId], float] = {}
        pair_report: list[dict[str, Any]] = []
        for index, first in enumerate(devices):
            for second in devices[index + 1 :]:
                value, server_id = self._cfrg_pair_dissimilarity(first, second)
                if value is not None:
                    pair_values[(first, second)] = value
                pair_report.append(
                    {
                        "first": first,
                        "second": second,
                        "dissimilarity": value,
                        "minimizing_server": server_id,
                    }
                )

        tree: Dict[DeviceId, list[Tuple[float, DeviceId]]] = {
            device_id: [] for device_id in devices
        }
        unvisited = set(devices)
        roots: list[DeviceId] = []
        mst_edges: list[dict[str, Any]] = []
        while unvisited:
            root = min(unvisited, key=self._id_sort_key)
            roots.append(root)
            component = {root}
            unvisited.remove(root)
            while unvisited:
                frontier = []
                for source in component:
                    for target in unvisited:
                        pair = tuple(sorted((source, target), key=self._id_sort_key))
                        value = pair_values.get(pair)
                        if value is not None:
                            frontier.append((value, source, target))
                if not frontier:
                    break
                value, source, target = min(
                    frontier,
                    key=lambda item: (
                        item[0],
                        self._id_sort_key(item[1]),
                        self._id_sort_key(item[2]),
                    ),
                )
                component.add(target)
                unvisited.remove(target)
                tree[source].append((value, target))
                tree[target].append((value, source))
                mst_edges.append(
                    {"source": source, "target": target, "weight": value}
                )

        order: list[DeviceId] = []
        visited: Set[DeviceId] = set()

        for root in roots:
            if root in visited:
                continue
            queue = deque([root])
            visited.add(root)
            while queue:
                device_id = queue.popleft()
                order.append(device_id)
                for _, neighbor in sorted(
                    tree[device_id],
                    key=lambda item: (item[0], self._id_sort_key(item[1])),
                ):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order, {
            "roots": roots,
            "traversal": "BFS",
            "pairwise_compatibility": pair_report,
            "edges": mst_edges,
        }

    def _cfrg_interval_groups(
        self,
        order: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> tuple[list[GroupSpec], list[dict[str, Any]], dict[str, Any]]:
        count = len(order)
        if count == 0:
            return [], [], {"objective": 0.0, "predecessors": []}
        group_limit = max(1, int(self.config.max_group_size))
        dp = [-float("inf")] * (count + 1)
        predecessor = [-1] * (count + 1)
        interval_cache: Dict[Tuple[int, int], dict[str, Any]] = {}
        dp[0] = 0.0

        for end in range(1, count + 1):
            for length in range(1, min(group_limit, end) + 1):
                start = end - length
                info = self._cfrg_interval_info(order[start:end], ownership)
                interval_cache[(start, end)] = info
                if not info["feasible"] or dp[start] == -float("inf"):
                    continue
                value = dp[start] + info["utility"]
                if value > dp[end] + 1e-12:
                    dp[end] = value
                    predecessor[end] = start

        if dp[count] == -float("inf"):
            return [], [], {
                "objective": None,
                "predecessors": predecessor,
                "infeasible": True,
            }

        intervals = []
        cursor = count
        while cursor > 0:
            start = predecessor[cursor]
            if start < 0:
                return [], [], {
                    "objective": None,
                    "predecessors": predecessor,
                    "infeasible": True,
                }
            intervals.append((start, cursor))
            cursor = start
        intervals.reverse()

        groups: list[GroupSpec] = []
        records: list[dict[str, Any]] = []
        for group_index, interval in enumerate(intervals):
            info = dict(interval_cache[interval])
            group = self._make_group(
                info["origin_server"],
                f"g{group_index}",
                tuple(info["members"]),
                set(info["required_features"]),
            )
            groups.append(group)
            info["group_id"] = group.id
            info["destinations"] = {
                server_id: sorted(features)
                for server_id, features in info["destinations"].items()
            }
            info["required_features"] = sorted(info["required_features"])
            records.append(info)

        return groups, records, {
            "objective": dp[count],
            "predecessors": predecessor,
            "intervals": [list(interval) for interval in intervals],
            "group_limit": group_limit,
        }

    def _cfrg_interval_info(
        self,
        members: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> dict[str, Any]:
        delivery = self._cfrg_delivery_info(members, ownership)
        if delivery is None:
            return {
                "feasible": False,
                "members": list(members),
                "utility": -float("inf"),
            }
        singleton_bandwidth = 0.0
        singleton_details = []
        for device_id in members:
            singleton = self._cfrg_delivery_info([device_id], ownership)
            if singleton is None:
                return {
                    "feasible": False,
                    "members": list(members),
                    "utility": -float("inf"),
                }
            singleton_bandwidth += singleton["bandwidth"]
            singleton_details.append(
                {
                    "device": device_id,
                    "origin_server": singleton["origin_server"],
                    "bandwidth": singleton["bandwidth"],
                }
            )
        bandwidth_saving = singleton_bandwidth - delivery["bandwidth"]
        backhaul_weight = sum(
            self.evaluator.wired_weight(delivery["origin_server"], server_id)
            for server_id in delivery["destinations"]
        )
        forwarding_cost = self.config.c_fwd * backhaul_weight
        utility = self.config.c_bw * bandwidth_saving - forwarding_cost
        return {
            "feasible": True,
            **delivery,
            "singleton_details": singleton_details,
            "bandwidth_without_grouping": singleton_bandwidth,
            "bandwidth_saving": bandwidth_saving,
            "backhaul_weight": backhaul_weight,
            "forwarding_cost": forwarding_cost,
            "utility": utility,
        }

    def _cfrg_delivery_info(
        self,
        members: Sequence[DeviceId],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> dict[str, Any] | None:
        if not members:
            return None
        feasible_servers = set(self.servers)
        for device_id in members:
            feasible_servers &= self._cfrg_reachable_servers(device_id)
        if not feasible_servers:
            return None
        origin = max(
            feasible_servers,
            key=lambda server_id: (
                min(
                    self.evaluator.channel_gain(device_id, server_id) ** 2
                    for device_id in members
                ),
                server_id,
            ),
        )
        worst_channel_gain = min(
            self.evaluator.channel_gain(device_id, origin) ** 2
            for device_id in members
        )
        member_set = set(members)
        destinations: Dict[ServerId, Set[str]] = defaultdict(set)
        for (server_id, feature), owner in ownership.items():
            if owner in member_set:
                destinations[server_id].add(feature)
        required_features = set().union(*destinations.values()) if destinations else set()
        group = self._make_group(
            origin,
            "candidate",
            tuple(members),
            required_features,
        )
        dependencies: list[tuple[AssignmentKey, ServerId]] = []
        assignments = getattr(self, "_cfrg_assignments", {})
        latest_start = getattr(self, "_cfrg_latest_start", {})
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                assigned_servers = self.evaluator.participating_servers(assignments, key)
                for target_server in assigned_servers:
                    delivered = destinations.get(target_server, set())
                    if set(subtask.required_features) & delivered:
                        dependencies.append((key, target_server))

        if dependencies:
            budgets = []
            for key, target_server in dependencies:
                task = self.tasks[key[0]]
                if task.deadline == float("inf"):
                    remaining = self.evaluator.bandwidth_time_budget()
                else:
                    remaining = latest_start.get(key, task.deadline)
                if origin != target_server:
                    remaining -= (
                        self.evaluator.group_feature_volume(group)
                        / self.evaluator.wired_rate(origin, target_server)
                    )
                budgets.append(remaining)
            time_budget = min(budgets)
        else:
            time_budget = self.evaluator.bandwidth_time_budget()
        if time_budget <= 0.0:
            return None

        bandwidth = self.evaluator.derive_group_bandwidth_for_budget(
            group,
            time_budget,
        )
        return {
            "members": list(members),
            "feasible_servers": sorted(feasible_servers),
            "origin_server": origin,
            "worst_channel_gain": worst_channel_gain,
            "destinations": dict(destinations),
            "required_features": required_features,
            "dependent_subtasks": [
                {
                    "subtask": self._assignment_label(key),
                    "destination": target_server,
                }
                for key, target_server in dependencies
            ],
            "time_budget": time_budget,
            "bandwidth": bandwidth,
        }

    def _cfrg_backhaul_plan(
        self,
        group_records: Sequence[Mapping[str, Any]],
    ) -> tuple[
        Set[Tuple[ServerId, str, ServerId]],
        list[ChainedBackhaulDecision],
    ]:
        backhaul: Set[Tuple[ServerId, str, ServerId]] = set()
        plan: list[ChainedBackhaulDecision] = []
        for record in group_records:
            origin = str(record["origin_server"])
            group_id = str(record["group_id"])
            transmitted_features = sorted(
                {
                    feature
                    for device_id in record["members"]
                    for feature in self.devices[device_id].features
                }
            )
            for target_server, features in record["destinations"].items():
                if target_server == origin or not features:
                    continue
                backhaul.add((origin, group_id, target_server))
                plan.append(
                    ChainedBackhaulDecision(
                        source_server=origin,
                        group_id=group_id,
                        target_server=target_server,
                        features=transmitted_features,
                    )
                )
        plan.sort(key=lambda item: (item.source_server, item.group_id, item.target_server))
        return backhaul, plan

    def _cfrg_subtask_features(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, str]]],
        ownership: Mapping[Tuple[ServerId, str], DeviceId],
    ) -> Dict[Tuple[str, str, str], Set[str]]:
        owned_by_server: Dict[ServerId, Set[str]] = defaultdict(set)
        for server_id, feature in ownership:
            owned_by_server[server_id].add(feature)
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        subtask_by_key = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        for key, pairs in assignments.items():
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            for server_id in {server_id for server_id, _ in pairs}:
                subtask_features[self._server_feature_key(key, server_id)] = (
                    set(subtask.required_features) & owned_by_server.get(server_id, set())
                )
        return subtask_features
