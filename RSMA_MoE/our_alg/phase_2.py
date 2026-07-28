from __future__ import annotations

from dataclasses import replace
import math
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

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
    ExpertId,
    FormulationEvaluator,
    GroupSpec,
    ServerId,
    SubtaskSpec,
)


class Phase2GroupingBackhaulMixin:
    """Phase 2: RSMA-aware IoT grouping and minimum-cost data/backhaul repair."""

    def _run_phase2_for_assignments(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        activated,
        used_memory,
    ) -> ChainedPipelineResult:
        assignments = {key: list(value) for key, value in assignments.items()}
        selected_probability = dict(selected_probability)
        selected_experts_by_key = {
            key: set(value) for key, value in selected_experts_by_key.items()
        }
        activated = {server_id: set(value) for server_id, value in activated.items()}
        used_memory = dict(used_memory)

        original_scheduler_violations = list(self.scheduler_violations)
        self.scheduler_violations = []
        try:
            candidate_features = self._candidate_server_features(assignments)
            candidate_groups = self._build_distance_groups(candidate_features)
            if self.activate_all_candidate_groups:
                active_groups, subtask_features, required_prob, rec_loss, perf_loss = self._use_all_candidate_groups(
                    assignments,
                    selected_probability,
                    selected_experts_by_key,
                    candidate_groups,
                    activated,
                    used_memory,
                )
            else:
                active_groups, subtask_features, required_prob, rec_loss, perf_loss = self._repair_loss_with_groups(
                    assignments,
                    selected_probability,
                    selected_experts_by_key,
                    candidate_groups,
                    activated,
                    used_memory,
                )
            local_scheduler_violations = list(self.scheduler_violations)
        finally:
            self.scheduler_violations = original_scheduler_violations

        data_req = self._subtask_data_requirements(assignments, subtask_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(active_groups)
        backhaul, plan = self._derive_backhaul(data_req, feature_to_groups)
        groups, budgets = self._derive_group_bandwidths(active_groups, assignments, data_req, backhaul, feature_to_groups)
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        evaluation = evaluator.evaluate(assignments, groups, backhaul, subtask_features=subtask_features)
        violations = [*local_scheduler_violations, *evaluation.violations]
        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment=assignments,
            subtask_data_requirements=data_req,
            server_required_features={sid: sorted(value) for sid, value in server_features.items()},
            rsma_groups=self._group_map(groups),
            group_required_features={self._group_label(group): sorted(group.required_features) for group in groups},
            group_time_budgets={self._group_key_label(key): value for key, value in budgets.items()},
            rsma_group_bandwidths=self._group_bandwidth_map(groups),
            backhaul_plan=plan,
            selected_probability=selected_probability,
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

    def _derive_backhaul(self, subtask_data_requirements, feature_to_groups):
        backhaul = set()
        features_by_edge: Dict[Tuple[str, str, str], Set[str]] = {}
        groups_by_key: Dict[Tuple[str, str], GroupSpec] = {}
        for groups in feature_to_groups.values():
            for group in groups:
                groups_by_key[(group.server_id, group.id)] = group

        for server_map in subtask_data_requirements.values():
            for target_server, features in server_map.items():
                requested = set(features)
                local_available = set()
                for feature in requested:
                    if any(group.server_id == target_server for group in feature_to_groups.get(feature, [])):
                        local_available.add(feature)
                missing = requested - local_available
                if not missing:
                    continue
                chosen_groups = self._minimum_cost_backhaul_cover(target_server, missing, feature_to_groups)
                if not chosen_groups:
                    for feature in sorted(missing):
                        self.scheduler_violations.append(
                            f"Backhaul: feature {feature} required by {target_server} has no active source group"
                        )
                    continue
                covered = set()
                for group in chosen_groups:
                    edge = (group.server_id, group.id, target_server)
                    backhaul.add(edge)
                    group_features = group.required_features & missing
                    features_by_edge.setdefault(edge, set()).update(group_features)
                    covered.update(group_features)
                for feature in sorted(missing - covered):
                    self.scheduler_violations.append(
                        f"Backhaul: feature {feature} required by {target_server} is not covered by selected groups"
                    )

        plan = [
            ChainedBackhaulDecision(src, group_id, dst, sorted(features))
            for (src, group_id, dst), features in sorted(features_by_edge.items())
        ]
        return backhaul, plan

    def _minimum_cost_backhaul_cover(self, target_server, missing_features, feature_to_groups):
        features = sorted(missing_features)
        feature_index = {feature: index for index, feature in enumerate(features)}
        full_mask = (1 << len(features)) - 1
        candidates = []
        seen = set()
        for feature in features:
            for group in feature_to_groups.get(feature, []):
                if group.server_id == target_server:
                    continue
                group_key = (group.server_id, group.id)
                if group_key in seen:
                    continue
                seen.add(group_key)
                mask = 0
                for item in group.required_features & missing_features:
                    mask |= 1 << feature_index[item]
                if not mask:
                    continue
                edge_cost = self.config.c_fwd * self.evaluator.wired_weight(group.server_id, target_server)
                backhaul_time = self.evaluator.group_feature_volume(group) / max(
                    self.evaluator.wired_rate(group.server_id, target_server),
                    1e-12,
                )
                candidates.append((edge_cost, backhaul_time, mask, group))

        dp = {0: (0.0, 0.0, tuple())}
        for edge_cost, backhaul_time, mask, group in candidates:
            next_dp = dict(dp)
            for state, (cost, time_cost, groups) in dp.items():
                next_state = state | mask
                next_cost = cost + edge_cost
                next_time = time_cost + backhaul_time
                old = next_dp.get(next_state)
                candidate = (next_cost, next_time, groups + (group,))
                if old is None or candidate[:2] < old[:2]:
                    next_dp[next_state] = candidate
            dp = next_dp
        result = dp.get(full_mask)
        return [] if result is None else list(result[2])

    def _best_local_group_loss_repair(
        self,
        subtask,
        key,
        target_servers,
        candidate_groups,
        active_groups,
        active_group_budgets,
        subtask_features,
        assignments,
        probability,
        threshold,
    ):
        return self._best_group_loss_repair_by_efficiency(
            subtask=subtask,
            key=key,
            target_servers=target_servers,
            candidate_groups=candidate_groups,
            active_groups=active_groups,
            active_backhaul=set(),
            active_group_budgets=active_group_budgets,
            subtask_features=subtask_features,
            assignments=assignments,
            probability=probability,
            threshold=threshold,
            local_only=True,
        )

    def _best_backhaul_group_loss_repair(
        self,
        subtask,
        key,
        subtask_features,
        probability,
        threshold,
        target_servers,
        candidate_groups,
        active_groups,
        active_backhaul,
        active_group_budgets,
        assignments,
    ):
        return self._best_group_loss_repair_by_efficiency(
            subtask=subtask,
            key=key,
            target_servers=target_servers,
            candidate_groups=candidate_groups,
            active_groups=active_groups,
            active_backhaul=active_backhaul,
            active_group_budgets=active_group_budgets,
            subtask_features=subtask_features,
            assignments=assignments,
            probability=probability,
            threshold=threshold,
            local_only=False,
        )

    def _best_group_loss_repair_by_efficiency(
        self,
        *,
        subtask,
        key,
        target_servers,
        candidate_groups,
        active_groups,
        active_backhaul,
        active_group_budgets,
        subtask_features,
        assignments,
        probability,
        threshold,
        local_only: bool,
    ):
        needed = set(subtask.required_features)
        current_rec = self._average_reconstruction_for_servers(
            subtask,
            key,
            target_servers,
            subtask_features,
            assignments,
        )
        current_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, current_rec)
        required_improvement = max(0.0, current_loss - threshold)
        if required_improvement <= 1e-12:
            return None

        best = None
        for target_server in target_servers:
            selected = set(subtask_features.get(self._server_feature_key(key, target_server), set()))
            missing = needed - selected
            if not missing:
                continue
            for group in candidate_groups:
                is_local = group.server_id == target_server
                if local_only != is_local:
                    continue
                added_features = group.required_features & missing
                if not added_features:
                    continue

                trial_features = {feature_key: set(value) for feature_key, value in subtask_features.items()}
                trial_features.setdefault(self._server_feature_key(key, target_server), set()).update(
                    group.required_features & needed
                )
                new_rec = self._average_reconstruction_for_servers(
                    subtask,
                    key,
                    target_servers,
                    trial_features,
                    assignments,
                )
                new_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, new_rec)
                raw_improvement = current_loss - new_loss
                if raw_improvement <= 1e-12:
                    continue

                bounded_improvement = min(raw_improvement, required_improvement)
                cost = self._group_incremental_cost(
                    group,
                    key,
                    [target_server],
                    active_groups,
                    active_backhaul,
                    active_group_budgets,
                    assignments,
                )
                cost = max(cost, 1e-12)
                coverage = len(added_features)
                overshoot = max(0.0, threshold - new_loss)
                efficiency = bounded_improvement / cost
                item = (
                    efficiency,
                    bounded_improvement,
                    -cost,
                    -overshoot,
                    coverage,
                    group.server_id,
                    group.id,
                    target_server,
                    group,
                )
                if best is None or item > best:
                    best = item

        if best is None:
            return None
        return best[-1], best[-2]

    def _repair_loss_with_groups(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        candidate_groups,
        activated,
        used_memory,
    ):
        self.post_pruning_report.update(
            {
                "enabled": True,
                "mode": "offline_final_group_pruning",
                "note": "Applied once after Phase 3 because the algorithm is evaluated offline.",
            }
        )
        active_groups: Dict[Tuple[str, str], GroupSpec] = {}
        active_backhaul = set()
        active_group_budgets: Dict[Tuple[str, str], float] = {}
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        reported_unavailable_features: Set[Tuple[AssignmentKey, ServerId, str]] = set()

        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                probability = selected_probability.get(label, 0.0)
                threshold = self.evaluator.conformal_loss_threshold(subtask)
                target_servers = self.evaluator.participating_servers(assignments, key)

                while True:
                    avg_rec = self._average_reconstruction_for_servers(
                        subtask,
                        key,
                        target_servers,
                        subtask_features,
                        assignments,
                    )
                    loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, avg_rec)
                    if loss <= threshold + 1e-12:
                        break

                    self._record_unavailable_missing_features(
                        subtask,
                        key,
                        target_servers,
                        subtask_features,
                        candidate_groups,
                        reported_unavailable_features,
                    )
                    plan = self._minimum_cost_group_repair_plan(
                        subtask=subtask,
                        key=key,
                        target_servers=target_servers,
                        candidate_groups=candidate_groups,
                        active_groups=active_groups,
                        active_backhaul=active_backhaul,
                        active_group_budgets=active_group_budgets,
                        subtask_features=subtask_features,
                        assignments=assignments,
                        probability=probability,
                        threshold=threshold,
                    )
                    if plan:
                        self._apply_group_repair_plan(
                            subtask,
                            key,
                            plan,
                            assignments,
                            active_groups,
                            active_backhaul,
                            active_group_budgets,
                            subtask_features,
                        )
                        continue

                    if not self.allow_expert_loss_repair:
                        break
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
                    probability = self.evaluator.selection_probability(subtask, selected_experts)
                    selected_probability[label] = probability
                    target_servers = self.evaluator.participating_servers(assignments, key)

                avg_rec = self._average_reconstruction_for_servers(
                    subtask,
                    key,
                    target_servers,
                    subtask_features,
                    assignments,
                )
                loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, avg_rec)
                if loss > threshold + 1e-12:
                    self.scheduler_violations.append(
                        f"Loss repair: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                required_probability[label] = self._required_probability_with_reconstruction(subtask, avg_rec)
                reconstruction_loss[label] = avg_rec
                performance_loss[label] = loss

        return list(active_groups.values()), subtask_features, required_probability, reconstruction_loss, performance_loss

    def _minimum_cost_group_repair_plan(
        self,
        *,
        subtask,
        key,
        target_servers,
        candidate_groups,
        active_groups,
        active_backhaul,
        active_group_budgets,
        subtask_features,
        assignments,
        probability,
        threshold,
    ):
        ordered_servers = list(target_servers)
        if not ordered_servers:
            return []
        required_features = sorted(subtask.required_features)
        if not required_features:
            return []
        feature_index = {feature: index for index, feature in enumerate(required_features)}
        full_mask = (1 << len(required_features)) - 1

        def mask_for(features) -> int:
            mask = 0
            for feature in features:
                index = feature_index.get(feature)
                if index is not None:
                    mask |= 1 << index
            return mask

        initial_state = tuple(
            mask_for(subtask_features.get(self._server_feature_key(key, server_id), set()))
            for server_id in ordered_servers
        )
        if self._loss_for_feature_state(
            subtask,
            key,
            ordered_servers,
            initial_state,
            required_features,
            assignments,
            probability,
        ) <= threshold + 1e-12:
            return []

        group_options = []
        for group in candidate_groups:
            additions = []
            for server_index, target_server in enumerate(ordered_servers):
                missing_mask = full_mask & ~initial_state[server_index]
                add_mask = mask_for(group.required_features) & missing_mask
                if add_mask:
                    additions.append((server_index, target_server, add_mask))
            if not additions:
                continue
            options = []
            subset_indices = []
            max_pair_size = min(2, len(additions))
            for size in range(1, max_pair_size + 1):
                subset_indices.extend(combinations(range(len(additions)), size))
            if len(additions) > 2:
                subset_indices.append(tuple(range(len(additions))))
            seen_subsets = set()
            for subset in subset_indices:
                if subset in seen_subsets:
                    continue
                seen_subsets.add(subset)
                target_ids = []
                delta = [0 for _ in ordered_servers]
                actions = []
                for option_index in subset:
                    server_index, target_server, add_mask = additions[option_index]
                    target_ids.append(target_server)
                    delta[server_index] |= add_mask
                    actions.append((group, target_server))
                cost = self._group_incremental_cost(
                    group,
                    key,
                    target_ids,
                    active_groups,
                    active_backhaul,
                    active_group_budgets,
                    assignments,
                )
                options.append((max(cost, 1e-12), tuple(delta), tuple(actions)))
            group_options.append(options)

        if not group_options:
            return []

        dp = {initial_state: (0.0, tuple())}
        for options in group_options:
            next_dp = dict(dp)
            for state, (state_cost, state_actions) in dp.items():
                for option_cost, delta, actions in options:
                    next_state = tuple(mask | add for mask, add in zip(state, delta))
                    if next_state == state:
                        continue
                    next_cost = state_cost + option_cost
                    old = next_dp.get(next_state)
                    if old is None or next_cost < old[0] - 1e-12:
                        next_dp[next_state] = (next_cost, state_actions + actions)
            dp = self._prune_group_repair_states(
                dp=next_dp,
                subtask=subtask,
                key=key,
                ordered_servers=ordered_servers,
                required_features=required_features,
                assignments=assignments,
                probability=probability,
                threshold=threshold,
            )

        best = None
        for state, (cost, actions) in dp.items():
            loss = self._loss_for_feature_state(
                subtask,
                key,
                ordered_servers,
                state,
                required_features,
                assignments,
                probability,
            )
            if loss > threshold + 1e-12:
                continue
            labels = tuple((group.server_id, group.id, target_server) for group, target_server in actions)
            item = (cost, len(actions), labels, actions)
            if best is None or item[:3] < best[:3]:
                best = item
        if best is None:
            return []
        return list(best[3])

    def _prune_group_repair_states(
        self,
        *,
        dp,
        subtask,
        key,
        ordered_servers,
        required_features,
        assignments,
        probability,
        threshold,
    ):
        if len(dp) <= 768:
            return dp
        ranked = []
        for state, (cost, actions) in dp.items():
            loss = self._loss_for_feature_state(
                subtask,
                key,
                ordered_servers,
                state,
                required_features,
                assignments,
                probability,
            )
            loss_gap = max(0.0, loss - threshold)
            feature_count = sum(mask.bit_count() for mask in state)
            ranked.append((loss_gap, cost, -feature_count, state, actions))
        ranked.sort()
        return {state: (cost, actions) for loss_gap, cost, _, state, actions in ranked[:768]}

    def _loss_for_feature_state(
        self,
        subtask,
        key,
        ordered_servers,
        state,
        required_features,
        assignments,
        probability,
    ) -> float:
        features_by_server = {}
        for server_id, mask in zip(ordered_servers, state):
            features = {
                feature
                for index, feature in enumerate(required_features)
                if mask & (1 << index)
            }
            features_by_server[self._server_feature_key(key, server_id)] = features
        rec = self._average_reconstruction_for_servers(
            subtask,
            key,
            ordered_servers,
            features_by_server,
            assignments,
        )
        return self.evaluator.performance_loss_from_probability_and_reconstruction(probability, rec)

    def _apply_group_repair_plan(
        self,
        subtask,
        key,
        plan,
        assignments,
        active_groups,
        active_backhaul,
        active_group_budgets,
        subtask_features,
    ) -> None:
        group_budget = self._subtask_bandwidth_budget(key, assignments)
        needed = set(subtask.required_features)
        for group, target_server in plan:
            group_key = (group.server_id, group.id)
            active_groups[group_key] = group
            active_group_budgets[group_key] = min(
                active_group_budgets.get(group_key, group_budget),
                group_budget,
            )
            if group.server_id != target_server:
                active_backhaul.add((group.server_id, group.id, target_server))
            self._server_feature_set(subtask_features, key, target_server).update(
                group.required_features & needed
            )

    def _build_distance_groups(self, server_required_features: Mapping[ServerId, Set[str]]) -> list[GroupSpec]:
        groups: list[GroupSpec] = []
        globally_required = set()
        for features in server_required_features.values():
            globally_required.update(features)

        for server_id in self.servers:
            local_devices = [
                device_id
                for device_id, device in self.devices.items()
                if device.home_server == server_id
            ]
            local_required = globally_required & self._server_local_features(local_devices)
            if not local_required:
                continue
            candidates = [
                device_id
                for device_id in local_devices
                if self.devices[device_id].features & local_required
            ]
            if not candidates:
                self._record_local_missing_features(server_id, local_required, groups)
                continue

            ordered = self._compatibility_order(server_id, candidates, local_required)
            for members in self._dp_group_order(server_id, ordered, local_required):
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
            self._record_local_missing_features(server_id, local_required, groups)
        return groups

    def _compatibility_order(
        self,
        server_id: ServerId,
        device_ids: Sequence[DeviceId],
        local_required: Set[str],
    ) -> list[DeviceId]:
        ordered = sorted(device_ids, key=self._id_sort_key)
        if len(ordered) <= 2:
            return ordered

        remaining = set(ordered)
        start = min(ordered, key=lambda did: (self.devices[did].distance.get(server_id, float("inf")), self._id_sort_key(did)))
        remaining.remove(start)
        tree: Dict[DeviceId, list[tuple[float, DeviceId]]] = {device_id: [] for device_id in ordered}
        frontier: list[tuple[float, DeviceId, DeviceId]] = [
            (self._compatibility_dissimilarity(server_id, start, other, local_required), start, other)
            for other in remaining
        ]
        while remaining and frontier:
            weight, source, target = min(
                (edge for edge in frontier if edge[2] in remaining),
                key=lambda edge: (edge[0], self._id_sort_key(edge[1]), self._id_sort_key(edge[2])),
            )
            remaining.remove(target)
            tree[source].append((weight, target))
            tree[target].append((weight, source))
            for other in remaining:
                frontier.append((self._compatibility_dissimilarity(server_id, target, other, local_required), target, other))

        result: list[DeviceId] = []
        visited: Set[DeviceId] = set()

        def visit(device_id: DeviceId) -> None:
            visited.add(device_id)
            result.append(device_id)
            for _, next_id in sorted(tree[device_id], key=lambda item: (item[0], self._id_sort_key(item[1]))):
                if next_id not in visited:
                    visit(next_id)

        visit(start)
        result.extend(device_id for device_id in ordered if device_id not in visited)
        return result

    def _compatibility_dissimilarity(
        self,
        server_id: ServerId,
        first: DeviceId,
        second: DeviceId,
        local_required: Set[str],
    ) -> float:
        first_pos = self._device_position(first, server_id)
        second_pos = self._device_position(second, server_id)
        max_dist = max(
            max(self.devices[first].distance.get(server_id, 0.0), self.devices[second].distance.get(server_id, 0.0)),
            1.0,
        )
        spatial_distance = math.sqrt(self._squared_distance(first_pos, second_pos)) / max_dist
        coherence_penalty = 1.0 - self.evaluator.spatial_correlation(first, second, server_id)
        first_features = self.devices[first].features & local_required
        second_features = self.devices[second].features & local_required
        union_size = len(first_features | second_features)
        feature_penalty = 1.0 - (len(first_features & second_features) / union_size if union_size else 0.0)
        return (0.45 * spatial_distance) + (0.35 * coherence_penalty) + (0.20 * feature_penalty)

    def _dp_group_order(
        self,
        server_id: ServerId,
        ordered: Sequence[DeviceId],
        local_required: Set[str],
    ) -> list[list[DeviceId]]:
        n = len(ordered)
        if n == 0:
            return []
        q = max(1, self.config.max_group_size)
        dp = [-float("inf")] * (n + 1)
        prev = [-1] * (n + 1)
        dp[0] = 0.0
        for end in range(1, n + 1):
            for length in range(1, min(q, end) + 1):
                start = end - length
                group_members = list(ordered[start:end])
                value = dp[start] + self._group_utility(server_id, group_members, local_required)
                if value > dp[end]:
                    dp[end] = value
                    prev[end] = start

        groups: list[list[DeviceId]] = []
        cursor = n
        while cursor > 0:
            start = prev[cursor]
            if start < 0:
                start = max(0, cursor - q)
            groups.append(list(ordered[start:cursor]))
            cursor = start
        groups.reverse()
        return groups

    def _group_utility(
        self,
        server_id: ServerId,
        members: Sequence[DeviceId],
        local_required: Set[str],
    ) -> float:
        features = self._group_local_features(members, local_required)
        if not members or not features:
            return -1e9
        group = self._make_group(server_id, "candidate", tuple(members), features)
        try:
            common_efficiency, private_efficiencies = self.evaluator.group_spectral_efficiencies(group)
            bandwidth_demand = self.evaluator.derive_group_bandwidth_for_budget(
                group,
                self.evaluator.bandwidth_time_budget(),
            )
            common_volume = self.evaluator.group_common_volume(group)
            private_volume = sum(self.evaluator.group_private_volume(group, device_id) for device_id in group.devices)
            private_efficiency = min(private_efficiencies.values()) if private_efficiencies else 0.0
            served_volume = common_volume + private_volume
            effective_rate_score = common_efficiency + private_efficiency
            return (served_volume * effective_rate_score) / max(bandwidth_demand, 1e-12)
        except Exception:
            return -self.evaluator.group_feature_volume(group)


