from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import ChainedPipelineResult
from utils.formulation import (
    AssignmentKey,
    ExpertId,
    FormulationEvaluator,
    GroupSpec,
    ServerId,
    SubtaskSpec,
)


class Phase3RefinementPruningMixin:
    """Phase 3: deployment refinement and offline final group pruning."""

    def _phase3_refine(
        self,
        assignments,
        current_result: ChainedPipelineResult,
    ) -> ChainedPipelineResult:
        best_result = current_result
        working_assignments = {key: list(value) for key, value in assignments.items()}
        accepted_moves: list[Dict[str, Any]] = []
        candidate_evaluations = 0
        max_iterations = int(self.phase3_report.get("max_iterations", 3))
        max_candidates = int(self.phase3_report.get("max_candidates_per_strategy", 20))
        bandwidth_tolerance = float(self.phase3_report.get("bandwidth_increase_tolerance", 0.0))
        strategies = ("relocate_same_expert",)

        for iteration in range(max_iterations):
            accepted_this_iteration = False
            for strategy in strategies:
                best_candidate = None
                for move in self._phase3_candidate_moves(working_assignments, strategy, max_candidates):
                    trial_assignments = move.pop("assignments")
                    selected_probability, selected_experts = self._selection_metadata_from_assignments(trial_assignments)
                    activated, used_memory = self._activation_state_from_assignments(trial_assignments)
                    trial_result = self._run_phase2_for_assignments(
                        trial_assignments,
                        selected_probability,
                        selected_experts,
                        activated,
                        used_memory,
                    )
                    candidate_evaluations += 1
                    if trial_result.violations:
                        continue
                    bandwidth_limit = best_result.bandwidth_cost * (1.0 + bandwidth_tolerance)
                    if trial_result.bandwidth_cost > bandwidth_limit + 1e-9:
                        continue
                    if trial_result.total_cost >= best_result.total_cost - 1e-9:
                        continue
                    candidate = (trial_result.total_cost, move, trial_result)
                    if best_candidate is None or candidate[0] < best_candidate[0]:
                        best_candidate = candidate

                if best_candidate is None:
                    continue

                _, accepted_move, accepted_result = best_candidate
                accepted_move = dict(accepted_move)
                accepted_move["iteration"] = iteration + 1
                accepted_move["cost_before"] = round(best_result.total_cost, 6)
                accepted_move["cost_after"] = round(accepted_result.total_cost, 6)
                accepted_move["saved_cost"] = round(best_result.total_cost - accepted_result.total_cost, 6)
                accepted_moves.append(accepted_move)
                best_result = accepted_result
                working_assignments = {
                    key: list(value) for key, value in accepted_result.subtask_assignment.items()
                }
                accepted_this_iteration = True
                break

            if not accepted_this_iteration:
                break

        self.phase3_report.update(
            {
                "iterations": len(accepted_moves),
                "candidate_evaluations": candidate_evaluations,
                "accepted_moves": accepted_moves,
                "cost_before": round(current_result.total_cost, 6),
                "cost_after": round(best_result.total_cost, 6),
                "saved_cost": round(current_result.total_cost - best_result.total_cost, 6),
            }
        )
        setattr(best_result, "phase3_report", self.phase3_report)
        return best_result

    def _run_offline_group_pruning(self, result: ChainedPipelineResult) -> ChainedPipelineResult:
        groups = self._groups_from_result(result)
        subtask_features = self._subtask_features_from_result(result)
        if not groups or not subtask_features:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "removed_groups": 0,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "No groups or subtask feature map available for pruning.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result

        pruned_groups, pruned_features = self._post_prune_groups(
            result.subtask_assignment,
            result.selected_probability,
            groups,
            subtask_features,
        )
        data_req = self._subtask_data_requirements(result.subtask_assignment, pruned_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(pruned_groups)
        backhaul, plan = self._derive_backhaul(data_req, feature_to_groups)
        resolved_groups, budgets = self._derive_group_bandwidths(
            pruned_groups,
            result.subtask_assignment,
            data_req,
            backhaul,
            feature_to_groups,
        )
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        evaluation = evaluator.evaluate(
            result.subtask_assignment,
            resolved_groups,
            backhaul,
            subtask_features=pruned_features,
        )
        if evaluation.violations or evaluation.objective.total_cost > result.total_cost + 1e-9:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "reverted": True,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "Pruned candidate was not feasible or did not improve cost.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result

        pruned_result = ChainedPipelineResult(
            expert_placement=result.expert_placement,
            subtask_assignment=result.subtask_assignment,
            subtask_data_requirements=data_req,
            server_required_features={sid: sorted(value) for sid, value in server_features.items()},
            rsma_groups=self._group_map(resolved_groups),
            group_required_features={self._group_label(group): sorted(group.required_features) for group in resolved_groups},
            group_time_budgets={self._group_key_label(key): value for key, value in budgets.items()},
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=plan,
            selected_probability=result.selected_probability,
            required_probability=result.required_probability,
            reconstruction_loss=result.reconstruction_loss,
            performance_loss=result.performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
        )
        setattr(pruned_result, "phase3_report", self.phase3_report)
        setattr(pruned_result, "post_pruning_report", self.post_pruning_report)
        return pruned_result

    def _groups_from_result(self, result: ChainedPipelineResult) -> list[GroupSpec]:
        group_features_by_server: Dict[ServerId, list[tuple[str, Set[str]]]] = {}
        for label, features in result.group_required_features.items():
            if ":" not in label:
                continue
            server_id, group_id = label.split(":", 1)
            group_features_by_server.setdefault(server_id, []).append((group_id, set(features)))

        fallback_required = set()
        for features in result.server_required_features.values():
            fallback_required.update(features)

        groups: list[GroupSpec] = []
        for server_id, member_groups in result.rsma_groups.items():
            feature_items = group_features_by_server.get(server_id, [])
            for index, members in enumerate(member_groups):
                if index < len(feature_items):
                    group_id, features = feature_items[index]
                else:
                    group_id = f"g{len(groups)}"
                    features = self._group_local_features(members, fallback_required)
                if not features:
                    continue
                groups.append(
                    self._make_group(
                        server_id=server_id,
                        group_id=group_id,
                        members=tuple(members),
                        required_features=features,
                    )
                )
        return groups

    def _subtask_features_from_result(self, result: ChainedPipelineResult):
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        for label, server_features in result.subtask_data_requirements.items():
            if ":" not in label:
                continue
            task_id, subtask_id = label.split(":", 1)
            for server_id, features in server_features.items():
                subtask_features[(task_id, subtask_id, server_id)] = set(features)
        return subtask_features

    def _phase3_candidate_moves(self, assignments, strategy: str, max_candidates: int):
        if strategy == "relocate_same_expert":
            yield from self._relocate_same_expert_moves(assignments, max_candidates)
            return
        if strategy == "remove_pair":
            count = 0
            for key in sorted(assignments):
                pairs = list(assignments[key])
                if len(pairs) <= 1:
                    continue
                for pair in pairs:
                    trial = {item_key: list(value) for item_key, value in assignments.items()}
                    trial[key] = [item for item in pairs if item != pair]
                    if not trial[key]:
                        continue
                    count += 1
                    yield {
                        "strategy": strategy,
                        "subtask": self._assignment_label(key),
                        "removed_pair": list(pair),
                        "assignments": trial,
                    }
                    if count >= max_candidates:
                        return
            return

        if strategy == "close_expert":
            active_pairs = sorted({pair for pairs in assignments.values() for pair in pairs})
            for pair in active_pairs[:max_candidates]:
                trial = {}
                feasible = True
                affected = 0
                for key, pairs in assignments.items():
                    next_pairs = [item for item in pairs if item != pair]
                    if len(next_pairs) != len(pairs):
                        affected += 1
                    if not next_pairs:
                        feasible = False
                        break
                    trial[key] = next_pairs
                if not feasible or affected <= 0:
                    continue
                yield {
                    "strategy": strategy,
                    "closed_pair": list(pair),
                    "affected_subtasks": affected,
                    "assignments": trial,
                }
            return

        if strategy == "replace_pair":
            count = 0
            subtask_by_key = self._subtask_by_key()
            for key in sorted(assignments):
                subtask = subtask_by_key.get(key)
                if subtask is None:
                    continue
                pairs = list(assignments[key])
                selected_experts = {expert_id for _, expert_id in pairs}
                ranked_experts = self._ranked_experts(subtask)[: max(self.top_k * 3, self.top_k + 4)]
                for old_pair in pairs:
                    for expert_id in ranked_experts:
                        if expert_id in selected_experts and expert_id != old_pair[1]:
                            continue
                        for server_id in self._feasible_servers_for_replacement(expert_id):
                            new_pair = (server_id, expert_id)
                            if new_pair == old_pair:
                                continue
                            if expert_id != old_pair[1] and any(existing_expert == expert_id for _, existing_expert in pairs):
                                continue
                            trial_pairs = [new_pair if item == old_pair else item for item in pairs]
                            if len({expert for _, expert in trial_pairs}) != len(trial_pairs):
                                continue
                            trial = {item_key: list(value) for item_key, value in assignments.items()}
                            trial[key] = trial_pairs
                            count += 1
                            yield {
                                "strategy": strategy,
                                "subtask": self._assignment_label(key),
                                "old_pair": list(old_pair),
                                "new_pair": list(new_pair),
                                "assignments": trial,
                            }
                            if count >= max_candidates:
                                return

    def _relocate_same_expert_moves(self, assignments, max_candidates: int):
        candidates = []
        for key in sorted(assignments):
            pairs = list(assignments[key])
            if len(pairs) <= 1:
                continue
            current_servers = {server_id for server_id, _ in pairs}
            for old_pair in pairs:
                old_server, expert_id = old_pair
                for target_server in sorted(current_servers):
                    if target_server == old_server:
                        continue
                    if expert_id not in self.servers[target_server].stored_experts:
                        continue
                    new_pair = (target_server, expert_id)
                    if new_pair in pairs:
                        continue
                    trial_pairs = [new_pair if item == old_pair else item for item in pairs]
                    if len(set(trial_pairs)) != len(trial_pairs):
                        continue
                    trial = {item_key: list(value) for item_key, value in assignments.items()}
                    trial[key] = trial_pairs
                    if not self._assignment_memory_feasible(trial):
                        continue
                    old_forwarding = self._subtask_pair_spread_cost(pairs)
                    new_forwarding = self._subtask_pair_spread_cost(trial_pairs)
                    estimated_saving = old_forwarding - new_forwarding
                    candidates.append(
                        (
                            -estimated_saving,
                            self._assignment_label(key),
                            old_pair,
                            new_pair,
                            trial,
                        )
                    )
        candidates.sort()
        for _, label, old_pair, new_pair, trial in candidates[:max_candidates]:
            yield {
                "strategy": "relocate_same_expert",
                "subtask": label,
                "old_pair": list(old_pair),
                "new_pair": list(new_pair),
                "assignments": trial,
            }

    def _subtask_pair_spread_cost(self, pairs: Sequence[tuple[ServerId, ExpertId]]) -> float:
        servers = sorted({server_id for server_id, _ in pairs})
        total = 0.0
        for index, source_server in enumerate(servers):
            for target_server in servers[index + 1 :]:
                total += self.evaluator.wired_weight(source_server, target_server)
        return total

    def _assignment_memory_feasible(self, assignments) -> bool:
        _, used_memory = self._activation_state_from_assignments(assignments)
        return all(
            used_memory.get(server_id, 0.0) <= self.servers[server_id].gpu_memory + 1e-12
            for server_id in self.servers
        )
    def _selection_metadata_from_assignments(self, assignments):
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        subtask_by_key = self._subtask_by_key()
        for key, pairs in assignments.items():
            selected_experts = {expert_id for _, expert_id in pairs}
            selected_experts_by_key[key] = selected_experts
            subtask = subtask_by_key.get(key)
            if subtask is not None:
                selected_probability[self._assignment_label(key)] = self.evaluator.selection_probability(
                    subtask,
                    selected_experts,
                )
        return selected_probability, selected_experts_by_key

    def _activation_state_from_assignments(self, assignments):
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                activated.setdefault(server_id, set()).add(expert_id)
        used_memory = {
            server_id: sum(self.experts[expert_id].memory for expert_id in experts if expert_id in self.experts)
            for server_id, experts in activated.items()
        }
        return activated, used_memory

    def _subtask_by_key(self) -> Dict[AssignmentKey, SubtaskSpec]:
        return {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }

    def _feasible_servers_for_replacement(self, expert_id: ExpertId) -> list[ServerId]:
        expert = self.experts[expert_id]
        feasible: list[ServerId] = []
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            if expert.memory <= server.gpu_memory:
                feasible.append(server_id)
        return sorted(feasible)

    def _post_prune_groups(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        selected_probability: Mapping[str, float],
        groups: Sequence[GroupSpec],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
    ) -> tuple[list[GroupSpec], Dict[Tuple[str, str, str], Set[str]]]:
        current_groups = list(groups)
        current_features = {key: set(value) for key, value in subtask_features.items()}
        current_eval = self._evaluate_prune_candidate(assignments, current_groups, current_features)
        current_cost = current_eval.objective.total_cost
        self.post_pruning_report.update(
            {
                "removed_groups": 0,
                "cost_before": current_cost,
                "cost_after": current_cost,
            }
        )
        if current_eval.violations:
            return current_groups, current_features

        removed = 0
        candidates = sorted(
            current_groups,
            key=lambda group: self._group_prune_priority(group, assignments),
            reverse=True,
        )
        for group in candidates:
            group_key = (group.server_id, group.id)
            if not any((item.server_id, item.id) == group_key for item in current_groups):
                continue
            trial_groups = [item for item in current_groups if (item.server_id, item.id) != group_key]
            trial_features = self._served_subtask_features(assignments, current_features, trial_groups)
            trial_eval = self._evaluate_prune_candidate(assignments, trial_groups, trial_features)
            if trial_eval.violations:
                continue
            trial_cost = trial_eval.objective.total_cost
            if trial_cost <= current_cost + 1e-9:
                current_groups = trial_groups
                current_features = trial_features
                current_cost = trial_cost
                removed += 1

        self.post_pruning_report.update(
            {
                "removed_groups": removed,
                "cost_after": current_cost,
            }
        )
        return current_groups, current_features

    def _served_subtask_features(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        requested_features: Mapping[Tuple[str, str, str], Set[str]],
        groups: Sequence[GroupSpec],
    ) -> Dict[Tuple[str, str, str], Set[str]]:
        available_features: Set[str] = set()
        for group in groups:
            available_features.update(group.required_features)
        served: Dict[Tuple[str, str, str], Set[str]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                required = set(subtask.required_features)
                for server_id in self.evaluator.participating_servers(assignments, key):
                    feature_key = self._server_feature_key(key, server_id)
                    desired = set(requested_features.get(feature_key, set()))
                    kept = desired & required & available_features
                    if kept:
                        served[feature_key] = kept
        return served

    def _evaluate_prune_candidate(
        self,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
    ):
        data_req = self._subtask_data_requirements(assignments, subtask_features)
        feature_to_groups = self._feature_to_groups(groups)
        backhaul, _ = self._derive_backhaul(data_req, feature_to_groups)
        resolved_groups, _ = self._derive_group_bandwidths(groups, assignments, data_req, backhaul, feature_to_groups)
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        return evaluator.evaluate(assignments, resolved_groups, backhaul, subtask_features=subtask_features)

    def _group_prune_priority(
        self,
        group: GroupSpec,
        assignments: Mapping[AssignmentKey, Sequence[tuple[ServerId, ExpertId]]],
    ) -> float:
        try:
            budget = self.evaluator.bandwidth_time_budget()
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, budget)
        except Exception:
            bandwidth = self.evaluator.group_feature_volume(group)
        return (self.config.c_bw * bandwidth) + self.config.c_fwd

    def _recompute_loss_metadata(
        self,
        assignments,
        selected_probability,
        groups,
        subtask_features,
    ):
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                target_servers = self.evaluator.participating_servers(assignments, key)
                probability = selected_probability.get(label, 0.0)
                avg_rec = self._average_reconstruction_for_servers(
                    subtask,
                    key,
                    target_servers,
                    subtask_features,
                    assignments,
                )
                required_probability[label] = self._required_probability_with_reconstruction(subtask, avg_rec)
                reconstruction_loss[label] = avg_rec
                performance_loss[label] = self.evaluator.performance_loss_from_probability_and_reconstruction(
                    probability,
                    avg_rec,
                )
        return list(groups), subtask_features, required_probability, reconstruction_loss, performance_loss



