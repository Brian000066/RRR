from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import heapq
from itertools import product
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
    InfeasibleBandwidthBudget,
    ServerId,
    SubtaskSpec,
)


class Phase3RefinementPruningMixin:
    """Phase III: FAR flow-aligned remapping and redundancy pruning."""

    def _phase3_refine(
        self,
        assignments,
        current_result: ChainedPipelineResult,
    ) -> ChainedPipelineResult:
        """Run FAR exactly after SCOPE and CFRG have produced a feasible state."""
        initial_cost = current_result.total_cost
        self._far_fixed_groups = tuple(
            replace(
                group,
                bandwidth=0.0,
                required_features=self.evaluator.group_transmitted_features(group),
            )
            for group in self._groups_from_result(current_result)
        )
        self.phase3_report["fixed_group_membership"] = [
            {
                "origin_server": group.server_id,
                "group_id": group.id,
                "members": list(group.devices),
            }
            for group in self._far_fixed_groups
        ]
        working = {
            key: list(value)
            for key, value in current_result.subtask_assignment.items()
        }
        best_result, working, pruning_report = self._far_prune_redundancy(
            working,
            current_result,
        )
        context = self._scope_context()
        structures = list(self.phase1_report.get("shared_structures", []))
        max_iterations = max(1, int(self.phase3_report.get("max_iterations", 2)))
        max_joint = max(1, int(self.phase3_report.get("max_joint_assignments", 32)))
        max_private = max(1, int(self.phase3_report.get("max_private_candidates", 48)))
        accepted_moves: list[dict[str, Any]] = []
        trace_report: list[dict[str, Any]] = []
        candidate_evaluations = 0

        for iteration in range(max_iterations):
            improved = False
            for structure in structures:
                paths = self._far_enumerate_paths(structure)
                placement = self._far_placement(working)
                blocks, diagnostics = self._far_marked_blocks(
                    structure,
                    paths,
                    placement,
                    context,
                )
                trace_report.append(
                    {
                        "iteration": iteration + 1,
                        "structure": structure.get("id"),
                        "paths": diagnostics,
                        "marked_blocks": blocks,
                    }
                )
                for block in blocks:
                    candidate_sets = self._far_shared_candidate_sets(
                        block,
                        paths,
                        placement,
                        structure,
                        context,
                    )
                    best_candidate = None
                    for choice in self._far_joint_choices(candidate_sets, max_joint):
                        trial_placement = self._far_apply_shared_move(
                            placement,
                            structure,
                            choice,
                            context,
                        )
                        if trial_placement == placement:
                            continue
                        trial_result = self._far_repair_and_evaluate(trial_placement)
                        candidate_evaluations += 1
                        if trial_result is None:
                            continue
                        if trial_result.total_cost >= best_result.total_cost - 1e-9:
                            continue
                        item = (trial_result.total_cost, choice, trial_result)
                        if best_candidate is None or item[0] < best_candidate[0]:
                            best_candidate = item
                    if best_candidate is None:
                        continue
                    _, choice, accepted_result = best_candidate
                    cost_before = best_result.total_cost
                    best_result = accepted_result
                    working = {
                        key: list(value)
                        for key, value in best_result.subtask_assignment.items()
                    }
                    accepted_moves.append(
                        {
                            "stage": "shared_structure",
                            "iteration": iteration + 1,
                            "structure": structure.get("id"),
                            "block": list(block),
                            "joint_assignment": dict(choice),
                            "cost_before": cost_before,
                            "cost_after": best_result.total_cost,
                            "saved_cost": cost_before - best_result.total_cost,
                        }
                    )
                    improved = True
            if not improved:
                break

        shared_keys = self._far_shared_occurrences(structures, context)
        private_report: list[dict[str, Any]] = []
        private_evaluations = 0
        for key in self._far_private_order(shared_keys):
            if private_evaluations >= max_private:
                break
            placement = self._far_placement(working)
            candidates, references = self._far_private_candidates(
                key,
                placement,
                shared_keys,
                context,
            )
            best_candidate = None
            evaluated_servers = []
            for server_id in candidates:
                if private_evaluations >= max_private:
                    break
                if server_id == placement.get(key):
                    continue
                trial_placement = dict(placement)
                trial_placement[key] = server_id
                trial_result = self._far_repair_and_evaluate(trial_placement)
                private_evaluations += 1
                candidate_evaluations += 1
                evaluated_servers.append(server_id)
                if trial_result is None:
                    continue
                if trial_result.total_cost >= best_result.total_cost - 1e-9:
                    continue
                item = (trial_result.total_cost, server_id, trial_result)
                if best_candidate is None or item[:2] < best_candidate[:2]:
                    best_candidate = item
            record = {
                "subtask": self._assignment_label(key),
                "references": [self._assignment_label(item) for item in references],
                "candidate_servers": candidates,
                "evaluated_servers": evaluated_servers,
                "accepted": best_candidate is not None,
            }
            if best_candidate is not None:
                _, selected_server, accepted_result = best_candidate
                cost_before = best_result.total_cost
                best_result = accepted_result
                working = {
                    item_key: list(value)
                    for item_key, value in best_result.subtask_assignment.items()
                }
                record.update(
                    {
                        "selected_server": selected_server,
                        "cost_before": cost_before,
                        "cost_after": best_result.total_cost,
                        "saved_cost": cost_before - best_result.total_cost,
                    }
                )
                accepted_moves.append({"stage": "private_propagation", **record})
            private_report.append(record)

        self.phase3_report.update(
            {
                "mode": "FAR",
                "redundancy_pruning": pruning_report,
                "shared_structure_traces": trace_report,
                "private_propagation": private_report,
                "candidate_evaluations": candidate_evaluations,
                "accepted_moves": accepted_moves,
                "cost_before": initial_cost,
                "cost_after": best_result.total_cost,
                "saved_cost": initial_cost - best_result.total_cost,
                "constraint_violations_after": len(best_result.violations),
            }
        )
        setattr(best_result, "phase3_report", self.phase3_report)
        return best_result

    def _far_placement(self, assignments) -> Dict[AssignmentKey, ServerId]:
        return {
            key: pairs[0][0]
            for key, pairs in assignments.items()
            if pairs
        }

    def _far_enumerate_paths(self, structure) -> list[list[str]]:
        nodes = list(structure.get("node_types", []))
        edges = [tuple(edge) for edge in structure.get("edges", [])]
        successors: Dict[str, list[str]] = {node: [] for node in nodes}
        indegree = {node: 0 for node in nodes}
        for source, target in edges:
            if source in successors and target in successors:
                successors[source].append(target)
                indegree[target] += 1
        sources = [node for node in nodes if indegree[node] == 0]
        sinks = {node for node in nodes if not successors[node]}
        paths: list[list[str]] = []

        def visit(node, path, seen):
            if len(paths) >= 128:
                return
            if node in sinks:
                paths.append(path + [node])
                return
            for target in sorted(successors[node]):
                if target not in seen:
                    visit(target, path + [node], seen | {target})

        for source in sorted(sources):
            visit(source, [], {source})
        return paths or [[node] for node in nodes]

    def _far_node_type_server(self, node_type, structure, placement, context):
        for task_id in structure.get("support_tasks", []):
            key = context["occurrence_by_task_type"].get((task_id, node_type))
            if key in placement:
                return placement[key]
        for key in context["occurrences_by_type"].get(node_type, []):
            if key in placement:
                return placement[key]
        return structure.get("placement_by_type", {}).get(node_type)

    def _far_marked_blocks(self, structure, paths, placement, context):
        blocks: list[list[str]] = []
        diagnostics = []
        for path in paths:
            servers = [
                self._far_node_type_server(node_type, structure, placement, context)
                for node_type in path
            ]
            if not servers or any(server_id is None for server_id in servers):
                continue
            trace = [servers[0]]
            spans = []
            for index in range(len(servers) - 1):
                physical = self._far_shortest_path(servers[index], servers[index + 1])
                start = len(trace) - 1
                trace.extend(physical[1:])
                spans.append((start, len(trace) - 1))
            marked, reasons = self._far_detect_detours(trace, spans)
            ordered = sorted(marked)
            path_blocks: list[list[str]] = []
            if ordered:
                current = [ordered[0]]
                for index in ordered[1:]:
                    if index == current[-1] + 1:
                        current.append(index)
                    else:
                        path_blocks.append([path[item] for item in current])
                        current = [index]
                path_blocks.append([path[item] for item in current])
            for block in path_blocks:
                if block not in blocks:
                    blocks.append(block)
            diagnostics.append(
                {
                    "node_path": path,
                    "node_servers": servers,
                    "physical_trace": trace,
                    "detours": reasons,
                    "marked_nodes": [path[index] for index in ordered],
                }
            )
        return blocks, diagnostics

    def _far_detect_detours(self, trace, spans):
        marked: Set[int] = set()
        reasons: list[dict[str, Any]] = []
        directed: Dict[Tuple[ServerId, ServerId], list[int]] = defaultdict(list)
        for index in range(len(trace) - 1):
            directed[(trace[index], trace[index + 1])].append(index)
        for (source, target), positions in directed.items():
            reverse = directed.get((target, source), [])
            if not reverse or str(source) > str(target):
                continue
            all_positions = positions + reverse
            self._far_mark_trace_range(marked, spans, min(all_positions), max(all_positions) + 1)
            reasons.append(
                {
                    "type": "reversed_link",
                    "link": [source, target],
                    "forward_positions": positions,
                    "reverse_positions": reverse,
                }
            )
        for left in range(len(trace)):
            for right in range(left + 2, len(trace)):
                segment = trace[left : right + 1]
                if trace[left] == trace[right] and segment == list(reversed(segment)):
                    self._far_mark_trace_range(marked, spans, left, right)
                    reasons.append(
                        {"type": "palindrome", "range": [left, right], "trace": segment}
                    )
        return marked, reasons

    @staticmethod
    def _far_mark_trace_range(marked, spans, left, right):
        for transition, (start, end) in enumerate(spans):
            if end >= left and start <= right:
                marked.add(transition)
                marked.add(transition + 1)

    def _far_physical_adjacency(self):
        adjacency: Dict[ServerId, list[Tuple[float, ServerId]]] = {
            server_id: [] for server_id in self.servers
        }
        for link in getattr(self, "physical_wired_links", ()):
            source = str(link.get("src"))
            target = str(link.get("dst"))
            weight = float(link.get("weight", 1.0))
            if source in adjacency and target in adjacency:
                adjacency[source].append((weight, target))
                adjacency[target].append((weight, source))
        return adjacency

    def _far_shortest_path(self, source, target):
        if source == target:
            return [source]
        adjacency = self._far_physical_adjacency()
        if not adjacency.get(source):
            return [source, target]
        queue = [(0.0, source, (source,))]
        best = {source: 0.0}
        while queue:
            cost, server_id, path = heapq.heappop(queue)
            if server_id == target:
                return list(path)
            if cost > best.get(server_id, float("inf")) + 1e-12:
                continue
            for weight, neighbor in adjacency.get(server_id, []):
                candidate = cost + weight
                if candidate < best.get(neighbor, float("inf")) - 1e-12:
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor, path + (neighbor,)))
        return [source, target]

    def _far_shared_candidate_sets(self, block, paths, placement, structure, context):
        candidate_sets = {}
        for node_type in block:
            candidates = {
                self._far_node_type_server(node_type, structure, placement, context)
            }
            for path in paths:
                for index, item in enumerate(path):
                    if item != node_type:
                        continue
                    for neighbor_index in (index - 1, index + 1):
                        if 0 <= neighbor_index < len(path):
                            server_id = self._far_node_type_server(
                                path[neighbor_index], structure, placement, context
                            )
                            if server_id is not None:
                                candidates.add(server_id)
            candidate_sets[node_type] = sorted(
                server_id for server_id in candidates if server_id in self.servers
            )
        return candidate_sets

    @staticmethod
    def _far_joint_choices(candidate_sets, limit):
        node_types = list(candidate_sets)
        for count, values in enumerate(
            product(*(candidate_sets[node_type] for node_type in node_types))
        ):
            if count >= limit:
                break
            yield dict(zip(node_types, values))

    def _far_apply_shared_move(self, placement, structure, choice, context):
        trial = dict(placement)
        for node_type, server_id in choice.items():
            for task_id in structure.get("support_tasks", []):
                key = context["occurrence_by_task_type"].get((task_id, node_type))
                if key is not None:
                    trial[key] = server_id
        return trial

    def _far_repair_and_evaluate(self, placement):
        """Repair X/Y by marginal loss reduction, then locally rerun CFRG."""
        subtask_by_key = self._subtask_by_key()
        keys_by_server: Dict[ServerId, list[AssignmentKey]] = defaultdict(list)
        for key, server_id in placement.items():
            if server_id not in self.servers or key not in subtask_by_key:
                return None
            keys_by_server[server_id].append(key)

        assignments: Dict[AssignmentKey, list[Tuple[ServerId, ExpertId]]] = {
            key: [] for key in placement
        }
        activated: Dict[ServerId, Set[ExpertId]] = {
            server_id: set() for server_id in self.servers
        }
        used_memory = {server_id: 0.0 for server_id in self.servers}
        probabilities = {key: 0.0 for key in placement}

        for server_id, keys in keys_by_server.items():
            server = self.servers[server_id]
            while True:
                unsatisfied = [
                    key
                    for key in keys
                    if self.evaluator.performance_loss_from_probability_and_reconstruction(
                        probabilities[key], 0.0
                    )
                    > self.evaluator.conformal_loss_threshold(subtask_by_key[key]) + 1e-12
                ]
                if not unsatisfied:
                    break
                candidates = []
                for expert_id in server.stored_experts:
                    if expert_id in activated[server_id] or expert_id not in self.experts:
                        continue
                    memory = self.experts[expert_id].memory
                    if used_memory[server_id] + memory > server.gpu_memory + 1e-12:
                        continue
                    reduction = 0.0
                    for key in unsatisfied:
                        contribution = self.evaluator.expert_contribution(
                            subtask_by_key[key], expert_id
                        )
                        old_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                            probabilities[key], 0.0
                        )
                        new_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                            probabilities[key] + contribution, 0.0
                        )
                        reduction += max(0.0, old_loss - new_loss)
                    candidates.append((reduction, -memory, expert_id))
                if not candidates:
                    return None
                reduction, _, expert_id = max(candidates)
                if reduction <= 1e-12:
                    return None
                activated[server_id].add(expert_id)
                used_memory[server_id] += self.experts[expert_id].memory
                for key in keys:
                    contribution = self.evaluator.expert_contribution(
                        subtask_by_key[key], expert_id
                    )
                    if contribution > 0.0:
                        assignments[key].append((server_id, expert_id))
                        probabilities[key] += contribution

        if any(not pairs for pairs in assignments.values()):
            return None
        selected_probability, _ = self._selection_metadata_from_assignments(assignments)
        result = self._far_evaluate_fixed_groups(assignments, selected_probability)
        return None if result is None or result.violations else result

    def _far_evaluate_fixed_groups(self, assignments, selected_probability):
        """Evaluate FAR placement while preserving Phase-II IoT group membership."""
        groups = [
            replace(group, bandwidth=0.0)
            for group in getattr(self, "_far_fixed_groups", ())
        ]
        if not groups:
            return None

        available_features: Set[str] = set()
        for group in groups:
            available_features.update(
                self.evaluator.group_transmitted_features(group)
            )

        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                received = set(subtask.required_features) & available_features
                if set(subtask.required_features) - received:
                    return None
                for server_id in self.evaluator.participating_servers(assignments, key):
                    subtask_features[
                        self._server_feature_key(key, server_id)
                    ] = set(received)

        data_req = self._subtask_data_requirements(assignments, subtask_features)
        feature_to_groups = self._feature_to_groups(groups)
        backhaul, backhaul_plan = self._derive_backhaul(data_req, feature_to_groups)
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                groups,
                assignments,
                data_req,
                backhaul,
                feature_to_groups,
            )
        except InfeasibleBandwidthBudget:
            return None

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
        if evaluation.violations:
            return None

        (
            _,
            _,
            required_probability,
            reconstruction_loss,
            performance_loss,
        ) = self._recompute_loss_metadata(
            assignments,
            selected_probability,
            resolved_groups,
            subtask_features,
        )
        server_features = self._server_required_features(data_req)

        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment={key: list(value) for key, value in assignments.items()},
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
            selected_probability=dict(selected_probability),
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
        )

    def _far_shared_occurrences(self, structures, context):
        shared = set()
        for structure in structures:
            for task_id in structure.get("support_tasks", []):
                for node_type in structure.get("node_types", []):
                    key = context["occurrence_by_task_type"].get((task_id, node_type))
                    if key is not None:
                        shared.add(key)
        return shared

    def _far_private_order(self, shared_keys):
        ordered = []
        for task in sorted(self.tasks.values(), key=lambda item: item.id):
            remaining = {subtask.id: subtask for subtask in task.subtasks}
            emitted = set()
            while remaining:
                ready = [
                    subtask
                    for subtask in remaining.values()
                    if all(
                        predecessor in emitted or predecessor not in remaining
                        for predecessor in subtask.predecessors
                    )
                ]
                if not ready:
                    ready = list(remaining.values())
                for subtask in sorted(ready, key=lambda item: item.id):
                    key = (task.id, subtask.id)
                    if key not in shared_keys:
                        ordered.append(key)
                    emitted.add(subtask.id)
                    remaining.pop(subtask.id, None)
        return ordered

    def _far_private_candidates(self, key, placement, shared_keys, context):
        references = [
            neighbor
            for neighbor in context["neighbors"].get(key, set())
            if neighbor in shared_keys and neighbor in placement
        ]
        if not references:
            subtask = context["subtask_by_key"][key]
            references = [
                (key[0], predecessor)
                for predecessor in subtask.predecessors
                if (key[0], predecessor) in placement
            ]
        candidates = {placement[key]}
        adjacency = self._far_physical_adjacency()
        for reference in references:
            server_id = placement[reference]
            candidates.add(server_id)
            candidates.update(neighbor for _, neighbor in adjacency.get(server_id, []))
        return sorted(candidates), references

    def _far_prune_redundancy(self, assignments, current_result):
        """Jointly prune experts/backhauls with one acceptance rule.

        Every tentative removal is fully reevaluated. A removal is eligible
        only when all constraints remain feasible and its resulting total cost
        is strictly lower than the current total cost. Among all eligible
        removals in one round, FAR commits the lowest-cost candidate.
        """
        result = current_result
        working = {key: list(value) for key, value in assignments.items()}
        removed_experts: list[dict[str, Any]] = []
        removed_backhaul: list[dict[str, Any]] = []
        accepted_removals: list[dict[str, Any]] = []
        evaluation_trace: list[dict[str, Any]] = []
        round_index = 0

        while True:
            round_index += 1
            candidates = []

            for pair in sorted({pair for pairs in working.values() for pair in pairs}):
                affected = [key for key, pairs in working.items() if pair in pairs]
                trial = {
                    key: [item for item in pairs if item != pair]
                    for key, pairs in working.items()
                }
                trace = {
                    "round": round_index,
                    "type": "expert",
                    "server": pair[0],
                    "expert": pair[1],
                    "affected_subtasks": [
                        self._assignment_label(key) for key in affected
                    ],
                    "cost_before": result.total_cost,
                }
                if any(not trial[key] for key in affected):
                    trace.update(
                        {
                            "feasible": False,
                            "accepted": False,
                            "reason": "an affected subtask would have no selected expert",
                        }
                    )
                    evaluation_trace.append(trace)
                    continue
                if not self._far_full_feature_loss_feasible(trial, affected):
                    trace.update(
                        {
                            "feasible": False,
                            "accepted": False,
                            "reason": "performance-loss precheck failed",
                        }
                    )
                    evaluation_trace.append(trace)
                    continue
                selected_probability, _ = self._selection_metadata_from_assignments(trial)
                trial_result = self._far_evaluate_fixed_groups(
                    trial,
                    selected_probability,
                )
                feasible = trial_result is not None and not trial_result.violations
                cost_after = None if trial_result is None else trial_result.total_cost
                improving = (
                    feasible
                    and cost_after is not None
                    and cost_after < result.total_cost - 1e-9
                )
                trace.update(
                    {
                        "feasible": feasible,
                        "cost_after": cost_after,
                        "improving": improving,
                        "accepted": False,
                    }
                )
                evaluation_trace.append(trace)
                if improving:
                    candidates.append(
                        (
                            float(cost_after),
                            0,
                            (pair[0], pair[1]),
                            "expert",
                            trial_result,
                            trial,
                            trace,
                        )
                    )

            for edge in sorted(result.backhaul):
                trial_result = self._far_remove_backhaul_tuple(result, edge)
                feasible = trial_result is not None and not trial_result.violations
                cost_after = None if trial_result is None else trial_result.total_cost
                improving = (
                    feasible
                    and cost_after is not None
                    and cost_after < result.total_cost - 1e-9
                )
                trace = {
                    "round": round_index,
                    "type": "backhaul",
                    "origin": edge[0],
                    "group": edge[1],
                    "destination": edge[2],
                    "cost_before": result.total_cost,
                    "feasible": feasible,
                    "cost_after": cost_after,
                    "improving": improving,
                    "accepted": False,
                }
                if trial_result is None and self._far_last_backhaul_rejection:
                    trace.update(self._far_last_backhaul_rejection)
                evaluation_trace.append(trace)
                if improving:
                    candidates.append(
                        (
                            float(cost_after),
                            1,
                            (edge[0], edge[1], edge[2]),
                            "backhaul",
                            trial_result,
                            {
                                key: list(value)
                                for key, value in trial_result.subtask_assignment.items()
                            },
                            trace,
                        )
                    )

            if not candidates:
                break

            (
                _,
                _,
                _,
                candidate_type,
                accepted_result,
                accepted_assignments,
                accepted_trace,
            ) = min(candidates, key=lambda item: item[:3])
            accepted_trace["accepted"] = True
            accepted_trace["saved_cost"] = (
                result.total_cost - accepted_result.total_cost
            )
            accepted_removals.append(dict(accepted_trace))

            stored_record = {
                key: value
                for key, value in accepted_trace.items()
                if key not in {"round", "type", "feasible", "improving", "accepted"}
            }
            if candidate_type == "expert":
                removed_experts.append(stored_record)
            else:
                removed_backhaul.append(stored_record)

            result = accepted_result
            working = accepted_assignments

        return result, working, {
            "acceptance_rule": (
                "tentative removal -> update affected variables -> full feasibility "
                "check -> accept iff total cost strictly decreases"
            ),
            "removed_experts": removed_experts,
            "removed_backhaul_tuples": removed_backhaul,
            "accepted_removals": accepted_removals,
            "candidate_evaluations": evaluation_trace,
        }

    def _far_full_feature_loss_feasible(self, assignments, keys) -> bool:
        subtasks = self._subtask_by_key()
        for key in keys:
            subtask = subtasks.get(key)
            if subtask is None:
                continue
            experts = {expert_id for _, expert_id in assignments.get(key, [])}
            probability = self.evaluator.selection_probability(subtask, experts)
            loss = self.evaluator.performance_loss_from_probability_and_reconstruction(
                probability, 0.0
            )
            if loss > self.evaluator.conformal_loss_threshold(subtask) + 1e-12:
                return False
        return True

    def _far_remove_backhaul_tuple(self, result, edge):
        self._far_last_backhaul_rejection = None
        groups = self._groups_from_result(result)
        if not groups:
            return None
        remaining_backhaul = set(result.backhaul) - {edge}
        remaining_plan = [
            decision
            for decision in result.backhaul_plan
            if (
                decision.source_server,
                decision.group_id,
                decision.target_server,
            )
            != edge
        ]
        available: Dict[ServerId, Set[str]] = defaultdict(set)
        for group in groups:
            available[group.server_id].update(
                self.evaluator.group_transmitted_features(group)
            )
        for decision in remaining_plan:
            available[decision.target_server].update(decision.features)

        subtask_by_key = self._subtask_by_key()
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        missing_required_features: list[dict[str, Any]] = []
        for key in sorted(result.subtask_assignment):
            subtask = subtask_by_key.get(key)
            if subtask is None:
                continue
            required = set(subtask.required_features)
            for server_id in self.evaluator.participating_servers(
                result.subtask_assignment,
                key,
            ):
                missing = required - available.get(server_id, set())
                if missing:
                    missing_required_features.append(
                        {
                            "subtask": self._assignment_label(key),
                            "server": server_id,
                            "missing_features": sorted(missing),
                        }
                    )
                    continue
                subtask_features[
                    self._server_feature_key(key, server_id)
                ] = set(required)

        if missing_required_features:
            self._far_last_backhaul_rejection = {
                "reason": "required feature would become unreachable",
                "missing_required_features": missing_required_features,
            }
            return None

        data_req = self._subtask_data_requirements(
            result.subtask_assignment, subtask_features
        )
        feature_to_groups = self._feature_to_groups(groups)
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                groups,
                result.subtask_assignment,
                data_req,
                remaining_backhaul,
                feature_to_groups,
            )
        except InfeasibleBandwidthBudget as exc:
            self._far_last_backhaul_rejection = {
                "reason": "non-positive uplink time budget",
                "detail": str(exc),
            }
            return None
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(
            self.servers,
            self.devices,
            self.experts,
            self.tasks,
            eval_config,
        )
        evaluation = evaluator.evaluate(
            result.subtask_assignment,
            resolved_groups,
            remaining_backhaul,
            subtask_features=subtask_features,
        )
        _, _, required_probability, reconstruction_loss, performance_loss = (
            self._recompute_loss_metadata(
                result.subtask_assignment,
                result.selected_probability,
                resolved_groups,
                subtask_features,
            )
        )
        server_features = self._server_required_features(data_req)
        return ChainedPipelineResult(
            expert_placement=result.expert_placement,
            subtask_assignment=result.subtask_assignment,
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
                self._group_key_label(key): value for key, value in budgets.items()
            },
            rsma_group_bandwidths=self._group_bandwidth_map(resolved_groups),
            backhaul_plan=remaining_plan,
            selected_probability=result.selected_probability,
            required_probability=required_probability,
            reconstruction_loss=reconstruction_loss,
            performance_loss=performance_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=remaining_backhaul,
            violations=list(evaluation.violations),
            evaluation=evaluation,
        )

    def _legacy_phase3_refine(
        self,
        assignments,
        current_result: ChainedPipelineResult,
    ) -> ChainedPipelineResult:
        if not getattr(self, "_far_fixed_groups", None):
            self._far_fixed_groups = tuple(
                replace(
                    group,
                    bandwidth=0.0,
                    required_features=self.evaluator.group_transmitted_features(group),
                )
                for group in self._groups_from_result(current_result)
            )
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
                    selected_probability, _ = self._selection_metadata_from_assignments(trial_assignments)
                    trial_result = self._far_evaluate_fixed_groups(
                        trial_assignments,
                        selected_probability,
                    )
                    candidate_evaluations += 1
                    if trial_result is None or trial_result.violations:
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
        try:
            resolved_groups, budgets = self._derive_group_bandwidths(
                pruned_groups,
                result.subtask_assignment,
                data_req,
                backhaul,
                feature_to_groups,
            )
        except InfeasibleBandwidthBudget:
            self.post_pruning_report.update(
                {
                    "enabled": True,
                    "mode": "offline_final_group_pruning",
                    "reverted": True,
                    "cost_before": round(result.total_cost, 6),
                    "cost_after": round(result.total_cost, 6),
                    "note": "Pruned candidate has no positive uplink time budget.",
                }
            )
            setattr(result, "post_pruning_report", self.post_pruning_report)
            return result
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
        if current_eval is None:
            return current_groups, current_features
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
            if trial_eval is None or trial_eval.violations:
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
        try:
            resolved_groups, _ = self._derive_group_bandwidths(
                groups,
                assignments,
                data_req,
                backhaul,
                feature_to_groups,
            )
        except InfeasibleBandwidthBudget:
            return None
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
