from __future__ import annotations

from typing import Mapping, Sequence, Set, Tuple

from utils.formulation import (
    AssignmentKey,
    ExpertId,
    FormulationConfig,
    FormulationEvaluator,
    GroupSpec,
    ObjectiveBreakdown,
    ServerId,
)


def cal_objective_cost(
    evaluator: FormulationEvaluator,
    assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    groups: Sequence[GroupSpec],
    backhaul: Set[Tuple[ServerId, str, ServerId]] | None = None,
) -> ObjectiveBreakdown:
    return evaluator.compute_objective(assignments, groups, backhaul or set())


def cal_total_cost(
    evaluator: FormulationEvaluator,
    assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    groups: Sequence[GroupSpec],
    backhaul: Set[Tuple[ServerId, str, ServerId]] | None = None,
) -> float:
    return cal_objective_cost(evaluator, assignments, groups, backhaul).total_cost


def activation_cost(num_activated_experts: int, config: FormulationConfig) -> float:
    return num_activated_experts * config.c_act


def bandwidth_cost(groups: Sequence[GroupSpec], config: FormulationConfig) -> float:
    return sum(group.bandwidth for group in groups) * config.c_bw


def forwarding_cost(bits: float, wired_rate: float, config: FormulationConfig) -> float:
    if wired_rate == float("inf"):
        return 0.0
    return config.c_fwd * bits / wired_rate
