"""GSSGD phase 1 adapted to this project: CCM + SPCCM selection.

This keeps the original GSSGD selection idea and only maps SIoT locations to
this project's IoT feature indexes. DCG logic is not included.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

from utils.formulation import DeviceId, DeviceSpec, FormulationEvaluator, ServerId


def phase1_dbg_selection(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    local_device_ids: Iterable[DeviceId],
    required_features: Set[str],
) -> Tuple[List[DeviceId], List[DeviceId]]:
    """Select a local SIoT set by CCM first, then SPCCM until coverage is met."""
    candidates = [device_id for device_id in local_device_ids if devices[device_id].features & required_features]
    candidates = sorted(candidates, key=_id_sort_key)
    if not candidates or not required_features:
        return [], candidates

    selected: List[DeviceId] = []
    first, max_coverage = _compute_ccm(devices, evaluator, server_id, candidates, required_features)
    if first is not None:
        selected.append(first)

    selected = _compute_spccm(
        devices=devices,
        evaluator=evaluator,
        server_id=server_id,
        candidates=candidates,
        selected=selected,
        required_features=required_features,
        max_coverage=max_coverage,
    )
    unselected = [device_id for device_id in candidates if device_id not in selected]
    return selected, unselected


def _compute_ccm(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    candidates: Sequence[DeviceId],
    required_features: Set[str],
) -> tuple[DeviceId | None, int]:
    coverages = [len(devices[device_id].features & required_features) for device_id in candidates]
    gains = [abs(evaluator.channel_gain(device_id, server_id)) for device_id in candidates]
    max_coverage = max(coverages, default=0)
    min_coverage = min(coverages, default=0)
    max_gain = max(gains, default=0.0)
    min_gain = min(gains, default=0.0)

    best: tuple[float, str, DeviceId] | None = None
    for device_id, coverage_value, gain_value in zip(candidates, coverages, gains):
        coverage = _normalize(coverage_value, min_coverage, max_coverage)
        gain = _normalize(gain_value, min_gain, max_gain)
        ccm = coverage * gain
        item = (ccm, str(device_id), device_id)
        if best is None or item > best:
            best = item
    return (None if best is None else best[2]), max(max_coverage, 1)


def _compute_spccm(
    devices: dict[DeviceId, DeviceSpec],
    evaluator: FormulationEvaluator,
    server_id: ServerId,
    candidates: Sequence[DeviceId],
    selected: List[DeviceId],
    required_features: Set[str],
    max_coverage: int,
) -> List[DeviceId]:
    covered = _covered_features(devices, selected, required_features)
    while not covered.issuperset(required_features):
        remaining = [device_id for device_id in candidates if device_id not in selected]
        if not remaining:
            break

        best: tuple[float, int, str, DeviceId] | None = None
        for device_id in remaining:
            additional = (devices[device_id].features & required_features) - covered
            if not additional:
                continue
            phase_alignment = _phase_alignment(evaluator, server_id, [*selected, device_id])
            additional_coverage = len(additional) / max(max_coverage, 1)
            spccm = phase_alignment * additional_coverage
            item = (spccm, len(additional), str(device_id), device_id)
            if best is None or item > best:
                best = item
        if best is None:
            break
        selected.append(best[3])
        covered.update(devices[best[3]].features & required_features)
    return selected


def _covered_features(
    devices: dict[DeviceId, DeviceSpec],
    device_ids: Sequence[DeviceId],
    required_features: Set[str],
) -> Set[str]:
    covered: Set[str] = set()
    for device_id in device_ids:
        covered.update(devices[device_id].features & required_features)
    return covered


def _phase_alignment(evaluator: FormulationEvaluator, server_id: ServerId, device_ids: Sequence[DeviceId]) -> float:
    if not device_ids:
        return 0.0
    terms = [evaluator.db_phase_term(device_id, server_id) for device_id in device_ids]
    return abs(sum(terms)) / len(terms)


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 1.0 if value > 0 else 0.0
    return (value - minimum) / (maximum - minimum)


def _id_sort_key(value: str) -> tuple[str, int, str]:
    prefix, _, suffix = str(value).rpartition("_")
    try:
        return prefix, int(suffix), str(value)
    except ValueError:
        return str(value), 0, str(value)
