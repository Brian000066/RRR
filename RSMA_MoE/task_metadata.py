from __future__ import annotations

import random

import networkx as nx


def make_gating_weights(num_experts: int, rng: random.Random) -> list[float]:
    """Create normalized gating weights whose sum is exactly 1.0."""
    raw_weights = [rng.random() for _ in range(num_experts)]
    total = sum(raw_weights)

    if total == 0:
        return [1.0 / num_experts for _ in range(num_experts)]

    weights = [round(weight / total, 6) for weight in raw_weights]
    weights[-1] = round(1.0 - sum(weights[:-1]), 6)
    return weights


def make_iot_features(
    rng: random.Random,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
) -> list[int]:
    """Choose IoT feature indexes required by one node."""
    min_count, max_count = iot_features_per_node_range

    if min_count < 0:
        raise ValueError("iot_features_per_node_range min cannot be negative.")
    if max_count < min_count:
        raise ValueError(
            "iot_features_per_node_range must be (min_count, max_count)."
        )
    if num_iot_features < 1:
        raise ValueError("num_iot_features must be at least 1.")
    if max_count > num_iot_features:
        raise ValueError(
            "iot_features_per_node_range max cannot exceed num_iot_features."
        )

    feature_count = rng.randint(min_count, max_count)
    return sorted(rng.sample(range(num_iot_features), feature_count))


def make_node_attributes(
    node_name: str,
    layer: int,
    num_experts: int,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
    rng: random.Random,
) -> dict[str, object]:
    """Create task-oriented attributes for one node."""
    return {
        "layer": layer,
        "prompt": f"Complete subtask {node_name} in layer {layer}.",
        "output_key": f"{node_name}_output",
        "required_data": {
            "upstream_outputs": [],
            "iot_features": make_iot_features(
                rng=rng,
                num_iot_features=num_iot_features,
                iot_features_per_node_range=iot_features_per_node_range,
            ),
        },
        "gating_weights": make_gating_weights(num_experts, rng),
    }


def update_required_upstream_outputs(graph: nx.DiGraph) -> None:
    """Fill node required_data with outputs from predecessor nodes."""
    for node in graph.nodes:
        predecessor_outputs = [
            {
                "source_node": predecessor,
                "output_key": graph.nodes[predecessor]["output_key"],
            }
            for predecessor in sorted(
                graph.predecessors(node),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
        ]
        graph.nodes[node]["required_data"]["upstream_outputs"] = (
            predecessor_outputs
        )


def validate_num_experts(num_experts: int) -> None:
    """Validate expert count for gating weights."""
    if num_experts < 1:
        raise ValueError("NUM_EXPERTS must be at least 1.")


def make_task_deadline_seconds(
    deadline_seconds_range: tuple[int, int],
    rng: random.Random,
) -> int:
    """Create one task deadline as a relative duration in seconds."""
    min_seconds, max_seconds = deadline_seconds_range

    if min_seconds < 1:
        raise ValueError("Task deadline must be at least 1 second.")
    if max_seconds < min_seconds:
        raise ValueError(
            "TASK_DEADLINE_SECONDS_RANGE must be "
            "(min_seconds, max_seconds)."
        )

    return rng.randint(min_seconds, max_seconds)
