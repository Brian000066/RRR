"""
Generate layered DAGs and export each one as PNG, GraphML, and JSON.

Install dependencies:
    pip install networkx matplotlib

Run:
    python main.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx

from task_metadata import (
    make_node_attributes,
    make_task_deadline_seconds,
    update_required_upstream_outputs,
    validate_num_experts,
)


@dataclass
class DAGSpec:
    num_nodes: int
    depth: int
    max_width: int
    edge_probability: float = 0.20
    allow_skip_edges: bool = False
    allow_early_branch_end: bool = False
    single_source: bool = True
    single_sink: bool = False
    exact_layer_widths: Optional[list[int]] = None


def build_dag_specs(num_dags: int, common_spec: DAGSpec) -> list[DAGSpec]:
    """Create one shared DAGSpec for each DAG to generate."""
    return [
        DAGSpec(
            num_nodes=common_spec.num_nodes,
            depth=common_spec.depth,
            max_width=common_spec.max_width,
            edge_probability=common_spec.edge_probability,
            allow_skip_edges=common_spec.allow_skip_edges,
            allow_early_branch_end=common_spec.allow_early_branch_end,
            single_source=common_spec.single_source,
            single_sink=common_spec.single_sink,
            exact_layer_widths=(
                list(common_spec.exact_layer_widths)
                if common_spec.exact_layer_widths is not None
                else None
            ),
        )
        for _ in range(num_dags)
    ]


def validate_spec(spec: DAGSpec) -> None:
    """Validate DAG generation settings."""
    if spec.num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if spec.depth < 1:
        raise ValueError("depth must be at least 1.")
    if spec.max_width < 1:
        raise ValueError("max_width must be at least 1.")
    if not 0.0 <= spec.edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1.")
    if spec.depth > spec.num_nodes:
        raise ValueError(
            f"depth={spec.depth} cannot be greater than "
            f"num_nodes={spec.num_nodes}."
        )

    if spec.exact_layer_widths is not None:
        widths = spec.exact_layer_widths

        if len(widths) != spec.depth:
            raise ValueError("exact_layer_widths length must equal depth.")
        if sum(widths) != spec.num_nodes:
            raise ValueError("exact_layer_widths sum must equal num_nodes.")
        if any(width < 1 for width in widths):
            raise ValueError("Each layer width must be at least 1.")
        if any(width > spec.max_width for width in widths):
            raise ValueError("No layer width can be greater than max_width.")
        if spec.single_source and widths[0] != 1:
            raise ValueError(
                "single_source=True requires the first layer width to be 1."
            )
        if spec.single_sink and widths[-1] != 1:
            raise ValueError(
                "single_sink=True requires the last layer width to be 1."
            )
        return

    capacity = spec.depth * spec.max_width
    if spec.single_source:
        capacity -= spec.max_width - 1
    if spec.single_sink and spec.depth > 1:
        capacity -= spec.max_width - 1

    if spec.num_nodes > capacity:
        raise ValueError(
            f"These settings can hold at most {capacity} nodes, "
            f"but num_nodes={spec.num_nodes}. Increase depth or max_width."
        )


def allocate_layer_widths(spec: DAGSpec, rng: random.Random) -> list[int]:
    """Distribute num_nodes across depth layers."""
    validate_spec(spec)

    if spec.exact_layer_widths is not None:
        return list(spec.exact_layer_widths)

    widths = [1] * spec.depth
    remaining = spec.num_nodes - spec.depth

    locked_layers: set[int] = set()
    if spec.single_source:
        locked_layers.add(0)
    if spec.single_sink:
        locked_layers.add(spec.depth - 1)

    available_layers = [
        layer for layer in range(spec.depth) if layer not in locked_layers
    ]

    while remaining > 0:
        candidates = [
            layer
            for layer in available_layers
            if widths[layer] < spec.max_width
        ]
        if not candidates:
            raise ValueError("Cannot fit num_nodes with this depth/max_width.")

        minimum_width = min(widths[layer] for layer in candidates)
        narrowest_layers = [
            layer for layer in candidates if widths[layer] == minimum_width
        ]
        chosen_layer = rng.choice(narrowest_layers)
        widths[chosen_layer] += 1
        remaining -= 1

    return widths


def make_layered_nodes(
    graph_index: int,
    layer_widths: Sequence[int],
) -> tuple[list[list[str]], dict[str, int]]:
    """Create node names grouped by layer."""
    layers: list[list[str]] = []
    node_to_layer: dict[str, int] = {}

    node_counter = 0
    for layer_index, width in enumerate(layer_widths):
        current_layer: list[str] = []

        for _ in range(width):
            node_name = f"v{graph_index}_{node_counter}"
            current_layer.append(node_name)
            node_to_layer[node_name] = layer_index
            node_counter += 1

        layers.append(current_layer)

    return layers, node_to_layer


def generate_dag(
    graph_index: int,
    spec: DAGSpec,
    rng: random.Random,
    num_experts: int,
    num_iot_features: int,
    iot_features_per_node_range: tuple[int, int],
    deadline_seconds: int,
) -> nx.DiGraph:
    """Generate one layered directed acyclic graph."""
    validate_num_experts(num_experts)
    layer_widths = allocate_layer_widths(spec, rng)
    layers, node_to_layer = make_layered_nodes(
        graph_index=graph_index,
        layer_widths=layer_widths,
    )

    graph = nx.DiGraph(
        graph_index=graph_index,
        num_nodes=spec.num_nodes,
        depth=spec.depth,
        max_width=spec.max_width,
        edge_probability=spec.edge_probability,
        allow_skip_edges=spec.allow_skip_edges,
        allow_early_branch_end=spec.allow_early_branch_end,
        single_source=spec.single_source,
        single_sink=spec.single_sink,
        task_id=f"task_{graph_index}",
        task_name=f"Task {graph_index}",
        deadline_seconds=deadline_seconds,
        num_experts=num_experts,
        layer_widths=layer_widths,
    )

    for node, layer in node_to_layer.items():
        graph.add_node(
            node,
            **make_node_attributes(
                node_name=node,
                layer=layer,
                num_experts=num_experts,
                num_iot_features=num_iot_features,
                iot_features_per_node_range=iot_features_per_node_range,
                rng=rng,
            ),
        )

    # Add mandatory edges between adjacent layers so each non-source has a
    # parent. If allow_early_branch_end is false, each non-sink also gets a
    # child; otherwise, some branches may end before the deepest layer.
    for layer_index in range(spec.depth - 1):
        current_layer = layers[layer_index]
        next_layer = layers[layer_index + 1]

        for child_index, child in enumerate(next_layer):
            parent = current_layer[child_index % len(current_layer)]
            graph.add_edge(parent, child)

        if not spec.allow_early_branch_end:
            for parent_index, parent in enumerate(current_layer):
                if graph.out_degree(parent) == 0:
                    child = next_layer[parent_index % len(next_layer)]
                    graph.add_edge(parent, child)

    # Add optional random edges.
    for source_layer in range(spec.depth - 1):
        if spec.allow_skip_edges:
            possible_target_layers = range(source_layer + 1, spec.depth)
        else:
            possible_target_layers = [source_layer + 1]

        for target_layer in possible_target_layers:
            for source in layers[source_layer]:
                for target in layers[target_layer]:
                    if graph.has_edge(source, target):
                        continue
                    if rng.random() < spec.edge_probability:
                        graph.add_edge(source, target)

    update_required_upstream_outputs(graph)

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Generated graph is not a DAG.")

    return graph


def layered_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Calculate stable positions for a layered graph drawing."""
    layer_to_nodes: dict[int, list[str]] = {}

    for node, data in graph.nodes(data=True):
        layer = int(data["layer"])
        layer_to_nodes.setdefault(layer, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    max_layer_width = max(len(nodes) for nodes in layer_to_nodes.values())

    for layer, nodes in sorted(layer_to_nodes.items()):
        nodes = sorted(nodes, key=lambda name: int(name.rsplit("_", 1)[1]))
        width = len(nodes)

        if width == 1:
            x_values = [0.0]
        else:
            x_values = [index - (width - 1) / 2 for index in range(width)]

        if width > 1 and max_layer_width > 1:
            scale = (max_layer_width - 1) / (width - 1)
            x_values = [x * scale for x in x_values]

        for node, x_value in zip(nodes, x_values):
            positions[node] = (x_value, -float(layer))

    return positions


def visualize_dag(
    graph: nx.DiGraph,
    output_path: Path,
    dpi: int = 180,
) -> None:
    """Save one DAG as a PNG image."""
    positions = layered_positions(graph)

    depth = int(graph.graph["depth"])
    layer_widths = list(graph.graph["layer_widths"])
    widest_layer = max(layer_widths)

    figure_width = max(7.0, widest_layer * 1.7)
    figure_height = max(5.0, depth * 1.8)

    plt.figure(figsize=(figure_width, figure_height))

    nx.draw_networkx(
        graph,
        pos=positions,
        with_labels=True,
        arrows=True,
        node_size=1500,
        font_size=9,
        width=1.2,
        arrowsize=18,
        connectionstyle="arc3,rad=0.03",
    )

    graph_index = graph.graph["graph_index"]
    plt.title(
        f"DAG {graph_index}: "
        f"nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, "
        f"depth={depth}, "
        f"layer widths={layer_widths}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_dag_json(graph: nx.DiGraph, output_path: Path) -> None:
    """Save one DAG as JSON."""
    data = {
        "graph": dict(graph.graph),
        "nodes": [
            {
                "id": node,
                **attributes,
            }
            for node, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
            }
            for source, target in graph.edges()
        ],
        "topological_order": list(nx.topological_sort(graph)),
    }

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_multiple_dags(
    specs: Sequence[DAGSpec],
    output_dir: Path,
    seed: int = 42,
    image_dpi: int = 180,
    num_experts: int = 1,
    num_iot_features: int = 1,
    iot_features_per_node_range: tuple[int, int] = (1, 1),
    task_deadline_seconds_range: tuple[int, int] = (60, 60),
) -> list[nx.DiGraph]:
    """Generate and export multiple DAGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    graphs: list[nx.DiGraph] = []

    for graph_index, spec in enumerate(specs, start=1):
        deadline_seconds = make_task_deadline_seconds(
            task_deadline_seconds_range,
            rng,
        )
        graph = generate_dag(
            graph_index=graph_index,
            spec=spec,
            rng=rng,
            num_experts=num_experts,
            num_iot_features=num_iot_features,
            iot_features_per_node_range=iot_features_per_node_range,
            deadline_seconds=deadline_seconds,
        )
        graphs.append(graph)

        image_path = output_dir / f"dag_{graph_index}.png"
        graphml_path = output_dir / f"dag_{graph_index}.graphml"
        json_path = output_dir / f"dag_{graph_index}.json"

        visualize_dag(
            graph=graph,
            output_path=image_path,
            dpi=image_dpi,
        )

        graphml_graph = graph.copy()
        for key, value in list(graphml_graph.graph.items()):
            if isinstance(value, (list, dict, tuple, set)):
                graphml_graph.graph[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )
        for _, attributes in graphml_graph.nodes(data=True):
            for key, value in list(attributes.items()):
                if isinstance(value, (list, dict, tuple, set)):
                    attributes[key] = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
        nx.write_graphml(graphml_graph, graphml_path)

        save_dag_json(graph, json_path)

        print(
            f"[DAG {graph_index}] "
            f"nodes={graph.number_of_nodes()}, "
            f"edges={graph.number_of_edges()}, "
            f"deadline_seconds={deadline_seconds}, "
            f"layer_widths={graph.graph['layer_widths']}"
        )
        print(f"  PNG:     {image_path}")
        print(f"  GraphML: {graphml_path}")
        print(f"  JSON:    {json_path}")

    return graphs
