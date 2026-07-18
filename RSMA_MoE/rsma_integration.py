from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx


BASE_DIR = Path(__file__).resolve().parent
RSMA_DIR = BASE_DIR / "26_rsma_network_2026"
if str(RSMA_DIR) not in sys.path:
    sys.path.insert(0, str(RSMA_DIR))

from jrgep_scheduler_standalone import JRGEPScheduler  # noqa: E402
from utils.formulation import FormulationConfig  # noqa: E402


def feature_name(feature_index: int | str) -> str:
    return f"feature_{feature_index}"


def expert_id(expert_index: int | str) -> str:
    return f"expert_{expert_index}"


def node_to_subtask(
    node_id: str,
    node: Mapping[str, Any],
    predecessors: Sequence[str],
    loss_threshold: float | None,
) -> dict[str, Any]:
    required_data = node.get("required_data", {})
    iot_features = required_data.get("iot_features", [])
    gating_weights = list(node.get("gating_weights", []))
    expert_confidence = list(node.get("expert_confidence", []))

    return {
        "id": str(node_id),
        "required_features": [feature_name(index) for index in iot_features],
        "predecessors": [str(item) for item in predecessors],
        "output_tokens": 256,
        "feature_volume_bits": 8_000.0 * max(len(iot_features), 1),
        "loss_threshold": loss_threshold,
        "prompt": node.get("prompt", ""),
        "gating_weights": gating_weights,
        "expert_confidence": expert_confidence,
        "reconstruction_loss": float(node.get("reconstruction_loss", 0.0)),
        "calibration_losses": list(node.get("calibration_losses", [])),
    }


def graphs_to_tasks(
    graphs: Iterable[nx.DiGraph],
    loss_threshold: float | None,
) -> list[dict[str, Any]]:
    """Convert generated DAG objects into JRGEP scheduler task specs."""
    tasks: list[dict[str, Any]] = []

    for graph in graphs:
        graph_meta = dict(graph.graph)
        task_id = str(graph_meta.get("task_id", f"task_{graph_meta.get('graph_index', len(tasks) + 1)}"))
        subtasks: list[dict[str, Any]] = []

        for node_id in nx.topological_sort(graph):
            node = graph.nodes[node_id]
            required_data = node.get("required_data", {})
            upstream_outputs = required_data.get("upstream_outputs", [])
            predecessors = [
                str(item.get("source_node"))
                for item in upstream_outputs
                if item.get("source_node") is not None
            ]
            if not predecessors:
                predecessors = [str(pred) for pred in graph.predecessors(node_id)]
            subtasks.append(
                node_to_subtask(
                    str(node_id),
                    node,
                    predecessors,
                    loss_threshold,
                )
            )

        tasks.append(
            {
                "id": task_id,
                "deadline": float(graph_meta.get("deadline_seconds", float("inf"))),
                "subtasks": subtasks,
            }
        )

    return tasks


def load_generated_dag_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dag_json_to_task(
    data: Mapping[str, Any],
    loss_threshold: float | None = None,
) -> dict[str, Any]:
    """Convert one saved generated_dags/dag_N.json file to a scheduler task."""
    graph_meta = dict(data.get("graph", {}))
    task_id = str(graph_meta.get("task_id", graph_meta.get("graph_index", "task")))

    predecessors_by_node: dict[str, list[str]] = {}
    for edge in data.get("edges", []):
        predecessors_by_node.setdefault(str(edge["target"]), []).append(str(edge["source"]))

    subtasks: list[dict[str, Any]] = []
    for node in data.get("nodes", []):
        node_id = str(node.get("id"))
        required_data = node.get("required_data", {})
        upstream_outputs = required_data.get("upstream_outputs", [])
        predecessors = [
            str(item.get("source_node"))
            for item in upstream_outputs
            if item.get("source_node") is not None
        ] or predecessors_by_node.get(node_id, [])
        subtasks.append(
            node_to_subtask(
                node_id,
                node,
                predecessors,
                loss_threshold,
            )
        )

    return {
        "id": task_id,
        "deadline": float(graph_meta.get("deadline_seconds", float("inf"))),
        "subtasks": subtasks,
    }


def load_tasks_from_generated_dir(
    generated_dir: Path,
    loss_threshold: float | None = None,
) -> list[dict[str, Any]]:
    return [
        dag_json_to_task(load_generated_dag_json(path), loss_threshold)
        for path in sorted(generated_dir.glob("dag_*.json"))
    ]


def build_simple_experts(num_experts: int) -> dict[str, dict[str, float | int]]:
    """Create indexed experts."""
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")

    experts: dict[str, dict[str, float | int]] = {}
    for index in range(num_experts):
        experts[expert_id(index)] = {
            "index": index,
            "memory": 256.0,
            "latency": 0.05 + index * 0.01,
        }
    return experts


def build_simple_servers(
    num_servers: int,
    experts: Mapping[str, Mapping[str, Any]],
    experts_per_server: int,
    gpu_memory: float = 8192.0,
    default_wired_rate: float = 1e9,
    wired_rate_range: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Create edge servers with globally unique expert storage and random wired links."""
    if num_servers < 1:
        raise ValueError("num_servers must be at least 1.")
    if experts_per_server < 1:
        raise ValueError("experts_per_server must be at least 1.")

    expert_ids = sorted(experts, key=lambda item: int(item.rsplit("_", 1)[1]))
    experts_per_server = min(experts_per_server, len(expert_ids))
    total_slots = num_servers * experts_per_server
    if len(expert_ids) > total_slots:
        raise ValueError(
            "not enough server expert slots for unique expert storage: "
            f"{len(expert_ids)} experts > {total_slots} slots"
        )

    experts_by_server: list[list[str]] = [[] for _ in range(num_servers)]
    for expert_offset, stored_expert_id in enumerate(expert_ids):
        start_server = expert_offset % num_servers
        for hop in range(num_servers):
            server_index = (start_server + hop) % num_servers
            if len(experts_by_server[server_index]) < experts_per_server:
                experts_by_server[server_index].append(stored_expert_id)
                break

    rng = rng or random.Random()
    rate_min, rate_max = wired_rate_range or (default_wired_rate, default_wired_rate)
    if rate_min <= 0.0 or rate_max <= 0.0:
        raise ValueError("wired rates must be positive.")
    if rate_min > rate_max:
        rate_min, rate_max = rate_max, rate_min

    servers: list[dict[str, Any]] = []
    for server_index, stored in enumerate(experts_by_server):
        servers.append(
            {
                "id": f"server_{server_index}",
                "gpu_memory": gpu_memory,
                "stored_experts": stored,
                "active_experts": [],
                "wired_rates": {
                    f"server_{other}": rng.uniform(rate_min, rate_max)
                    for other in range(num_servers)
                    if other != server_index
                },
            }
        )
    return servers


def build_server_positions(num_servers: int, area_size: float) -> list[tuple[float, float]]:
    cell_size = math.ceil(math.sqrt(num_servers))
    step = int(area_size) // cell_size
    positions: list[tuple[float, float]] = []
    for row in range(cell_size):
        for col in range(cell_size):
            if len(positions) >= num_servers:
                break
            positions.append((float(row * step + step // 2), float(col * step + step // 2)))
    return positions



def geometric_angle(device_position: tuple[float, float], server_position: tuple[float, float]) -> float:
    """Counterclockwise angle from MEC horizontal axis to the IoT device."""
    angle = math.atan2(
        device_position[1] - server_position[1],
        device_position[0] - server_position[0],
    )
    return angle if angle >= 0.0 else angle + (2.0 * math.pi)

def channel_from_distance(
    distance: float,
    num_antennas: int,
    rng: random.Random,
) -> tuple[list[complex], float, float]:
    """Original SIoT-RSMA style complex channel vector for one IoT-server link."""
    safe_distance = max(distance, 1.0)
    path_loss_db = 50.0 + 15.0 * math.log10(safe_distance)
    loss = 10 ** (path_loss_db / 10.0)
    shadowing = 10 ** (rng.gauss(0.0, 1.0) / 10.0)

    vector = [
        complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) / (loss * shadowing)
        for _ in range(num_antennas)
    ]
    channel_gain = math.sqrt(sum(abs(value) ** 2 for value in vector))
    channel_phase = math.atan2(
        sum(value.imag for value in vector),
        sum(value.real for value in vector),
    )
    return vector, channel_gain, channel_phase


def vector_to_json(vector: Sequence[complex]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in vector]


def spatial_correlation(vector_a: Sequence[complex], vector_b: Sequence[complex]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 1.0
    numerator = abs(sum(a.conjugate() * b for a, b in zip(vector_a, vector_b))) ** 2
    denominator = sum(abs(a) ** 2 for a in vector_a) * sum(abs(b) ** 2 for b in vector_b)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(float(numerator / denominator), 1.0))


def build_simple_devices(
    num_iot_features: int,
    num_iot_devices: int,
    num_servers: int,
    features_per_device_range: tuple[int, int],
    connected_servers_per_device: int | None,
    area_size: float,
    cell_radius: float,
    num_antennas: int,
    rng: random.Random,
    max_power: float = 50.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create IoT devices with multi-cell associations and original-like channels."""
    if num_iot_features < 1:
        raise ValueError("num_iot_features must be at least 1.")
    if num_iot_devices < 1:
        raise ValueError("num_iot_devices must be at least 1.")
    if area_size <= 0 or cell_radius <= 0:
        raise ValueError("area_size and cell_radius must be positive.")
    if num_antennas < 1:
        raise ValueError("num_antennas must be at least 1.")
    if connected_servers_per_device is not None:
        if connected_servers_per_device < 1 or connected_servers_per_device > num_servers:
            raise ValueError("connected_servers_per_device must be None or between 1 and num_servers.")

    min_features, max_features = features_per_device_range
    if min_features < 1 or max_features < min_features:
        raise ValueError("features_per_device_range must be valid and positive.")
    max_features = min(max_features, num_iot_features)

    server_positions = build_server_positions(num_servers, area_size)
    devices: list[dict[str, Any]] = []
    association_map: list[dict[str, Any]] = []
    feature_pool = list(range(num_iot_features))

    while len(devices) < num_iot_devices:
        x = rng.uniform(0.0, area_size)
        y = rng.uniform(0.0, area_size)
        distances = [
            math.dist((x, y), server_position)
            for server_position in server_positions
        ]
        connected = [
            index
            for index, distance in enumerate(distances)
            if distance <= cell_radius
        ]
        if not connected:
            continue

        if connected_servers_per_device is not None and len(connected) > connected_servers_per_device:
            connected = sorted(rng.sample(connected, connected_servers_per_device))

        device_index = len(devices)
        feature_count = rng.randint(min_features, max_features)
        features = set(rng.sample(feature_pool, feature_count))
        features.add(device_index % num_iot_features)

        channel_gain: dict[str, float] = {}
        channel_vector: dict[str, list[list[float]]] = {}
        raw_channel_vectors: dict[str, list[complex]] = {}
        channel_phase: dict[str, float] = {}
        phase_angle: dict[str, float] = {}
        distance_by_server: dict[str, float] = {}
        server_ids = [f"server_{index}" for index in connected]
        for server_index in connected:
            server_id = f"server_{server_index}"
            vector, gain, channel_phase_value = channel_from_distance(
                distances[server_index],
                num_antennas,
                rng,
            )
            channel_gain[server_id] = gain
            raw_channel_vectors[server_id] = vector
            channel_vector[server_id] = vector_to_json(vector)
            channel_phase[server_id] = channel_phase_value
            phase_angle[server_id] = geometric_angle((x, y), server_positions[server_index])
            distance_by_server[server_id] = round(float(distances[server_index]), 6)

        home_server_index = min(connected, key=lambda index: distances[index])
        association_map.append({
            "device_id": device_index,
            "connected_edge_server_ids": list(connected),
            "distances": [round(float(distances[index]), 2) for index in connected],
            "angles": [
                round(float(geometric_angle((x, y), server_positions[index])), 6)
                for index in connected
            ],
        })
        devices.append(
            {
                "id": f"device_{device_index}",
                "home_server": f"server_{home_server_index}",
                "connected_edge_server_ids": server_ids,
                "features": [feature_name(index) for index in sorted(features)],
                "position": [round(x, 6), round(y, 6)],
                "channel_gain": channel_gain,
                "channel_vector": channel_vector,
                "channel_phase": channel_phase,
                "_raw_channel_vector": raw_channel_vectors,
                "spatial_correlation": {},
                "phase_angle": phase_angle,
                "distance": distance_by_server,
                "max_power": max_power,
            }
        )

    for server_index in range(num_servers):
        server_id = f"server_{server_index}"
        connected_device_indices = [
            index
            for index, device in enumerate(devices)
            if server_id in device["connected_edge_server_ids"]
        ]
        for index in connected_device_indices:
            device = devices[index]
            correlations = {}
            vector = device["_raw_channel_vector"][server_id]
            for other_index in connected_device_indices:
                other = devices[other_index]
                other_id = other["id"]
                other_vector = other["_raw_channel_vector"][server_id]
                correlations[other_id] = spatial_correlation(vector, other_vector)
            device["spatial_correlation"][server_id] = correlations

    for device in devices:
        device.pop("_raw_channel_vector", None)

    topology = {
        "area_size": area_size,
        "cell_radius": cell_radius,
        "num_antennas": num_antennas,
        "channel_model": "geometry_based_distance_angle_with_complex_vector_channel",
        "association_policy": "all_edge_servers_within_cell_radius",
        "association_map": association_map,
        "server_positions": {
            f"server_{index}": [round(x, 6), round(y, 6)]
            for index, (x, y) in enumerate(server_positions)
        },
    }
    return devices, topology


def result_to_jsonable(result: Any, network_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    evaluation = result.evaluation
    payload = {
        "objective": {
            "total_cost": int(result.total_cost + 0.5),
            "activation_cost": int(result.activation_cost + 0.5),
            "bandwidth_cost": int(result.bandwidth_cost + 0.5),
            "forwarding_cost": int(result.forwarding_cost + 0.5),
        },
        "expert_placement": result.expert_placement,
        "subtask_assignment": {
            f"{task_id}:{subtask_id}": pairs
            for (task_id, subtask_id), pairs in result.subtask_assignment.items()
        },
        "rsma_groups": result.rsma_groups,
        "rsma_group_bandwidths": result.rsma_group_bandwidths,
        "backhaul": [list(item) for item in sorted(result.backhaul)],
        "violations": result.violations,
        "task_finish_time": evaluation.timing.task_finish_time if evaluation else {},
    }
    if network_context is not None:
        payload["network_model"] = dict(network_context)
    return payload


def run_rsma_scheduler(
    graphs: Sequence[nx.DiGraph],
    output_path: Path,
    num_experts: int,
    num_iot_features: int,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (2, 6),
    connected_servers_per_device: int | None = None,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
    default_wired_rate: float = 1e9,
    wired_rate_range: tuple[float, float] | None = None,
    c_bw: float = 1e-3,
    c_act: float = 1.0,
    c_fwd: float = 1.0,
    uplink_time_budget: float | None = None,
    bandwidth_time_fraction: float = 1.0,
    min_bandwidth: float = 0.0,
    default_feature_bits: float = 12000.0,
    wavelength: float = 0.125,
    noise_power: float = 1e-18,
    common_power_ratio: float = 0.6,
    max_device_power: float = 1.2589e-3,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    beamforming_correlation_weight: float = 0.0,
    loss_threshold: float | None = None,
    lambda_reconstruction: float = 0.1,
    calibration_alpha: float = 0.1,
    random_seed: int = 42,
    max_group_size: int = 4,
) -> Any:
    """Run the RSMA/JRGEP scheduler on generated DAG tasks."""
    rng = random.Random(random_seed)
    tasks = graphs_to_tasks(graphs, loss_threshold)
    experts = build_simple_experts(num_experts)
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        gpu_memory=server_gpu_memory,
        default_wired_rate=default_wired_rate,
        wired_rate_range=wired_rate_range,
        rng=rng,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=num_servers,
        features_per_device_range=features_per_device_range,
        connected_servers_per_device=connected_servers_per_device,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        rng=rng,
        max_power=max_device_power,
    )

    scheduler = JRGEPScheduler(
        servers=servers,
        devices=devices,
        experts=experts,
        tasks=tasks,
        config=FormulationConfig(
            max_group_size=max_group_size,
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            derive_bandwidth=True,
            uplink_time_budget=uplink_time_budget,
            bandwidth_time_fraction=bandwidth_time_fraction,
            min_bandwidth=min_bandwidth,
            default_wired_rate=default_wired_rate,
            noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            beamforming_correlation_weight=beamforming_correlation_weight,
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
        ),
    )
    result = scheduler.schedule()
    network_context = {
        "experts": experts,
        "servers": servers,
        "devices": devices,
        "topology": topology,
        "rsma_parameters": {
            "num_servers": num_servers,
            "num_iot_devices": num_iot_devices,
            "features_per_device_range": features_per_device_range,
            "connected_servers_per_device": connected_servers_per_device,
            "experts_per_server": experts_per_server,
            "server_gpu_memory": server_gpu_memory,
            "default_wired_rate": default_wired_rate,
            "wired_rate_range": wired_rate_range,
            "bandwidth_mode": "derived",
            "uplink_time_budget": uplink_time_budget,
            "bandwidth_time_fraction": bandwidth_time_fraction,
            "min_bandwidth": min_bandwidth,
            "default_feature_bits": default_feature_bits,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "beamforming_correlation_weight": beamforming_correlation_weight,
            "max_group_size": max_group_size,
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_to_jsonable(result, network_context), file, ensure_ascii=False, indent=2)

    return result










