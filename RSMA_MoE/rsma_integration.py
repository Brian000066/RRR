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

def build_feature_bits(
    num_iot_features: int,
    default_feature_bits: float,
    feature_bits_range: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> dict[str, float]:
    if num_iot_features < 1:
        return {}
    rng = rng or random.Random()
    if feature_bits_range is None:
        low = high = float(default_feature_bits)
    else:
        low, high = feature_bits_range
        low = float(low)
        high = float(high)
        if low <= 0.0 or high <= 0.0:
            raise ValueError("feature_bits_range values must be positive.")
        if low > high:
            low, high = high, low
    return {
        feature_name(index): (low if low == high else rng.uniform(low, high))
        for index in range(num_iot_features)
    }


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


def normalize_float_range(
    value_range: tuple[float, float] | None,
    default: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    low, high = value_range or default
    low = float(low)
    high = float(high)
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"{label} values must be positive.")
    if low > high:
        low, high = high, low
    return low, high


def build_simple_experts(
    num_experts: int,
    memory_range: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> dict[str, dict[str, float | int]]:
    """Create indexed experts with heterogeneous model sizes."""
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1.")

    rng = rng or random.Random()
    memory_min, memory_max = normalize_float_range(memory_range, (256.0, 256.0), "expert_memory_range")

    experts: dict[str, dict[str, float | int]] = {}
    for index in range(num_experts):
        memory = memory_min if memory_min == memory_max else rng.uniform(memory_min, memory_max)
        experts[expert_id(index)] = {
            "index": index,
            "memory": memory,
            "latency": 0.05 + index * 0.01,
        }
    return experts


def build_simple_servers(
    num_servers: int,
    experts: Mapping[str, Mapping[str, Any]],
    experts_per_server: int,
    gpu_memory: float = 8192.0,
    gpu_memory_range: tuple[float, float] | None = None,
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
    gpu_min, gpu_max = normalize_float_range(gpu_memory_range, (gpu_memory, gpu_memory), "server_gpu_memory_range")
    rate_min, rate_max = wired_rate_range or (1e9, 1e9)
    if rate_min <= 0.0 or rate_max <= 0.0:
        raise ValueError("wired rates must be positive.")
    if rate_min > rate_max:
        rate_min, rate_max = rate_max, rate_min

    servers: list[dict[str, Any]] = []
    for server_index, stored in enumerate(experts_by_server):
        server_gpu_memory = gpu_min if gpu_min == gpu_max else rng.uniform(gpu_min, gpu_max)
        servers.append(
            {
                "id": f"server_{server_index}",
                "gpu_memory": server_gpu_memory,
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
    """SIoT-RSMA scalar channel gain for one IoT-server link.

    The SIoT code draws link distances in [10, 300] m and computes
    |h| = |small-scale fading| / (path loss * shadowing). We keep the server
    antenna count as experiment metadata, but this channel model is scalar.
    """
    del num_antennas
    siot_distance = min(max(distance, 10.0), 300.0)
    path_loss_db = 50.0 + 15.0 * math.log10(siot_distance)
    loss = 10 ** (path_loss_db / 10.0)
    shadowing = 10 ** (rng.gauss(0.0, 1.0) / 10.0)
    small_scale_fading = complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
    channel_gain = abs(small_scale_fading) / (loss * shadowing)
    channel_phase = math.atan2(small_scale_fading.imag, small_scale_fading.real)
    return [], channel_gain, channel_phase

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




def circular_feature_window(feature_pool: Sequence[int], start: int, size: int) -> list[int]:
    if not feature_pool:
        return []
    size = max(1, min(size, len(feature_pool)))
    return [feature_pool[(start + offset) % len(feature_pool)] for offset in range(size)]


def fill_features_from_pool(
    features: set[int],
    pool: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> None:
    if len(features) >= target_count:
        return
    candidates = [feature for feature in pool if feature not in features]
    if not candidates:
        return
    take_count = min(target_count - len(features), len(candidates))
    features.update(rng.sample(candidates, take_count))

def build_server_feature_pools(
    feature_pool: Sequence[int],
    num_servers: int,
    overlap_ratio: float,
    rng: random.Random,
) -> dict[int, list[int]]:
    """Split global feature indexes into server pools with limited neighbor overlap."""
    if num_servers < 1:
        return {}
    overlap_ratio = min(max(overlap_ratio, 0.0), 1.0)
    chunks = [list(chunk) for chunk in _split_evenly(list(feature_pool), num_servers)]
    avg_chunk_size = max(1, math.ceil(len(feature_pool) / num_servers))
    overlap_count = max(1, round(avg_chunk_size * overlap_ratio)) if overlap_ratio > 0.0 else 0
    pools: dict[int, set[int]] = {index: set(chunks[index]) for index in range(num_servers)}
    for server_index in range(num_servers):
        if overlap_count <= 0:
            continue
        for neighbor_index in ((server_index - 1) % num_servers, (server_index + 1) % num_servers):
            neighbor_chunk = chunks[neighbor_index]
            if neighbor_chunk:
                pools[server_index].update(rng.sample(neighbor_chunk, min(overlap_count, len(neighbor_chunk))))
    return {index: sorted(values) for index, values in pools.items()}


def _split_evenly(values: Sequence[int], parts: int) -> list[list[int]]:
    return [
        list(values[round(index * len(values) / parts): round((index + 1) * len(values) / parts)])
        for index in range(parts)
    ]


def build_position_feature_pools(
    server_feature_pools: Mapping[int, Sequence[int]],
    sector_count: int,
    ring_count: int,
) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    sector_pools: dict[tuple[int, int], list[int]] = {}
    ring_pools: dict[tuple[int, int], list[int]] = {}
    for server_index, pool in server_feature_pools.items():
        pool = list(pool)
        window_size = max(1, math.ceil(len(pool) / max(2, sector_count // 2)))
        for sector_index in range(sector_count):
            sector_pools[(server_index, sector_index)] = circular_feature_window(
                pool,
                sector_index * max(1, window_size // 2),
                window_size,
            )
        ring_window_size = max(1, math.ceil(len(pool) / max(1, ring_count)))
        for ring_index in range(ring_count):
            ring_pools[(server_index, ring_index)] = circular_feature_window(
                pool,
                ring_index * ring_window_size,
                ring_window_size + max(1, ring_window_size // 2),
            )
    return sector_pools, ring_pools


def spatial_feature_set(
    *,
    feature_pool: Sequence[int],
    server_feature_pools: Mapping[int, Sequence[int]],
    sector_feature_pools: Mapping[tuple[int, int], Sequence[int]],
    ring_feature_pools: Mapping[tuple[int, int], Sequence[int]],
    home_server_index: int,
    connected_server_indices: Sequence[int],
    home_distance: float,
    home_angle: float,
    cell_radius: float,
    feature_count: int,
    rng: random.Random,
    global_random_fraction: float = 0.15,
    sector_count: int = 8,
    ring_count: int = 3,
) -> set[int]:
    """Assign feature indexes with positive, imperfect correlation to IoT position."""
    if feature_count <= 0:
        return set()

    global_random_fraction = min(max(global_random_fraction, 0.0), 1.0)
    local_target = max(1, round(feature_count * (1.0 - global_random_fraction)))
    sector_index = int((home_angle / (2.0 * math.pi)) * sector_count) % sector_count
    distance_ratio = min(max(home_distance / max(cell_radius, 1e-12), 0.0), 0.999999)
    ring_index = int(distance_ratio * ring_count)

    overlap_pool: set[int] = set()
    for server_index in connected_server_indices:
        if server_index != home_server_index:
            overlap_pool.update(server_feature_pools.get(server_index, []))

    features: set[int] = set()
    fill_features_from_pool(
        features,
        sector_feature_pools.get((home_server_index, sector_index), []),
        max(1, round(feature_count * 0.35)),
        rng,
    )
    fill_features_from_pool(
        features,
        ring_feature_pools.get((home_server_index, ring_index), []),
        max(len(features), round(feature_count * 0.55)),
        rng,
    )
    fill_features_from_pool(
        features,
        sorted(overlap_pool),
        max(len(features), round(feature_count * 0.70)),
        rng,
    )
    fill_features_from_pool(
        features,
        server_feature_pools.get(home_server_index, []),
        max(len(features), local_target),
        rng,
    )
    fill_features_from_pool(features, feature_pool, feature_count, rng)
    return set(sorted(features))


def build_simple_devices(
    num_iot_features: int,
    num_iot_devices: int,
    num_servers: int,
    features_per_device_range: tuple[int, int],
    area_size: float,
    cell_radius: float,
    num_antennas: int,
    rng: random.Random,
    max_power: float = 50.0,
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
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
    min_features, max_features = features_per_device_range
    if min_features < 1 or max_features < min_features:
        raise ValueError("features_per_device_range must be valid and positive.")
    max_features = min(max_features, num_iot_features)

    server_positions = build_server_positions(num_servers, area_size)
    devices: list[dict[str, Any]] = []
    association_map: list[dict[str, Any]] = []
    feature_pool = list(range(num_iot_features))
    sector_count = 8
    ring_count = 3
    server_feature_pools = build_server_feature_pools(
        feature_pool,
        num_servers,
        server_feature_overlap_ratio,
        rng,
    )
    sector_feature_pools, ring_feature_pools = build_position_feature_pools(
        server_feature_pools,
        sector_count,
        ring_count,
    )

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

        device_index = len(devices)
        feature_count = rng.randint(min_features, max_features)

        channel_gain: dict[str, float] = {}
        channel_vector: dict[str, list[list[float]]] = {}
        raw_channel_vectors: dict[str, list[complex]] = {}
        channel_phase: dict[str, float] = {}
        phase_angle: dict[str, float] = {}
        distance_by_server: dict[str, float] = {}
        server_ids = [f"server_{index}" for index in connected]
        for server_index in connected:
            server_id = f"server_{server_index}"
            link_distance = min(max(distances[server_index], 10.0), cell_radius)
            vector, gain, channel_phase_value = channel_from_distance(
                link_distance,
                num_antennas,
                rng,
            )
            channel_gain[server_id] = gain
            raw_channel_vectors[server_id] = vector
            channel_vector[server_id] = vector_to_json(vector)
            channel_phase[server_id] = channel_phase_value
            phase_angle[server_id] = geometric_angle((x, y), server_positions[server_index])
            distance_by_server[server_id] = round(float(link_distance), 6)

        home_server_index = min(connected, key=lambda index: distances[index])
        home_distance = min(max(distances[home_server_index], 10.0), cell_radius)
        home_angle = geometric_angle((x, y), server_positions[home_server_index])
        features = spatial_feature_set(
            feature_pool=feature_pool,
            server_feature_pools=server_feature_pools,
            sector_feature_pools=sector_feature_pools,
            ring_feature_pools=ring_feature_pools,
            home_server_index=home_server_index,
            connected_server_indices=connected,
            home_distance=home_distance,
            home_angle=home_angle,
            cell_radius=cell_radius,
            feature_count=feature_count,
            rng=rng,
            global_random_fraction=global_random_feature_fraction,
            sector_count=sector_count,
            ring_count=ring_count,
        )
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
        "channel_model": "siot_scalar_pathloss_shadowing_fading",
        "distance_model": "geometry_based_link_distance_clamped_to_siot_10_300m",
        "angle_model": "geometry_based_counterclockwise_from_server_horizontal_axis",
        "feature_model": "server_pool_position_correlated_with_global_randomness",
        "feature_model_parameters": {
            "server_feature_overlap_ratio": server_feature_overlap_ratio,
            "global_random_feature_fraction": global_random_feature_fraction,
            "sector_count": sector_count,
            "ring_count": ring_count,
            "server_feature_pool_sizes": {
                f"server_{index}": len(pool)
                for index, pool in server_feature_pools.items()
            },
        },
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
    expert_memory_range: tuple[float, float] | None = None,
    feature_bits_range: tuple[float, float] | None = None,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (5, 12),
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
    server_gpu_memory_range: tuple[float, float] | None = None,
    wired_rate_range: tuple[float, float] | None = None,
    c_bw: float = 1e-3,
    c_act: float = 1.0,
    c_fwd: float = 1.0,
    bandwidth_time_fraction: float = 1.0,
    default_feature_bits: float = 12000.0,
    wavelength: float = 0.125,
    noise_power: float = 1e-18,
    common_power_ratio: float = 0.6,
    max_device_power: float = 1.2589e-3,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    loss_threshold: float | None = None,
    lambda_reconstruction: float = 0.1,
    calibration_alpha: float = 0.1,
    random_seed: int = 42,
    max_group_size: int = 5,
    min_rate: float = 1.0,
    gssgd_beamforming_gain_threshold: float = 0.0,
) -> Any:
    """Run the RSMA/JRGEP scheduler on generated DAG tasks."""
    rng = random.Random(random_seed)
    tasks = graphs_to_tasks(graphs, loss_threshold)
    feature_bits_by_name = build_feature_bits(num_iot_features, default_feature_bits, feature_bits_range, rng)
    experts = build_simple_experts(num_experts, memory_range=expert_memory_range, rng=rng)
    servers = build_simple_servers(
        num_servers=num_servers,
        experts=experts,
        experts_per_server=experts_per_server,
        gpu_memory=server_gpu_memory,
        gpu_memory_range=server_gpu_memory_range,
        wired_rate_range=wired_rate_range,
        rng=rng,
    )
    devices, topology = build_simple_devices(
        num_iot_features=num_iot_features,
        num_iot_devices=num_iot_devices,
        num_servers=num_servers,
        features_per_device_range=features_per_device_range,
        server_feature_overlap_ratio=server_feature_overlap_ratio,
        global_random_feature_fraction=global_random_feature_fraction,
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
            min_rate=min_rate,
            gssgd_beamforming_gain_threshold=gssgd_beamforming_gain_threshold,
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            derive_bandwidth=True,
            bandwidth_time_fraction=bandwidth_time_fraction,
                noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            default_feature_bits=default_feature_bits,
            feature_bits_by_name=feature_bits_by_name,
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
            "expert_memory_range": expert_memory_range,
            "experts_per_server": experts_per_server,
            "server_gpu_memory_range": server_gpu_memory_range,
            "wired_rate_range": wired_rate_range,
            "bandwidth_mode": "derived",
            "bandwidth_time_fraction": bandwidth_time_fraction,
            "default_feature_bits": default_feature_bits,
            "feature_bits_range": feature_bits_range,
            "feature_bits_by_name": feature_bits_by_name,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "max_group_size": max_group_size,
            "min_rate": min_rate,
            "gssgd_beamforming_gain_threshold": gssgd_beamforming_gain_threshold,
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_to_jsonable(result, network_context), file, ensure_ascii=False, indent=2)

    return result
