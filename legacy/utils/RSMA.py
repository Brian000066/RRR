from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from utils.formulation import DeviceSpec, FormulationConfig, GroupSpec


def build_equal_power_group(
    server_id: str,
    group_id: str,
    device_ids: Sequence[str],
    devices: Mapping[str, DeviceSpec],
    config: FormulationConfig,
) -> GroupSpec:
    common_power = {
        device_id: devices[device_id].max_power * config.common_power_ratio
        for device_id in device_ids
    }
    private_power = {
        device_id: devices[device_id].max_power * (1.0 - config.common_power_ratio)
        for device_id in device_ids
    }
    return GroupSpec(
        server_id=server_id,
        id=group_id,
        devices=tuple(device_ids),
        bandwidth=0.0,
        common_power=common_power,
        private_power=private_power,
    )


def compute_group_rates(
    evaluator,
    group: GroupSpec,
) -> Tuple[float, Dict[str, float]]:
    common_rates, private_rates = evaluator.compute_rates([group])
    common = common_rates[(group.server_id, group.id)]
    private = {
        device_id: private_rates[(group.server_id, group.id, device_id)]
        for device_id in group.devices
    }
    return common, private


def min_private_rate(private_rates: Mapping[str, float]) -> float:
    return min(private_rates.values(), default=0.0)
