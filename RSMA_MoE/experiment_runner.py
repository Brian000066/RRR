from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from compared_method.hybrid_topk_distance_backhaul import (
    run_chained_pipeline as run_hybrid_topk_distance_backhaul,
)
from compared_method.hybrid_topk_gssgd_backhaul import run_hybrid_topk_gssgd_backhaul
from compared_method.WDMoE import run_wdmoe_distance, run_wdmoe_gssgd
from config import ExperimentConfig
from generate_dags import build_dag_specs, generate_multiple_dags
from rsma_integration import run_rsma_scheduler


@dataclass(frozen=True)
class RunSeeds:
    dag_seed: int
    network_seed: int


@dataclass(frozen=True)
class AverageResult:
    total_cost: float
    activation_cost: float
    bandwidth_cost: float
    forwarding_cost: float
    violation_count: float
    violations: tuple[Any, ...] = ()


def rounded_cost(value: float) -> int:
    return int(value + 0.5)


def resolve_run_seeds(config: ExperimentConfig, run_index: int = 0) -> RunSeeds:
    if not config.auto_random_seed:
        return RunSeeds(
            dag_seed=config.random_seed + run_index,
            network_seed=config.network_random_seed + run_index,
        )

    rng = random.SystemRandom()
    return RunSeeds(
        dag_seed=rng.randint(1, 2_147_483_647),
        network_seed=rng.randint(1, 2_147_483_647),
    )


def generate_task_graphs(config: ExperimentConfig, seeds: RunSeeds, verbose: bool = True, export_artifacts: bool = True):
    specs = build_dag_specs(config.num_dags, config.common_dag_spec)
    graphs = generate_multiple_dags(
        specs=specs,
        output_dir=config.output_dir,
        seed=seeds.dag_seed,
        image_dpi=config.image_dpi,
        num_experts=config.num_experts,
        num_iot_features=config.num_iot_features,
        iot_features_per_node_range=config.iot_features_per_node_range,
        task_deadline_seconds_range=config.task_deadline_seconds_range,
        reconstruction_sigma=config.reconstruction_sigma,
        reconstruction_error_range=config.reconstruction_error_range,
        num_calibration_samples=config.num_calibration_samples,
        calibration_loss_range=config.calibration_loss_range,
        verbose=verbose,
        export_artifacts=export_artifacts,
    )
    if verbose:
        print(f"Run seeds: DAG={seeds.dag_seed}, network={seeds.network_seed}")
        print(f"\nDone: generated {len(graphs)} DAG(s).")
        print(f"Output directory: {config.output_dir.resolve()}")
    return graphs


def common_scheduler_kwargs(config: ExperimentConfig, seeds: RunSeeds) -> dict[str, Any]:
    return {
        "num_experts": config.num_experts,
        "num_iot_features": config.num_iot_features,
        "num_servers": config.num_edge_servers,
        "num_iot_devices": config.num_iot_devices,
        "features_per_device_range": config.iot_features_per_device_range,
        "server_feature_overlap_ratio": config.iot_server_feature_overlap_ratio,
        "global_random_feature_fraction": config.iot_global_random_feature_fraction,
        "experts_per_server": config.experts_per_server,
        "server_gpu_memory": config.server_gpu_memory,
        "wired_rate_range": config.wired_rate_range,
        "c_bw": config.c_bw,
        "c_act": config.c_act,
        "c_fwd": config.c_fwd,
        "bandwidth_time_fraction": config.bandwidth_time_fraction,
        "default_feature_bits": config.default_feature_bits,
        "wavelength": config.wavelength,
        "noise_power": config.noise_power,
        "common_power_ratio": config.common_power_ratio,
        "max_device_power": config.max_device_power,
        "area_size": config.area_size,
        "cell_radius": config.cell_radius,
        "num_antennas": config.num_antennas,
        "random_seed": seeds.network_seed,
        "max_group_size": config.max_group_size,
        "min_rate": config.min_rate,
        "gssgd_beamforming_gain_threshold": config.gssgd_beamforming_gain_threshold,
        "loss_threshold": config.loss_threshold,
        "lambda_reconstruction": config.lambda_reconstruction,
        "calibration_alpha": config.calibration_alpha,
        "reconstruction_sigma": config.reconstruction_sigma,
    }


def topk_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "top_k": config.topk_k,
        "rank_by": config.topk_rank_by,
    }


def grouping_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "clusters_per_server": config.distance_grouping_clusters_per_server,
        "kmeans_iterations": config.distance_grouping_kmeans_iterations,
    }


def wdmoe_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "wdmoe_initial_threshold": config.wdmoe_initial_threshold,
        "wdmoe_threshold_step": config.wdmoe_threshold_step,
        "wdmoe_max_threshold": config.wdmoe_max_threshold,
        "wdmoe_wlr_target_ratio": config.wdmoe_wlr_target_ratio,
    }

def safe_usage(cost: float, unit_cost: float) -> float:
    if unit_cost == 0:
        return 0.0
    return cost / unit_cost


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def print_cost_line(label: str, cost: float, unit_cost: float, usage_unit: str) -> None:
    usage = safe_usage(cost, unit_cost)
    print(
        f"  {label}: usage={format_number(usage)} {usage_unit}, "
        f"unit_cost={unit_cost:g}, cost={rounded_cost(cost)}"
    )


def print_method_result(method_name: str, result_path, result: Any, config: ExperimentConfig) -> None:
    print(f"{method_name} result: {result_path.resolve()}")
    print(f"{method_name} total cost: {rounded_cost(result.total_cost)}")
    print("Cost breakdown:")
    print_cost_line("Activation", result.activation_cost, config.c_act, "server-expert activations")
    print_cost_line("Bandwidth", result.bandwidth_cost, config.c_bw, "Hz")
    print_cost_line("Forwarding", result.forwarding_cost, config.c_fwd, "forwarding events")
    print(f"{method_name} violations: {len(result.violations)}")


METHOD_DISPLAY_NAMES = {
    "rsma": "RSMA",
    "hybrid_topk_gssgd_backhaul": "Hybrid TopK+GSSGD",
    "hybrid_topk_distance_backhaul": "Hybrid TopK+Distance",
    "wdmoe_gssgd": "WDMoE+GSSGD",
    "wdmoe_distance": "WDMoE+Distance",
}


COST_COLORS = {
    "Activation": "#4C78A8",
    "Bandwidth": "#F58518",
    "Forwarding": "#54A24B",
}


def cost_components(result: Any) -> list[tuple[str, float]]:
    return [
        ("Activation", result.activation_cost),
        ("Bandwidth", result.bandwidth_cost),
        ("Forwarding", result.forwarding_cost),
    ]


def show_cost_plot(results: dict[str, Any], title: str = "Cost Comparison") -> None:
    if not results:
        return

    import matplotlib.pyplot as plt

    method_names = [METHOD_DISPLAY_NAMES.get(key, key) for key in results]
    total_costs = [rounded_cost(result.total_cost) for result in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    total_axis = axes[0]
    bars = total_axis.bar(method_names, total_costs, color="#4C78A8")
    total_axis.set_title("Total Cost")
    total_axis.set_ylabel("Cost")
    total_axis.tick_params(axis="x", rotation=18)
    total_axis.bar_label(bars, padding=3)
    total_axis.grid(axis="y", linestyle="--", alpha=0.3)

    breakdown_axis = axes[1]
    bottoms = [0 for _ in method_names]
    for label in ("Activation", "Bandwidth", "Forwarding"):
        values = [rounded_cost(getattr(result, f"{label.lower()}_cost")) for result in results.values()]
        bars = breakdown_axis.bar(
            method_names,
            values,
            bottom=bottoms,
            label=label,
            color=COST_COLORS[label],
        )
        labels = [str(value) if value else "" for value in values]
        breakdown_axis.bar_label(bars, labels=labels, label_type="center")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    breakdown_axis.set_title("Cost Breakdown")
    breakdown_axis.set_ylabel("Cost")
    breakdown_axis.tick_params(axis="x", rotation=18)
    breakdown_axis.legend()
    breakdown_axis.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()


def run_enabled_methods(
    config: ExperimentConfig,
    graphs,
    seeds: RunSeeds,
    *,
    print_details: bool = True,
    show_plot: bool = True,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    base_kwargs = common_scheduler_kwargs(config, seeds)

    if config.run_rsma_scheduler:
        result_path = config.results_dir / "rsma_schedule_result.json"
        result = run_rsma_scheduler(graphs=graphs, output_path=result_path, **base_kwargs)
        if print_details:
            print_method_result("RSMA", result_path, result, config)
        results["rsma"] = result

    if config.run_hybrid_topk_gssgd_backhaul:
        result_path = config.results_dir / "hybrid_topk_gssgd_backhaul_result.json"
        result = run_hybrid_topk_gssgd_backhaul(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
        )
        if print_details:
            print_method_result("Hybrid TopK+GSSGD", result_path, result, config)
        results["hybrid_topk_gssgd_backhaul"] = result

    if config.run_hybrid_topk_distance_backhaul:
        result_path = config.results_dir / "hybrid_topk_distance_backhaul_result.json"
        result = run_hybrid_topk_distance_backhaul(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
        )
        if print_details:
            print_method_result("Hybrid TopK+Distance", result_path, result, config)
        results["hybrid_topk_distance_backhaul"] = result


    if config.run_wdmoe_gssgd:
        result_path = config.results_dir / "wdmoe_gssgd_result.json"
        result = run_wdmoe_gssgd(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
            **wdmoe_kwargs(config),
        )
        if print_details:
            print_method_result("WDMoE+GSSGD", result_path, result, config)
        results["wdmoe_gssgd"] = result

    if config.run_wdmoe_distance:
        result_path = config.results_dir / "wdmoe_distance_result.json"
        result = run_wdmoe_distance(
            graphs=graphs,
            output_path=result_path,
            **base_kwargs,
            **topk_kwargs(config),
            **grouping_kwargs(config),
            **wdmoe_kwargs(config),
        )
        if print_details:
            print_method_result("WDMoE+Distance", result_path, result, config)
        results["wdmoe_distance"] = result

    if show_plot:
        show_cost_plot(results)
    return results


def print_run_summary(run_number: int, results: dict[str, Any]) -> None:
    parts = []
    for method_key, result in results.items():
        name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        parts.append(
            f"{name}: total={rounded_cost(result.total_cost)}, "
            f"violations={len(result.violations)}"
        )
    print(f"Run {run_number}: " + " | ".join(parts))


def average_run_results(run_results: list[dict[str, Any]]) -> dict[str, AverageResult]:
    if not run_results:
        return {}

    method_keys = list(run_results[0].keys())
    averages: dict[str, AverageResult] = {}
    for method_key in method_keys:
        method_results = [result_set[method_key] for result_set in run_results if method_key in result_set]
        count = len(method_results)
        if count == 0:
            continue
        averages[method_key] = AverageResult(
            total_cost=sum(result.total_cost for result in method_results) / count,
            activation_cost=sum(result.activation_cost for result in method_results) / count,
            bandwidth_cost=sum(result.bandwidth_cost for result in method_results) / count,
            forwarding_cost=sum(result.forwarding_cost for result in method_results) / count,
            violation_count=sum(len(result.violations) for result in method_results) / count,
        )
    return averages


def print_average_results(averages: dict[str, AverageResult], config: ExperimentConfig, num_runs: int) -> None:
    print(f"\nAverage result over {num_runs} run(s)")
    print("=" * 72)
    for method_key, result in averages.items():
        method_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        print(f"{method_name} average total cost: {result.total_cost:.2f} ({rounded_cost(result.total_cost)})")
        print("Average cost breakdown:")
        print_cost_line("Activation", result.activation_cost, config.c_act, "server-expert activations")
        print_cost_line("Bandwidth", result.bandwidth_cost, config.c_bw, "Hz")
        print_cost_line("Forwarding", result.forwarding_cost, config.c_fwd, "forwarding events")
        print(f"{method_name} average violations: {result.violation_count:.2f}")
        print("-" * 72)


def run_single_experiment(
    config: ExperimentConfig,
    *,
    run_index: int = 0,
    verbose: bool = True,
    print_details: bool = True,
    show_plot: bool = True,
) -> dict[str, Any]:
    seeds = resolve_run_seeds(config, run_index=run_index)
    graphs = generate_task_graphs(config, seeds, verbose=verbose, export_artifacts=verbose)
    return run_enabled_methods(
        config,
        graphs,
        seeds,
        print_details=print_details,
        show_plot=show_plot,
    )


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    if config.num_runs <= 1:
        return run_single_experiment(config)

    print(f"Running {config.num_runs} independent run(s) for average results.")
    print(f"Each run generates {config.num_dags} DAG(s).")
    run_results: list[dict[str, Any]] = []
    for run_index in range(config.num_runs):
        seeds = resolve_run_seeds(config, run_index=run_index)
        print(
            f"\nRun {run_index + 1}/{config.num_runs}: "
            f"DAG seed={seeds.dag_seed}, network seed={seeds.network_seed}"
        )
        graphs = generate_task_graphs(config, seeds, verbose=False, export_artifacts=False)
        results = run_enabled_methods(
            config,
            graphs,
            seeds,
            print_details=False,
            show_plot=False,
        )
        print_run_summary(run_index + 1, results)
        run_results.append(results)

    averages = average_run_results(run_results)
    print_average_results(averages, config, config.num_runs)
    show_cost_plot(averages, title=f"Average Cost Comparison ({config.num_runs} runs)")
    return averages




