from pathlib import Path

from generate_dags import DAGSpec, build_dag_specs, generate_multiple_dags


# ============================================================
# Main settings
# Change parameters here, then run:
#     python main.py
# ============================================================

NUM_DAGS = 5
RANDOM_SEED = 42
OUTPUT_DIR = Path("generated_dags")
IMAGE_DPI = 180

COMMON_DAG_SPEC = DAGSpec(
    num_nodes=15,
    depth=8,
    max_width=10,
    edge_probability=0.2,
    allow_skip_edges=False,
    allow_early_branch_end=True,
    single_source=True,
    single_sink=False,
)

NUM_EXPERTS = 8
TASK_DEADLINE_SECONDS_RANGE = (30, 120)
NUM_IOT_FEATURES = 6
IOT_FEATURES_PER_NODE_RANGE = (1, 3)


def main() -> None:
    specs = build_dag_specs(NUM_DAGS, COMMON_DAG_SPEC)
    graphs = generate_multiple_dags(
        specs=specs,
        output_dir=OUTPUT_DIR,
        seed=RANDOM_SEED,
        image_dpi=IMAGE_DPI,
        num_experts=NUM_EXPERTS,
        num_iot_features=NUM_IOT_FEATURES,
        iot_features_per_node_range=IOT_FEATURES_PER_NODE_RANGE,
        task_deadline_seconds_range=TASK_DEADLINE_SECONDS_RANGE,
    )

    print(f"\nDone: generated {len(graphs)} DAG(s).")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
