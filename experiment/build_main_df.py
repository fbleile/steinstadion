import argparse
import os
import pandas as pd
from pathlib import Path
from definitions import DEFAULT_DATA_GEN_TYPES

import matplotlib.pyplot as plt
import seaborn as sns
from plot_config import METRICS_NAMING, METHODS_CONFIG

def find_latest_summary_dir(base_path):
    summary_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("summary_")]
    if not summary_dirs:
        return None
    return sorted(summary_dirs)[-1]  # alphabetical highest

def find_summary_csv(summary_dir, gen_type):
    for file in summary_dir.glob(f"df-{gen_type}-summary_*.csv"):
        return file
    return None

def load_all_summaries(results_dir, args):
    all_dfs = []
    for gen_type in args.only_gen_types:
        gen_path = results_dir / gen_type
        if not gen_path.exists():
            print(f"Skipping {gen_type} (folder does not exist)")
            continue

        latest_summary = find_latest_summary_dir(gen_path)
        if not latest_summary:
            print(f"No summary directory found in {gen_type}")
            continue

        csv_path = find_summary_csv(latest_summary / args.mode, gen_type)
        if not csv_path:
            print(f"No summary CSV found in {latest_summary}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df["gen_type"] = gen_type
            all_dfs.append(df)
            print(f"Loaded summary from {csv_path}")
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
    
    if not all_dfs:
        print("No valid summary files found.")
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)

def plot_relative_metric(df: pd.DataFrame, metric: str, benchmark_method: str):
    assert metric in METRICS_NAMING, f"Unknown metric '{metric}'. Available: {list(METRICS_NAMING.keys())}"
    assert 'val' in df.columns, "DataFrame must contain a 'val' column."
    assert 'method' in df.columns, "DataFrame must contain a 'method' column."
    assert 'metric' in df.columns, "DataFrame must contain a 'metric' column."
    assert 'gen_type' in df.columns, "DataFrame must contain a 'gen_type' column."
    assert 'data_idx' in df.columns, "DataFrame must contain a 'data_idx' column."
    assert 'env_idx' in df.columns, "DataFrame must contain a 'env_idx' column."
    
    # Filter for the metric
    metric_df = df[df['metric'] == metric].copy()
    
    dups = metric_df[metric_df.duplicated(subset=['gen_type', 'metric', 'data_idx', 'env_idx', 'method'], keep=False)]
    
    # If there are any, print the full dataframe and raise an error
    if not dups.empty:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 0):
            print("Duplicate rows found for (gen_type, metric, data_idx, env_idx, method):")
            print(dups)
        raise AssertionError("DataFrame rows must be unique per (gen_type, metric, data_idx, env_idx, method)")
    
    assert dups.empty, f"Duplicate rows found for (gen_type, metric, data_idx, env_idx, method):\n{dups.head()}"
    


    # Create a multi-index for easier alignment
    keys = ['gen_type', 'data_idx', 'env_idx']
    benchmark_df = metric_df[metric_df['method'] == benchmark_method][keys + ['val']]
    benchmark_df = benchmark_df.rename(columns={'val': 'val_benchmark'})

    # Merge benchmark values into full df
    metric_df = metric_df.merge(benchmark_df, on=keys, how='left')

    # Compute relative performance
    rel_col = f"{metric}_rel"
    metric_df[rel_col] = metric_df['val'] / metric_df['val_benchmark']
    metric_df = metric_df.dropna(subset=[rel_col])

    # Mapping for display names and colors
    method_config_map = {cfg[0]: {"color": cfg[1], "display_name": cfg[3]} for cfg in METHODS_CONFIG}
    metric_df["display_name"] = metric_df["method"].map(lambda m: method_config_map.get(m, {}).get("display_name", m))
    metric_df["color"] = metric_df["method"].map(lambda m: method_config_map.get(m, {}).get("color", "#777777"))

    # Plot per gen_type
    gen_types = metric_df["gen_type"].unique()
    for gen_type in gen_types:
        sub_df = metric_df[metric_df["gen_type"] == gen_type]

        plt.figure(figsize=(10, 5))
        ax = sns.boxplot(
            data=sub_df,
            x="display_name",
            y=rel_col,
            palette=sub_df.set_index("display_name")["color"].to_dict()
        )
        ax.set_title(f"{METRICS_NAMING[metric]} (relative to {benchmark_method}) for {gen_type}")
        ax.set_ylabel(f"{METRICS_NAMING[metric]} (relative)")
        ax.set_xlabel("Method")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Aggregate and plot summary CSVs.")
    
    parser.add_argument(
        "--only_gen_types",
        nargs="+",
        choices=DEFAULT_DATA_GEN_TYPES,
        default=DEFAULT_DATA_GEN_TYPES,
        help="Subset of data generation types to include."
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "median"],
        default="mean",
        help="Specify the aggregation mode to use (mean or median)."
    )
    args = parser.parse_args()

    results_dir = Path("results")
    master_df = load_all_summaries(results_dir, args)
    
    print(master_df)

    if not master_df.empty:
        plot_relative_metric(master_df, 'wasser_test', 'kds-linear_u_diag', )
        # kds-linear_u_diag ,  , 
