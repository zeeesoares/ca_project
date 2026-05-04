#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            metrics = row["token_bucket_metrics"]
            overhead = row.get("overhead_vs_local") or {}

            rows.append(
                {
                    "benchmark_id": row["benchmark_id"],
                    "timestamp": row["timestamp"],
                    "arch": row["arch"],
                    "target": row["target"],
                    "throughput_mib_per_second": row["throughput_mib_per_second"],
                    "chunk_size_bytes": row["chunk_size_bytes"],
                    "chunk_rate_ratio": row["chunk_rate_ratio"],
                    "actual_time_seconds": metrics["actual_time_seconds"],
                    "expected_time_seconds": metrics["expected_time_seconds"],
                    "actual_throughput_bps": metrics[
                        "actual_effective_throughput_bytes_per_second"
                    ],
                    "expected_throughput_bps": metrics[
                        "expected_effective_throughput_bytes_per_second"
                    ],
                    "throughput_ratio": metrics["throughput_ratio_actual_expected"],
                    "time_ratio": metrics["time_ratio_actual_expected"],
                    "pfs_local_time_ratio": overhead.get("time_ratio_pfs_local"),
                    "pfs_local_throughput_ratio": overhead.get(
                        "throughput_ratio_pfs_local"
                    ),
                    "nr_of_concurrent_jobs": row.get("nr_of_concurrent_jobs"),
                    "dry_run": row.get("dry_run", False),
                }
            )

    return pd.DataFrame(rows)


def latest_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    latest_ids = []

    for arch in sorted(df["arch"].unique()):
        arch_df = df[df["arch"] == arch]
        latest_id = arch_df.sort_values("timestamp")["benchmark_id"].iloc[-1]
        latest_ids.append(latest_id)

    return df[df["benchmark_id"].isin(latest_ids)]


def plot_heatmap(
    *,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output: Path,
):
    pivot = df.pivot_table(
        index="chunk_size_bytes",
        columns="throughput_mib_per_second",
        values=value_col,
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 7))

    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Throughput MiB/s")
    ax.set_ylabel("Chunk size bytes")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_col)

    description = describe_concurrent_jobs(df)

    if description:
        fig.text(
            0.5,
            0.01,
            description.strip(),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output)
    plt.close(fig)


def plot_lines(
    *,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output: Path,
):
    fig, ax = plt.subplots(figsize=(11, 7))

    for chunk_size, group in df.groupby("chunk_size_bytes"):
        group = group.sort_values("throughput_mib_per_second")

        ax.plot(
            group["throughput_mib_per_second"],
            group[value_col],
            marker="o",
            label=f"chunk={chunk_size}",
        )

    ax.set_title(title)
    ax.set_xlabel("Throughput MiB/s")
    ax.set_ylabel(value_col)
    ax.set_xscale("log", base=2)
    ax.legend()

    description = describe_concurrent_jobs(df)

    if description:
        fig.text(
            0.5,
            0.01,
            description.strip(),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output)
    plt.close(fig)


def describe_concurrent_jobs(df: pd.DataFrame) -> str:
    if "nr_of_concurrent_jobs" not in df.columns:
        return ""

    jobs = df["nr_of_concurrent_jobs"].dropna()

    if jobs.empty:
        return ""

    min_jobs = int(jobs.min())
    max_jobs = int(jobs.max())
    mean_jobs = jobs.mean()

    if min_jobs == max_jobs:
        return f"\nConcurrent jobs: {min_jobs}"

    return (
        f"\nConcurrent jobs: min={min_jobs}, "
        f"mean={mean_jobs:.1f}, max={max_jobs}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot token bucket benchmark results."
    )
    parser.add_argument(
        "--results-file",
        required=True,
        type=Path,
        help="Aggregate JSONL results file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where plots will be written",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all benchmarks instead of only latest per architecture",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(args.results_file)

    if df.empty:
        raise RuntimeError("No benchmark data found.")

    if not args.all:
        df = latest_benchmarks(df)

    for arch in sorted(df["arch"].unique()):
        arch_df = df[df["arch"] == arch]

        for target in sorted(arch_df["target"].unique()):
            target_df = arch_df[arch_df["target"] == target]

            prefix = args.output_dir / f"arch_{arch}__target_{target}"

            plot_heatmap(
                df=target_df,
                value_col="throughput_ratio",
                title=f"{arch} {target}: actual/expected throughput ratio",
                output=Path(f"{prefix}__throughput_ratio_heatmap.png"),
            )

            plot_heatmap(
                df=target_df,
                value_col="actual_throughput_bps",
                title=f"{arch} {target}: actual throughput B/s",
                output=Path(f"{prefix}__actual_throughput_heatmap.png"),
            )

            plot_lines(
                df=target_df,
                value_col="throughput_ratio",
                title=f"{arch} {target}: throughput ratio by chunk size",
                output=Path(f"{prefix}__throughput_ratio_lines.png"),
            )

        pfs_df = arch_df[arch_df["target"] == "PFS"]

        if not pfs_df.empty and pfs_df["pfs_local_time_ratio"].notna().any():
            prefix = args.output_dir / f"arch_{arch}__pfs_overhead"

            plot_heatmap(
                df=pfs_df,
                value_col="pfs_local_time_ratio",
                title=f"{arch}: PFS / Local time ratio",
                output=Path(f"{prefix}__time_ratio_heatmap.png"),
            )

            plot_heatmap(
                df=pfs_df,
                value_col="pfs_local_throughput_ratio",
                title=f"{arch}: PFS / Local throughput ratio",
                output=Path(f"{prefix}__throughput_ratio_heatmap.png"),
            )

    print(f"Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
