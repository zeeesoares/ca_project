#!/usr/bin/env python3
"""
Aggregate per-worker JSONL results into a cumulative comparison table.

Reads all *.jsonl files under RESULTS_DIR, groups them by
(mode, policy, n_workers, pfs_bw_bps), and prints:

  - Per-worker mean stall time, mean migration time, mean throughput
  - Aggregate row: mean ± std across workers in the same run group
  - Jain's fairness index on per-worker throughputs within each group

Also saves a flat CSV for further analysis / plotting.

Usage:
    python -m benchmarks.e2e.collect \
        --results-dir /path/to/results \
        --output-csv  /path/to/summary.csv

    # Or process a specific experiment tag:
    python -m benchmarks.e2e.collect \
        --results-dir /path/to/results/my_experiment_tag \
        --output-csv  /path/to/summary.csv
"""

import argparse
import json
import math
import pandas as pd
from pathlib import Path

from benchmarks.e2e.plot_accumulated_bw import plot_accumulated_bandwidth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_all_jsonl(results_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results_dir.rglob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No JSONL data found under {results_dir}")
    return pd.DataFrame(rows)


def jain_fairness(values) -> float:
    xs = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not xs:
        return float("nan")
    n = len(xs)
    return (sum(xs) ** 2) / (n * sum(x ** 2 for x in xs))


def fmt(val, fmt_str=".3f", suffix=""):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  N/A  "
    return f"{val:{fmt_str}}{suffix}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_worker_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per worker_id within each group."""
    return (
        df.groupby(
            ["experiment_tag", "mode", "policy", "n_workers", "pfs_bw_bps", "worker_id"],
            dropna=False,
        )
        .agg(
            mean_stall_s=("stall_s", "mean"),
            std_stall_s=("stall_s", "std"),
            mean_migration_s=("migration_s", "mean"),
            std_migration_s=("migration_s", "std"),
            mean_throughput_mbps=("effective_throughput_bps",
                                  lambda x: x.mean() / (1000 ** 2)),
            std_throughput_mbps=("effective_throughput_bps",
                                 lambda x: x.std() / (1000 ** 2)),
            n_checkpoints=("checkpoint_index", "count"),
            timeout_count=("timed_out", "sum"),
            concurrent_jobs=("concurrent_jobs", "max"),
        )
        .reset_index()
    )


def group_aggregate(pw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across workers within each group."""
    rows = []
    for keys, g in pw.groupby(
        ["experiment_tag", "mode", "policy", "n_workers", "pfs_bw_bps"], dropna=False
    ):
        tag, mode, policy, n_workers, pfs_bw_bps = keys
        tputs = g["mean_throughput_mbps"].tolist()
        rows.append(
            {
                "experiment_tag":   tag,
                "mode":             mode,
                "policy":           policy,
                "n_workers":        n_workers,
                "pfs_bw_mbps":      pfs_bw_bps / (1000 ** 2) if pfs_bw_bps else None,
                "mean_stall_s":     g["mean_stall_s"].mean(),
                "std_stall_s":      g["mean_stall_s"].std(),
                "mean_migration_s": g["mean_migration_s"].mean(),
                "std_migration_s":  g["mean_migration_s"].std(),
                "mean_throughput_mbps": g["mean_throughput_mbps"].mean(),
                "std_throughput_mbps":  g["mean_throughput_mbps"].std(),
                "jain_fairness":    jain_fairness(tputs),
                "total_timeout_count": g["timeout_count"].sum(),
                "concurrent_jobs":   g["concurrent_jobs"].max(),
            }
        )
    return pd.DataFrame(rows)


def generate_accumulated_bw_plots(
    df: pd.DataFrame,
    results_dir: Path,
    plots_dir: Path,
) -> None:
    """
    Generate one accumulated bandwidth graph per experiment_tag.

    Expects migrater logs under:
        <results-dir>/<experiment_tag>/migrater_logs/migrater_*.log

    If --results-dir already points to one experiment directory, also checks:
        <results-dir>/migrater_logs/migrater_*.log
    """

    for experiment_tag, group in df.groupby("experiment_tag", dropna=False):
        if experiment_tag is None or pd.isna(experiment_tag):
            continue

        experiment_tag = str(experiment_tag)

        candidate_dirs = [
            results_dir / experiment_tag / "migrater_logs",
            results_dir / "migrater_logs",
        ]

        log_paths = []
        for log_dir in candidate_dirs:
            if log_dir.exists():
                log_paths = sorted(log_dir.glob("migrater_*.log"))
                if log_paths:
                    break

        if not log_paths:
            print(f"No migrater logs found for {experiment_tag}; skipping accumulated graph")
            continue

        bw_values = pd.to_numeric(group["pfs_bw_bps"], errors="coerce").dropna().unique()
        if len(bw_values) == 0:
            print(f"No pfs_bw_bps found for {experiment_tag}; skipping accumulated graph")
            continue

        orch_bw = float(bw_values[0])

        output_path = plots_dir / experiment_tag / "accumulated_bandwidth.png"

        print(f"Generating accumulated bandwidth graph for {experiment_tag}")
        print(f"  Logs: {len(log_paths)}")
        print(f"  Output: {output_path}")

        plot_accumulated_bandwidth(
            log_paths=log_paths,
            orch_bw=orch_bw,
            output_path=output_path,
        )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

COL_W = {
    "mode":        14,
    "policy":      24,
    "n_workers":    9,
    "pfs_bw":       9,
    "stall":       18,
    "migration":   18,
    "throughput":  18,
    "fairness":    10,
    "concurrent_jobs": 18,
}

def _hdr():
    return (
        f"{'Mode':<{COL_W['mode']}}"
        f"{'Policy':<{COL_W['policy']}}"
        f"{'#Workers':>{COL_W['n_workers']}}"
        f"{'PFS BW':>{COL_W['pfs_bw']}}"
        f"{'Stall (s) ±σ':>{COL_W['stall']}}"
        f"{'Migration (s) ±σ':>{COL_W['migration']}}"
        f"{'Tput (MB/s) ±σ':>{COL_W['throughput']}}"
        f"{'Jain':>{COL_W['fairness']}}"
        f"{'Concurrent Jobs':>{COL_W['concurrent_jobs']}}"
    )


def _row(r):
    # Use r["col"] to avoid conflicts with pandas Series methods (e.g. .mode()).
    row_mode    = r["mode"]
    row_policy  = str(r["policy"])
    row_nw      = int(r["n_workers"])
    row_bw      = r["pfs_bw_mbps"]
    row_stall_m = r["mean_stall_s"]
    row_stall_s = r["std_stall_s"]
    row_mig_m   = r["mean_migration_s"]
    row_mig_s   = r["std_migration_s"]
    row_tput_m  = r["mean_throughput_mbps"]
    row_tput_s  = r["std_throughput_mbps"]
    row_jain    = r["jain_fairness"]

    pfs_bw = f"{row_bw:.0f} MB/s" if row_bw and not math.isnan(float(row_bw)) else "N/A"
    stall  = f"{row_stall_m:.3f} ±{row_stall_s if not math.isnan(float(row_stall_s or 0)) else 0:.3f}"
    mig    = f"{row_mig_m:.2f} ±{row_mig_s if not math.isnan(float(row_mig_s or 0)) else 0:.2f}"
    tput   = f"{row_tput_m:.1f} ±{row_tput_s if not math.isnan(float(row_tput_s or 0)) else 0:.1f}"
    jain   = f"{row_jain:.4f}" if not math.isnan(float(row_jain)) else "  N/A"
    return (
        f"{row_mode:<{COL_W['mode']}}"
        f"{row_policy:<{COL_W['policy']}}"
        f"{row_nw:>{COL_W['n_workers']}}"
        f"{pfs_bw:>{COL_W['pfs_bw']}}"
        f"{stall:>{COL_W['stall']}}"
        f"{mig:>{COL_W['migration']}}"
        f"{tput:>{COL_W['throughput']}}"
        f"{jain:>{COL_W['fairness']}}"
        f"{int(r['concurrent_jobs']):>{COL_W['concurrent_jobs']}}"
    )


def print_table(agg: pd.DataFrame, pw: pd.DataFrame):
    sep = "-" * (sum(COL_W.values()) + 4)

    print()
    print("=" * (sum(COL_W.values()) + 4))
    print(" CUMULATIVE RESULTS — Baseline vs Orchestrated")
    print("=" * (sum(COL_W.values()) + 4))
    print()

    # Sort: baseline first, then by (policy, n_workers)
    agg_sorted = agg.sort_values(
        ["experiment_tag", "mode", "policy", "n_workers"],
        na_position="first",
        key=lambda col: col.fillna("").astype(str),
    )

    current_n = None
    print(_hdr())
    print(sep)

    for _, r in agg_sorted.iterrows():
        if r["n_workers"] != current_n:
            if current_n is not None:
                print(sep)
            current_n = r["n_workers"]
        print(_row(r))

    print(sep)
    print()

    # Per-worker detail
    print("PER-WORKER DETAIL")
    print("-" * 80)
    pw_sorted = pw.sort_values(
        ["mode", "policy", "n_workers", "worker_id"],
        na_position="first",
        key=lambda col: col.fillna("").astype(str),
    )

    for _, r in pw_sorted.iterrows():
        bps = r["pfs_bw_bps"]
        pfs = f"{float(bps) / (1024**2):.0f} MB/s" if bps and not math.isnan(float(bps)) else "N/A"
        timeouts = int(r["timeout_count"]) if r["timeout_count"] else 0
        print(
            f"  {r['mode']:<14} {str(r['policy']):<24} n={int(r['n_workers'])}  "
            f"bw={pfs:>10}  {r['worker_id']:<12}  "
            f"stall={r['mean_stall_s']:.3f}s  "
            f"migr={r['mean_migration_s']:.2f}s  "
            f"tput={r['mean_throughput_mbps']:.1f} MB/s"
            + (f"  [TIMEOUTS: {timeouts}]" if timeouts > 0 else "")
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results into a cumulative comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory to scan for *.jsonl result files (recursive)",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Optional CSV path for the aggregate table",
    )
    parser.add_argument(
        "--detail-csv", type=Path, default=None,
        help="Optional CSV path for per-worker detail",
    )
    parser.add_argument(
        "--accumulated-bw-plots",
        action="store_true",
        help="Generate accumulated bandwidth plots from migrater logs",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory where accumulated bandwidth plots are written",
    )
    args = parser.parse_args()

    df = read_all_jsonl(args.results_dir)
    print(f"Loaded {len(df)} checkpoint records from {args.results_dir}")

    for col in ["stall_s", "migration_s", "effective_throughput_bps",
                "n_workers", "pfs_bw_bps", "concurrent_jobs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pw  = per_worker_stats(df)
    agg = group_aggregate(pw)

    print_table(agg, pw)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.output_csv, index=False)
        print(f"Aggregate CSV saved: {args.output_csv}")

    if args.detail_csv:
        args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
        pw.to_csv(args.detail_csv, index=False)
        print(f"Per-worker CSV saved: {args.detail_csv}")

    if args.accumulated_bw_plots:
        plots_dir = args.plots_dir or (args.results_dir / "plots")
        generate_accumulated_bw_plots(
            df=df,
            results_dir=args.results_dir,
            plots_dir=plots_dir,
        )


if __name__ == "__main__":
    main()
