#!/usr/bin/env python3
"""
Visual tables and comparison charts for the e2e benchmark results.

Produces:
  1. summary_table.png       — styled comparison table (all groups)
  2. stall_comparison.png    — bar chart: stall time baseline vs orchestrated
  3. migration_throughput.png — bar chart: migration time + throughput per policy
  4. fairness.png            — Jain fairness index per policy

Usage:
    python -m benchmarks.e2e.plot \
        --results-dir /path/to/results \
        --output-dir  /path/to/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from benchmarks.e2e.collect import COL_W, read_all_jsonl, per_worker_stats, group_aggregate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _nan(v):
    import pandas as pd
    if v is None:
        return True
    if isinstance(v, str):
        return v.lower() in ("nan", "none", "")
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(agg: pd.DataFrame, pw: pd.DataFrame, output: Path):
    """
    Render a matplotlib table with one row per (mode, policy, n_workers).
    Cells are color-coded: stall time green=low/red=high, fairness green=high.
    """
    agg = agg.sort_values(
        ["experiment_tag", "n_workers", "mode", "policy"],
        key=lambda c: c.fillna("zzz").astype(str),
    ).reset_index(drop=True)

    col_labels = [
        "Tag","Mode", "Policy", "#W", "PFS BW",
        "Stall (s)", "σ stall", "Migration (s)", "σ migr",
        "Tput (MB/s)", "σ tput", "Jobs",
    ]

    rows = []
    for _, r in agg.iterrows():
        bw  = f"{r['pfs_bw_mbps']:.0f}" if not _nan(r["pfs_bw_mbps"]) else "N/A"
        rows.append([
            r["experiment_tag"],
            r["mode"],
            str(r["policy"]) if not _nan(r["policy"]) else "—",
            str(int(r["n_workers"])),
            bw,
            f"{r['mean_stall_s']:.3f}",
            f"{r['std_stall_s']:.3f}" if not _nan(r["std_stall_s"]) else "—",
            f"{r['mean_migration_s']:.2f}",
            f"{r['std_migration_s']:.2f}" if not _nan(r["std_migration_s"]) else "—",
            f"{r['mean_throughput_mbps']:.1f}",
            f"{r['std_throughput_mbps']:.1f}" if not _nan(r["std_throughput_mbps"]) else "—",
            f"{r['concurrent_jobs']:>{COL_W['concurrent_jobs']}}"
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)

    fig_h = max(2.0, 0.45 * n_rows + 1.2)
    fig_w = 16
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))

    # Color headers
    header_color = "#2c3e50"
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Color data rows
    stall_vals = [
        float(r[5]) for r in rows if r[5] not in ("—", "N/A")
    ]
    stall_max = max(stall_vals) if stall_vals else 1.0

    for i, (row_data, (_, agg_row)) in enumerate(zip(rows, agg.iterrows())):
        ri = i + 1  
        is_baseline = agg_row["mode"] == "baseline"
        
        # 1. Cor de fundo alternada para a linha toda
        base_bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        for j in range(n_cols):
            tbl[ri, j].set_facecolor(base_bg)

        # 2. Colorir a coluna MODE (Índice 1)
        # Rosa para baseline, Verde para orchestrated
        tbl[ri, 1].set_facecolor("#fde8e8" if is_baseline else "#e8f5e9")

        # 3. Colorir a coluna STALL (Índice 5) com gradiente
        try:
            stall = float(row_data[5]) # Mudado de 4 para 5
            t = stall / stall_max if stall_max > 0 else 0
            # Red-Green gradient
            r_c = 0.9 * t + 0.6 * (1 - t)
            g_c = 0.6 * t + 0.9 * (1 - t)
            tbl[ri, 5].set_facecolor((r_c, g_c, 0.6))
        except ValueError:
            pass
            
        # 4. Colorir a coluna PFS BW (Índice 4) 
        # Apenas para dar um destaque visual se for N/A
        if row_data[4] == "N/A":
            tbl[ri, 4].set_facecolor("#f2f2f2")

    fig.suptitle(
        "Benchmark Summary — Baseline vs Orchestrated",
        fontsize=12, fontweight="bold", y=0.98,
    )
    save(fig, output)


# ---------------------------------------------------------------------------
# 2. Stall time comparison
# ---------------------------------------------------------------------------

def plot_stall_comparison(agg: pd.DataFrame, output: Path):
    """
    Grouped bar: stall time (mean ± std) for baseline vs each orchestrated policy.
    One group per n_workers value.
    """
    n_worker_vals = sorted(agg["n_workers"].unique())
    policies = (
        ["baseline"]
        + sorted(agg[agg["mode"] == "orchestrated"]["policy"]
                 .dropna().unique().tolist())
    )

    x = np.arange(len(n_worker_vals))
    width = 0.7 / len(policies)
    cmap  = plt.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(max(8, len(n_worker_vals) * 2.5), 6))

    for i, pol in enumerate(policies):
        means, errs = [], []
        for nw in n_worker_vals:
            if pol == "baseline":
                sub = agg[(agg["mode"] == "baseline") & (agg["n_workers"] == nw)]
            else:
                sub = agg[
                    (agg["mode"] == "orchestrated")
                    & (agg["policy"] == pol)
                    & (agg["n_workers"] == nw)
                ]
            if sub.empty:
                means.append(0); errs.append(0)
            else:
                means.append(float(sub["mean_stall_s"].iloc[0]))
                errs.append(float(sub["std_stall_s"].fillna(0).iloc[0]))

        offset = (i - (len(policies) - 1) / 2) * width
        ax.bar(
            x + offset, means, width * 0.9,
            yerr=errs, capsize=4,
            label=pol, color=cmap(i % 10),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={nw}" for nw in n_worker_vals])
    ax.set_ylabel("Mean stall time (s)  [lower is better]")
    ax.set_title("Training Stall Time: Baseline vs Orchestrated")
    ax.legend(title="Policy / Mode", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    save(fig, output)


# ---------------------------------------------------------------------------
# 3. Migration time + throughput (side by side)
# ---------------------------------------------------------------------------

def plot_migration_throughput(agg: pd.DataFrame, output: Path):
    policies = (
        ["baseline"]
        + sorted(agg[agg["mode"] == "orchestrated"]["policy"]
                 .dropna().unique().tolist())
    )
    n_worker_vals = sorted(agg["n_workers"].unique())
    cmap = plt.colormaps["tab10"]

    fig, (ax_mig, ax_tput) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (ax_mig,  "mean_migration_s",      "Mean migration time (s)",   "Migration Time"),
        (ax_tput, "mean_throughput_mbps",   "Mean throughput (MB/s)",    "Effective Throughput"),
    ]:
        x     = np.arange(len(n_worker_vals))
        width = 0.7 / len(policies)
        std_col = metric.replace("mean_", "std_")

        for i, pol in enumerate(policies):
            means, errs = [], []
            for nw in n_worker_vals:
                if pol == "baseline":
                    sub = agg[(agg["mode"] == "baseline") & (agg["n_workers"] == nw)]
                else:
                    sub = agg[
                        (agg["mode"] == "orchestrated")
                        & (agg["policy"] == pol)
                        & (agg["n_workers"] == nw)
                    ]
                if sub.empty:
                    means.append(0); errs.append(0)
                else:
                    means.append(float(sub[metric].iloc[0]))
                    e = sub[std_col].fillna(0).iloc[0]
                    errs.append(float(e))

            offset = (i - (len(policies) - 1) / 2) * width
            ax.bar(x + offset, means, width * 0.9,
                   yerr=errs, capsize=3,
                   label=pol, color=cmap(i % 10))

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={nw}" for nw in n_worker_vals])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    save(fig, output)

def plot_workload_throughput_comparison(agg: pd.DataFrame, output: Path):
    """
    Gera um gráfico científico composto por 3 subplots (Burst, Steady, Dynamic Churn).
    Garante que a Baseline aparece sempre em cada cenário como termo de comparação.
    """
    # 1. Mapeamento e limpeza das tags reais do teu log
    def clean_tag(t):
        t = str(t).lower()
        if "burst" in t: return "Burst"
        if "steady" in t: return "Steady"
        if "dynamic" in t: return "Dynamic Churn"
        return "Baseline"

    agg = agg.copy()
    agg["workload_group"] = agg["experiment_tag"].apply(clean_tag)
    
    # Isolar a linha da baseline
    baseline_row = agg[agg["mode"] == "baseline"]
    if baseline_row.empty:
        print("Aviso: Dados da Baseline não encontrados.")
        return

    # Isolar políticas orquestradas
    orch_data = agg[agg["mode"] == "orchestrated"]
    policies = sorted(orch_data["policy"].dropna().unique().tolist())
    
    workload_groups = ["Burst", "Steady", "Dynamic Churn"]
    
    # Criar a janela com 3 gráficos lado a lado (1 linha, 3 colunas)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Cores sóbrias e com alto contraste para a Baseline
    colors = {
        "Baseline": "#dc2626",           # Vermelho Vivo
        "active-fair-share": "#2563eb",  # Azul
        "age-priority": "#16a34a",       # Verde
        "epoch-priority": "#ea580c",     # Laranja
        "uniform-fair-share": "#7c3aed"  # Roxo
    }

    for ax, group in zip(axes, workload_groups):
        group_orch = orch_data[orch_data["workload_group"] == group]
        
        # A Baseline entra SEMPRE à esquerda em cada um dos 3 gráficos
        display_items = [("Baseline", float(baseline_row["mean_throughput_mbps"].iloc[0]), 
                          float(baseline_row["std_throughput_mbps"].fillna(0).iloc[0]))]
        
        for pol in policies:
            sub = group_orch[group_orch["policy"] == pol]
            if not sub.empty:
                display_items.append((
                    pol, 
                    float(sub["mean_throughput_mbps"].iloc[0]), 
                    float(sub["std_throughput_mbps"].fillna(0).iloc[0])
                ))
            else:
                display_items.append((pol, 0.0, 0.0))

        names = [item[0] for item in display_items]
        means = [item[1] for item in display_items]
        errs  = [item[2] for item in display_items]
        
        x_pos = np.arange(len(names))
        bar_colors = [colors.get(n, "#64748b") for n in names]
        
        # Desenhar as barras
        bars = ax.bar(x_pos, means, yerr=errs, capsize=4, color=bar_colors, edgecolor="#1e293b", width=0.6)
        
        # Desenhar os valores numéricos por cima de cada barra para análise imediata
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Estética do Subplot
        ax.set_title(f"Workload: {group}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        # Quebra as strings longas com um '\n' para não se sobreporem no eixo X
        ax.set_xticklabels([n.replace("-", "\n").title() for n in names], fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Label do eixo Y comum a todos (apenas no primeiro painel da esquerda)
    axes[0].set_ylabel("Effective Throughput (MB/s)", fontsize=11, fontweight="bold")
    
    # Criar uma legenda única global centralizada no topo da figura
    legend_patches = [mpatches.Patch(color=color, label=name.replace("-", " ").title()) 
                      for name, color in colors.items() if name == "Baseline" or name in policies]
    
    fig.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, 1.08), 
               ncol=len(legend_patches), frameon=True, fontsize=10)

    fig.tight_layout()
    save(fig, output)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual tables and charts for e2e benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir",  required=True, type=Path)
    args = parser.parse_args()

    df  = read_all_jsonl(args.results_dir)
    for col in ["stall_s", "migration_s", "effective_throughput_bps",
                "n_workers", "pfs_bw_bps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pw  = per_worker_stats(df)
    agg = group_aggregate(pw)

    print(f"Loaded {len(df)} records → {len(agg)} groups")
    print(f"Writing plots to {args.output_dir}/")

    out = args.output_dir
    plot_summary_table(agg, pw, out / "summary_table.png")
    plot_stall_comparison(agg, out / "stall_comparison.png")
    plot_workload_throughput_comparison(agg, out / "workload_throughput_comparison.png")
    plot_migration_throughput(agg, out / "migration_throughput.png")
    print("Done.")


if __name__ == "__main__":
    main()
