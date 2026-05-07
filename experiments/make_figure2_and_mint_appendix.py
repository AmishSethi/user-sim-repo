"""Regenerate Figure 2 summary panels and benchmark spread figures.

Inputs:
    tau2_bench_data/figure2_data.json

Outputs:
    figures/fig2.{pdf,svg,png}
    src/figures/fig2.{pdf,svg}
    figures/*_agent_spread.{pdf,svg,png}
    src/figures/*_agent_spread.{pdf,svg}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tau2_bench_data" / "figure2_data.json"
FIG = ROOT / "figures"
SRC_FIG = ROOT / "src" / "figures"

PANEL = [
    "llama3.1-8b",
    "qwen2.5-7b",
    "gemma3-12b",
    "llama3.3-70b",
    "gpt-oss-120b",
    "deepseek-r1",
]

FIXED_AGENT_ORDER = [
    "gpt-oss-120b",
    "deepseek-r1",
    "gemma3-12b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "llama3.3-70b",
]

MODEL_LABEL = {
    "llama3.1-8b": "Llama-3.1-8B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "gemma3-12b": "Gemma-3-12B",
    "llama3.3-70b": "Llama-3.3-70B",
    "gpt-oss-120b": "gpt-oss-120B",
    "deepseek-r1": "DeepSeek-R1",
}

WRAPPED_LABEL = {
    "llama3.1-8b": "Llama-3.1-\n8B",
    "qwen2.5-7b": "Qwen2.5-\n7B",
    "gemma3-12b": "Gemma-3-\n12B",
    "llama3.3-70b": "Llama-3.3-\n70B",
    "gpt-oss-120b": "gpt-oss-\n120B",
    "deepseek-r1": "DeepSeek-\nR1",
}

BENCH_ORDER = [
    "tau2-bench telecom",
    "tau-bench airline",
    "MINT",
    "24-task customer-service",
    "AgentClinic-MedQA",
]
BENCH_LABEL = {
    "MINT": "MINT",
    "24-task customer-service": "24-task customer\nservice",
    "AgentClinic-MedQA": "AgentClinic",
    "tau-bench airline": r"$\tau$-bench airline",
    "tau2-bench telecom": r"$\tau^2$-bench telecom",
}

SPREAD_OUTPUTS = {
    "MINT": "mint_agent_spread",
    "24-task customer-service": "customer_service_agent_spread",
    "AgentClinic-MedQA": "agentclinic_agent_spread",
    "tau-bench airline": "taubench_airline_agent_spread",
    "tau2-bench telecom": "tau2_agent_spread",
}

SIM_COLORS = {
    "llama3.1-8b": "#7D8FA8",
    "qwen2.5-7b": "#E38624",
    "gemma3-12b": "#6AAE22",
    "llama3.3-70b": "#168A80",
    "gpt-oss-120b": "#6647D9",
    "deepseek-r1": "#C92A2A",
}

TEAL = "#129389"
GREY = "#9CA5B1"
GRID = "#D8DEE6"
THRESHOLD_RED = "#C92A2A"
TEXT = "#111827"
MUTED = "#6B7280"

BENCH_COLORS = {
    "tau2-bench telecom": "#6D4CC2",
    "tau-bench airline": "#D97706",
    "MINT": TEAL,
    "24-task customer-service": "#5278A7",
    "AgentClinic-MedQA": "#6AAE22",
}


def configure_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.5,
            "axes.titlesize": 8.8,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> dict:
    data = json.loads(DATA.read_text())
    if data["panel"] != PANEL:
        raise ValueError(f"Unexpected panel order in {DATA}: {data['panel']}")
    return data


def load_pivot(data: dict, benchmark: str) -> pd.DataFrame:
    """Return rows = simulated users, columns = fixed agents."""
    bench = data["benchmarks"][benchmark]
    if "success_matrix" in bench:
        rows = {
            row["Agent"]: row["Simulated user"]
            for row in bench["success_matrix"]
        }
        return pd.DataFrame(rows).loc[PANEL, PANEL].astype(float)
    matrix = bench["success_rate_pct"]
    return pd.DataFrame(matrix).T.loc[PANEL, PANEL].astype(float)


def worst_pair_kendall(pivot: pd.DataFrame) -> float:
    taus: list[float] = []
    for i, sim_a in enumerate(PANEL):
        rank_a = pivot.loc[sim_a].rank(ascending=False).to_numpy(dtype=float)
        for sim_b in PANEL[i + 1 :]:
            rank_b = pivot.loc[sim_b].rank(ascending=False).to_numpy(dtype=float)
            tau, _ = kendalltau(rank_a, rank_b)
            if tau == tau:
                taus.append(float(tau))
    if not taus:
        raise ValueError("No non-degenerate Kendall tau pairs found")
    return min(taus)


def summary_stats(data: dict) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for bench in BENCH_ORDER:
        bench_data = data["benchmarks"][bench]
        stats[bench] = {
            "max_swing": float(bench_data["max_same_agent_swing_pp"]),
            "worst_tau": float(bench_data["worst_pair_kendall_tau"]),
            "snr": float(bench_data["t1_variance"]["mean_snr"]),
        }
    return stats


def style_metric_axis(ax: plt.Axes, show_ylabels: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D1DB")
    ax.spines["bottom"].set_color("#C9D1DB")
    ax.tick_params(axis="y", length=0, pad=5)
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="x", color=GRID, linewidth=0.85)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def draw_metric_panel(
    ax: plt.Axes,
    title: str,
    values: list[float],
    threshold: float,
    xlim: tuple[float, float],
    ticks: list[float],
    tick_fmt,
    value_fmt,
    threshold_label: str,
    show_ylabels: bool,
) -> None:
    y = np.arange(len(BENCH_ORDER))
    colors = [BENCH_COLORS[bench] for bench in BENCH_ORDER]
    for yi, val, color in zip(y, values, colors):
        ax.hlines(yi, xlim[0], val, color=color, linewidth=2.2, zorder=2)
        ax.scatter(val, yi, s=36, color=color, edgecolor="white", linewidth=0.8, zorder=3)

    ax.axvline(threshold, color=THRESHOLD_RED, linestyle=(0, (2.2, 2.2)), linewidth=1.05, zorder=1)
    ax.text(
        threshold,
        1.025,
        threshold_label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.7,
        color=THRESHOLD_RED,
        clip_on=False,
    )

    ax.set_yticks(y, [BENCH_LABEL[b] for b in BENCH_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks, [tick_fmt(t) for t in ticks])
    ax.set_title(title, loc="left", pad=20, fontweight="normal")
    style_metric_axis(ax, show_ylabels)

    span = xlim[1] - xlim[0]
    for yi, val in enumerate(values):
        label_x = min(val + 0.025 * span, xlim[1] - 0.02 * span)
        ax.text(
            label_x,
            yi,
            value_fmt(val),
            ha="left",
            va="center",
            fontsize=7.8,
            color=TEXT,
            zorder=4,
        )


def render_fig2(data: dict) -> None:
    configure_style()
    stats = summary_stats(data)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.46), gridspec_kw={"wspace": 0.34})

    draw_metric_panel(
        axes[0],
        "(a) T2 score sensitivity",
        [stats[b]["max_swing"] for b in BENCH_ORDER],
        10.0,
        (0.0, 100.0),
        [0, 20, 40, 60, 80, 100],
        lambda x: f"{x:.0f}",
        lambda x: f"{x:.0f} pp",
        "10 pp ref. (lower better)",
        True,
    )
    axes[0].set_xlabel("Maximum fixed-agent range\n(percentage points)", labelpad=7)

    draw_metric_panel(
        axes[1],
        r"(b) T2 ranking stability",
        [stats[b]["worst_tau"] for b in BENCH_ORDER],
        0.90,
        (0.0, 1.0),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lambda x: f"{x:.2f}",
        lambda x: f"{x:.2f}",
        r"$\tau_K=0.90$ ref. (higher better)",
        False,
    )
    axes[1].set_xlabel(r"Worst-pair Kendall $\tau_K$" "\n(ranking agreement)", labelpad=7)

    draw_metric_panel(
        axes[2],
        "(c) T1 rerun-noise check",
        [stats[b]["snr"] for b in BENCH_ORDER],
        3.0,
        (0.0, 4.0),
        [0, 1, 2, 3, 4],
        lambda x: f"{x:.1f}x",
        lambda x: f"{x:.1f}x",
        r"$3\times$ ref. (higher better)",
        False,
    )
    axes[2].set_xlabel("Cross-simulator SD / rerun SD\n(signal-to-noise ratio)", labelpad=7)

    save_all(fig, FIG / "fig2")
    copy_to_src("fig2")


def order_agents_by_swing(pivot: pd.DataFrame) -> list[str]:
    ranges = pivot.max(axis=0) - pivot.min(axis=0)
    highs = pivot.max(axis=0)
    means = pivot.mean(axis=0)
    return (
        pd.DataFrame({"range": ranges, "high": highs, "mean": means})
        .sort_values(["range", "high", "mean"], ascending=[False, False, False])
        .index
        .tolist()
    )


def pp_label(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f} pp"
    return f"{value:.1f} pp"


def render_benchmark_spread(data: dict, benchmark: str) -> None:
    configure_style()
    pivot = load_pivot(data, benchmark)
    agent_order = FIXED_AGENT_ORDER
    x = np.arange(len(agent_order))
    jitter = np.linspace(-0.18, 0.18, len(PANEL))

    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    band_color = "#BFEFE6"
    edge_color = "#146C68"

    for i, agent in enumerate(agent_order):
        values = pivot[agent]
        low = float(values.min())
        high = float(values.max())
        swing = high - low

        ax.add_patch(
            plt.Rectangle(
                (i - 0.32, low),
                0.64,
                max(swing, 1.0),
                facecolor=band_color,
                edgecolor="none",
                alpha=0.52,
                zorder=1,
            )
        )
        ax.hlines([low, high], i - 0.32, i + 0.32, color=edge_color, linewidth=1.15, zorder=2)
        ax.text(
            i,
            high + 2.2,
            pp_label(swing),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="normal",
            color="#111827",
            zorder=5,
        )
        for j, sim in enumerate(PANEL):
            ax.scatter(
                i + jitter[j],
                float(values.loc[sim]),
                s=44,
                color=SIM_COLORS[sim],
                edgecolor="white",
                linewidth=0.85,
                zorder=4,
            )

    ax.set_ylabel("success (%)", color="#374151")
    ax.set_xlabel("Fixed agent under test", labelpad=8)
    y_min = 20 if float(pivot.min().min()) >= 20 else 0
    ax.set_ylim(y_min - 3, 105)
    ax.set_yticks(list(range(y_min, 101, 20)))
    ax.set_xticks(x, [WRAPPED_LABEL[a] for a in agent_order])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors="#374151")
    ax.grid(axis="y", color=GRID, linewidth=0.82)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D1DB")
    ax.spines["bottom"].set_color("#C9D1DB")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=SIM_COLORS[sim],
            markeredgecolor="white",
            markeredgewidth=0.85,
            markersize=6.3,
            label=MODEL_LABEL[sim],
        )
        for sim in PANEL
    ]
    ax.legend(
        handles=handles,
        title="Simulated user",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        title_fontsize=8.2,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=1.05,
    )

    stem = SPREAD_OUTPUTS[benchmark]
    save_all(fig, FIG / stem)
    copy_to_src(stem)


def save_all(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def copy_to_src(stem: str) -> None:
    SRC_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        shutil.copyfile(FIG / f"{stem}{suffix}", SRC_FIG / f"{stem}{suffix}")


def main() -> None:
    data = load_data()
    render_fig2(data)
    for benchmark in SPREAD_OUTPUTS:
        render_benchmark_spread(data, benchmark)
    print(f"wrote {FIG / 'fig2.pdf'}")
    for stem in SPREAD_OUTPUTS.values():
        print(f"wrote {FIG / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
