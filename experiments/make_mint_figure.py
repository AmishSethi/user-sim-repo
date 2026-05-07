"""Generate the headline simulator-fidelity figure for the paper.

The layout is intentionally conservative: three compact diagnostic panels show
the report-card checks, and one larger panel shows the MINT per-agent score
spread under six user simulators.
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
from scipy.stats import kendalltau


OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SRC_FIG = Path(__file__).resolve().parent.parent / "figures"
SRC_FIG.mkdir(parents=True, exist_ok=True)

PANEL = [
    "llama3.1-8b",
    "qwen2.5-7b",
    "gemma3-12b",
    "llama3.3-70b",
    "gpt-oss-120b",
    "deepseek-r1",
]

SHORT = {
    "llama3.1-8b": "Llama 8B",
    "qwen2.5-7b": "Qwen 7B",
    "gemma3-12b": "Gemma 12B",
    "llama3.3-70b": "Llama 70B",
    "gpt-oss-120b": "gpt-oss",
    "deepseek-r1": "DeepSeek",
}

BENCHMARKS = {
    "MINT": "mint_raw.jsonl",
    "24-task CS": "exp1_v2_raw.jsonl",
    "AgentClinic": "agentclinic_raw.jsonl",
}
BENCH_ORDER = ["MINT", "24-task CS", "AgentClinic"]

BENCH_COLORS = {
    "MINT": "#0B6E69",
    "24-task CS": "#C96E2C",
    "AgentClinic": "#7161A8",
}

SIM_COLORS = {
    "llama3.1-8b": "#4E79A7",
    "qwen2.5-7b": "#F28E2B",
    "gemma3-12b": "#59A14F",
    "llama3.3-70b": "#76B7B2",
    "gpt-oss-120b": "#E15759",
    "deepseek-r1": "#B07AA1",
}

SIM_MARKERS = {
    "llama3.1-8b": "o",
    "qwen2.5-7b": "s",
    "gemma3-12b": "^",
    "llama3.3-70b": "D",
    "gpt-oss-120b": "P",
    "deepseek-r1": "X",
}

VARIANCE_BENCH_NAMES = {
    "mint": "MINT",
    "24task": "24-task CS",
    "clinic": "AgentClinic",
}


def load_pivot(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((row["sim_alias"], row["agent_alias"], bool(row["success"])))

    df = pd.DataFrame(rows, columns=["sim", "agent", "success"])
    df = df[df["sim"].isin(PANEL) & df["agent"].isin(PANEL)]
    pivot = df.groupby(["sim", "agent"])["success"].mean().unstack("agent") * 100.0
    return pivot.loc[PANEL, PANEL]


def worst_pair_kendall(pivot: pd.DataFrame) -> float:
    taus = []
    for idx, sim_a in enumerate(PANEL):
        rank_a = pivot.loc[sim_a].rank(ascending=False).to_numpy(dtype=float)
        for sim_b in PANEL[idx + 1 :]:
            rank_b = pivot.loc[sim_b].rank(ascending=False).to_numpy(dtype=float)
            tau, _ = kendalltau(rank_a, rank_b)
            taus.append(tau)
    return float(np.nanmin(taus))


def load_stats() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    pivots = {name: load_pivot(OUT / filename) for name, filename in BENCHMARKS.items()}
    stats: dict[str, dict[str, float]] = {}
    for name, pivot in pivots.items():
        fixed_agent_cross_sim_ranges = pivot.max(axis=0) - pivot.min(axis=0)
        stats[name] = {
            "max_swing": float(fixed_agent_cross_sim_ranges.max()),
            "worst_tau": worst_pair_kendall(pivot),
        }

    variance = json.loads((OUT / "variance_summary.json").read_text())
    for name in BENCH_ORDER:
        cells = [
            cell
            for cell in variance["cells"]
            if VARIANCE_BENCH_NAMES[cell["bench"]] == name
        ]
        stats[name]["snr"] = float(np.mean([cell["snr"] for cell in cells]))

    return pivots, stats


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax.set_axisbelow(True)


def draw_metric_panel(
    ax: plt.Axes,
    title: str,
    values: list[float],
    threshold: float,
    xlim: tuple[float, float],
    value_fmt: str,
    show_ylabels: bool,
) -> None:
    y = np.arange(len(BENCH_ORDER))
    colors = [BENCH_COLORS[name] for name in BENCH_ORDER]
    ax.barh(y, values, color=colors, height=0.56)
    ax.set_yticks(y, BENCH_ORDER if show_ylabels else [""] * len(BENCH_ORDER))
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=9.7, fontweight="bold", pad=2)
    style_axis(ax)

    line_color = "#9A1B1E"
    ax.axvline(threshold, color=line_color, linestyle="--", linewidth=1.25)

    for yi, val in enumerate(values):
        ax.text(
            val + 0.025 * (xlim[1] - xlim[0]),
            yi,
            value_fmt.format(val),
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#111111",
        )


def draw_mint_panel(ax: plt.Axes, mint: pd.DataFrame) -> None:
    agent_order = list((mint.max(axis=0) - mint.min(axis=0)).sort_values(ascending=False).index)
    x = np.arange(len(agent_order))

    for i, agent in enumerate(agent_order):
        values = mint[agent]
        low, high = float(values.min()), float(values.max())
        ax.vlines(i, low, high, color="#5A5A5A", linewidth=1.8, zorder=1)
        ax.hlines([low, high], i - 0.08, i + 0.08, color="#333333", linewidth=1.7, zorder=2)
        ax.text(
            i,
            high + 2.5,
            f"{high - low:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
        for j, sim in enumerate(PANEL):
            jitter = (j - 2.5) * 0.045
            ax.scatter(
                i + jitter,
                float(mint.loc[sim, agent]),
                s=34,
                color=SIM_COLORS[sim],
                marker=SIM_MARKERS[sim],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    ax.set_title(
        "(d) MINT scores for each agent under six simulators",
        loc="left",
        fontsize=9.8,
        fontweight="bold",
        pad=7,
    )
    ax.text(
        0.01,
        0.965,
        "range labels are percentage-point swings",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color="#555555",
    )
    ax.set_ylabel("success (%)")
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(x, [SHORT[a] for a in agent_order], rotation=13, ha="right")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=SIM_MARKERS[sim],
            linestyle="",
            color=SIM_COLORS[sim],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            label=SHORT[sim],
        )
        for sim in PANEL
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=6,
        frameon=False,
        fontsize=7.4,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def render(out_pdf: Path, out_png: Path) -> None:
    pivots, stats = load_stats()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.65))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.55, 2.0], hspace=0.48, wspace=0.62)

    draw_metric_panel(
        fig.add_subplot(gs[0, 0]),
        "(a) max cross-sim\nswing, fixed agent (pp)",
        [stats[name]["max_swing"] for name in BENCH_ORDER],
        10,
        (0, 43),
        "{:.0f}",
        True,
    )
    draw_metric_panel(
        fig.add_subplot(gs[0, 1]),
        "(b) worst-pair\nranking $\\tau$",
        [stats[name]["worst_tau"] for name in BENCH_ORDER],
        0.9,
        (0, 1.0),
        "{:.2f}",
        False,
    )
    draw_metric_panel(
        fig.add_subplot(gs[0, 2]),
        "(c) T1\nsignal-to-noise",
        [stats[name]["snr"] for name in BENCH_ORDER],
        3,
        (0, 5.0),
        "{:.1f}",
        False,
    )
    draw_mint_panel(fig.add_subplot(gs[1, :]), pivots["MINT"])

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    out_pdf = FIG / "mint_main.pdf"
    out_png = FIG / "mint_main.png"
    render(out_pdf, out_png)
    shutil.copyfile(out_pdf, SRC_FIG / "mint_main.pdf")
    print(f"figure -> {out_pdf}")
