"""Generate tau2-bench telecom simulator-sensitivity figures.

The tau2 export stores the success matrix as outer key = fixed agent under
test and inner key = simulated user. Internally, this script converts it to
rows = simulated users and columns = fixed agents for plotting/statistics.
This script renders three candidate designs:

* option_a_heatmap: matrix heatmap plus a per-agent range strip.
* option_b_range: lightweight fixed-agent spread plot with simulator dots.
* option_c_band: vertical band plot matching the paper's Figure 2 style.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tau2_bench_data" / "tau2_full_matrix.json"
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

SIM_COLORS = {
    "llama3.1-8b": "#7D8FA8",
    "qwen2.5-7b": "#E38624",
    "gemma3-12b": "#6AAE22",
    "llama3.3-70b": "#168A80",
    "gpt-oss-120b": "#6647D9",
    "deepseek-r1": "#C92A2A",
}

HEATMAP_VARIANTS = {
    1: {"figsize": (7.15, 3.85), "xrot": 0, "xwrap": True, "strip_h": 0.42, "cbar": True},
    2: {"figsize": (7.10, 3.70), "xrot": 24, "xwrap": False, "strip_h": 0.34, "cbar": True},
    3: {"figsize": (7.10, 3.55), "xrot": 0, "xwrap": True, "strip_h": 0.34, "cbar": False},
    4: {"figsize": (7.20, 3.65), "xrot": 18, "xwrap": False, "strip_h": 0.38, "cbar": False},
}

RANGE_VARIANTS = {
    1: {"figsize": (7.15, 3.15), "legend_y": 1.22, "label_pad": 2.5, "show_mean": False, "jitter": 0.13},
    2: {"figsize": (7.15, 3.25), "legend_y": 1.12, "label_pad": 2.5, "show_mean": True, "jitter": 0.13},
    3: {"figsize": (7.05, 2.85), "legend_y": 1.14, "label_pad": 2.0, "show_mean": False, "jitter": 0.11},
    4: {"figsize": (7.20, 3.20), "legend_y": 1.11, "label_pad": 3.0, "show_mean": True, "jitter": 0.11},
}

BAND_VARIANTS = {
    1: {"figsize": (7.2, 3.15), "legend_y": 1.15, "label_pad": 2.4, "band_alpha": 0.58, "title": True},
    2: {"figsize": (7.2, 3.05), "legend_y": 1.28, "label_pad": 2.2, "band_alpha": 0.52, "title": False},
    3: {"figsize": (7.2, 3.28), "legend_y": 1.16, "label_pad": 2.8, "band_alpha": 0.50, "title": True},
}

FINAL_KIND = "option_c_band"
FINAL_VARIANT = 2


def configure_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def configure_figure2_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_tau2() -> tuple[pd.DataFrame, dict]:
    data = json.loads(DATA.read_text())
    panel = data["panel"]
    if panel != PANEL:
        raise ValueError(f"Unexpected panel order in {DATA}: {panel}")
    if "success_matrix" in data:
        rows = {
            row["Agent"]: row["Simulated user"]
            for row in data["success_matrix"]
        }
        pivot = pd.DataFrame(rows).loc[PANEL, PANEL]
    else:
        agent_by_sim = pd.DataFrame(data["success_rate_pct"]).T.loc[PANEL, PANEL]
        pivot = agent_by_sim.T.loc[PANEL, PANEL]
    return pivot.astype(float), data


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


def order_simulators_by_mean(pivot: pd.DataFrame) -> list[str]:
    means = pivot.mean(axis=1)
    highs = pivot.max(axis=1)
    return (
        pd.DataFrame({"mean": means, "high": highs})
        .sort_values(["mean", "high"], ascending=[False, False])
        .index
        .tolist()
    )


def labels(names: list[str], wrapped: bool) -> list[str]:
    source = WRAPPED_LABEL if wrapped else MODEL_LABEL
    return [source[name] for name in names]


def save_all(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def render_option_a_heatmap(out_prefix: Path, variant: int) -> None:
    configure_style()
    pivot, _ = load_tau2()
    cfg = HEATMAP_VARIANTS[variant]
    agent_order = order_agents_by_swing(pivot)
    sim_order = order_simulators_by_mean(pivot)
    matrix = pivot.loc[sim_order, agent_order]
    ranges = (pivot.max(axis=0) - pivot.min(axis=0)).loc[agent_order]

    cmap = LinearSegmentedColormap.from_list(
        "tau2_success",
        ["#F8FAFC", "#DBF3EF", "#8EDAD0", "#2A9D8F", "#146C68"],
    )

    fig = plt.figure(figsize=cfg["figsize"])
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2 if cfg["cbar"] else 1,
        height_ratios=[cfg["strip_h"], 3.0],
        width_ratios=[20, 0.75] if cfg["cbar"] else [1],
        hspace=0.06,
        wspace=0.08,
    )
    ax_strip = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1]) if cfg["cbar"] else None

    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(agent_order)), labels=labels(agent_order, cfg["xwrap"]))
    ax.set_yticks(np.arange(len(sim_order)), labels=labels(sim_order, False))
    ax.tick_params(axis="x", rotation=cfg["xrot"], length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)
    if cfg["xrot"]:
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    ax.set_xlabel("Fixed agent under test", labelpad=6)
    ax.set_ylabel("Simulated user", labelpad=6)

    ax.set_xticks(np.arange(-0.5, len(agent_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sim_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y, sim in enumerate(sim_order):
        for x, agent in enumerate(agent_order):
            value = int(matrix.loc[sim, agent])
            color = "white" if value >= 65 else "#111827"
            ax.text(x, y, f"{value}", ha="center", va="center", fontsize=7.8, color=color)

    ax_strip.set_xlim(-0.5, len(agent_order) - 0.5)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([0.5], labels=["Range"])
    ax_strip.tick_params(axis="y", length=0, pad=4)
    for i, agent in enumerate(agent_order):
        ax_strip.add_patch(
            plt.Rectangle(
                (i - 0.5, 0),
                1,
                1,
                facecolor="#F3F4F6",
                edgecolor="white",
                linewidth=1.25,
            )
        )
        ax_strip.text(
            i,
            0.5,
            f"{int(ranges.loc[agent])} pp",
            ha="center",
            va="center",
            fontsize=7.8,
            color="#111827",
        )
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    if cax is not None:
        cbar = fig.colorbar(im, cax=cax, ticks=[0, 25, 50, 75, 100])
        cbar.ax.tick_params(labelsize=7.3, length=2.5, pad=2)
        cbar.set_label("success (%)", labelpad=5, fontsize=8.0)
        cbar.outline.set_visible(False)

    save_all(fig, out_prefix)


def render_option_b_range(out_prefix: Path, variant: int) -> None:
    configure_style()
    pivot, _ = load_tau2()
    cfg = RANGE_VARIANTS[variant]
    agent_order = order_agents_by_swing(pivot)
    y = np.arange(len(agent_order))
    jitter = np.linspace(-cfg["jitter"], cfg["jitter"], len(PANEL))

    fig, ax = plt.subplots(figsize=cfg["figsize"])

    for i, agent in enumerate(agent_order):
        values = pivot[agent]
        low = float(values.min())
        high = float(values.max())
        mean = float(values.mean())
        swing = high - low
        ax.hlines(i, low, high, color="#8F9AA8", linewidth=1.25, zorder=1)
        ax.vlines([low, high], i - 0.11, i + 0.11, color="#8F9AA8", linewidth=1.05, zorder=1)
        if cfg["show_mean"]:
            ax.scatter(mean, i, marker="D", s=22, color="#111827", edgecolor="white", linewidth=0.5, zorder=4)
        for j, sim in enumerate(PANEL):
            ax.scatter(
                float(values.loc[sim]),
                i + jitter[j],
                s=34,
                color=SIM_COLORS[sim],
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )
        label_x = min(high + cfg["label_pad"], 102)
        ha = "left"
        if swing == 0:
            label_x = 2.5
            ha = "left"
        ax.text(
            label_x,
            i,
            f"{swing:.0f} pp",
            ha=ha,
            va="center",
            fontsize=7.8,
            color="#374151",
        )

    ax.set_xlabel("success (%)", labelpad=6)
    ax.set_ylabel("Fixed agent under test", labelpad=6)
    ax.set_xlim(-2, 106)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(y, labels=labels(agent_order, False))
    ax.invert_yaxis()
    ax.set_ylim(len(agent_order) - 0.5, -0.72)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.8)
    ax.grid(axis="y", visible=False)
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
            markeredgewidth=0.75,
            markersize=5.6,
            label=MODEL_LABEL[sim],
        )
        for sim in PANEL
    ]
    if cfg["show_mean"]:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                color="#111827",
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=5.0,
                label="mean",
            )
        )
    ax.legend(
        handles=handles,
        title="Simulated user",
        loc="upper center",
        bbox_to_anchor=(0.5, cfg["legend_y"]),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        title_fontsize=7.5,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.85,
    )

    save_all(fig, out_prefix)


def render_option_c_band(out_prefix: Path, variant: int) -> None:
    configure_figure2_style()
    pivot, _ = load_tau2()
    cfg = BAND_VARIANTS[variant]
    agent_order = order_agents_by_swing(pivot)
    x = np.arange(len(agent_order))
    jitter = np.linspace(-0.18, 0.18, len(PANEL))

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    band_color = "#BFEFE6"
    edge_color = "#146C68"

    for i, agent in enumerate(agent_order):
        values = pivot[agent]
        low = float(values.min())
        high = float(values.max())
        swing = high - low
        band_height = max(high - low, 1.0)

        ax.add_patch(
            plt.Rectangle(
                (i - 0.32, low),
                0.64,
                band_height,
                facecolor=band_color,
                edgecolor="none",
                alpha=cfg["band_alpha"],
                zorder=1,
            )
        )
        ax.hlines([low, high], i - 0.32, i + 0.32, color=edge_color, linewidth=1.15, zorder=2)

        label_y = high + cfg["label_pad"]
        if swing == 0:
            label_y = 5.5
        ax.text(
            i,
            label_y,
            f"{swing:.0f} pp",
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

    if cfg["title"]:
        ax.set_title(
            r"$\tau^2$-bench telecom success per fixed agent",
            loc="left",
            pad=11,
        )
    ax.set_ylabel("success (%)", color="#374151")
    ax.set_xlabel("Fixed agent under test", labelpad=8)
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x, labels(agent_order, wrapped=True))
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors="#374151")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.82)
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
        bbox_to_anchor=(0.5, cfg["legend_y"]),
        ncol=3,
        frameon=False,
        fontsize=7.8,
        title_fontsize=8.2,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=1.05,
    )

    save_all(fig, out_prefix)


def render(kind: str, out_prefix: Path, variant: int) -> None:
    if kind == "option_a_heatmap":
        render_option_a_heatmap(out_prefix, variant)
    elif kind == "option_b_range":
        render_option_b_range(out_prefix, variant)
    elif kind == "option_c_band":
        render_option_c_band(out_prefix, variant)
    else:
        raise ValueError(f"Unknown figure kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["option_a_heatmap", "option_b_range", "option_c_band"], default=FINAL_KIND)
    parser.add_argument("--variant", type=int, default=FINAL_VARIANT)
    parser.add_argument("--out-prefix", type=Path, default=FIG / "tau2_agent_spread")
    parser.add_argument("--copy-to-src", action="store_true")
    args = parser.parse_args()

    if args.kind == "option_a_heatmap" and args.variant not in HEATMAP_VARIANTS:
        raise ValueError(f"Heatmap variant must be one of {sorted(HEATMAP_VARIANTS)}")
    if args.kind == "option_b_range" and args.variant not in RANGE_VARIANTS:
        raise ValueError(f"Range variant must be one of {sorted(RANGE_VARIANTS)}")
    if args.kind == "option_c_band" and args.variant not in BAND_VARIANTS:
        raise ValueError(f"Band variant must be one of {sorted(BAND_VARIANTS)}")

    render(args.kind, args.out_prefix, args.variant)
    if args.copy_to_src:
        SRC_FIG.mkdir(parents=True, exist_ok=True)
        target = SRC_FIG / args.out_prefix.name
        shutil.copyfile(args.out_prefix.with_suffix(".pdf"), target.with_suffix(".pdf"))
        shutil.copyfile(args.out_prefix.with_suffix(".svg"), target.with_suffix(".svg"))
    print(f"{args.kind} -> {args.out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
