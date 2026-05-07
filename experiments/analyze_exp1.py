"""Aggregate Experiment 1 results and produce figures + tables.

Inputs:  experiments/results/exp1_raw.jsonl  (one DialogueResult per line)
Outputs: figures/ranking_flip.pdf
         experiments/results/exp1_summary.csv
         experiments/results/exp1_kendall.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr


RAW = Path(__file__).resolve().parent / "results" / "exp1_raw.jsonl"
OUTDIR = Path(__file__).resolve().parent / "results"
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_df() -> pd.DataFrame:
    rows = []
    for ln in RAW.read_text().splitlines():
        if not ln.strip():
            continue
        rows.append(json.loads(ln))
    df = pd.DataFrame(rows)
    return df


def main():
    df = load_df()
    print(f"loaded {len(df)} rows")
    # Aggregate success per (sim, agent)
    pivot = (df.groupby(["sim_alias", "agent_alias"])["success"].mean()
             .unstack("agent_alias") * 100.0)
    print("\nSuccess rate (%) per (sim, agent):")
    print(pivot.round(1))
    pivot.to_csv(OUTDIR / "exp1_summary.csv")

    # Per-sim agent rankings (tie-broken by mean)
    sims = list(pivot.index)
    agents = list(pivot.columns)
    rank = pd.DataFrame(index=sims, columns=agents, dtype=float)
    for s in sims:
        scores = pivot.loc[s]
        # rank ascending so higher score = lower rank number; we'll display as 1 = best
        ranks = scores.rank(ascending=False, method="average")
        rank.loc[s] = ranks
    print("\nAgent ranks under each simulator (1=best):")
    print(rank)
    rank.to_csv(OUTDIR / "exp1_ranks.csv")

    # Kendall tau between every pair of sims
    K = pd.DataFrame(index=sims, columns=sims, dtype=float)
    P = pd.DataFrame(index=sims, columns=sims, dtype=float)
    for s1 in sims:
        for s2 in sims:
            r1 = rank.loc[s1].astype(float).to_numpy()
            r2 = rank.loc[s2].astype(float).to_numpy()
            tau, _ = kendalltau(r1, r2)
            r, _ = pearsonr(pivot.loc[s1].astype(float).to_numpy(),
                            pivot.loc[s2].astype(float).to_numpy())
            K.loc[s1, s2] = tau
            P.loc[s1, s2] = r
    print("\nKendall's tau between agent rankings under different simulators:")
    print(K.round(2))
    K.to_csv(OUTDIR / "exp1_kendall.csv")
    P.to_csv(OUTDIR / "exp1_pearson.csv")

    # Variance of agent score across simulators
    agent_var = pivot.var(axis=0)
    print("\nVariance of agent success rate across simulators:")
    print(agent_var.round(2))

    # Figure: heatmap of success rate + Kendall's tau side panel
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    data = pivot.to_numpy()
    im = ax.imshow(data, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(sims)))
    ax.set_yticklabels(sims, fontsize=8)
    ax.set_xlabel("Agent backbone", fontsize=9)
    ax.set_ylabel("User-simulator backbone", fontsize=9)
    ax.set_title("Task-success rate (%) on the same 24-task benchmark",
                 fontsize=9)
    for i in range(len(sims)):
        for j in range(len(agents)):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center",
                    color="white" if data[i,j] < 55 else "black", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("success rate (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax2 = axes[1]
    Kvals = K.to_numpy().astype(float)
    np.fill_diagonal(Kvals, np.nan)
    im2 = ax2.imshow(Kvals, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
    ax2.set_xticks(range(len(sims)))
    ax2.set_xticklabels(sims, rotation=20, ha="right", fontsize=8)
    ax2.set_yticks(range(len(sims)))
    ax2.set_yticklabels(sims, fontsize=8)
    ax2.set_title("Kendall's tau between rankings", fontsize=9)
    for i in range(len(sims)):
        for j in range(len(sims)):
            v = Kvals[i, j]
            if np.isnan(v):
                continue
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     color="white" if abs(v) > 0.5 else "black", fontsize=8)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("Kendall tau", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = FIGDIR / "ranking_flip.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIGDIR / "ranking_flip.png", dpi=180, bbox_inches="tight")
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    main()
