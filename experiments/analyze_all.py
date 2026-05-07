"""Aggregate exp1 v2 + MINT + AgentClinic results into one combined figure
and per-benchmark stats. Produces:

  results/all_summary.json
  figures/ranking_flip_v2.pdf      (replaces old ranking_flip)
  figures/cross_benchmark.pdf      (new: 3-panel comparison)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

OUT = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "24-task CS":   OUT / "exp1_v2_raw.jsonl",
    "MINT (50)":    OUT / "mint_raw.jsonl",
    "AgentClinic":  OUT / "agentclinic_raw.jsonl",
}

# Stable agent / sim alias order
PANEL = ["llama3.1-8b", "qwen2.5-7b", "gemma3-12b",
         "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"]


def load_df(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            rows.append({"sim": r["sim_alias"], "agent": r["agent_alias"],
                         "success": bool(r["success"])})
    return pd.DataFrame(rows)


def per_benchmark_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["sim"].isin(PANEL) & df["agent"].isin(PANEL)]
    pv = (df.groupby(["sim", "agent"])["success"].mean()
          .unstack("agent") * 100.0)
    sims = [s for s in PANEL if s in pv.index]
    agents = [a for a in PANEL if a in pv.columns]
    return pv.loc[sims, agents]


def kendall_matrix(pv: pd.DataFrame) -> pd.DataFrame:
    sims = list(pv.index)
    K = pd.DataFrame(index=sims, columns=sims, dtype=float)
    for a in sims:
        ra = pv.loc[a].rank(ascending=False).to_numpy(dtype=float)
        for b in sims:
            rb = pv.loc[b].rank(ascending=False).to_numpy(dtype=float)
            tau, _ = kendalltau(ra, rb)
            K.loc[a, b] = tau
    return K


def render(name: str, pv: pd.DataFrame, K: pd.DataFrame, ax_pivot, ax_tau):
    data = pv.to_numpy()
    im = ax_pivot.imshow(data, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax_pivot.set_xticks(range(pv.shape[1]))
    ax_pivot.set_xticklabels(pv.columns, rotation=30, ha="right", fontsize=7)
    ax_pivot.set_yticks(range(pv.shape[0]))
    ax_pivot.set_yticklabels(pv.index, fontsize=7)
    ax_pivot.set_title(f"{name}: success (%) per (sim, agent)", fontsize=8)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = data[i, j]
            ax_pivot.text(j, i, f"{v:.0f}", ha="center", va="center",
                          color="white" if v < 55 else "black", fontsize=6)
    plt.colorbar(im, ax=ax_pivot, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)

    Kv = K.to_numpy().astype(float).copy()
    np.fill_diagonal(Kv, np.nan)
    im2 = ax_tau.imshow(Kv, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
    ax_tau.set_xticks(range(K.shape[0]))
    ax_tau.set_xticklabels(K.columns, rotation=30, ha="right", fontsize=7)
    ax_tau.set_yticks(range(K.shape[0]))
    ax_tau.set_yticklabels(K.index, fontsize=7)
    ax_tau.set_title(f"{name}: Kendall $\\tau$ between rankings", fontsize=8)
    for i in range(K.shape[0]):
        for j in range(K.shape[0]):
            v = Kv[i, j]
            if np.isnan(v):
                continue
            ax_tau.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=6)
    plt.colorbar(im2, ax=ax_tau, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)


def main():
    summary = {}
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 12.5))
    for row, (name, path) in enumerate(BENCHMARKS.items()):
        if not path.exists():
            print(f"[skip] {name}: {path} missing")
            for col in (0, 1):
                axes[row, col].text(0.5, 0.5, f"{name}\n(no data yet)",
                                    ha="center", va="center")
                axes[row, col].axis("off")
            continue
        df = load_df(path)
        if df.empty:
            print(f"[skip] {name}: empty file")
            continue
        pv = per_benchmark_pivot(df)
        K = kendall_matrix(pv)
        # Aggregate stats
        worst_pair_tau = float(np.nanmin(K.to_numpy() + np.eye(K.shape[0])))
        max_swing = float(pv.max(axis=0).max() - pv.min(axis=0).min())
        per_agent_var = float(pv.var(axis=0).mean())
        summary[name] = {
            "n_dialogues": int(len(df)),
            "n_sims": int(pv.shape[0]),
            "n_agents": int(pv.shape[1]),
            "mean_success_rate_pct": float(pv.values.mean()),
            "max_score_swing_pp": max_swing,
            "mean_per_agent_variance": per_agent_var,
            "worst_pair_kendall_tau": worst_pair_tau,
            "pivot": pv.round(1).to_dict(),
            "kendall": K.round(2).to_dict(),
        }
        render(name, pv, K, axes[row, 0], axes[row, 1])
        print(f"[ok]   {name}: {len(df)} dialogues, max swing {max_swing:.1f} pp, worst-pair tau {worst_pair_tau:.2f}")

    fig.tight_layout()
    out_pdf = FIG / "cross_benchmark.pdf"
    out_png = FIG / "cross_benchmark.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"figure -> {out_pdf}")

    (OUT / "all_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {OUT/'all_summary.json'}")


if __name__ == "__main__":
    main()
