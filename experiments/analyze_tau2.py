"""Aggregate tau2-bench results into a (sim, agent) success matrix
and compute the same statistics we report for MINT, 24-task, AgentClinic.

Reads:  experiments/results/tau2_raw.jsonl
Writes: experiments/results/tau2_summary.json (and prints a human report)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / "experiments" / "results"

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
            rows.append({
                "sim": r["sim_alias"],
                "agent": r["agent_alias"],
                "success": bool(r["success"]),
                "reward": float(r.get("reward", 0.0)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    raw = OUT / "tau2_raw.jsonl"
    if not raw.exists():
        print(f"missing: {raw}")
        return
    df = load_df(raw)
    print(f"Loaded {len(df)} (sim, agent, task) rows.")
    pv = (df.groupby(["sim", "agent"])["success"].mean().unstack("agent") * 100.0)
    pv = pv.reindex(index=[s for s in PANEL if s in pv.index],
                    columns=[a for a in PANEL if a in pv.columns])
    print("\nSuccess-rate matrix (rows=simulator, cols=agent), %:")
    print(pv.round(1).to_string())

    # Worst-pair Kendall tau
    sims = list(pv.index)
    K = pd.DataFrame(index=sims, columns=sims, dtype=float)
    for a in sims:
        ra = pv.loc[a].rank(ascending=False).to_numpy(dtype=float)
        for b in sims:
            rb = pv.loc[b].rank(ascending=False).to_numpy(dtype=float)
            tau, _ = kendalltau(ra, rb)
            K.loc[a, b] = tau
    np.fill_diagonal(K.values, np.nan)
    worst_pair = float(np.nanmin(K.to_numpy()))
    print(f"\nWorst-pair Kendall tau: {worst_pair:.3f}")

    # Per-agent score range
    print("\nPer-agent score range across simulators:")
    for a in pv.columns:
        col = pv[a]
        print(f"  {a:14s}: min={col.min():.1f}  max={col.max():.1f}  range={(col.max()-col.min()):.1f} pp")
    max_swing = float(pv.max(axis=0).max() - pv.min(axis=0).min())
    print(f"\nMax same-agent swing across cells: {max_swing:.1f} pp")
    print(f"Mean success rate (all cells): {pv.values.mean():.1f}%")

    summary = {
        "n_dialogues": int(len(df)),
        "n_sims": int(pv.shape[0]),
        "n_agents": int(pv.shape[1]),
        "mean_success_rate_pct": float(pv.values.mean()),
        "max_score_swing_pp": max_swing,
        "worst_pair_kendall_tau": worst_pair,
        "pivot": pv.round(1).to_dict(),
        "kendall": K.round(2).to_dict(),
    }
    (OUT / "tau2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'tau2_summary.json'}")


if __name__ == "__main__":
    main()
