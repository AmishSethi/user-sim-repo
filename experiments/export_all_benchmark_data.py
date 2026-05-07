"""Build a single JSON containing every benchmark probe and every
derived statistic the paper reports.

Each benchmark contains:

    success_matrix              list of {Agent, Simulated user dict}
                                grouped by Agent. The Simulated user dict
                                maps each simulator alias to the percent
                                success rate for that (sim, agent) cell.
    mean_success_rate_pct       overall mean over the matrix
    max_same_agent_swing_pp     max over agents of (max_sim - min_sim)
    per_agent_swing_pp          per-agent (max_sim - min_sim)
    worst_pair_kendall_tau      min over simulator pairs
    pairwise_kendall_tau        list of {Simulator A, Simulator B,
                                kendall_tau} entries, undirected pairs

For benchmarks where the K=10 variance experiment ran, also includes:

    t1_variance.n_cells, mean_within_cell_std_pp, mean_cross_sim_std_pp,
        mean_snr, min_snr, max_snr
    t1_variance.per_cell        list of per-cell records with uniform
                                schema {Agent, Simulated user, n_trials,
                                mean_pct, within_std_pp, cross_sim_std_pp,
                                snr}

Output: tau2_bench_data/all_benchmark_data.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / "experiments" / "results"
EXPORT_DIR = ROOT / "tau2_bench_data"
EXPORT_DIR.mkdir(exist_ok=True)

PANEL = ["llama3.1-8b", "qwen2.5-7b", "gemma3-12b",
         "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"]
PANEL_PRETTY = ["Llama-3.1-8B", "Qwen2.5-7B", "Gemma-3-12B",
                "Llama-3.3-70B", "gpt-oss-120B", "DeepSeek-R1"]

BENCHMARKS = {
    "MINT": {
        "file": "mint_raw.jsonl",
        "subset": "50 tasks across GSM8K, HotpotQA, MMLU",
        "n_dialogues": 1800,
        "n_tasks_per_cell": 50,
        "n_trials": 1,
        "max_steps": 8,
        "variance_key": "mint",
    },
    "24-task customer-service": {
        "file": "exp1_v2_raw.jsonl",
        "subset": "24 multi-turn customer-service tasks (airline, retail, account, billing)",
        "n_dialogues": 3024,
        "n_tasks_per_cell": 24,
        "n_trials": 3,
        "max_steps": 8,
        "variance_key": "24task",
    },
    "AgentClinic-MedQA": {
        "file": "agentclinic_raw.jsonl",
        "subset": "30-case subset of AgentClinic-MedQA",
        "n_dialogues": 1080,
        "n_tasks_per_cell": 30,
        "n_trials": 1,
        "max_steps": 8,
        "variance_key": "clinic",
    },
    "tau-bench airline": {
        "file": "taubench_raw.jsonl",
        "subset": "5 tasks from tau-bench airline (Yao et al. 2024)",
        "n_dialogues": 180,
        "n_tasks_per_cell": 5,
        "n_trials": 1,
        "max_steps": 40,
        "variance_key": None,
    },
    "tau2-bench telecom": {
        "file": "tau2_raw.jsonl",
        "subset": "20 tasks from tau2-bench telecom_full (Sierra Research 2025)",
        "n_dialogues": 720,
        "n_tasks_per_cell": 20,
        "n_trials": 1,
        "max_steps": 80,
        "variance_key": None,
    },
}


def load_pivot(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    rows = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            rows.append({"sim": r["sim_alias"], "agent": r["agent_alias"],
                         "success": bool(r["success"])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df["sim"].isin(PANEL) & df["agent"].isin(PANEL)]
    pv = (df.groupby(["sim", "agent"])["success"].mean().unstack("agent") * 100.0)
    pv = pv.reindex(index=PANEL, columns=PANEL)
    return pv


def kendall_pairs(pv: pd.DataFrame) -> tuple[float, list]:
    """Compute pairwise Kendall tau over simulator pairs.

    Returns (worst_pair, list of {Simulator A, Simulator B, kendall_tau}).
    """
    sims = list(pv.index)
    pairs = []
    taus = []
    for i, sa in enumerate(sims):
        ra = pv.loc[sa].rank(ascending=False).to_numpy(dtype=float)
        for j in range(i + 1, len(sims)):
            sb = sims[j]
            rb = pv.loc[sb].rank(ascending=False).to_numpy(dtype=float)
            t, _ = kendalltau(ra, rb)
            t_val = round(float(t), 3) if t == t else None
            pairs.append({
                "Simulator A": sa,
                "Simulator B": sb,
                "kendall_tau": t_val,
            })
            if t_val is not None:
                taus.append(t_val)
    worst = float(round(min(taus), 3)) if taus else None
    return worst, pairs


def success_matrix_grouped_by_agent(pv: pd.DataFrame) -> list:
    """Return [{Agent, Simulated user: {sim: pct, ...}}] for each agent."""
    rows = []
    for agent in PANEL:
        sim_dict = {sim: float(round(pv.loc[sim, agent], 1)) for sim in PANEL}
        rows.append({"Agent": agent, "Simulated user": sim_dict})
    return rows


def per_agent_swing(pv: pd.DataFrame) -> list:
    """Return [{Agent, swing_pp}] per agent."""
    swing = pv.max(axis=0) - pv.min(axis=0)
    return [{"Agent": agent, "swing_pp": float(round(swing[agent], 1))}
            for agent in PANEL]


def main() -> None:
    out: dict = {
        "panel": PANEL,
        "panel_pretty": PANEL_PRETTY,
        "schema": {
            "success_matrix": "list of one entry per agent. Each entry has \"Agent\" (alias) and \"Simulated user\" (mapping from simulator alias to percent success rate of that agent under that simulator).",
            "max_same_agent_swing_pp": "max over agents of (max_sim - min_sim) in percentage points",
            "per_agent_swing_pp": "list of one entry per agent giving the cross-simulator swing for that agent",
            "worst_pair_kendall_tau": "minimum Kendall tau over simulator pairs whose induced agent rankings are not both degenerate",
            "pairwise_kendall_tau": "list of undirected simulator pairs with the Kendall tau between their induced agent rankings",
            "t1_variance": "K=10 trial replication on diagnostic cells; per_cell rows give per-cell Agent, Simulated user, n_trials, mean_pct, within_std_pp, cross_sim_std_pp, snr",
            "bootstrap": "B=2000 task-resampling bootstrap; max_swing_ci90 and worst_tau_ci90 are 5th to 95th percentile",
        },
        "benchmarks": {},
    }

    var = {}
    vpath = OUT / "variance_summary.json"
    if vpath.exists():
        var_data = json.loads(vpath.read_text())
        for c in var_data["cells"]:
            var.setdefault(c["bench"], []).append(c)

    for name, meta in BENCHMARKS.items():
        pv = load_pivot(OUT / meta["file"])
        if pv is None:
            print(f"[skip] {name}: no data at {meta['file']}")
            continue
        worst_tau, tau_pairs = kendall_pairs(pv)
        bench_block = {
            **{k: v for k, v in meta.items() if k not in ("file", "variance_key")},
            "success_matrix": success_matrix_grouped_by_agent(pv),
            "mean_success_rate_pct": float(round(pv.values.mean(), 2)),
            "max_same_agent_swing_pp": float(round(
                (pv.max(axis=0) - pv.min(axis=0)).max(), 1)),
            "per_agent_swing_pp": per_agent_swing(pv),
            "worst_pair_kendall_tau": worst_tau,
            "pairwise_kendall_tau": tau_pairs,
        }
        # Variance experiment T1 receipt
        vkey = meta.get("variance_key")
        if vkey and vkey in var:
            cells = var[vkey]
            # Compute per-cell mean_pct from main matrix (cells store sim, agent, but not the per-cell main-run mean)
            per_cell_rows = []
            for c in cells:
                sim, agent = c["sim"], c["agent"]
                per_cell_rows.append({
                    "Agent": agent,
                    "Simulated user": sim,
                    "n_trials": 10,
                    "mean_pct": float(round(pv.loc[sim, agent], 1)),
                    "within_std_pp": float(round(c["within_std_pp"], 2)),
                    "cross_sim_std_pp": float(round(c["cross_sim_std_pp"], 2)),
                    "snr": float(round(c["snr"], 2)),
                })
            snrs = [c["snr"] for c in cells]
            bench_block["t1_variance"] = {
                "n_cells": len(cells),
                "n_trials_per_cell": 10,
                "mean_within_cell_std_pp": float(round(
                    sum(c["within_std_pp"] for c in cells) / len(cells), 2)),
                "max_within_cell_std_pp": float(round(
                    max(c["within_std_pp"] for c in cells), 2)),
                "mean_cross_sim_std_pp": float(round(
                    sum(c["cross_sim_std_pp"] for c in cells) / len(cells), 2)),
                "max_cross_sim_std_pp": float(round(
                    max(c["cross_sim_std_pp"] for c in cells), 2)),
                "mean_snr": float(round(sum(snrs) / len(snrs), 2)),
                "min_snr": float(round(min(snrs), 2)),
                "max_snr": float(round(max(snrs), 2)),
                "per_cell": per_cell_rows,
            }
        else:
            bench_block["t1_variance"] = None
        out["benchmarks"][name] = bench_block
        print(f"[ok]   {name}: max_swing={bench_block['max_same_agent_swing_pp']:.1f} pp, "
              f"worst_tau={worst_tau}, "
              f"mean={bench_block['mean_success_rate_pct']:.1f}%")

    # Headline summary table
    summary_rows = []
    for name, b in out["benchmarks"].items():
        snr = b["t1_variance"]["mean_snr"] if b["t1_variance"] else None
        summary_rows.append({
            "benchmark": name,
            "n_dialogues": b["n_dialogues"],
            "n_tasks_per_cell": b["n_tasks_per_cell"],
            "mean_success_rate_pct": b["mean_success_rate_pct"],
            "max_same_agent_swing_pp": b["max_same_agent_swing_pp"],
            "worst_pair_kendall_tau": b["worst_pair_kendall_tau"],
            "t1_mean_snr": snr,
        })
    out["summary_table"] = summary_rows

    out_path = EXPORT_DIR / "all_benchmark_data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print("\nSummary table (also embedded in JSON):")
    print(f"{'benchmark':28s} {'dialogs':>8s} {'mean%':>7s} {'maxSw':>7s} {'wstTau':>7s} {'snr':>5s}")
    for r in summary_rows:
        snr = f"{r['t1_mean_snr']:.2f}x" if r['t1_mean_snr'] is not None else "  -  "
        wt = f"{r['worst_pair_kendall_tau']:.2f}" if r['worst_pair_kendall_tau'] is not None else "  -  "
        print(f"{r['benchmark']:28s} {r['n_dialogues']:>8d} "
              f"{r['mean_success_rate_pct']:>7.1f} "
              f"{r['max_same_agent_swing_pp']:>7.1f} "
              f"{wt:>7s} {snr:>5s}")


if __name__ == "__main__":
    main()
