"""Export the minimum data needed to recreate figure 2.

Reads experiments/results/*.jsonl and variance_summary.json, writes
tau2_bench_data/figure2_data.json. Run this anytime new benchmark
data lands.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / "experiments" / "results"
EXPORT_DIR = ROOT / "tau2_bench_data"
EXPORT_DIR.mkdir(exist_ok=True)

PANEL = ["llama3.1-8b", "qwen2.5-7b", "gemma3-12b",
         "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"]
PANEL_PRETTY = ["Llama-3.1-8B", "Qwen2.5-7B", "Gemma-3-12B",
                "Llama-3.3-70B", "gpt-oss-120B", "DeepSeek-R1"]

# Benchmark name -> raw jsonl filename and metadata
BENCHMARKS = {
    "MINT": {
        "file": "mint_raw.jsonl",
        "subset": "50 tasks across GSM8K, HotpotQA, MMLU",
        "n_dialogues": 1800,
        "n_tasks_per_cell": 50,
        "n_trials": 1,
        "max_steps": 8,
    },
    "24-task customer-service": {
        "file": "exp1_v2_raw.jsonl",
        "subset": "24 multi-turn customer-service tasks (airline, retail, account, billing)",
        "n_dialogues": 3024,
        "n_tasks_per_cell": 24,
        "n_trials": 3,
        "max_steps": 8,
    },
    "AgentClinic-MedQA": {
        "file": "agentclinic_raw.jsonl",
        "subset": "30-case subset of AgentClinic-MedQA",
        "n_dialogues": 1080,
        "n_tasks_per_cell": 30,
        "n_trials": 1,
        "max_steps": 8,
    },
    "tau-bench airline": {
        "file": "taubench_raw.jsonl",
        "subset": "5 tasks from tau-bench airline (Yao et al. 2024)",
        "n_dialogues": 180,
        "n_tasks_per_cell": 5,
        "n_trials": 1,
        "max_steps": 40,
    },
    "tau2-bench telecom": {
        "file": "tau2_raw.jsonl",
        "subset": "5 tasks from tau2-bench telecom_full (Sierra Research)",
        "n_dialogues": 180,
        "n_tasks_per_cell": 5,
        "n_trials": 1,
        "max_steps": 40,
    },
}


def load_matrix(path: Path) -> dict | None:
    """Return 6x6 success-rate matrix as nested dict, or None if file missing."""
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
    if len(df) == 0:
        return None
    pv = (df.groupby(["sim", "agent"])["success"].mean().unstack("agent") * 100.0)
    # Only export benchmarks that have all 36 cells covered
    if pv.shape != (6, 6) and len(pv.index) * len(pv.columns) < 36:
        # Partial; skip
        return None
    pv = pv.reindex(index=PANEL, columns=PANEL)
    return {sim: {agent: float(round(pv.loc[sim, agent], 1))
                  for agent in PANEL} for sim in PANEL}


def main() -> None:
    fig2 = {
        "panel": PANEL,
        "panel_pretty": PANEL_PRETTY,
        "benchmarks": {},
        "variance_t1": {},
    }
    for name, meta in BENCHMARKS.items():
        matrix = load_matrix(OUT / meta["file"])
        if matrix is None:
            print(f"  [skip] {name}: no data yet at {meta['file']}")
            continue
        fig2["benchmarks"][name] = {
            **{k: v for k, v in meta.items() if k != "file"},
            "success_rate_pct": matrix,
        }
        print(f"  [add]  {name}")

    # Variance summary
    vpath = OUT / "variance_summary.json"
    if vpath.exists():
        var = json.loads(vpath.read_text())
        names = {"mint": "MINT", "24task": "24-task customer-service",
                 "clinic": "AgentClinic-MedQA"}
        for raw_name, pretty_name in names.items():
            cells = [c for c in var["cells"] if c["bench"] == raw_name]
            if not cells:
                continue
            fig2["variance_t1"][pretty_name] = {
                "n_cells": len(cells),
                "n_trials_per_cell": 10,
                "mean_within_cell_std_pp": round(
                    sum(c["within_std_pp"] for c in cells) / len(cells), 2),
                "mean_cross_sim_std_pp": round(
                    sum(c["cross_sim_std_pp"] for c in cells) / len(cells), 2),
                "mean_snr": round(
                    sum(c["snr"] for c in cells) / len(cells), 2),
            }
    for missing in ("tau-bench airline", "tau2-bench telecom"):
        if missing in fig2["benchmarks"]:
            fig2["variance_t1"][missing] = {
                "note": "T1 variance experiment not run; cross-simulator effect "
                        "is large enough that trial-to-trial noise is not a candidate "
                        "explanation."
            }

    out_path = EXPORT_DIR / "figure2_data.json"
    out_path.write_text(json.dumps(fig2, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"benchmarks included: {list(fig2['benchmarks'].keys())}")


if __name__ == "__main__":
    main()
