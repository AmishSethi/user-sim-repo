"""Compare within-cell trial-to-trial std with cross-simulator std.

For each diagnostic (benchmark, simulator, agent) cell from variance_raw.jsonl,
compute per-trial success rate and the std across the 10 trials. Then pull the
corresponding cross-simulator std from the full original run (where we had 6
simulators tested against the same agent on the same task list, with 1 or 3
trials per cell). The ratio of cross-sim std to within-cell trial std is the
signal-to-noise estimate.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"

# --- variance experiment data -------------------------------------------------
rows = []
for ln in (OUT / "variance_raw.jsonl").read_text().splitlines():
    if not ln.strip():
        continue
    d = json.loads(ln)
    if "error" in d:
        continue
    rows.append(d)
vdf = pd.DataFrame(rows)
print(f"variance_raw rows: {len(vdf)}")

# per-trial success rate (averaged over tasks in that benchmark)
per_trial = (vdf.groupby(["bench", "sim", "agent", "trial"])["success"]
              .mean().reset_index())
within_std = (per_trial.groupby(["bench", "sim", "agent"])["success"]
              .std().rename("within_std").reset_index())
within_mean = (per_trial.groupby(["bench", "sim", "agent"])["success"]
               .mean().rename("mean_success").reset_index())
cell = within_std.merge(within_mean, on=["bench", "sim", "agent"])
cell["within_std_pp"] = (cell["within_std"] * 100).round(2)
cell["mean_success_pp"] = (cell["mean_success"] * 100).round(1)

# --- cross-sim std from the original full runs --------------------------------
def load_full(path):
    rs = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        rs.append((d["sim_alias"], d["agent_alias"], bool(d["success"])))
    return pd.DataFrame(rs, columns=["sim", "agent", "success"])

fulls = {
    "24task":  load_full(OUT / "exp1_v2_raw.jsonl"),
    "mint":    load_full(OUT / "mint_raw.jsonl"),
    "clinic":  load_full(OUT / "agentclinic_raw.jsonl"),
}

# For each (bench, agent), cross-sim std is std of mean success over the
# 6 LLM simulators (we exclude the DSL row to be fair to the variance cells
# which only used LLM sims).
LLM_SIMS = {"llama3.1-8b", "qwen2.5-7b", "gemma3-12b",
            "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"}


def cross_sim_std_for(bench: str, agent: str) -> float:
    df = fulls[bench]
    sub = df[(df["sim"].isin(LLM_SIMS)) & (df["agent"] == agent)]
    means = sub.groupby("sim")["success"].mean() * 100.0
    return float(means.std())


# Annotate each cell with the cross-sim std at that benchmark+agent
cell["cross_sim_std_pp"] = cell.apply(
    lambda r: cross_sim_std_for(r["bench"], r["agent"]), axis=1).round(2)
cell["snr"] = (cell["cross_sim_std_pp"] /
               cell["within_std_pp"].replace(0, np.nan)).round(2)

print("\nDiagnostic-cell variance breakdown:")
print(cell.to_string(index=False))

# Aggregated comparison per benchmark
print("\nPer-benchmark summary:")
agg = (cell.groupby("bench")[["within_std_pp", "cross_sim_std_pp", "snr"]]
       .agg(["mean", "max"]).round(2))
print(agg.to_string())

# Headline for the paper: aggregate snr and worst-case cell within-std
print("\n--- Headline numbers ---")
print(f"  Mean within-cell trial-to-trial std (10 trials):  "
      f"{cell['within_std_pp'].mean():.2f} pp")
print(f"  Max within-cell trial-to-trial std:               "
      f"{cell['within_std_pp'].max():.2f} pp")
print(f"  Mean cross-sim std (paired benchmark/agent):      "
      f"{cell['cross_sim_std_pp'].mean():.2f} pp")
print(f"  Mean signal-to-noise ratio:                       "
      f"{cell['snr'].mean():.2f}x")
print(f"  Cells where cross-sim std > 2x within-cell std:   "
      f"{int((cell['snr'] >= 2).sum())} / {len(cell)}")

(OUT / "variance_summary.json").write_text(json.dumps({
    "cells": cell.to_dict(orient="records"),
    "headline": {
        "mean_within_cell_std_pp": float(cell["within_std_pp"].mean()),
        "max_within_cell_std_pp": float(cell["within_std_pp"].max()),
        "mean_cross_sim_std_pp": float(cell["cross_sim_std_pp"].mean()),
        "mean_snr": float(cell["snr"].mean()),
        "n_cells_snr_at_least_2x": int((cell["snr"] >= 2).sum()),
        "n_cells_total": int(len(cell)),
    },
}, indent=2))
print(f"\nwrote {OUT/'variance_summary.json'}")
