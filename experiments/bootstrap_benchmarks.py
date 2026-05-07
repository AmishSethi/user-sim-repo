"""Bootstrap-over-tasks confidence intervals for the headline T2 numbers
(max same-agent cross-simulator swing and worst-pair Kendall tau).

For each benchmark we treat each (sim, agent) cell as N task outcomes,
resample those tasks with replacement to produce B bootstrap matrices,
recompute the swing and Kendall tau on each, and report the 5th-95th
percentile CI plus the standard error.

Adds the bootstrap block to tau2_bench_data/all_benchmark_data.json
under each benchmark, and prints a summary table.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / "experiments" / "results"
EXPORT = ROOT / "tau2_bench_data" / "all_benchmark_data.json"

PANEL = ["llama3.1-8b", "qwen2.5-7b", "gemma3-12b",
         "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"]

BENCHMARKS = {
    "MINT":                       "mint_raw.jsonl",
    "24-task customer-service":   "exp1_v2_raw.jsonl",
    "AgentClinic-MedQA":          "agentclinic_raw.jsonl",
    "tau-bench airline":          "taubench_raw.jsonl",
    "tau2-bench telecom":         "tau2_raw.jsonl",
}

B = 2000
RNG = np.random.default_rng(42)


def load_cell_outcomes(path: Path) -> dict[tuple[str, str], list[int]]:
    """Return dict[(sim, agent)] -> list of 0/1 per task."""
    cells = defaultdict(list)
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r["sim_alias"] not in PANEL or r["agent_alias"] not in PANEL:
                continue
            cells[(r["sim_alias"], r["agent_alias"])].append(int(r["success"]))
    return cells


def matrix_from_cells(cells: dict, idx: np.ndarray | None) -> np.ndarray:
    """Build a 6x6 matrix of cell mean success rates. If idx is given,
    resample each cell's outcomes by idx (modulo cell length)."""
    M = np.zeros((6, 6))
    for i, sim in enumerate(PANEL):
        for j, agent in enumerate(PANEL):
            outs = cells.get((sim, agent), [])
            if not outs:
                M[i, j] = 0
                continue
            if idx is None:
                M[i, j] = np.mean(outs) * 100
            else:
                arr = np.asarray(outs)
                resampled = arr[idx % len(arr)]
                M[i, j] = resampled.mean() * 100
    return M


def stats_from_matrix(M: np.ndarray) -> tuple[float, float]:
    swing = (M.max(axis=0) - M.min(axis=0)).max()
    taus = []
    for i in range(6):
        ra = (-M[i]).argsort().argsort()
        for j in range(i + 1, 6):
            rb = (-M[j]).argsort().argsort()
            t, _ = kendalltau(ra, rb)
            if t == t:
                taus.append(t)
    worst_tau = float(np.nanmin(taus)) if taus else float("nan")
    return float(swing), worst_tau


def bootstrap(cells: dict, n_tasks: int, B: int = B) -> dict:
    swings, taus = [], []
    for _ in range(B):
        idx = RNG.integers(0, n_tasks, size=n_tasks)
        M = matrix_from_cells(cells, idx)
        s, t = stats_from_matrix(M)
        swings.append(s)
        if t == t:
            taus.append(t)
    return {
        "max_swing_mean":  round(float(np.mean(swings)), 2),
        "max_swing_se":    round(float(np.std(swings, ddof=1)), 2),
        "max_swing_ci90":  [round(float(np.percentile(swings, 5)), 1),
                            round(float(np.percentile(swings, 95)), 1)],
        "worst_tau_mean":  round(float(np.mean(taus)), 3),
        "worst_tau_se":    round(float(np.std(taus, ddof=1)), 3),
        "worst_tau_ci90":  [round(float(np.percentile(taus, 5)), 3),
                            round(float(np.percentile(taus, 95)), 3)],
        "n_resamples":     B,
        "n_tasks_per_cell": n_tasks,
    }


def main() -> None:
    with EXPORT.open() as f:
        all_data = json.load(f)
    print(f"Bootstrap (B={B}) over tasks per cell:\n")
    print(f"{'benchmark':28s} {'maxSw':>9s} {'maxSw 90% CI':>16s} {'wstTau':>8s} {'wstTau 90% CI':>17s}")
    for name, fname in BENCHMARKS.items():
        cells = load_cell_outcomes(OUT / fname)
        # Find n_tasks per cell from the data (assumes balanced)
        n_tasks = max(len(v) for v in cells.values())
        boot = bootstrap(cells, n_tasks)
        # Insert into all_data
        all_data["benchmarks"][name]["bootstrap"] = boot
        # Print row
        print(f"{name:28s} "
              f"{boot['max_swing_mean']:>5.1f} +/-{boot['max_swing_se']:<3.1f} "
              f"[{boot['max_swing_ci90'][0]:>4.1f}, {boot['max_swing_ci90'][1]:>5.1f}] "
              f"{boot['worst_tau_mean']:>5.2f} +/-{boot['worst_tau_se']:<5.3f}  "
              f"[{boot['worst_tau_ci90'][0]:>+5.2f}, {boot['worst_tau_ci90'][1]:>+5.2f}]")

    with EXPORT.open("w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nWrote bootstrap CIs into {EXPORT}")


if __name__ == "__main__":
    main()
