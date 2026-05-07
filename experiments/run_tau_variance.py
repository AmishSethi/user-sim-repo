"""K=10 trial variance experiment for tau-bench (airline) and
tau2-bench (telecom). For each benchmark we pick four diagnostic
(simulator, agent) cells spanning the score range observed in the
6x6 sweep, and re-run each NUM_TASKS task list NUM_TRIALS times.
Output: experiments/results/tau_airline_variance_raw.jsonl and
        experiments/results/tau2_telecom_variance_raw.jsonl
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TAU2 = ROOT / "experiments" / "external" / "tau2-bench"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_LOOKUP = {
    "llama3.1-8b":  "openrouter/meta-llama/llama-3.1-8b-instruct",
    "qwen2.5-7b":   "openrouter/qwen/qwen-2.5-7b-instruct",
    "gemma3-12b":   "openrouter/google/gemma-3-12b-it",
    "llama3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
    "deepseek-r1":  "openrouter/deepseek/deepseek-r1",
}

# Diagnostic cells span the score range from the main sweep.
# Each entry: (sim_alias, agent_alias, expected_pct_for_reference)
DIAG_CELLS = {
    "airline": [
        ("gemma3-12b",   "qwen2.5-7b",     0),    # floor
        ("deepseek-r1",  "deepseek-r1",   60),    # mid
        ("gpt-oss-120b", "gpt-oss-120b",  80),    # mid-high
        ("llama3.3-70b", "gpt-oss-120b", 100),    # ceiling
    ],
    "telecom": [
        ("llama3.1-8b",  "llama3.1-8b",    0),    # floor
        ("deepseek-r1",  "deepseek-r1",   10),    # low
        ("llama3.3-70b", "gpt-oss-120b",  70),    # mid-high
        ("gpt-oss-120b", "gpt-oss-120b",  95),    # ceiling
    ],
}

NUM_TRIALS = 10
NUM_TASKS = 5
PER_TRIAL_TIMEOUT = 1500


def run_cell_trial(sim_alias: str, agent_alias: str, trial: int,
                   domain: str, max_steps: int) -> list[dict]:
    save_path = f"/tmp/tauvar_{domain}_{sim_alias}_{agent_alias}_t{trial}"
    subprocess.run(["rm", "-rf", save_path], check=False)
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", domain,
        "--task-set-name", domain,
        "--agent-llm", PANEL_LOOKUP[agent_alias],
        "--user-llm", PANEL_LOOKUP[sim_alias],
        "--num-trials", "1",
        "--num-tasks", str(NUM_TASKS),
        "--max-steps", str(max_steps),
        "--max-concurrency", "5",
        "--seed", str(trial),
        "--save-to", save_path,
        "--log-level", "WARNING",
    ]
    try:
        subprocess.run(cmd, cwd=str(TAU2), capture_output=True,
                       text=True, timeout=PER_TRIAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [{"sim_alias": sim_alias, "agent_alias": agent_alias,
                 "trial": trial, "task_id": f"timeout_{i}",
                 "reward": 0.0, "success": False,
                 "termination": "timeout"} for i in range(NUM_TASKS)]
    rp = Path(save_path) / "results.json"
    out = []
    if rp.exists():
        with rp.open() as f:
            d = json.load(f)
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            rew = float(ri.get("reward", 0.0) or 0.0)
            out.append({
                "sim_alias": sim_alias,
                "agent_alias": agent_alias,
                "trial": trial,
                "task_id": s.get("task_id"),
                "reward": rew,
                "success": rew >= 0.5,
                "termination": s.get("termination_reason"),
            })
    while len(out) < NUM_TASKS:
        out.append({"sim_alias": sim_alias, "agent_alias": agent_alias,
                    "trial": trial, "task_id": f"missing_{len(out)}",
                    "reward": 0.0, "success": False,
                    "termination": "missing"})
    return out[:NUM_TASKS]


def main(domain: str, raw_filename: str, max_steps: int,
         parallel_workers: int) -> None:
    raw = OUT / raw_filename
    completed = set()
    if raw.exists():
        for ln in raw.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            completed.add((r["sim_alias"], r["agent_alias"], r["trial"]))
    diag = DIAG_CELLS[domain]
    todo = [(sa, aa, t) for sa, aa, _ in diag for t in range(NUM_TRIALS)
            if (sa, aa, t) not in completed]
    print(f"[{domain}] resuming with {len(completed)} cell-trials done; "
          f"running {len(todo)} more with {parallel_workers} workers",
          flush=True)

    t_global = time.time()
    finished = 0
    with raw.open("a") as f, ProcessPoolExecutor(max_workers=parallel_workers) as ex:
        futures = {ex.submit(run_cell_trial, sa, aa, t, domain, max_steps): (sa, aa, t)
                   for sa, aa, t in todo}
        for fut in as_completed(futures):
            sa, aa, t = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[ERR] {domain} sim={sa} agent={aa} trial={t}: {e}",
                      flush=True)
                continue
            n_pass = sum(1 for r in rows if r["success"])
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            finished += 1
            elapsed = time.time() - t_global
            print(f"[{domain}][{finished}/{len(todo)}] sim={sa:13s} "
                  f"agent={aa:13s} trial={t:2d} pass={n_pass}/{NUM_TASKS}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["airline", "telecom"])
    args = parser.parse_args()
    if args.domain == "airline":
        main("airline", "tau_airline_variance_raw.jsonl",
             max_steps=40, parallel_workers=4)
    else:
        main("telecom", "tau2_telecom_variance_raw.jsonl",
             max_steps=80, parallel_workers=4)
