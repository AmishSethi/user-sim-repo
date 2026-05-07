"""K=10 trial variance experiment for tau2-bench.

Picks four diagnostic (sim, agent) cells spanning the score range,
re-runs each NUM_TASKS task list NUM_TRIALS times. Used to compute the
T1 receipt for tau2.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TAU2 = ROOT / "experiments" / "external" / "tau2-bench"
OUT = ROOT / "experiments" / "results"
RAW = OUT / "tau2_variance_raw.jsonl"

PANEL_LOOKUP = {
    "llama3.1-8b":  "openrouter/meta-llama/llama-3.1-8b-instruct",
    "qwen2.5-7b":   "openrouter/qwen/qwen-2.5-7b-instruct",
    "gemma3-12b":   "openrouter/google/gemma-3-12b-it",
    "llama3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
    "deepseek-r1":  "openrouter/deepseek/deepseek-r1",
}

# 4 diagnostic cells; chosen post-hoc once tau2_summary.json exists
# to span the score range. For now use a placeholder set; the runner
# can be re-pointed once main results land.
DIAG_CELLS = [
    ("llama3.1-8b",  "llama3.1-8b"),
    ("gpt-oss-120b", "llama3.3-70b"),
    ("deepseek-r1",  "deepseek-r1"),
    ("llama3.3-70b", "gpt-oss-120b"),
]
NUM_TRIALS = 10
NUM_TASKS = 5
MAX_STEPS = 40
PER_TRIAL_TIMEOUT = 1500


def run_cell_trial(sim_alias: str, agent_alias: str, trial: int) -> list[dict]:
    save_path = f"/tmp/tau2_var_{sim_alias}_{agent_alias}_t{trial}"
    subprocess.run(["rm", "-rf", save_path], check=False)
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "telecom",
        "--task-set-name", "telecom",
        "--agent-llm", PANEL_LOOKUP[agent_alias],
        "--user-llm", PANEL_LOOKUP[sim_alias],
        "--num-trials", "1",
        "--num-tasks", str(NUM_TASKS),
        "--max-steps", str(MAX_STEPS),
        "--max-concurrency", "5",
        "--seed", str(trial),
        "--save-to", save_path,
        "--log-level", "WARNING",
    ]
    try:
        subprocess.run(cmd, cwd=str(TAU2), capture_output=True,
                       text=True, timeout=PER_TRIAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [{"reward": 0.0, "success": False, "termination": "timeout"}
                for _ in range(NUM_TASKS)]
    rp = Path(save_path) / "results.json"
    if not rp.exists():
        return [{"reward": 0.0, "success": False, "termination": "no_results"}
                for _ in range(NUM_TASKS)]
    with rp.open() as f:
        d = json.load(f)
    out = []
    for s in d.get("simulations", []):
        rew = float(s.get("reward_info", {}).get("reward", 0.0))
        out.append({
            "reward": rew,
            "success": rew >= 0.5,
            "task_id": s.get("task_id"),
            "termination": s.get("termination_reason"),
        })
    while len(out) < NUM_TASKS:
        out.append({"reward": 0.0, "success": False, "termination": "missing"})
    return out[:NUM_TASKS]


def main() -> None:
    completed = set()
    if RAW.exists():
        for ln in RAW.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
                completed.add((r["sim_alias"], r["agent_alias"], r["trial"]))
            except Exception:
                pass
    print(f"Resuming with {len(completed)} cell-trials done.", flush=True)
    total = len(DIAG_CELLS) * NUM_TRIALS
    todo = [(sa, aa, t) for sa, aa in DIAG_CELLS for t in range(NUM_TRIALS)
            if (sa, aa, t) not in completed]
    print(f"Running {len(todo)} cell-trials.", flush=True)
    with RAW.open("a") as f:
        for i, (sa, aa, t) in enumerate(todo, 1):
            t0 = time.time()
            print(f"[{i}/{len(todo)}] sim={sa:13s} agent={aa:13s} trial={t} ... ",
                  end="", flush=True)
            tasks = run_cell_trial(sa, aa, t)
            n_pass = sum(1 for x in tasks if x["success"])
            for tk in tasks:
                rec = {"sim_alias": sa, "agent_alias": aa, "trial": t, **tk}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"pass {n_pass}/{NUM_TASKS}  ({time.time()-t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
