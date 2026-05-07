"""Run tau2-bench telecom on the 6-model OpenRouter panel.

Runs the 6x6 (simulator, agent) matrix on a fixed 5-task subset of
tau2-bench telecom (the first five tasks). Each cell is one tau2
invocation that sweeps the five tasks internally.

Output: experiments/results/tau2_raw.jsonl  (one row per (sim,agent,task))
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TAU2 = ROOT / "experiments" / "external" / "tau2-bench"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "tau2_raw.jsonl"

PANEL = [
    ("llama3.1-8b",  "openrouter/meta-llama/llama-3.1-8b-instruct"),
    ("qwen2.5-7b",   "openrouter/qwen/qwen-2.5-7b-instruct"),
    ("gemma3-12b",   "openrouter/google/gemma-3-12b-it"),
    ("llama3.3-70b", "openrouter/meta-llama/llama-3.3-70b-instruct"),
    ("gpt-oss-120b", "openrouter/openai/gpt-oss-120b"),
    ("deepseek-r1",  "openrouter/deepseek/deepseek-r1"),
]

NUM_TASKS = 5
MAX_STEPS = 40
PER_CELL_TIMEOUT = 1500  # 25 min hard limit per cell


def run_cell(sim_alias: str, sim_model: str,
             agent_alias: str, agent_model: str) -> list[dict]:
    """Run all NUM_TASKS for one (sim, agent) cell. Returns list of per-task dicts."""
    save_path = f"/tmp/tau2_cell_{sim_alias}_{agent_alias}"
    # Clean previous output
    subprocess.run(["rm", "-rf", save_path], check=False)
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "telecom",
        "--task-set-name", "telecom",
        "--agent-llm", agent_model,
        "--user-llm", sim_model,
        "--num-trials", "1",
        "--num-tasks", str(NUM_TASKS),
        "--max-steps", str(MAX_STEPS),
        "--max-concurrency", "5",
        "--save-to", save_path,
        "--log-level", "WARNING",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(TAU2), capture_output=True, text=True,
            timeout=PER_CELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return [{"task_id": f"timeout_{i}", "reward": 0.0,
                 "duration": PER_CELL_TIMEOUT / NUM_TASKS,
                 "termination": "timeout", "n_messages": 0,
                 "success": False}
                for i in range(NUM_TASKS)]
    dur = time.time() - t0
    results_path = Path(save_path) / "results.json"
    if not results_path.exists():
        return [{"task_id": f"missing_{i}", "reward": 0.0,
                 "duration": dur / NUM_TASKS,
                 "termination": "no_results", "n_messages": 0,
                 "success": False, "stderr_tail": proc.stderr[-300:]}
                for i in range(NUM_TASKS)]
    with results_path.open() as f:
        d = json.load(f)
    out = []
    for s in d.get("simulations", []):
        reward = float(s.get("reward_info", {}).get("reward", 0.0))
        out.append({
            "task_id": s.get("task_id"),
            "reward": reward,
            "duration": float(s.get("duration", 0.0)),
            "termination": s.get("termination_reason"),
            "n_messages": len(s.get("messages", [])),
            "success": reward >= 0.5,
        })
    # Pad to NUM_TASKS if short (shouldn't usually happen)
    while len(out) < NUM_TASKS:
        out.append({"task_id": f"short_{len(out)}", "reward": 0.0,
                    "duration": 0.0, "termination": "missing",
                    "n_messages": 0, "success": False})
    return out[:NUM_TASKS]


def main() -> None:
    completed = set()
    if RAW.exists():
        for ln in RAW.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
                completed.add((r["sim_alias"], r["agent_alias"]))
            except Exception:
                pass
    print(f"Resuming with {len(completed)} cells already done.", flush=True)

    cells = [(sa, sm, aa, am)
             for sa, sm in PANEL for aa, am in PANEL]
    cells = [c for c in cells if (c[0], c[2]) not in completed]
    print(f"Running {len(cells)} cells.", flush=True)

    with RAW.open("a") as f:
        for i, (sim_alias, sim_model, agent_alias, agent_model) in enumerate(cells, 1):
            t0 = time.time()
            print(f"[{i}/{len(cells)}] sim={sim_alias:13s} agent={agent_alias:13s} ... ",
                  end="", flush=True)
            tasks = run_cell(sim_alias, sim_model, agent_alias, agent_model)
            cell_time = time.time() - t0
            n_pass = sum(1 for t in tasks if t["success"])
            for t in tasks:
                rec = {
                    "sim_alias": sim_alias,
                    "agent_alias": agent_alias,
                    **t,
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"pass {n_pass}/{NUM_TASKS}  ({cell_time:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
