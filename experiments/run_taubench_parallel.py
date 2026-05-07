"""Run original tau-bench (Yao et al. 2024) airline domain on the
6-model OpenRouter panel. Same parallel structure as run_tau2_parallel.py.

Output: experiments/results/taubench_raw.jsonl
"""
from __future__ import annotations
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TAU2 = ROOT / "experiments" / "external" / "tau2-bench"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "taubench_raw.jsonl"

PANEL = [
    ("llama3.1-8b",  "openrouter/meta-llama/llama-3.1-8b-instruct"),
    ("qwen2.5-7b",   "openrouter/qwen/qwen-2.5-7b-instruct"),
    ("gemma3-12b",   "openrouter/google/gemma-3-12b-it"),
    ("llama3.3-70b", "openrouter/meta-llama/llama-3.3-70b-instruct"),
    ("gpt-oss-120b", "openrouter/openai/gpt-oss-120b"),
    ("deepseek-r1",  "openrouter/deepseek/deepseek-r1"),
]

DOMAIN = "airline"
NUM_TASKS = 5
MAX_STEPS = 40
PARALLEL_CELLS = 4  # smaller pool to avoid colliding with the tau2 job
PER_CELL_TIMEOUT = 1800


def run_cell(args: tuple[str, str, str, str]) -> list[dict]:
    sim_alias, sim_model, agent_alias, agent_model = args
    save_path = f"/tmp/tau_{DOMAIN}_{sim_alias}_{agent_alias}"
    subprocess.run(["rm", "-rf", save_path], check=False)
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", DOMAIN,
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
        return [{"sim_alias": sim_alias, "agent_alias": agent_alias,
                 "task_id": f"timeout_{i}", "reward": 0.0,
                 "duration": PER_CELL_TIMEOUT / NUM_TASKS,
                 "termination": "timeout", "n_messages": 0,
                 "success": False}
                for i in range(NUM_TASKS)]
    dur = time.time() - t0
    rp = Path(save_path) / "results.json"
    out = []
    if rp.exists():
        with rp.open() as f:
            d = json.load(f)
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            rew = float(ri.get("reward", 0.0) or 0.0)
            msgs = s.get("messages") or []
            out.append({
                "sim_alias": sim_alias,
                "agent_alias": agent_alias,
                "task_id": s.get("task_id"),
                "reward": rew,
                "duration": float(s.get("duration", 0.0) or 0.0),
                "termination": s.get("termination_reason"),
                "n_messages": len(msgs),
                "success": rew >= 0.5,
            })
    while len(out) < NUM_TASKS:
        out.append({"sim_alias": sim_alias, "agent_alias": agent_alias,
                    "task_id": f"missing_{len(out)}", "reward": 0.0,
                    "duration": dur / NUM_TASKS, "termination": "missing",
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
    cells = [(sa, sm, aa, am) for sa, sm in PANEL for aa, am in PANEL]
    cells = [c for c in cells if (c[0], c[2]) not in completed]
    print(f"Resuming with {len(completed)} cells done. Running {len(cells)} more "
          f"with {PARALLEL_CELLS} parallel workers (domain={DOMAIN}).", flush=True)

    t_global = time.time()
    finished = 0
    with RAW.open("a") as f, ProcessPoolExecutor(max_workers=PARALLEL_CELLS) as ex:
        futures = {ex.submit(run_cell, c): c for c in cells}
        for fut in as_completed(futures):
            c = futures[fut]
            sim_alias, _, agent_alias, _ = c
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[ERR] sim={sim_alias} agent={agent_alias}: {e}",
                      flush=True)
                continue
            n_pass = sum(1 for r in rows if r["success"])
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            finished += 1
            elapsed = time.time() - t_global
            print(f"[{finished}/{len(cells)}] sim={sim_alias:13s} "
                  f"agent={agent_alias:13s} pass={n_pass}/{NUM_TASKS}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
