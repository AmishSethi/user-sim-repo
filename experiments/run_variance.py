"""Variance-control experiment.

Runs each diagnostic (simulator, agent) cell K=10 times on the same task
list, then compares within-cell std (across K trials) to cross-simulator std
(across simulator backbones from the original run). The point is to confirm
that the simulator-induced score swing reported in the paper is signal, not
inference-time stochasticity.
"""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path

from llm_client import LLMClient, PANEL_OPENROUTER
from tasks import all_tasks
from dialogue import run_dialogue_llm_user
from run_mint import load_mint_tasks, run_mint_dialogue
from run_agentclinic import load_agentclinic_cases, run_clinic_dialogue


# Diagnostic cells: spread across the score range observed in the original
# run, plus one same-family control per benchmark.
DIAG_CELLS = [
    # bench, simulator, agent
    ("24task", "qwen2.5-7b",   "llama3.1-8b"),     # low cell (38.9%)
    ("24task", "gpt-oss-120b", "llama3.1-8b"),     # high cell (75.0%)
    ("24task", "llama3.3-70b", "gemma3-12b"),      # high cell (88.9%)
    ("24task", "deepseek-r1",  "deepseek-r1"),     # same-family control

    ("mint",   "gemma3-12b",   "gemma3-12b"),      # low cell (30%)
    ("mint",   "gpt-oss-120b", "gpt-oss-120b"),    # high cell (92%)
    ("mint",   "deepseek-r1",  "llama3.1-8b"),     # mid cell (62%)
    ("mint",   "llama3.1-8b",  "gpt-oss-120b"),    # high cell (80%)

    ("clinic", "llama3.1-8b",  "gemma3-12b"),      # low cell (13.3%)
    ("clinic", "gpt-oss-120b", "deepseek-r1"),     # high cell (60%)
    ("clinic", "llama3.1-8b",  "llama3.1-8b"),     # same-family low (23.3%)
    ("clinic", "deepseek-r1",  "llama3.3-70b"),    # mid cell (40%)
]
N_TRIALS = 10
MAX_CONCURRENCY = 12


async def run_one(client, bench, sim, agent, trial, task):
    if bench == "24task":
        r = await run_dialogue_llm_user(client, task, sim, agent, trial)
        return {"bench": bench, "sim": sim, "agent": agent, "trial": trial,
                "task_id": task.task_id, "success": bool(r.success)}
    if bench == "mint":
        r = await run_mint_dialogue(client, task, sim, agent, trial)
        return {"bench": bench, "sim": sim, "agent": agent, "trial": trial,
                "task_id": task["task_id"], "success": bool(r.success)}
    if bench == "clinic":
        r = await run_clinic_dialogue(client, task, sim, agent, trial)
        return {"bench": bench, "sim": sim, "agent": agent, "trial": trial,
                "task_id": task["task_id"], "success": bool(r.success)}
    raise ValueError(bench)


async def main():
    out = Path(__file__).resolve().parent / "results"
    raw = out / "variance_raw.jsonl"

    client = LLMClient(PANEL_OPENROUTER)
    tasks24 = all_tasks()
    tasks_mint = load_mint_tasks()
    tasks_clinic = load_agentclinic_cases()
    print(f"24-task: {len(tasks24)} tasks; MINT: {len(tasks_mint)}; "
          f"AgentClinic: {len(tasks_clinic)}")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def boxed(coro):
        async with sem:
            try:
                return await coro
            except Exception as e:
                return {"error": str(e)}

    bench_tasks = {"24task": tasks24, "mint": tasks_mint, "clinic": tasks_clinic}

    coros = []
    for bench, sim, agent in DIAG_CELLS:
        for trial in range(N_TRIALS):
            for task in bench_tasks[bench]:
                coros.append(boxed(run_one(client, bench, sim, agent, trial, task)))

    print(f"queued {len(coros)} dialogues across {len(DIAG_CELLS)} cells "
          f"x {N_TRIALS} trials")

    t0 = time.time()
    done = 0
    with raw.open("w") as f:
        for task in asyncio.as_completed(coros):
            r = await task
            if isinstance(r, dict) and "error" in r:
                # missing bench/sim/agent context for error rows; we skip them
                # in analysis but log them
                f.write(json.dumps({"error": r["error"]}) + "\n")
            else:
                f.write(json.dumps(r) + "\n")
            done += 1
            if done % 200 == 0:
                pct = 100.0 * done / len(coros)
                print(f"  done {done}/{len(coros)} ({pct:.1f}%) at t={time.time()-t0:.1f}s")
    print(f"finished {done} dialogues in {time.time()-t0:.1f}s")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
