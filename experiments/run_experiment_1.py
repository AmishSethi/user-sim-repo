"""Experiment 1: Simulator-induced ranking flip on a multi-turn customer-service task.

For each (simulator backend, agent backend, trial) we run a dialogue and
deterministically score against the task's oracle. We log results so they can
be aggregated and figure-rendered separately.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from tasks import all_tasks
from llm_client import LLMClient, ServerConfig
from dialogue import run_dialogue_llm_user, run_dialogue_dsl_user, DialogueResult


from llm_client import PANEL_OPENROUTER, PANEL_ALIASES

# Six-model panel (3 small open-weight + 3 medium/large), all via OpenRouter.
SERVERS = PANEL_OPENROUTER

# Simulators include the DSL agenda baseline plus all six LLMs.
SIM_ALIASES   = PANEL_ALIASES + ["dsl"]
# Agents are all six LLMs (DSL is simulator-only by design).
AGENT_ALIASES = PANEL_ALIASES
TRIALS = 3
MAX_CONCURRENCY = 12


async def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "exp1_v2_raw.jsonl"

    client = LLMClient(SERVERS)
    # Wait until each LLM server is ready
    print("waiting for LLM servers ...")
    for s in SERVERS:
        for _ in range(120):
            ok = await client.health(s.alias)
            if ok:
                print(f"  server {s.alias} ready")
                break
            await asyncio.sleep(2.0)
        else:
            raise RuntimeError(f"server {s.alias} did not come up in time")

    tasks_list = all_tasks()
    print(f"loaded {len(tasks_list)} tasks")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    coros = []

    async def run_one(task, sim_alias, agent_alias, trial):
        async with sem:
            try:
                if sim_alias == "dsl":
                    result = await run_dialogue_dsl_user(client, task, agent_alias, trial,
                                                         seed=trial * 1009)
                else:
                    result = await run_dialogue_llm_user(client, task, sim_alias, agent_alias, trial)
                return result
            except Exception as e:
                return DialogueResult(task_id=task.task_id, sim_alias=sim_alias,
                                      agent_alias=agent_alias, trial=trial, turns=0,
                                      success=False, predicted=None,
                                      transcript=[{"role": "error", "content": str(e)}])

    for task in tasks_list:
        for sim in SIM_ALIASES:
            for agent in AGENT_ALIASES:
                for trial in range(TRIALS):
                    coros.append(run_one(task, sim, agent, trial))

    print(f"queued {len(coros)} dialogues")
    t0 = time.time()
    results: list[DialogueResult] = []
    done = 0
    with raw_path.open("w") as f:
        for coro in asyncio.as_completed(coros):
            r = await coro
            results.append(r)
            done += 1
            if done % 25 == 0:
                pct = 100.0 * done / len(coros)
                print(f"  done {done}/{len(coros)} ({pct:.1f}%) at t={time.time()-t0:.1f}s")
            f.write(json.dumps(asdict(r)) + "\n")

    print(f"finished {len(results)} dialogues in {time.time()-t0:.1f}s")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
