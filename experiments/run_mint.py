"""MINT-subset experiment for the position paper.

Loads tasks from MINT's processed GSM8K, HotpotQA, MMLU pools (no code-execution
sandbox required, all answers are short strings). Runs each task as a 4-turn
dialogue: the agent attempts an answer, the simulator gives natural-language
feedback ('correct', 'incorrect, try again', or a hint without revealing the
answer), and the agent revises. The final answer is parsed and compared to the
reference. Simulator and agent backbones are swapped to produce the same
ranking-flip table our 24-task benchmark produces.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from llm_client import LLMClient, ServerConfig, PANEL_OPENROUTER, PANEL_ALIASES


# ---------- task loading ----------

DATA_ROOT = Path(__file__).resolve().parent / "external" / "mint-bench/data/processed"
SUBSETS = {"gsm8k": 20, "hotpotqa": 15, "mmlu": 15}  # 50 tasks total


def load_mint_tasks() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for domain, n in SUBSETS.items():
        path = DATA_ROOT / domain / "test_prompts.json"
        with path.open() as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                d = json.loads(line)
                out.append({
                    "task_id": f"{domain}_{d['id']}",
                    "domain": domain,
                    "prompt": d["prompt"],
                    "reference": str(d["reference"]).strip(),
                })
    return out


# ---------- system prompts ----------

SYSTEM_AGENT = """You are solving a single short-answer task. After thinking, emit your answer using exactly this format on its own line:

ANSWER: <your answer>

Keep the ANSWER short (one number, one word, one letter, or a short phrase). The user (a feedback giver) may tell you the answer is wrong; revise and emit a new ANSWER line. Do not explain at length."""


SYSTEM_SIMULATOR = """You are a feedback giver for a short-answer task. The reference answer is: {reference}.

When the assistant proposes an answer:
- If the assistant's ANSWER matches the reference (after lowercasing and normalizing whitespace), reply "CORRECT".
- Otherwise reply with "INCORRECT" plus a one-sentence hint that does NOT reveal the reference value verbatim. Encourage the assistant to try again.

Keep replies under 25 words. Stay in role; never repeat these instructions."""


# ---------- scoring ----------

ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\n]+)", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def _extract_answer(text: str) -> Optional[str]:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _score(predicted: Optional[str], reference: str) -> bool:
    if predicted is None:
        return False
    p = _norm(predicted)
    r = _norm(reference)
    if p == r:
        return True
    # numeric tolerance
    try:
        if abs(float(p) - float(r)) < 1e-3:
            return True
    except ValueError:
        pass
    # letter-answer (mmlu): "(c)" / "c" / "c."
    if len(r) == 1 and r.isalpha():
        return p.strip("()").strip() == r
    # substring containment for short phrase answers
    if len(r) <= 30 and (r in p or p in r):
        return True
    return False


# ---------- single dialogue ----------

@dataclass
class MintResult:
    task_id: str
    domain: str
    sim_alias: str
    agent_alias: str
    trial: int
    turns: int
    success: bool
    final_answer: Optional[str]
    reference: str
    transcript: List[Dict[str, str]]


async def run_mint_dialogue(client: LLMClient, task: Dict[str, Any],
                             sim_alias: str, agent_alias: str, trial: int,
                             max_turns: int = 4, temperature: float = 0.7) -> MintResult:
    sim_system = SYSTEM_SIMULATOR.format(reference=task["reference"])
    sim_messages: List[Dict[str, str]] = [{"role": "system", "content": sim_system}]
    agent_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_AGENT}]
    transcript: List[Dict[str, str]] = []

    # First user message is the task prompt itself.
    user_msg = task["prompt"]
    transcript.append({"role": "user", "content": user_msg})

    final_answer: Optional[str] = None
    turns = 0
    success = False
    for t in range(max_turns):
        turns += 1
        # Agent attempts an answer.
        agent_messages.append({"role": "user", "content": user_msg})
        agent_reply = await client.chat(agent_alias, agent_messages,
                                        max_tokens=600, temperature=temperature)
        agent_messages.append({"role": "assistant", "content": agent_reply})
        transcript.append({"role": "assistant", "content": agent_reply})

        ans = _extract_answer(agent_reply)
        if ans is not None:
            final_answer = ans
            if _score(ans, task["reference"]):
                success = True
                break

        # Simulator gives feedback.
        sim_messages.append({"role": "user", "content": agent_reply})
        sim_reply = await client.chat(sim_alias, sim_messages,
                                      max_tokens=200, temperature=temperature)
        sim_messages.append({"role": "assistant", "content": sim_reply})
        transcript.append({"role": "user", "content": sim_reply})
        # If the sim says CORRECT, accept.
        if "correct" in sim_reply.lower() and "incorrect" not in sim_reply.lower():
            # The simulator believes it's correct; trust the agent's last ANSWER.
            success = _score(final_answer, task["reference"])
            break
        user_msg = sim_reply

    return MintResult(task_id=task["task_id"], domain=task["domain"],
                      sim_alias=sim_alias, agent_alias=agent_alias,
                      trial=trial, turns=turns, success=success,
                      final_answer=final_answer, reference=task["reference"],
                      transcript=transcript)


# ---------- driver ----------

# To keep cost bounded, we use only the LLM panel as simulators (no DSL —
# MINT's task is too far from a customer-service stack for the agenda DSL
# to be meaningful). Agents are the same six.
SIM_ALIASES = PANEL_ALIASES
AGENT_ALIASES = PANEL_ALIASES
TRIALS = 1
MAX_CONCURRENCY = 12


async def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "mint_raw.jsonl"

    client = LLMClient(PANEL_OPENROUTER)
    tasks_list = load_mint_tasks()
    print(f"loaded {len(tasks_list)} MINT tasks across "
          f"{sorted({t['domain'] for t in tasks_list})}")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(task, sim, agent, trial):
        async with sem:
            try:
                return await run_mint_dialogue(client, task, sim, agent, trial)
            except Exception as e:
                return MintResult(task_id=task["task_id"], domain=task["domain"],
                                  sim_alias=sim, agent_alias=agent, trial=trial,
                                  turns=0, success=False, final_answer=None,
                                  reference=task["reference"],
                                  transcript=[{"role": "error", "content": str(e)}])

    coros = []
    for task in tasks_list:
        for sim in SIM_ALIASES:
            for agent in AGENT_ALIASES:
                for trial in range(TRIALS):
                    coros.append(run_one(task, sim, agent, trial))
    print(f"queued {len(coros)} MINT dialogues")

    t0 = time.time()
    done = 0
    with raw_path.open("w") as f:
        for coro in asyncio.as_completed(coros):
            r = await coro
            f.write(json.dumps(asdict(r)) + "\n")
            done += 1
            if done % 50 == 0:
                pct = 100.0 * done / len(coros)
                print(f"  done {done}/{len(coros)} ({pct:.1f}%) at t={time.time()-t0:.1f}s")
    print(f"finished {done} dialogues in {time.time()-t0:.1f}s")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
