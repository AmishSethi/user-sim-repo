"""Run a multi-turn dialogue between an agent and a (LLM or DSL) simulator,
then deterministically score the outcome against the task's oracle."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from tasks import Task
from llm_client import LLMClient
from dsl_sim import DSLSimulator


SYSTEM_AGENT = """You are a customer-service agent. You help customers with their requests.

CRITICAL RULES:
- NEVER make up or invent information. Do not assume names, emails, IDs, or any other details.
- Ask the customer for ANY information you do not have, one piece at a time.
- Issue a tool call ONLY when you have all required arguments from the customer.
- When you finally have everything, issue exactly one tool call in this exact format on its own:

ACTION: <action_name>
ARGS: <valid JSON with arguments>

While gathering information, ask one short clear question (under 25 words). Do not include the ACTION/ARGS lines until you have what you need.
Available actions and their arguments are listed below. Choose the single action that completes the task.

actions:
- cancel_and_refund(booking_ref, refund_method)
- rebook(booking_ref, new_date)
- add_baggage(booking_ref, bags)
- lookup_refund_status(booking_ref)
- upgrade_seat(booking_ref, target_class)
- resend_receipt(booking_ref, email)
- initiate_return(order_id, reason, refund_to)
- initiate_exchange(order_id, to_size)
- lookup_rma(rma_id)
- cancel_order(order_id)
- price_match_credit(order_id, credit)
- reship_missing_item(order_id, item)
- send_password_reset(email)
- update_phone(email, new_phone)
- enable_2fa(email, method, phone)
- merge_accounts(keep, merge)
- delete_account(email, confirm)
- list_active_sessions(email)
- open_dispute(invoice_id, amount)
- schedule_downgrade(email, target_plan)
- update_card(email, new_card_last4)
- email_invoice(email, month)
- apply_promo(email, code)
- refund_duplicate(email, amount)
"""


SYSTEM_LLM_USER = """You are role-playing a customer in a customer service chat. Stay in character.

Persona: {persona}
Your goal: {goal}
Private info you may share if and only if asked:
{private_info_json}

Rules:
- Speak briefly (under 30 words per turn).
- Reveal information only when the agent asks for it.
- If the agent solves your goal, thank them and end the conversation.
- Do not break character. Do not reveal your goal verbatim unless asked or until you have to.
"""


def _parse_agent_action(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"ACTION:\s*(\w+)\s*\nARGS:\s*(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    name = m.group(1).strip()
    raw = m.group(2).strip()
    try:
        # Be permissive about minor JSON quirks
        args = json.loads(raw)
    except Exception:
        # try to recover via single-quote -> double-quote
        try:
            args = json.loads(raw.replace("'", '"'))
        except Exception:
            return None
    return {"action": name, "args": args}


def _normalize(v: Any) -> Any:
    if isinstance(v, str):
        # lowercase, collapse separators, drop common boilerplate suffixes
        s = v.strip().lower().replace("-", "_").replace(" ", "_")
        for suffix in ("_method", "_on_file", "_payment_method"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s
    return v


def _values_match(predicted: Any, expected: Any) -> bool:
    """Match predicted vs expected after normalization. For strings, also
    accept bidirectional substring containment so that, e.g., the oracle
    'original_card' accepts 'original_payment_method' / 'original_card_on_file'.
    Numeric tolerance is exact except for floats, which use a 0.01 tolerance."""
    if predicted is None:
        return False
    if isinstance(expected, bool):
        return bool(predicted) == expected
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        if isinstance(expected, float) or isinstance(predicted, float):
            return abs(float(predicted) - float(expected)) < 0.01
        return int(predicted) == int(expected)
    pn = _normalize(predicted)
    en = _normalize(expected)
    if pn == en:
        return True
    if isinstance(pn, str) and isinstance(en, str):
        # bidirectional substring containment, after dropping common stop tokens
        if en in pn or pn in en:
            return True
    return False


def _action_satisfies(predicted: Dict[str, Any], task: Task) -> bool:
    if predicted is None:
        return False
    if predicted.get("action") != task.oracle_action:
        return False
    pargs = predicted.get("args", {})
    for k, v in task.oracle_args.items():
        if k not in pargs:
            return False
        if not _values_match(pargs[k], v):
            return False
    return True


@dataclass
class DialogueResult:
    task_id: str
    sim_alias: str
    agent_alias: str
    trial: int
    turns: int
    success: bool
    predicted: Optional[Dict[str, Any]]
    transcript: List[Dict[str, str]]


async def run_dialogue_llm_user(client: LLMClient, task: Task,
                                sim_alias: str, agent_alias: str,
                                trial: int, max_turns: int = 8,
                                temperature: float = 0.7) -> DialogueResult:
    private_info_json = json.dumps(task.private_info, ensure_ascii=False)
    sim_system = SYSTEM_LLM_USER.format(persona=task.persona, goal=task.goal,
                                         private_info_json=private_info_json)

    sim_messages: List[Dict[str, str]] = [{"role": "system", "content": sim_system}]
    agent_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_AGENT}]
    transcript: List[Dict[str, str]] = []

    # First user message
    user_msg = task.initial_user_message
    transcript.append({"role": "user", "content": user_msg})

    predicted: Optional[Dict[str, Any]] = None
    turns = 0
    for t in range(max_turns):
        turns += 1
        # Agent reply
        agent_messages.append({"role": "user", "content": user_msg})
        agent_reply = await client.chat(agent_alias, agent_messages,
                                        max_tokens=600, temperature=temperature)
        agent_messages.append({"role": "assistant", "content": agent_reply})
        transcript.append({"role": "assistant", "content": agent_reply})

        predicted = _parse_agent_action(agent_reply)
        if predicted is not None:
            break

        # Simulator reply
        sim_messages.append({"role": "user", "content": agent_reply})
        sim_reply = await client.chat(sim_alias, sim_messages,
                                      max_tokens=400, temperature=temperature)
        sim_messages.append({"role": "assistant", "content": sim_reply})
        transcript.append({"role": "user", "content": sim_reply})
        user_msg = sim_reply

        if any(p in sim_reply.lower() for p in ["that's all", "thank you", "no thanks", "all set", "we're good"]):
            # let the agent close out one more turn
            agent_messages.append({"role": "user", "content": sim_reply})
            agent_reply = await client.chat(agent_alias, agent_messages,
                                            max_tokens=600, temperature=temperature)
            transcript.append({"role": "assistant", "content": agent_reply})
            predicted = _parse_agent_action(agent_reply)
            break

    success = _action_satisfies(predicted, task)
    return DialogueResult(task_id=task.task_id, sim_alias=sim_alias, agent_alias=agent_alias,
                          trial=trial, turns=turns, success=success,
                          predicted=predicted, transcript=transcript)


async def run_dialogue_dsl_user(client: LLMClient, task: Task,
                                agent_alias: str, trial: int,
                                max_turns: int = 8,
                                temperature: float = 0.7,
                                seed: int = 0) -> DialogueResult:
    import random
    sim = DSLSimulator(persona=task.persona, goal=task.goal,
                       private_info={k: str(v) for k, v in task.private_info.items()},
                       rng=random.Random(seed),
                       initial=task.initial_user_message)

    agent_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_AGENT}]
    transcript: List[Dict[str, str]] = []

    user_msg = sim.initial_message()
    transcript.append({"role": "user", "content": user_msg})
    predicted: Optional[Dict[str, Any]] = None
    turns = 0
    for t in range(max_turns):
        turns += 1
        agent_messages.append({"role": "user", "content": user_msg})
        agent_reply = await client.chat(agent_alias, agent_messages,
                                        max_tokens=600, temperature=temperature)
        agent_messages.append({"role": "assistant", "content": agent_reply})
        transcript.append({"role": "assistant", "content": agent_reply})

        predicted = _parse_agent_action(agent_reply)
        if predicted is not None:
            break
        user_msg = sim.respond(transcript, agent_reply)
        transcript.append({"role": "user", "content": user_msg})
        if sim.is_done():
            agent_messages.append({"role": "user", "content": user_msg})
            agent_reply = await client.chat(agent_alias, agent_messages,
                                            max_tokens=600, temperature=temperature)
            transcript.append({"role": "assistant", "content": agent_reply})
            predicted = _parse_agent_action(agent_reply)
            break

    success = _action_satisfies(predicted, task)
    return DialogueResult(task_id=task.task_id, sim_alias="dsl", agent_alias=agent_alias,
                          trial=trial, turns=turns, success=success,
                          predicted=predicted, transcript=transcript)
