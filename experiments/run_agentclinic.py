"""AgentClinic-subset experiment for the position paper.

Loads cases from agentclinic_medqa.jsonl (USMLE-style cases). Runs each case as
a doctor-patient dialogue: the doctor (agent) asks questions, the patient
(simulator) responds based on the case data, and after a fixed number of turns
the doctor must commit to a diagnosis. The diagnosis is matched against the
case's Correct_Diagnosis. By swapping the patient simulator across the same
panel of LLMs we use in the 24-task and MINT experiments, we measure whether
the same doctor's success rate is invariant to which LLM is playing patient.
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

DATA_PATH = Path(__file__).resolve().parent / "external" / "AgentClinic/agentclinic_medqa.jsonl"
N_CASES = 30  # subset size


def load_agentclinic_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with DATA_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= N_CASES:
                break
            d = json.loads(line)
            osce = d["OSCE_Examination"]
            patient = osce["Patient_Actor"]
            out.append({
                "task_id": f"medqa_{i:03d}",
                "objective": osce["Objective_for_Doctor"],
                "demographics": patient["Demographics"],
                "history": patient["History"],
                "symptoms": patient["Symptoms"],
                "past_history": patient.get("Past_Medical_History", "Unknown"),
                "social_history": patient.get("Social_History", "Unknown"),
                "review": patient.get("Review_of_Systems", "Unknown"),
                "physical_exam": osce.get("Physical_Examination_Findings", {}),
                "test_results": osce.get("Test_Results", {}),
                "correct_diagnosis": osce["Correct_Diagnosis"],
            })
    return out


# ---------- system prompts ----------

SYSTEM_DOCTOR = """You are a physician taking a clinical history. You can ask the patient questions and you can request physical exam or test results. After at most {max_turns} questions, you must commit to a diagnosis.

When you are ready to commit, emit on its own line:

DIAGNOSIS: <your single most likely diagnosis>

Keep each question short (under 25 words). Do not lecture. Do not ask multiple questions at once."""


SYSTEM_PATIENT = """You are a patient in a clinical encounter. Stay strictly in character. Use the following case information to answer the doctor's questions truthfully but only when asked. Speak in first person, briefly (under 25 words).

Demographics: {demographics}
Brief history: {history}
Primary symptom: {primary}
Other symptoms: {secondary}
Past medical history: {past_history}
Social history: {social_history}
Review of systems: {review}

Rules:
- Do not volunteer information the doctor did not ask about.
- Do not state the diagnosis. You do not know it.
- If asked about something not in your case, answer plausibly with "I don't know" or "no" as appropriate.
- Stay in character. Do not break role."""


# ---------- scoring (lenient, name-based) ----------

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # drop common modifier words
    for stop in ("disorder", "disease", "syndrome", "condition", "type", "primary"):
        s = re.sub(rf"\b{stop}\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


DIAG_RE = re.compile(r"DIAGNOSIS\s*:\s*([^\n]+)", re.IGNORECASE)


def _extract_diagnosis(text: str) -> Optional[str]:
    matches = DIAG_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip().rstrip(".").strip()


def _score(predicted: Optional[str], reference: str) -> bool:
    """Lenient match: normalized substring containment in either direction.
    For a strict eval one would have a clinician adjudicate; this approximation
    is enough for the simulator-swap variance question."""
    if predicted is None:
        return False
    p = _norm(predicted)
    r = _norm(reference)
    if not p or not r:
        return False
    if p == r:
        return True
    # token-overlap heuristic: at least 60% of reference tokens appear in predicted
    rt = set(r.split())
    pt = set(p.split())
    if not rt:
        return False
    overlap = len(rt & pt) / len(rt)
    return overlap >= 0.6


# ---------- single dialogue ----------

@dataclass
class ClinicResult:
    task_id: str
    sim_alias: str
    agent_alias: str
    trial: int
    turns: int
    success: bool
    diagnosis: Optional[str]
    reference: str
    transcript: List[Dict[str, str]]


async def run_clinic_dialogue(client: LLMClient, case: Dict[str, Any],
                               sim_alias: str, agent_alias: str, trial: int,
                               max_turns: int = 6, temperature: float = 0.7) -> ClinicResult:
    sim_system = SYSTEM_PATIENT.format(
        demographics=case["demographics"],
        history=case["history"],
        primary=case["symptoms"].get("Primary_Symptom", "Unknown"),
        secondary=", ".join(case["symptoms"].get("Secondary_Symptoms", [])),
        past_history=case["past_history"],
        social_history=case["social_history"],
        review=case["review"],
    )
    agent_system = SYSTEM_DOCTOR.format(max_turns=max_turns)

    sim_messages: List[Dict[str, str]] = [{"role": "system", "content": sim_system}]
    agent_messages: List[Dict[str, str]] = [{"role": "system", "content": agent_system}]
    transcript: List[Dict[str, str]] = []

    # Doctor opens.
    opener = f"Hello, I'm the physician seeing you today. {case['objective']} What brings you in?"
    transcript.append({"role": "assistant", "content": opener})
    agent_messages.append({"role": "assistant", "content": opener})

    diagnosis: Optional[str] = None
    turns = 0
    user_msg: Optional[str] = None

    # First the patient responds to the opener.
    sim_messages.append({"role": "user", "content": opener})
    patient_reply = await client.chat(sim_alias, sim_messages, max_tokens=200,
                                      temperature=temperature)
    sim_messages.append({"role": "assistant", "content": patient_reply})
    transcript.append({"role": "user", "content": patient_reply})
    user_msg = patient_reply

    for t in range(max_turns):
        turns += 1
        agent_messages.append({"role": "user", "content": user_msg})
        agent_reply = await client.chat(agent_alias, agent_messages,
                                        max_tokens=400, temperature=temperature)
        agent_messages.append({"role": "assistant", "content": agent_reply})
        transcript.append({"role": "assistant", "content": agent_reply})

        diagnosis = _extract_diagnosis(agent_reply)
        if diagnosis is not None:
            break

        # Patient replies.
        sim_messages.append({"role": "user", "content": agent_reply})
        patient_reply = await client.chat(sim_alias, sim_messages, max_tokens=200,
                                          temperature=temperature)
        sim_messages.append({"role": "assistant", "content": patient_reply})
        transcript.append({"role": "user", "content": patient_reply})
        user_msg = patient_reply

    # If the doctor never committed, ask once more for a diagnosis.
    if diagnosis is None:
        agent_messages.append({"role": "user", "content":
            "(Time is up. Please commit to your single most likely diagnosis using DIAGNOSIS: ...)"})
        agent_reply = await client.chat(agent_alias, agent_messages,
                                        max_tokens=200, temperature=0.0)
        transcript.append({"role": "assistant", "content": agent_reply})
        diagnosis = _extract_diagnosis(agent_reply)

    success = _score(diagnosis, case["correct_diagnosis"])
    return ClinicResult(task_id=case["task_id"], sim_alias=sim_alias,
                        agent_alias=agent_alias, trial=trial, turns=turns,
                        success=success, diagnosis=diagnosis,
                        reference=case["correct_diagnosis"], transcript=transcript)


# ---------- driver ----------

SIM_ALIASES = PANEL_ALIASES
AGENT_ALIASES = PANEL_ALIASES
TRIALS = 1
MAX_CONCURRENCY = 12


async def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "agentclinic_raw.jsonl"

    client = LLMClient(PANEL_OPENROUTER)
    cases = load_agentclinic_cases()
    print(f"loaded {len(cases)} AgentClinic MedQA cases")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(case, sim, agent, trial):
        async with sem:
            try:
                return await run_clinic_dialogue(client, case, sim, agent, trial)
            except Exception as e:
                return ClinicResult(task_id=case["task_id"], sim_alias=sim,
                                    agent_alias=agent, trial=trial, turns=0,
                                    success=False, diagnosis=None,
                                    reference=case["correct_diagnosis"],
                                    transcript=[{"role": "error", "content": str(e)}])

    coros = []
    for case in cases:
        for sim in SIM_ALIASES:
            for agent in AGENT_ALIASES:
                for trial in range(TRIALS):
                    coros.append(run_one(case, sim, agent, trial))
    print(f"queued {len(coros)} AgentClinic dialogues")

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
