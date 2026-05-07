"""A small DSL-style agenda-based user simulator.

The simulator is parameterized by a *frame* that captures the user's *goal*,
*known slots* (private info), and a *reveal policy*. Each turn the simulator
either (a) reveals a slot the agent has asked for, (b) asserts the goal in
short form if it has not yet been stated, (c) offers a brief acknowledgement,
or (d) ends the conversation when the goal is satisfied.

Compared to an unconstrained LLM-as-user the DSL simulator is deliberately
brittle and stochastic in well-defined ways: there is no sycophancy, no
volunteered information beyond the persona-specific reveal policy, and a
small probability of paraphrasing each utterance. We use it as a
verifiability baseline in Experiment 1.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _paraphrase(s: str, rng: random.Random) -> str:
    """Apply a tiny set of stochastic surface variations."""
    if rng.random() < 0.35:
        # add a short hedge
        s = rng.choice(["", "Sure, ", "Okay, ", "Sorry, ", "Hi, "]) + s
    if rng.random() < 0.20:
        s = s.replace(",", "")
    if rng.random() < 0.15 and not s.endswith("?"):
        s = s + " thanks"
    return s.strip()


@dataclass
class DSLSimulator:
    persona: str
    goal: str
    private_info: Dict[str, str]
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    revealed: Dict[str, bool] = field(default_factory=dict)
    initial: Optional[str] = None
    asserted_goal: bool = False
    done: bool = False

    def __post_init__(self) -> None:
        for k in self.private_info:
            self.revealed.setdefault(k, False)

    def _slot_match(self, agent_text: str, slot: str) -> bool:
        # Heuristic: the agent asks for `slot` if it mentions a paraphrase of
        # the slot key. We accept many surface forms.
        agent_lower = agent_text.lower()
        synonyms = {
            "name": ["name", "who am i speaking", "who is this", "your full name"],
            "booking_ref": ["booking", "reservation", "reference", "confirmation", "pnr"],
            "flight": ["flight number", "flight"],
            "date": ["date", "when", "what day"],
            "card_last4": ["card", "last four", "last 4", "ending in"],
            "email": ["email", "e-mail"],
            "phone": ["phone", "number"],
            "order_id": ["order", "order number", "order id"],
            "rma_id": ["rma", "return number", "return id"],
            "invoice_id": ["invoice", "invoice number"],
            "promo": ["promo", "promotion", "code"],
            "new_card_last4": ["new card", "new last four", "new card last"],
            "missing_item": ["which item", "what was missing"],
            "to_size": ["what size", "size you want", "new size"],
            "from_size": ["current size", "size you have"],
            "primary_email": ["primary email", "older email", "email to keep"],
            "duplicate_email": ["other email", "duplicate"],
            "duplicate_amount": ["amount", "how much"],
            "disputed_amount": ["amount", "how much"],
            "month": ["which month", "what month"],
            "previous_cancellation": ["cancelled", "cancellation"],
            "refund_status": ["status", "refund status"],
            "current_plan": ["current plan", "plan you're on"],
            "target_plan": ["plan you want", "downgrade to", "upgrade to"],
            "discount_amount": ["amount", "how much"],
            "seat_class_current": ["class", "current class"],
            "route": ["route", "from where", "destination"],
            "old_date": ["original date", "current date"],
            "new_date": ["new date", "what date"],
            "reason": ["reason", "why", "what happened"],
            "item": ["which item", "what is the item", "what item"],
        }
        for syn in synonyms.get(slot, [slot]):
            if syn in agent_lower:
                return True
        return False

    def initial_message(self) -> str:
        if self.initial is None:
            self.initial = "Hi, I need help."
        return _paraphrase(self.initial, self.rng)

    def respond(self, dialogue_history: List[Dict[str, str]], last_agent: str) -> str:
        # Decide what to do based on the agent's last turn.
        if any(token in last_agent.lower() for token in ["all set", "anything else", "is there anything else", "have a great"]):
            self.done = True
            return _paraphrase("No, that's all. Thank you.", self.rng)

        if "?" in last_agent or any(w in last_agent.lower() for w in ["could you", "can you", "may i", "what is", "what's", "please provide", "share", "tell me"]):
            for slot, value in self.private_info.items():
                if not self.revealed.get(slot, False) and self._slot_match(last_agent, slot):
                    self.revealed[slot] = True
                    return _paraphrase(f"{value}.", self.rng)
            # asked something we don't have a direct match for - say yes / give the goal
            if not self.asserted_goal:
                self.asserted_goal = True
                return _paraphrase(self.goal + ".", self.rng)
            # offer the next unrevealed slot
            for slot, value in self.private_info.items():
                if not self.revealed.get(slot, False):
                    self.revealed[slot] = True
                    return _paraphrase(f"{slot.replace('_',' ')}: {value}.", self.rng)
            return _paraphrase("Yes.", self.rng)

        # Agent didn't ask a question - either confirm or wait.
        if not self.asserted_goal:
            self.asserted_goal = True
            return _paraphrase(self.goal + ".", self.rng)
        return _paraphrase("Okay.", self.rng)

    def is_done(self) -> bool:
        return self.done
