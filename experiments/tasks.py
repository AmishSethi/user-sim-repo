"""Multi-turn customer-service tasks used to test simulator-induced ranking flips.

Each task fixes a user *persona*, *goal*, *private info*, and an *oracle policy*
giving the action(s) the agent should take. The agent does not see the persona
or oracle policy. The simulator is initialized with persona, goal, and private
info. Success is measured by whether the agent ultimately calls the right
action with the right arguments before the dialogue ends.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Task:
    task_id: str
    domain: str
    persona: str
    goal: str
    private_info: Dict[str, Any]
    oracle_action: str
    oracle_args: Dict[str, Any]
    rubric: str
    initial_user_message: str

    def goal_brief(self) -> str:
        return f"{self.persona}\nYou want to: {self.goal}"


def all_tasks() -> List[Task]:
    """Return the curated multi-turn customer-service task set."""
    tasks: List[Task] = []

    # ---------- Airline domain (6 tasks) ----------
    tasks.append(Task(
        task_id="air_001",
        domain="airline",
        persona="You are Ana Ruiz, 34. You speak briefly and politely.",
        goal="cancel your flight UA123 on June 14 and ask for a full refund to original card",
        private_info={"name": "Ana Ruiz", "booking_ref": "X7FQ2L", "flight": "UA123",
                      "date": "2026-06-14", "card_last4": "4421"},
        oracle_action="cancel_and_refund",
        oracle_args={"booking_ref": "X7FQ2L", "refund_method": "original_card"},
        rubric="agent must (1) verify booking_ref X7FQ2L, (2) cancel flight, (3) issue refund to original card",
        initial_user_message="Hi, I need to cancel a flight please.",
    ))
    tasks.append(Task(
        task_id="air_002",
        domain="airline",
        persona="You are Marcus, 52. You are a frequent flyer and a bit impatient.",
        goal="rebook your flight DL445 from June 20 to June 22 (same time)",
        private_info={"name": "Marcus Hill", "booking_ref": "PB3K9N", "flight": "DL445",
                      "old_date": "2026-06-20", "new_date": "2026-06-22"},
        oracle_action="rebook",
        oracle_args={"booking_ref": "PB3K9N", "new_date": "2026-06-22"},
        rubric="agent must verify booking, rebook to 2026-06-22 same flight number",
        initial_user_message="Need to move my June 20 flight to June 22.",
    ))
    tasks.append(Task(
        task_id="air_003",
        domain="airline",
        persona="You are Priya, 28. You rarely volunteer information unless asked.",
        goal="add a checked bag to your existing booking AA009 SFO->JFK",
        private_info={"name": "Priya Shah", "booking_ref": "QH8L2C", "flight": "AA009",
                      "route": "SFO-JFK"},
        oracle_action="add_baggage",
        oracle_args={"booking_ref": "QH8L2C", "bags": 1},
        rubric="agent must verify booking and add 1 checked bag",
        initial_user_message="Hi, I'd like to add a bag.",
    ))
    tasks.append(Task(
        task_id="air_004",
        domain="airline",
        persona="You are Hugo, 67. You worry about being charged twice.",
        goal="confirm whether your original ticket was already refunded for cancelled flight LH22",
        private_info={"name": "Hugo Schmidt", "booking_ref": "ZR4M8T", "flight": "LH22",
                      "previous_cancellation": True, "refund_status": "completed"},
        oracle_action="lookup_refund_status",
        oracle_args={"booking_ref": "ZR4M8T"},
        rubric="agent must look up booking and confirm refund completed",
        initial_user_message="I'm calling to check on a refund.",
    ))
    tasks.append(Task(
        task_id="air_005",
        domain="airline",
        persona="You are Tomoko, 41. You are concise.",
        goal="upgrade your seat on flight JL44 (booking: SD2W7P) to premium economy if available",
        private_info={"name": "Tomoko Sato", "booking_ref": "SD2W7P", "flight": "JL44",
                      "seat_class_current": "economy"},
        oracle_action="upgrade_seat",
        oracle_args={"booking_ref": "SD2W7P", "target_class": "premium_economy"},
        rubric="agent must verify booking and offer premium-economy upgrade",
        initial_user_message="Can I upgrade my seat please?",
    ))
    tasks.append(Task(
        task_id="air_006",
        domain="airline",
        persona="You are Devon, 22. You forget details and need prompting.",
        goal="get a flight receipt resent to your email for tax purposes",
        private_info={"name": "Devon Park", "booking_ref": "NV0J9X",
                      "email": "devon.park@ex.com"},
        oracle_action="resend_receipt",
        oracle_args={"booking_ref": "NV0J9X", "email": "devon.park@ex.com"},
        rubric="agent must look up booking and resend receipt to email on file",
        initial_user_message="Hi, I need a receipt please.",
    ))

    # ---------- Retail returns (6 tasks) ----------
    tasks.append(Task(
        task_id="ret_001",
        domain="retail",
        persona="You are Alex, 30. You are matter-of-fact.",
        goal="return a damaged blender (order 88210) for a refund",
        private_info={"name": "Alex Wei", "order_id": "88210", "item": "blender",
                      "reason": "damaged"},
        oracle_action="initiate_return",
        oracle_args={"order_id": "88210", "reason": "damaged", "refund_to": "original"},
        rubric="agent must verify order, accept damaged-item return, refund to original",
        initial_user_message="I need to return something that arrived broken.",
    ))
    tasks.append(Task(
        task_id="ret_002",
        domain="retail",
        persona="You are Jamie, 39. You are polite and detail-oriented.",
        goal="exchange shoes (order 91005) from size 9 to size 10",
        private_info={"name": "Jamie Cole", "order_id": "91005", "item": "running shoes",
                      "from_size": 9, "to_size": 10},
        oracle_action="initiate_exchange",
        oracle_args={"order_id": "91005", "to_size": 10},
        rubric="agent must look up order and arrange exchange to size 10",
        initial_user_message="I need to exchange a pair of shoes for a different size.",
    ))
    tasks.append(Task(
        task_id="ret_003",
        domain="retail",
        persona="You are Riya, 26. You are friendly but firm.",
        goal="get the status of return RMA-44219",
        private_info={"name": "Riya Patel", "rma_id": "RMA-44219", "status": "approved"},
        oracle_action="lookup_rma",
        oracle_args={"rma_id": "RMA-44219"},
        rubric="agent must look up the RMA and report its status",
        initial_user_message="What's the status on my return?",
    ))
    tasks.append(Task(
        task_id="ret_004",
        domain="retail",
        persona="You are Bo, 19. You are casual and brief.",
        goal="cancel order 77310 (placed yesterday) before it ships",
        private_info={"name": "Bo Kim", "order_id": "77310"},
        oracle_action="cancel_order",
        oracle_args={"order_id": "77310"},
        rubric="agent must verify order is unshipped and cancel it",
        initial_user_message="cancel my order pls",
    ))
    tasks.append(Task(
        task_id="ret_005",
        domain="retail",
        persona="You are Lila, 60. You take time to give details.",
        goal="apply a price-match for the item you ordered (order 81222) that dropped in price",
        private_info={"name": "Lila Hart", "order_id": "81222", "item": "lamp",
                      "discount_amount": 12.50},
        oracle_action="price_match_credit",
        oracle_args={"order_id": "81222", "credit": 12.50},
        rubric="agent must verify order and credit the price-match amount",
        initial_user_message="I noticed the lamp I bought is now cheaper.",
    ))
    tasks.append(Task(
        task_id="ret_006",
        domain="retail",
        persona="You are Naomi, 35. You are calm and concise.",
        goal="report a missing item from order 99001 (a phone case) and get it reshipped",
        private_info={"name": "Naomi Singh", "order_id": "99001", "missing_item": "phone case"},
        oracle_action="reship_missing_item",
        oracle_args={"order_id": "99001", "item": "phone case"},
        rubric="agent must verify order and arrange reshipment of phone case",
        initial_user_message="One item from my order didn't arrive.",
    ))

    # ---------- Account / auth (6 tasks) ----------
    tasks.append(Task(
        task_id="acc_001",
        domain="account",
        persona="You are Karim, 45. You are polite and patient.",
        goal="reset the password on your account (email: karim.t@example.com)",
        private_info={"name": "Karim Toure", "email": "karim.t@example.com"},
        oracle_action="send_password_reset",
        oracle_args={"email": "karim.t@example.com"},
        rubric="agent must verify email and trigger password reset",
        initial_user_message="I can't log in. Can you help me reset my password?",
    ))
    tasks.append(Task(
        task_id="acc_002",
        domain="account",
        persona="You are Sara, 31. You are slightly suspicious of phishing.",
        goal="update the phone number on your account from old to +1-555-201-9933",
        private_info={"name": "Sara Mendes", "email": "s.mendes@ex.com",
                      "new_phone": "+1-555-201-9933"},
        oracle_action="update_phone",
        oracle_args={"email": "s.mendes@ex.com", "new_phone": "+1-555-201-9933"},
        rubric="agent must verify identity and update phone number on file",
        initial_user_message="I'd like to update my phone number on my account.",
    ))
    tasks.append(Task(
        task_id="acc_003",
        domain="account",
        persona="You are Owen, 24. You speak casually.",
        goal="enable two-factor authentication via SMS",
        private_info={"name": "Owen Lee", "email": "owen@l.io", "phone": "+1-555-300-7012"},
        oracle_action="enable_2fa",
        oracle_args={"email": "owen@l.io", "method": "sms", "phone": "+1-555-300-7012"},
        rubric="agent must enable 2FA via SMS using the user's phone",
        initial_user_message="hey can u turn on 2fa on my account",
    ))
    tasks.append(Task(
        task_id="acc_004",
        domain="account",
        persona="You are Imani, 38. You are warm and conversational.",
        goal="merge two accounts you accidentally created with different emails into the older one",
        private_info={"name": "Imani Brown", "primary_email": "imani.b@ex.com",
                      "duplicate_email": "i.brown88@ex.com"},
        oracle_action="merge_accounts",
        oracle_args={"keep": "imani.b@ex.com", "merge": "i.brown88@ex.com"},
        rubric="agent must verify both emails and merge into primary",
        initial_user_message="I think I have two accounts. Can we combine them?",
    ))
    tasks.append(Task(
        task_id="acc_005",
        domain="account",
        persona="You are Felix, 50. You are brief and a little gruff.",
        goal="delete your account and confirm data deletion",
        private_info={"name": "Felix Wong", "email": "felix.w@ex.com"},
        oracle_action="delete_account",
        oracle_args={"email": "felix.w@ex.com", "confirm": True},
        rubric="agent must verify identity and delete the account, confirming deletion",
        initial_user_message="I want to close my account.",
    ))
    tasks.append(Task(
        task_id="acc_006",
        domain="account",
        persona="You are Yuki, 29. You are inquisitive.",
        goal="ask which devices are currently logged in to your account",
        private_info={"name": "Yuki Tanaka", "email": "yuki.t@ex.com"},
        oracle_action="list_active_sessions",
        oracle_args={"email": "yuki.t@ex.com"},
        rubric="agent must verify identity and list active sessions",
        initial_user_message="Can I see what devices are logged in?",
    ))

    # ---------- Billing (6 tasks) ----------
    tasks.append(Task(
        task_id="bil_001",
        domain="billing",
        persona="You are Chen, 33. You are polite and concise.",
        goal="dispute a $42 charge on invoice INV-7791 you don't recognize",
        private_info={"name": "Chen Zhao", "invoice_id": "INV-7791", "disputed_amount": 42.00},
        oracle_action="open_dispute",
        oracle_args={"invoice_id": "INV-7791", "amount": 42.00},
        rubric="agent must verify invoice and open a dispute for the $42 charge",
        initial_user_message="There's a charge on my bill I don't recognize.",
    ))
    tasks.append(Task(
        task_id="bil_002",
        domain="billing",
        persona="You are Esme, 41. You are warm.",
        goal="downgrade your subscription from Pro to Basic at the next renewal",
        private_info={"name": "Esme Hall", "email": "esme@ex.com",
                      "current_plan": "Pro", "target_plan": "Basic"},
        oracle_action="schedule_downgrade",
        oracle_args={"email": "esme@ex.com", "target_plan": "Basic"},
        rubric="agent must verify identity and schedule downgrade at next renewal",
        initial_user_message="Hi, I'd like to change my plan.",
    ))
    tasks.append(Task(
        task_id="bil_003",
        domain="billing",
        persona="You are Otis, 58. You give details slowly.",
        goal="update your billing card to one ending in 1188",
        private_info={"name": "Otis Green", "email": "otis.g@ex.com", "new_card_last4": "1188"},
        oracle_action="update_card",
        oracle_args={"email": "otis.g@ex.com", "new_card_last4": "1188"},
        rubric="agent must verify identity and update card-on-file last4 to 1188",
        initial_user_message="I need to update my credit card on file.",
    ))
    tasks.append(Task(
        task_id="bil_004",
        domain="billing",
        persona="You are Mira, 27. You are direct.",
        goal="get a copy of last month's invoice emailed to you",
        private_info={"name": "Mira Kapoor", "email": "mira.k@ex.com",
                      "month": "April 2026"},
        oracle_action="email_invoice",
        oracle_args={"email": "mira.k@ex.com", "month": "April 2026"},
        rubric="agent must verify identity and email last month's invoice",
        initial_user_message="Can you send me last month's invoice?",
    ))
    tasks.append(Task(
        task_id="bil_005",
        domain="billing",
        persona="You are Reza, 36. You are precise.",
        goal="apply promo code SPRING25 to your next renewal",
        private_info={"name": "Reza Karimi", "email": "reza.k@ex.com", "promo": "SPRING25"},
        oracle_action="apply_promo",
        oracle_args={"email": "reza.k@ex.com", "code": "SPRING25"},
        rubric="agent must verify identity and apply SPRING25 to next renewal",
        initial_user_message="I have a promo code I'd like to apply.",
    ))
    tasks.append(Task(
        task_id="bil_006",
        domain="billing",
        persona="You are Pat, 48. You are blunt.",
        goal="ask for a refund on a duplicate $99 charge that posted twice",
        private_info={"name": "Pat Quinn", "email": "pat.q@ex.com", "duplicate_amount": 99.00},
        oracle_action="refund_duplicate",
        oracle_args={"email": "pat.q@ex.com", "amount": 99.00},
        rubric="agent must verify identity and refund duplicate $99",
        initial_user_message="I was charged twice. I want a refund on the duplicate.",
    ))

    return tasks


if __name__ == "__main__":
    ts = all_tasks()
    print(f"{len(ts)} tasks")
    print(f"domains: {sorted(set(t.domain for t in ts))}")
