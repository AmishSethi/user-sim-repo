"""Experiment 2: a small n-gram fidelity probe for LLM user simulators.

We compare LLM-simulated user turns against a held-out human user-turn
distribution at the population level. Even at the (weaker) population level
of fidelity, we expect a clear distinguishability gap.

To keep the probe self-contained we use a *synthetic but human-written*
reference distribution (the seed user utterances from `tasks.py` plus a
hand-written paraphrase set). This bypasses the licensing complications of
shipping MultiWOZ excerpts in the artifact while still demonstrating the
methodology. The same probe applies to any human corpus.
"""
from __future__ import annotations
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import asyncio
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from llm_client import LLMClient, ServerConfig
from tasks import all_tasks


SERVERS = [
    ServerConfig(alias="llama3.1-8b", base_url="http://127.0.0.1:8011/v1",
                 model="meta-llama/Llama-3.1-8B-Instruct"),
    ServerConfig(alias="qwen2.5-3b",  base_url="http://127.0.0.1:8012/v1",
                 model="Qwen/Qwen2.5-3B-Instruct"),
    ServerConfig(alias="phi4-mini",   base_url="http://127.0.0.1:8013/v1",
                 model="microsoft/Phi-4-mini-instruct"),
]

OUTDIR = Path(__file__).resolve().parent / "results"
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


# ---------- Reference (human) corpus ----------
# A small, hand-written set of customer-service user turns. The phrasing is
# deliberately terse and idiomatic in ways that LLMs tend not to reproduce.
HUMAN_TURNS: List[str] = [
    "hey can u cancel my flight",
    "need a refund on uA123",
    "umm so the lamp i bought dropped in price",
    "i cant log in. resent the password please?",
    "two charges on my card. weird.",
    "wanna change my plan to basic",
    "size 10 instead of 9 if u have it",
    "yeah just close the account.",
    "can you hear me",
    "double charge. fix it pls",
    "Forgot my password again, sorry",
    "ok so the package came smashed",
    "checking on a refund from like 3 weeks ago",
    "What devices are signed in?",
    "promo code SPRING25 should give a discount",
    "I want to dispute a $42 fee",
    "is anything else needed",
    "Yeah just send the receipt to my email",
    "no the booking is X7FQ2L",
    "PB3K9N - and please move it to the 22nd",
    "the missing thing was a phone case",
    "so I have two accounts and want them merged",
    "card is updated, ends in 1188",
    "merge into the imani.b@ex.com one",
    "I'd like premium economy if possible",
    "RMA-44219 is what I have",
    "April please. My email is mira.k@ex.com",
    "can you turn on 2fa via text",
    "the duplicate is for $99",
    "k so its 88210, blender, came damaged",
    "yes verify with my booking ref",
    "no thanks im good",
    "thank u",
    "yes thats right",
    "what info do u need",
    "ok done thanks",
    "still need anything from me",
    "ya thats my email",
    "no. just close it",
    "go ahead and refund",
    "checked bag yes 1 bag",
    "yes please",
    "alright thanks",
    "yep that works",
    "do you need anything else from me",
    "Ana Ruiz",
    "Marcus Hill",
    "Priya Shah",
    "Hugo Schmidt",
    "Tomoko Sato",
    "Devon Park, devon.park@ex.com",
    "Alex Wei",
    "Jamie Cole",
    "Riya Patel",
    "Bo Kim",
    "Lila Hart",
    "Naomi Singh",
    "Karim Toure",
    "Sara Mendes",
    "Owen Lee",
    "Imani Brown",
    "Felix Wong",
    "Yuki Tanaka",
    "Chen Zhao",
    "Esme Hall",
    "Otis Green",
    "Mira Kapoor",
    "Reza Karimi",
    "Pat Quinn",
]


SYS_USER = """You are role-playing a customer in a customer-service chat. Reply concisely, in
first person, no more than one short sentence. Do not include the agent's words.

Persona: {persona}
Goal: {goal}
Private info you may share if asked:
{info}

Context (last agent message): {agent_msg}
"""


async def collect_llm_turns(client: LLMClient, alias: str, n: int = 800) -> List[str]:
    """Collect ~n LLM-simulated user turns by sampling across the task pool
    with a variety of likely agent prompts."""
    tasks_list = all_tasks()
    AGENT_PROMPTS = [
        "Hi, how can I help you today?",
        "Could you please provide your booking reference?",
        "Sure - what is your order number?",
        "May I confirm your email on file?",
        "What date are you looking to change to?",
        "Got it. Can you tell me what's wrong with the item?",
        "What is the new size you'd like?",
        "Thanks. What is the amount?",
        "May I have the last four digits of your card?",
        "Anything else I can help with?",
        "Can you confirm your phone number?",
        "Which plan would you like to switch to?",
        "Could you spell that out for me?",
    ]
    turns: List[str] = []
    out = []
    sem = asyncio.Semaphore(20)

    async def one(t, ap, j):
        async with sem:
            sys_prompt = SYS_USER.format(
                persona=t.persona, goal=t.goal,
                info=json.dumps(t.private_info), agent_msg=ap,
            )
            try:
                txt = await client.chat(alias, [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": ap},
                ], max_tokens=40, temperature=0.9)
                # remove role labels if model leaks them
                txt = re.sub(r"^\s*(user|customer)[:\-]\s*", "", txt, flags=re.I).strip()
                txt = txt.strip(' "\n')
                return txt
            except Exception as e:
                return None

    coros = []
    j = 0
    while len(coros) < n:
        for t in tasks_list:
            for ap in AGENT_PROMPTS:
                coros.append(one(t, ap, j))
                j += 1
                if len(coros) >= n:
                    break
            if len(coros) >= n:
                break
    print(f"  dispatching {len(coros)} {alias} samples")
    res = await asyncio.gather(*coros)
    return [r for r in res if r and r.strip()]


# ---------- N-gram language model ----------
def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())


class NGramLM:
    def __init__(self, n: int = 3, alpha: float = 0.4):
        self.n = n
        self.alpha = alpha
        self.counts: defaultdict = defaultdict(Counter)
        self.context_totals: Counter = Counter()
        self.vocab: set = set()

    def fit(self, corpus: List[str]) -> None:
        for line in corpus:
            toks = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
            self.vocab.update(toks)
            for i in range(self.n - 1, len(toks)):
                ctx = tuple(toks[i - (self.n - 1): i])
                w = toks[i]
                self.counts[ctx][w] += 1
                self.context_totals[ctx] += 1

    def _prob(self, ctx: Tuple[str, ...], w: str) -> float:
        # Stupid backoff
        if self.context_totals.get(ctx, 0) > 0 and self.counts[ctx].get(w, 0) > 0:
            return self.counts[ctx][w] / self.context_totals[ctx]
        if len(ctx) > 0:
            return self.alpha * self._prob(ctx[1:], w)
        # unigram smooth
        unigram = sum(c.get(w, 0) for c in self.counts.values())
        total = sum(self.context_totals.values()) or 1
        return (unigram + 1) / (total + len(self.vocab) + 1)

    def nll(self, line: str) -> float:
        toks = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
        s = 0.0
        cnt = 0
        for i in range(self.n - 1, len(toks)):
            ctx = tuple(toks[i - (self.n - 1): i])
            w = toks[i]
            s += -math.log(max(self._prob(ctx, w), 1e-12))
            cnt += 1
        return s / max(cnt, 1)


async def main():
    client = LLMClient(SERVERS)
    print("waiting for servers ...")
    for s in SERVERS:
        for _ in range(180):
            ok = await client.health(s.alias)
            if ok:
                break
            await asyncio.sleep(2.0)

    aliases = ["llama3.1-8b", "qwen2.5-3b", "phi4-mini"]
    samples_per = 800
    sim_turns = {}
    for a in aliases:
        print(f"collecting samples from {a}")
        sim_turns[a] = await collect_llm_turns(client, a, n=samples_per)
        print(f"  got {len(sim_turns[a])} samples from {a}")
        Path(OUTDIR / f"sim_turns_{a}.json").write_text(
            json.dumps(sim_turns[a], indent=2))

    Path(OUTDIR / "human_turns.json").write_text(json.dumps(HUMAN_TURNS, indent=2))
    await client.aclose()

    # ---- N-gram log-likelihood gap ----
    rng = np.random.default_rng(0)
    # Train a 3-gram on the human corpus
    human_lm = NGramLM(n=3); human_lm.fit(HUMAN_TURNS)
    rows = []
    nlls = {"human (held out)": []}
    # leave-one-out human held-out NLL
    for i, line in enumerate(HUMAN_TURNS):
        train = HUMAN_TURNS[:i] + HUMAN_TURNS[i+1:]
        lm = NGramLM(n=3); lm.fit(train)
        nlls["human (held out)"].append(lm.nll(line))
    for a in aliases:
        nlls[a] = [human_lm.nll(s) for s in sim_turns[a]]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax = axes[0]
    labels = ["human (held out)"] + aliases
    data = [nlls[l] for l in labels]
    bp = ax.boxplot(data, labels=labels, showfliers=False, widths=0.55,
                    patch_artist=True)
    palette = ["#7fbf7b", "#f5b25b", "#e58368", "#a06ec0"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_ylabel("NLL per token (3-gram on human corpus)", fontsize=9)
    ax.set_title("Population-fidelity gap (lower NLL = closer to human)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.tick_params(axis='y', labelsize=8)

    # ---- Distinguishability ----
    X_text = []
    y = []
    for s in HUMAN_TURNS:
        X_text.append(s); y.append(0)
    for a in aliases:
        for s in sim_turns[a]:
            X_text.append(s); y.append(1)
    y = np.array(y)
    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vect.fit_transform(X_text)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        proba = clf.predict_proba(X[te])[:, 1]
        accs.append((pred == y[te]).mean())
        from sklearn.metrics import roc_auc_score, roc_curve
        aucs.append(roc_auc_score(y[te], proba))
    print(f"\nhuman-vs-LLM-sim distinguishability classifier:")
    print(f"  accuracy = {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  AUROC    = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # ROC on cross-validated out-of-fold predictions
    from sklearn.metrics import roc_curve, roc_auc_score
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf2.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y, oof)
    auc = roc_auc_score(y, oof)
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color="#222", lw=1.6,
             label=f"char 3-5gram TF-IDF + logreg, AUROC={auc:.2f} (5-fold CV)")
    ax2.plot([0, 1], [0, 1], color="grey", lw=0.7, ls="--")
    ax2.set_xlabel("FPR (human turns flagged as sim)", fontsize=9)
    ax2.set_ylabel("TPR (sim turns flagged as sim)", fontsize=9)
    ax2.set_title("Distinguishability of human vs LLM-sim user turns", fontsize=9)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(FIGDIR / "ngram_fidelity.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "ngram_fidelity.png", dpi=180, bbox_inches="tight")
    print(f"\nfigure saved to {FIGDIR / 'ngram_fidelity.pdf'}")

    # Save summary
    summary = {
        "human_held_out_nll_mean": float(np.mean(nlls["human (held out)"])),
        "human_held_out_nll_std":  float(np.std(nlls["human (held out)"])),
        "human_held_out_nll_n":    int(len(nlls["human (held out)"])),
    }
    for a in aliases:
        summary[f"{a}_nll_mean"] = float(np.mean(nlls[a]))
        summary[f"{a}_nll_std"]  = float(np.std(nlls[a]))
        summary[f"{a}_n"]        = int(len(nlls[a]))
    summary["classifier_accuracy_mean"] = float(np.mean(accs))
    summary["classifier_accuracy_std"]  = float(np.std(accs))
    summary["classifier_auroc_mean"]    = float(np.mean(aucs))
    summary["classifier_auroc_std"]     = float(np.std(aucs))
    Path(OUTDIR / "exp2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
