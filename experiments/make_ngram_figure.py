"""Regenerate the appendix n-gram separability figure from saved turns.

The model calls were run earlier by `run_experiment_2_ngram.py`. This script
only reloads the saved reference and simulator turns, recomputes the simple
3-gram likelihood and TF-IDF discriminator diagnostics, and writes the figure.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results"
FIG = ROOT / "figures"
SRC_FIG = ROOT / "src/figures"
FIG.mkdir(parents=True, exist_ok=True)
SRC_FIG.mkdir(parents=True, exist_ok=True)

ALIASES = ["llama3.1-8b", "qwen2.5-3b", "phi4-mini"]
LABELS = {
    "human": "author ref",
    "llama3.1-8b": "Llama 8B",
    "qwen2.5-3b": "Qwen 3B",
    "phi4-mini": "Phi-mini",
}
COLORS = {
    "human": "#9AA6B5",
    "llama3.1-8b": "#0F9688",
    "qwen2.5-3b": "#D47728",
    "phi4-mini": "#7E63B6",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


class NGramLM:
    def __init__(self, n: int = 3, alpha: float = 0.4):
        self.n = n
        self.alpha = alpha
        self.counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()

    def fit(self, corpus: list[str]) -> None:
        for line in corpus:
            toks = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
            self.vocab.update(toks)
            for idx in range(self.n - 1, len(toks)):
                ctx = tuple(toks[idx - (self.n - 1) : idx])
                word = toks[idx]
                self.counts[ctx][word] += 1
                self.context_totals[ctx] += 1

    def _prob(self, ctx: tuple[str, ...], word: str) -> float:
        if self.context_totals.get(ctx, 0) > 0 and self.counts[ctx].get(word, 0) > 0:
            return self.counts[ctx][word] / self.context_totals[ctx]
        if ctx:
            return self.alpha * self._prob(ctx[1:], word)
        unigram = sum(counter.get(word, 0) for counter in self.counts.values())
        total = sum(self.context_totals.values()) or 1
        return (unigram + 1) / (total + len(self.vocab) + 1)

    def nll(self, line: str) -> float:
        toks = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
        loss = 0.0
        count = 0
        for idx in range(self.n - 1, len(toks)):
            ctx = tuple(toks[idx - (self.n - 1) : idx])
            loss += -math.log(max(self._prob(ctx, toks[idx]), 1e-12))
            count += 1
        return loss / max(count, 1)


def load_turns() -> tuple[list[str], dict[str, list[str]]]:
    human = json.loads((OUT / "human_turns.json").read_text())
    sims = {
        alias: json.loads((OUT / f"sim_turns_{alias}.json").read_text())
        for alias in ALIASES
    }
    return human, sims


def compute_nlls(human: list[str], sims: dict[str, list[str]]) -> dict[str, list[float]]:
    nlls: dict[str, list[float]] = {"human": []}
    for idx, line in enumerate(human):
        lm = NGramLM(n=3)
        lm.fit(human[:idx] + human[idx + 1 :])
        nlls["human"].append(lm.nll(line))

    human_lm = NGramLM(n=3)
    human_lm.fit(human)
    for alias, turns in sims.items():
        nlls[alias] = [human_lm.nll(turn) for turn in turns]
    return nlls


def compute_discriminator(
    human: list[str],
    sims: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    sim_turns = [turn for alias in ALIASES for turn in sims[alias]]
    texts = human + sim_turns
    labels = np.array([0] * len(human) + [1] * len(sim_turns))

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    features = vectorizer.fit_transform(texts)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(labels))
    aucs: list[float] = []
    accs: list[float] = []
    for train_idx, test_idx in splitter.split(features, labels):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(features[train_idx], labels[train_idx])
        scores = clf.predict_proba(features[test_idx])[:, 1]
        oof[test_idx] = scores
        aucs.append(roc_auc_score(labels[test_idx], scores))
        accs.append(float((clf.predict(features[test_idx]) == labels[test_idx]).mean()))

    fpr, tpr, _ = roc_curve(labels, oof)
    return fpr, tpr, float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs))


def render() -> None:
    human, sims = load_turns()
    nlls = compute_nlls(human, sims)
    fpr, tpr, auc_mean, auc_std, acc_mean = compute_discriminator(human, sims)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.6),
        gridspec_kw={"width_ratios": [1.04, 1.0], "wspace": 0.32},
    )

    ax = axes[0]
    order = ["human"] + ALIASES
    positions = np.arange(len(order))[::-1]
    data = [nlls[key] for key in order]
    box = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#777777", "linewidth": 0.9},
        capprops={"color": "#777777", "linewidth": 0.9},
    )
    for patch, key in zip(box["boxes"], order):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.82 if key == "human" else 0.72)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(0.8)

    human_mean = float(np.mean(nlls["human"]))
    ax.axvline(human_mean, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_yticks(positions, [LABELS[key] for key in order])
    ax.set_xlabel("3-gram NLL per token")
    ax.set_title("(a) Mean NLL stays close", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="x", color="#E1E4E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(4.8, 7.85)

    for key, y_pos in zip(order, positions):
        mean = float(np.mean(nlls[key]))
        ax.plot(mean, y_pos, marker="o", color="#111111", markersize=3.2, zorder=4)
        ax.text(7.72, y_pos, f"{mean:.2f}", ha="right", va="center", fontsize=7.8)

    ax2 = axes[1]
    ax2.fill_between(fpr, tpr, fpr, color="#CDEDE7", alpha=0.85, linewidth=0)
    ax2.plot(fpr, tpr, color="#0B6E69", linewidth=1.8)
    ax2.plot([0, 1], [0, 1], color="#808080", linewidth=0.9, linestyle="--")
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.01, 1.01)
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title("(b) Classifier separates turns", loc="left", fontweight="bold", pad=4)
    ax2.grid(color="#E1E4E8", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.96,
        0.08,
        f"AUROC {auc_mean:.3f} +/- {auc_std:.3f}\naccuracy {100 * acc_mean:.1f}%",
        ha="right",
        va="bottom",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D0D0D0"},
    )

    fig.savefig(FIG / "ngram_fidelity.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(FIG / "ngram_fidelity.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
    shutil.copyfile(FIG / "ngram_fidelity.pdf", SRC_FIG / "ngram_fidelity.pdf")

    summary = {
        "classifier_auroc_mean": auc_mean,
        "classifier_auroc_std": auc_std,
        "classifier_accuracy_mean": acc_mean,
        "nll_means": {key: float(np.mean(vals)) for key, vals in nlls.items()},
    }
    print(json.dumps(summary, indent=2))
    print(f"figure -> {FIG / 'ngram_fidelity.pdf'}")


if __name__ == "__main__":
    render()
