"""Regenerate the lexical-separability figure from saved turns.

This script is a cleaner, repo-relative replacement for the older
`make_ngram_figure.py`. It rebuilds both panels from saved data:

* panel (a): leave-one-out author-reference NLL and simulator NLLs under a
  3-gram language model trained on the author reference;
* panel (b): cross-validated character 3-5gram TF-IDF logistic-regression ROC.

By default the script writes to `figures/ngram_fidelity.{pdf,svg,png}`. Use
`--out-prefix` to write inspection copies elsewhere.
"""
from __future__ import annotations

import argparse
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
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
SRC_FIG = ROOT / "src" / "figures"

ALIASES = ["llama3.1-8b", "qwen2.5-3b", "phi4-mini"]
ORDER = ["human", *ALIASES]

LABELS = {
    "human": "Author ref.",
    "llama3.1-8b": "Llama 8B",
    "qwen2.5-3b": "Qwen 3B",
    "phi4-mini": "Phi-mini",
}

COLORS = {
    "human": "#9CA5B1",
    "llama3.1-8b": "#129389",
    "qwen2.5-3b": "#E38624",
    "phi4-mini": "#6647D9",
}

GRID = "#D8DEE6"
SPINE = "#C9D1DB"
TEXT = "#111827"
MUTED = "#6B7280"
TEAL_DARK = "#146C68"
TEAL_FILL = "#BFEFE6"


def configure_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 8.8,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


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
            tokens = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
            self.vocab.update(tokens)
            for idx in range(self.n - 1, len(tokens)):
                context = tuple(tokens[idx - (self.n - 1) : idx])
                word = tokens[idx]
                self.counts[context][word] += 1
                self.context_totals[context] += 1

    def _prob(self, context: tuple[str, ...], word: str) -> float:
        if self.context_totals.get(context, 0) > 0 and self.counts[context].get(word, 0) > 0:
            return self.counts[context][word] / self.context_totals[context]
        if context:
            return self.alpha * self._prob(context[1:], word)
        unigram = sum(counter.get(word, 0) for counter in self.counts.values())
        total = sum(self.context_totals.values()) or 1
        return (unigram + 1) / (total + len(self.vocab) + 1)

    def nll(self, line: str) -> float:
        tokens = ["<s>"] * (self.n - 1) + tokenize(line) + ["</s>"]
        loss = 0.0
        count = 0
        for idx in range(self.n - 1, len(tokens)):
            context = tuple(tokens[idx - (self.n - 1) : idx])
            loss += -math.log(max(self._prob(context, tokens[idx]), 1e-12))
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
    accuracies: list[float] = []

    for train_idx, test_idx in splitter.split(features, labels):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(features[train_idx], labels[train_idx])
        scores = clf.predict_proba(features[test_idx])[:, 1]
        oof[test_idx] = scores
        aucs.append(roc_auc_score(labels[test_idx], scores))
        accuracies.append(float((clf.predict(features[test_idx]) == labels[test_idx]).mean()))

    fpr, tpr, _ = roc_curve(labels, oof)
    return fpr, tpr, float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accuracies))


def style_axis(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.82)
    ax.set_axisbelow(True)


def draw_nll_panel(ax: plt.Axes, nlls: dict[str, list[float]]) -> None:
    positions = np.arange(len(ORDER))[::-1]
    data = [nlls[key] for key in ORDER]
    box = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT, "linewidth": 1.05},
        whiskerprops={"color": MUTED, "linewidth": 0.95},
        capprops={"color": MUTED, "linewidth": 0.95},
    )

    for patch, key in zip(box["boxes"], ORDER):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.70 if key != "human" else 0.66)
        patch.set_edgecolor(MUTED)
        patch.set_linewidth(0.8)

    human_mean = float(np.mean(nlls["human"]))
    ax.axvline(human_mean, color=MUTED, linestyle=(0, (3, 2.2)), linewidth=1.0, zorder=1)
    ax.text(
        human_mean,
        1.015,
        "author mean",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.7,
        color=MUTED,
        clip_on=False,
    )

    ax.set_yticks(positions, [LABELS[key] for key in ORDER])
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.set_xlim(4.75, 8.02)
    ax.set_xticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
    ax.set_xlabel("3-gram NLL per token", labelpad=7)
    ax.set_title("(a) Mean NLL stays close", loc="left", pad=20, fontweight="normal")
    style_axis(ax, "x")

    for key, y_pos in zip(ORDER, positions):
        mean = float(np.mean(nlls[key]))
        ax.plot(mean, y_pos, marker="o", color=TEXT, markersize=3.3, zorder=4)
        ax.text(7.91, y_pos, f"{mean:.2f}", ha="right", va="center", fontsize=7.4, color=TEXT)


def draw_roc_panel(
    ax: plt.Axes,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_mean: float,
    auc_std: float,
    acc_mean: float,
) -> None:
    ax.fill_between(fpr, tpr, fpr, color=TEAL_FILL, alpha=0.68, linewidth=0)
    ax.plot(fpr, tpr, color=TEAL_DARK, linewidth=1.7)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.9, linestyle=(0, (3, 2.2)))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("false positive rate", labelpad=7)
    ax.set_ylabel("true positive rate", labelpad=7)
    ax.set_title("(b) Classifier separates turns", loc="left", pad=20, fontweight="normal")
    style_axis(ax, "both")

    ax.text(
        0.92,
        0.14,
        f"AUROC {auc_mean:.3f} +/- {auc_std:.3f}\naccuracy {100 * acc_mean:.1f}%",
        ha="right",
        va="center",
        fontsize=7.3,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": SPINE,
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )


def save_all(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def render(out_prefix: Path, copy_to_src: bool) -> dict[str, float | dict[str, float]]:
    configure_style()
    human, sims = load_turns()
    nlls = compute_nlls(human, sims)
    fpr, tpr, auc_mean, auc_std, acc_mean = compute_discriminator(human, sims)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.34})
    draw_nll_panel(axes[0], nlls)
    draw_roc_panel(axes[1], fpr, tpr, auc_mean, auc_std, acc_mean)
    save_all(fig, out_prefix)

    if copy_to_src:
        SRC_FIG.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_prefix.with_suffix(".pdf"), SRC_FIG / f"{out_prefix.name}.pdf")
        shutil.copyfile(out_prefix.with_suffix(".svg"), SRC_FIG / f"{out_prefix.name}.svg")

    return {
        "classifier_auroc_mean": auc_mean,
        "classifier_auroc_std": auc_std,
        "classifier_accuracy_mean": acc_mean,
        "nll_means": {key: float(np.mean(vals)) for key, vals in nlls.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-prefix", type=Path, default=FIG / "ngram_fidelity")
    parser.add_argument("--copy-to-src", action="store_true")
    args = parser.parse_args()

    summary = render(args.out_prefix, args.copy_to_src)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
