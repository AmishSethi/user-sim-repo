# Experiment Plan

Two illustrative empirical analyses to support the position.

## Experiment 1 — Simulator-induced ranking flip

**Claim being tested**: The ranking of agent systems on a multi-turn customer-service style benchmark changes when the user simulator backend changes, even though the tasks, judge, and protocol are held constant.

**Setup**
- 24 multi-turn customer service tasks across four domains (airline, retail, account/auth, billing) inspired by tau-bench / tau²-bench style. Each task fixes a *user persona*, *goal*, *private info* (e.g., booking reference), and an *oracle policy* (the action(s) the agent should take to satisfy the goal).
- **Agent models** (4): `Llama-3.1-8B-Instruct`, `Qwen2.5-3B-Instruct`, `Phi-3-mini-4k-instruct`, `Phi-4-mini-instruct`. Held fixed across simulator conditions.
- **User-simulator backends** (4):
  - `Sim_LL3`: Llama-3.1-8B-Instruct prompted as user
  - `Sim_QW`: Qwen2.5-3B-Instruct prompted as user
  - `Sim_PHI`: Phi-4-mini-instruct prompted as user
  - `Sim_DSL`: a scripted, agenda-based simulator drawing utterance templates from a small grammar with stochastic slot revelation order
- **Judge**: a fixed deterministic checker that combines (a) regex/slot match against oracle policy and (b) an LLM-judge tiebreaker (`Llama-3.1-8B-Instruct` with a structured rubric). Held fixed.
- **Protocol**: For each `(agent, simulator)` pair we run *K=3* trials per task, max 8 turns each. We log conversations, predicted action, and judge outcomes.
- **Metrics**: per-agent success rate by simulator; Kendall's τ between rankings across simulator pairs; standard deviation of agent score across simulators.

**Expected pattern (supporting position)**: Mean agent success rate varies meaningfully across simulator backends; the *ranking* of the four agents is not preserved (Kendall's τ < 1). The DSL simulator is notably more stringent / different than the LLM simulators.

## Experiment 2 — N-gram fidelity probe

**Claim being tested**: Even at the population level — a level of fidelity weaker than individual fidelity — LLM user simulators do not match human user-utterance distributions; the gap is detectable by a simple n-gram baseline.

**Setup**
- Source of *real* human user text: MultiWOZ 2.2 user turns (public, license-permitting), and a held-out portion of public dialogue corpora.
- For each LLM simulator backend listed in Experiment 1, sample N=2000 user turns conditioned on matched dialogue contexts.
- Train a 3-gram backoff language model on the held-in portion of human turns.
- Compute: (a) per-token NLL of human held-out vs simulator turns under the human n-gram model; (b) per-token NLL under a simulator-trained n-gram (cross-NLL); (c) distinguishability of human vs simulator turns via a logistic-regression classifier on character/word n-gram features.

**Expected pattern**: Simulator turns have lower NLL under the simulator-trained n-gram than under the human-trained n-gram (distribution shift is detectable cheaply); a logistic-regression classifier can separate human from LLM-simulated user turns at well above chance.

## Why these experiments

We deliberately keep both small. The role of the empirical evidence is *illustrative*, not definitive. The position paper argues that the field needs systematic, large-scale fidelity measurement; here we demonstrate that even toy probes are sufficient to surface the problem.

## Outputs

- `figures/ranking_flip.pdf`: heatmap of agent success by simulator + Kendall's τ table.
- `figures/ngram_fidelity.pdf`: density plot of NLL gap and classifier ROC.
- `experiments/results/` with raw logs and aggregated CSVs.
