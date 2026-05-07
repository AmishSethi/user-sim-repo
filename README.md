# Simulator-Fidelity Reproducibility Pack

Code, raw data, and processed exports for the empirical probes in
*Position: LLM-User Agent Benchmarks Need Simulator-Fidelity Reports Before Human-Facing Claims*
(NeurIPS 2026 Position Paper Track).

The paper argues that interactive multi-turn agent benchmarks should not be
published as evidence of human-facing capability without a quantitative
measurement of user-simulator fidelity. This repo contains everything needed
to regenerate the headline numbers, the supporting probes, and every figure
in the paper.

## What's in here

```
.
├── experiments/                   # all runner, analysis, and figure code
│   ├── run_*.py                   # one runner per benchmark probe
│   ├── analyze_*.py               # per-probe and cross-probe analysis
│   ├── bootstrap_benchmarks.py    # task-resampling 90% CIs
│   ├── export_all_benchmark_data.py
│   ├── make_*figure*.py           # figure-generation scripts
│   ├── start_servers.sh           # vLLM launcher for the n-gram probe
│   ├── EXPERIMENT_PLAN.md         # pre-registration-style experiment plan
│   └── results/                   # raw .jsonl + per-probe summary .json
└── tau2_bench_data/               # post-processed exports used by the paper
    ├── all_benchmark_data.json    # complete dump (all five probes)
    ├── tau2_full_matrix.json      # τ²-bench telecom (headline)
    ├── other_benchmarks_summary.json
    └── README.md                  # schema reference for the JSONs
```

## Headline numbers (reproduce these)

| Benchmark | Dialogues | Tasks/cell | Mean | Max swing | Worst $\tau_K$ | T1 SNR |
|---|---:|---:|---:|---:|---:|---:|
| τ²-bench telecom (headline) | 720 | 20 | 10.4% | **95.0 pp** | 0.35 | 1.82× |
| MINT | 1,800 | 50 | 65.4% | 40.0 pp | 0.33 | 3.52× |
| 24-task customer-service | 3,024 | 24 | 68.4% | 20.8 pp | 0.41 | 1.33× |
| AgentClinic-MedQA | 1,080 | 30 | 34.0% | 16.7 pp | 0.33 | 1.36× |
| τ-bench airline | 180 | 5 | 74.4% | 80.0 pp | 0.00 | 1.16× |

All five probes use the same six-backbone OpenRouter panel as both evaluated
agents and simulated users:
Llama-3.1-8B, Qwen2.5-7B, Gemma-3-12B, Llama-3.3-70B, gpt-oss-120B, DeepSeek-R1.

## Quickstart

### 1. Environment

```bash
git clone <this-repo>
cd simulator-fidelity-reproduce
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API key

The five main probes call OpenRouter for both the agent side and the
simulated-user side. Get a key at <https://openrouter.ai>; export it:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

The n-gram fidelity probe (Appendix A) uses local vLLM servers instead;
see "n-gram fidelity probe" below.

### 3. External benchmark frameworks

Three probes (τ²-bench, MINT, AgentClinic) wrap upstream public benchmarks.
Clone them at the exact commits we used:

```bash
mkdir -p experiments/external
cd experiments/external

git clone https://github.com/sierra-research/tau2-bench.git
( cd tau2-bench && git checkout 7483cc60e4957fb2cf834c05a41059a715ee287c )

git clone https://github.com/xingyaoww/mint-bench.git
( cd mint-bench && git checkout 3f7f12c10bf763be1e6dbdeb42feb57624121f61 )

git clone https://github.com/SamuelSchmidgall/AgentClinic.git
( cd AgentClinic && git checkout b6570edefb940857a7c334350656b29f9d984f24 )

cd ../..
```

The 24-task customer-service probe and τ-bench airline are entirely in this
repo; no external clone needed for them.

### 4. Reproduce per probe

The runners are independent. Each one writes `experiments/results/<probe>_raw.jsonl`,
the per-dialogue log used by the analysis scripts. Resume is supported via the
existing jsonl, so partial runs can be re-launched.

```bash
cd experiments

# τ²-bench telecom (headline; 6×6×20 = 720 dialogues, ~3 hours)
python run_tau2_parallel.py
python analyze_tau2.py
# T1 receipt: 4 cells × 10 reruns × 5 tasks
python run_tau2_variance.py

# MINT (6×6×50 = 1800 dialogues, ~2 hours)
python run_mint.py
# T1 receipt
python run_variance.py

# 24-task customer-service probe (6×6×3×24 = 2592 dialogues, ~1 hour)
python run_experiment_1.py
python analyze_exp1.py

# AgentClinic-MedQA (6×6×30 = 1080 dialogues, ~1 hour)
python run_agentclinic.py

# τ-bench airline (6×6×5 = 180 dialogues, ~30 min)
python run_taubench_parallel.py
python run_tau_variance.py
```

Wall time depends on OpenRouter rate limits and which models are warm. Total
end-to-end is roughly 6–8 hours for the entire panel; individual probes finish
much faster. The runners parallelize cells, so you can also kill and restart.

### 5. Aggregate and produce paper exports

```bash
# Per-cell variance summary (consumed by export_all_benchmark_data.py)
python analyze_variance.py

# Build the master JSON used by the paper
python export_all_benchmark_data.py

# Bootstrap-over-tasks 90% CIs (writes back into all_benchmark_data.json)
python bootstrap_benchmarks.py
```

The result is `tau2_bench_data/all_benchmark_data.json`, which contains every
6×6 success-rate matrix, per-agent and worst-pair metrics, T1 variance receipts,
and bootstrap CIs.

### 6. Generate figures

```bash
# Figure 2: τ²-bench agent-spread plot
python make_tau2_figure.py

# Figure 4: 4-panel supporting-spread plot
python make_figure2_and_mint_appendix.py
python make_mint_figure.py

# Figure 3: n-gram fidelity (panel a + panel b)
python make_ngram_fidelity_clean.py
```

PDFs land in `figures/` at the repo root.

## n-gram fidelity probe (Appendix A)

This probe uses local vLLM servers, not OpenRouter, because it samples ~2400
fresh user turns from each of three open-weight simulators and trains a
discriminator on the resulting text.

```bash
# 1. Launch three vLLM servers (Llama-3.1-8B, Qwen2.5-3B, Phi-4-mini-instruct)
bash experiments/start_servers.sh

# 2. Wait for readiness
for p in 8011 8012 8013; do
  until curl -sf http://127.0.0.1:$p/v1/models > /dev/null; do sleep 5; done
done

# 3. Run the probe (~150 s on a single H100)
cd experiments
python run_experiment_2_ngram.py
```

`start_servers.sh` pins each server to a specific GPU (`CUDA_VISIBLE_DEVICES=9`,
`0`, `3`); edit those lines for your hardware. All three fit on one 80 GB H100
with `--gpu-memory-utilization 0.25`.

## Existing logs from our runs

`experiments/results/` already contains the raw output from the runs reported
in the paper:

| File | Contents | Source |
|---|---|---|
| `tau2_raw.jsonl` | τ²-bench telecom main sweep, 720 dialogues | `run_tau2_parallel.py` |
| `tau2_telecom_variance_raw.jsonl` | T1 K=10 reruns on 4 telecom cells | `run_tau2_variance.py` |
| `mint_raw.jsonl` | MINT main sweep, 1800 dialogues | `run_mint.py` |
| `variance_raw.jsonl` | T1 K=10 reruns on 4 MINT cells | `run_variance.py` |
| `exp1_v2_raw.jsonl` | 24-task customer-service main sweep | `run_experiment_1.py` |
| `agentclinic_raw.jsonl` | AgentClinic-MedQA main sweep | `run_agentclinic.py` |
| `taubench_raw.jsonl` | τ-bench airline main sweep | `run_taubench_parallel.py` |
| `tau_airline_variance_raw.jsonl` | T1 K=10 reruns on 4 airline cells | `run_tau_variance.py` |
| `human_turns.json`, `sim_turns_*.json` | n-gram fidelity raw turns | `run_experiment_2_ngram.py` |
| `*_summary.json`, `variance_summary.json` | per-probe aggregates | analysis scripts |

These let you skip the API calls and go straight to analysis or figures:

```bash
cd experiments
python export_all_benchmark_data.py         # rebuilds tau2_bench_data/all_benchmark_data.json
python bootstrap_benchmarks.py              # adds bootstrap CIs
python make_tau2_figure.py                  # rebuilds Figure 2
```

## Hardware notes

- **OpenRouter probes** (τ², MINT, customer-service, AgentClinic, τ-bench airline):
  no GPU needed locally; you pay OpenRouter for inference. Total cost for the
  full panel was approximately \$80–\$120 at our run.
- **n-gram fidelity probe** (Appendix A only): single H100 sufficient. Three
  vLLM servers fit at `--gpu-memory-utilization 0.25` each.

## Citation

```
@inproceedings{usersim2026position,
  title     = {Position: {LLM}-User Agent Benchmarks Need Simulator-Fidelity Reports Before Human-Facing Claims},
  author    = {Anonymous Submission},
  booktitle = {Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  note      = {Position Paper Track}
}
```

(Update once the paper is published.)

## License

Code is released under Apache 2.0; see `LICENSE`. The raw-data jsonls under
`experiments/results/` are released under CC-BY-4.0.

The external benchmark frameworks (τ²-bench, MINT, AgentClinic) retain their
upstream licenses; see the original repositories.
