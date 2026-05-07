# Benchmark data

This directory contains the empirical data behind every claim in the paper. All of it is derived from runs in `experiments/results/`; the JSON files here are the post-processed views.

## What each file is

| File | Contents |
|---|---|
| `all_benchmark_data.json` | Complete dump of every benchmark we ran. Per benchmark: 6×6 success-rate matrix, mean success, max same-agent cross-sim swing, per-agent swing, worst-pair Kendall τ, full pairwise Kendall τ matrix, and (where available) the K=10 variance T1 receipt with per-cell rows. Includes a `summary_table` with one row per benchmark in figure-2 display order. |
| `tau2_full_matrix.json` | Just τ²-bench telecom (the main showcase). Full 6×6 success-rate matrix plus summary metrics and the K=10 T1 variance block. |
| `other_benchmarks_summary.json` | Headline metrics (mean, max swing, worst-pair τ, T1 SNR) for the other four benchmarks (MINT, 24-task CS, AgentClinic, τ-bench airline). No matrices. |
| `figure2_data.json` | Earlier export keyed for the figure-2 renderer; superseded by `all_benchmark_data.json` but retained for any tooling that already reads it. |
| `tau2_bench_data.json` | Original τ²-only export from the 5-task pilot; superseded by `tau2_full_matrix.json`. Retained for anyone reading the older filename. |

## Headline numbers

| Benchmark | Dialogues | Tasks per cell | Mean | Max same-agent swing | Worst-pair Kendall τ | T1 mean SNR |
|---|---|---|---|---|---|---|
| MINT | 1,800 | 50 | 65.4% | 40.0 pp | 0.33 | 3.52× |
| 24-task customer-service | 3,024 | 24 | 68.4% | 20.8 pp | 0.41 | 1.33× |
| AgentClinic-MedQA | 1,080 | 30 | 34.0% | 16.7 pp | 0.33 | 1.36× |
| τ-bench airline | 180 | 5 | 74.4% | 80.0 pp | 0.00 | 1.16× |
| τ²-bench telecom | 720 | 20 | 10.4% | 95.0 pp | 0.35 | 1.82× (3.06× at high-effect cells) |

Five benchmarks, 6×6 simulator-agent panel, 6,804 main-sweep dialogues, 800 variance dialogues, all served through OpenRouter on the canonical six-backbone panel (Llama-3.1-8B, Qwen2.5-7B, Gemma-3-12B, Llama-3.3-70B, gpt-oss-120B, DeepSeek-R1).

## Reproducing the data from raw runs

The raw `*.jsonl` files in `../experiments/results/` are the per-dialogue logs. To regenerate the JSONs in this folder from the raw logs:

```bash
cd ../experiments
python3 export_all_benchmark_data.py    # all_benchmark_data.json
python3 export_figure2_data.py          # figure2_data.json
```

Both scripts read the raw `*.jsonl` files, recompute every aggregate, and overwrite the corresponding output JSON in this folder.

## Schema for `success_matrix` blocks

Every benchmark's `success_matrix` is a nested dict with rows = simulator at the user's seat, columns = agent under test, values = success rate in percent.

```json
"success_matrix": {
  "<sim_alias>": {
    "<agent_alias>": <pct in [0, 100]>,
    ...
  },
  ...
}
```

The canonical alias order is `panel`: `["llama3.1-8b", "qwen2.5-7b", "gemma3-12b", "llama3.3-70b", "gpt-oss-120b", "deepseek-r1"]`.
