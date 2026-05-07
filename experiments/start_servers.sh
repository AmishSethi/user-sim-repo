#!/usr/bin/env bash
# Start three vLLM OpenAI-compatible servers, one per backbone model.
#
# Edit before first use:
#   * CUDA_VISIBLE_DEVICES on each line: pick a free GPU on your machine
#     (the defaults of 9 / 0 / 3 were picked for one specific shared cluster
#      and almost certainly do not match your hardware).
#   * --gpu-memory-utilization on each line: lower it if you are sharing
#     a GPU with other workloads, raise it if you have the GPU to yourself.
#     The Llama-3.1-8B server in bf16 with KV cache for max-len 4096 needs
#     about 0.40 of an 80 GB H100; the two smaller models need about 0.20.
#   * REPO_ROOT below: this should be the directory containing this script.
#   * conda env name: replace 'vllm-final' if your env is named differently.
#
# Run from anywhere with:    bash <path-to>/start_servers.sh
# Servers go to ports 8011 (Llama), 8012 (Qwen), 8013 (Phi). Logs go to
# server_logs/ alongside this script. Each server takes 1-2 min on a warm
# HuggingFace cache, longer on first download.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$REPO_ROOT/server_logs"

# Activate conda env (script may be invoked from a non-interactive shell).
# Comment out if you manage your environment differently.
source /usr/local/conda/etc/profile.d/conda.sh
conda activate vllm-final

# ---------------------------------------------------------------------------
# Server 1: Llama-3.1-8B-Instruct  ->  port 8011  (alias "llama3.1-8b")
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=9 nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8011 --host 127.0.0.1 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --disable-log-requests \
  > "$REPO_ROOT/server_logs/llama8b.log" 2>&1 &
echo "started llama8b PID $!"

sleep 4

# ---------------------------------------------------------------------------
# Server 2: Qwen2.5-3B-Instruct  ->  port 8012  (alias "qwen2.5-3b")
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8012 --host 127.0.0.1 \
  --gpu-memory-utilization 0.18 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --disable-log-requests \
  > "$REPO_ROOT/server_logs/qwen3b.log" 2>&1 &
echo "started qwen3b PID $!"

sleep 4

# ---------------------------------------------------------------------------
# Server 3: Phi-4-mini-instruct  ->  port 8013  (alias "phi4-mini")
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-4-mini-instruct \
  --port 8013 --host 127.0.0.1 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --disable-log-requests \
  > "$REPO_ROOT/server_logs/phi4mini.log" 2>&1 &
echo "started phi4mini PID $!"

echo
echo "All three vLLM servers are launching."
echo "First-time model downloads can take several minutes; subsequent runs"
echo "load from the HuggingFace cache in 30-90 seconds per model."
echo
echo "Wait for readiness with:"
echo "  for p in 8011 8012 8013; do"
echo "    until curl -sf http://127.0.0.1:\$p/v1/models > /dev/null; do sleep 5; done"
echo "  done"
