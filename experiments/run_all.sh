#!/usr/bin/env bash
# Run the full experimental pipeline end-to-end.
#
# Assumes start_servers.sh has already been edited for your hardware. Will
# launch the servers, wait for them to be ready, then run both experiments
# and produce the figures in ../figures/.
#
# Total wall-clock on a single H100 with a warm HuggingFace cache:
#   start_servers.sh    ~ 90 s
#   run_experiment_1.py ~ 150 s
#   analyze_exp1.py     ~ 5 s
#   run_experiment_2.py ~ 150 s
# = roughly 6-7 minutes total.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== launching vLLM servers ==="
bash "$REPO_ROOT/start_servers.sh"

echo "=== waiting for servers (max 5 min each) ==="
for p in 8011 8012 8013; do
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$p/v1/models" > /dev/null 2>&1; then
      echo "  port $p ready"
      break
    fi
    sleep 5
  done
done

echo "=== experiment 1: simulator-induced ranking variance ==="
python run_experiment_1.py

echo "=== experiment 1 analysis ==="
python analyze_exp1.py

echo "=== experiment 2: n-gram fidelity probe ==="
python run_experiment_2_ngram.py

echo
echo "=== done ==="
echo "Figures written to ../figures/"
echo "Aggregated results in results/"
