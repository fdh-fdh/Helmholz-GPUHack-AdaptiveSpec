#!/usr/bin/env bash
set -euo pipefail

python src/benchmark.py \
  --mode target-only \
  --target-config config/target_model.yaml
