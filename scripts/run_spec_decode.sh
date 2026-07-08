#!/usr/bin/env bash
set -euo pipefail

python src/benchmark.py \
  --mode spec-decode \
  --target-config config/target_model.yaml \
  --draft-config config/draft_model.yaml
