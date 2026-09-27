#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/train_real_dataset.sh DATASET_ROOT [options]" >&2
    exit 1
fi
DATASET_ROOT="$1"
shift
python "$PROJECT_ROOT/scripts/reconstruct_dataset.py" \
    --dataset-root "$DATASET_ROOT" \
    --record-glob '**/scale_noaxis_8' \
    --touches 20 \
    --output-root "$PROJECT_ROOT/outputs/real_world_dataset" "$@"
