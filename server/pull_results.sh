#!/usr/bin/env bash
# Run on YOUR machine: copies checkpoints, samples and logs off the server,
# so nothing is lost if the server disappears. rsync resumes partial files.
#   bash server/pull_results.sh prakriti-gpu      (an SSH host alias)
set -euo pipefail
HOST="${1:?usage: pull_results.sh <ssh-host>}"
DEST="${2:-./server_results}"
mkdir -p "$DEST"
rsync -az --partial --info=progress2 \
  --include='*/' \
  --include='state/***' --include='status.json' --include='logs/***' \
  --include='ckpt/latest.json' --include='ckpt/step_*.pt' \
  --include='samples/***' --include='reports/***' --include='export/***' \
  --exclude='*' \
  "$HOST:prakriti/work/" "$DEST/"
