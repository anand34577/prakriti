#!/usr/bin/env bash
# Pre-adversarial (stage-2 sub-epoch 3) vs final checkpoint, same evaluation set.
set -euo pipefail
cd ~/prakriti
export PRAKRITI_WORK=$PWD/work PYTHONPATH=$PWD
PRAKRITI_EXPORT=$PWD/work/export_pre envs/kokoro/bin/python -m prakriti.s3.finish work/s3/logs/epoch_2nd_00002.pth
PRAKRITI_EXPORT=$PWD/work/export_pre PRAKRITI_EVAL_NAME=prakriti_pre SKIP_SPEED=1 envs/kokoro/bin/python -m prakriti.s3.evaluate
LD_LIBRARY_PATH=$(cat envs/eval/cuda12_libs) envs/eval/bin/python -m prakriti.s1.score
envs/eval/bin/python prakriti/s3/summary.py > work/compare_summary.txt
echo COMPARE_DONE
