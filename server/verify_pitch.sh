#!/usr/bin/env bash
# Calibrated vs uncalibrated: same eval set, same scorers; plus listening samples.
set -euo pipefail
cd ~/vaani
export PYTHONPATH=$PWD/pk PRAKRITI_WORK=$PWD/work
M=$PWD/work/prakriti_model
PRAKRITI_EXPORT=$M PRAKRITI_EVAL_NAME=prakriti SKIP_SPEED=1 envs/kokoro/bin/python -m prakriti.s3.evaluate
envs/kokoro/bin/python -m prakriti.s3.samples $M work/samples_calibrated
LD_LIBRARY_PATH=$(cat envs/eval/cuda12_libs) envs/eval/bin/python -m prakriti.s1.score
envs/train/bin/python pk/prakriti/s3/tone.py prakriti > /dev/null
echo VERIFY_DONE
