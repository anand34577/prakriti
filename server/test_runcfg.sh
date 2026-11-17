#!/usr/bin/env bash
# Checks train.sh's runcfg helper writes nested keys where the trainer reads them.
set -euo pipefail
cd ~/prakriti
PY=envs/train/bin/python
OUT=$(mktemp -d)
cp work/s1/train/config.yml "$OUT/config.yml"
eval "$(sed -n '/^runcfg() {/,/^}/p' prakriti/s3/train.sh)"
runcfg "$OUT/out.yml" "loss_params.joint_epoch=2" "epochs_2nd=4"
$PY -c "
import yaml, sys
c = yaml.safe_load(open('$OUT/out.yml'))
ok = c['loss_params']['joint_epoch'] == 2 and c['epochs_2nd'] == 4 and 'joint_epoch' not in c
print('runcfg nested keys:', 'OK' if ok else 'FAIL', c['loss_params']['joint_epoch'], c['epochs_2nd'])
sys.exit(0 if ok else 1)
"
rm -rf "$OUT"
