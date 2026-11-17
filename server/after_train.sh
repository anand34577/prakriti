#!/usr/bin/env bash
# Restarts the pipeline once training is done so the finishing stages run.
W=~/prakriti/work
while [ ! -f $W/state/s3_train.done ]; do sleep 60; done
while systemctl is-active --quiet prakriti-pipeline; do sleep 20; done
sudo systemctl start prakriti-pipeline
echo "[$(date +%T)] finishing stages started" >> $W/logs/launcher.log
