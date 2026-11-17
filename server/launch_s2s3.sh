#!/usr/bin/env bash
# Waits for the running textgen + Rasa jobs, proves training resume works on a
# tiny run, then hands everything to the systemd pipeline.
set -uo pipefail
cd ~/prakriti
W=$PWD/work
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $W/logs/launcher.log; }

while tmux has-session -t textgen 2>/dev/null || tmux has-session -t rasa 2>/dev/null; do sleep 30; done
log "textgen + rasa finished"
grep -q "coverage round" $W/logs/textgen.log || { log "textgen did not reach coverage; stop"; exit 1; }
[ "$(wc -l < $W/s2/ledger_real_shards.jsonl)" -ge 25 ] || { log "rasa incomplete; stop"; exit 1; }

# ---- resume test on the session-1 trial data (tiny epochs) ----
T=$W/s3test; rm -rf $T; mkdir -p $T
envs/train/bin/python - <<EOF
import yaml
c = yaml.safe_load(open("$W/s1/train/config.yml")); c["log_dir"] = "$T/logs"
yaml.safe_dump(c, open("$T/config.yml", "w"), allow_unicode=True, sort_keys=False)
EOF
export S3_DIR=$T S1_EPOCHS=3 S1_STEPS=4 S2_EPOCHS=2 S2_STEPS=3 PRAKRITI_WORK=$W
setsid bash prakriti/s3/train.sh > $T/run_a.log 2>&1 &
PID=$!
for i in $(seq 1 120); do [ -n "$(ls $T/logs/epoch_1st_*.pth 2>/dev/null)" ] && break; sleep 5; done
kill -9 -- -$PID 2>/dev/null; sleep 5
log "killed test run after: $(ls $T/logs | tr '\n' ' ')"
bash prakriti/s3/train.sh > $T/run_b.log 2>&1
if grep -q "\[resume\] continuing at epoch" $T/run_b.log && grep -q "training complete" $T/run_b.log; then
  log "RESUME TEST PASSED"
else
  log "RESUME TEST FAILED - see $T/run_b.log"; exit 1
fi
unset S3_DIR S1_EPOCHS S1_STEPS S2_EPOCHS S2_STEPS

# ---- hand over to systemd ----
mkdir -p $W/state
for s in s2_textgen s2_rasa; do echo '{"by":"launcher"}' > $W/state/$s.done; done
sudo systemctl reset-failed prakriti-pipeline 2>/dev/null
sudo systemctl start prakriti-pipeline
log "pipeline started under systemd"
