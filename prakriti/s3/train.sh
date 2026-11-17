#!/usr/bin/env bash
# Session 3 training: Stage 1 then Stage 2, each resuming from its newest checkpoint.
#   S1_EPOCHS x S1_STEPS  and  S2_EPOCHS x S2_STEPS  set the budget (virtual epochs).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${PRAKRITI_WORK:-$REPO/work}"
OUT="${S3_DIR:-$WORK/s3}"; LOGS="$OUT/logs"; mkdir -p "$LOGS"
PY="$REPO/envs/train/bin/python"
ACC="$REPO/envs/train/bin/accelerate launch --mixed_precision no --num_processes 1 --num_machines 1 --dynamo_backend no"
S1_EPOCHS=${S1_EPOCHS:-10}; S1_STEPS=${S1_STEPS:-700}
S2_EPOCHS=${S2_EPOCHS:-10}; S2_STEPS=${S2_STEPS:-400}; S2_JOINT=${S2_JOINT:-3}
S2_BATCH=${S2_BATCH:-6}; S2_SLM_MAXLEN=${S2_SLM_MAXLEN:-500}  # joint (SLM adversarial) phase needs extra memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$REPO"
cd "$REPO/third_party/kikiri-tts/StyleTTS2"

latest_valid() {  # latest_valid <glob-prefix>: newest checkpoint whose zip directory is intact
  "$PY" - "$@" <<'EOF'
import glob, os, sys, zipfile
for p in sorted(glob.glob(sys.argv[1] + "*.pth"), reverse=True):
    try:
        with zipfile.ZipFile(p) as z:
            z.namelist()
        print(p); break
    except Exception:
        os.replace(p, p + ".corrupt")
        print(f"[resume] set aside corrupt {os.path.basename(p)}", file=sys.stderr)
EOF
}
rm -f "$LOGS"/*.tmp

runcfg() {  # runcfg <out.yml> key=value ...  (yaml values parsed as yaml)
  "$PY" - "$OUT/config.yml" "$@" <<'EOF'
import sys, yaml
src, dst, *kv = sys.argv[1:]
c = yaml.safe_load(open(src))
for item in kv:
    k, v = item.split("=", 1)
    parts = k.split(".")  # dotted keys reach nested sections, e.g. loss_params.joint_epoch
    d = c
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = yaml.safe_load(v)
yaml.safe_dump(c, open(dst, "w"), allow_unicode=True, sort_keys=False)
EOF
}

# ---------- Stage 1 ----------
if [ ! -f "$LOGS/first_stage.pth" ]; then
  last=$(latest_valid "$LOGS/epoch_1st_")
  if [ -n "$last" ]; then
    echo "[stage1] resuming from $(basename "$last")"
    runcfg "$OUT/run1.yml" "epochs_1st=$S1_EPOCHS" "pretrained_model=$last" "load_only_params=false"
  else
    runcfg "$OUT/run1.yml" "epochs_1st=$S1_EPOCHS"
  fi
  PRAKRITI_STEPS_PER_EPOCH=$S1_STEPS $ACC train_first.py --config_path "$OUT/run1.yml" 2>&1 | tee -a "$OUT/stage1.log"
  [ -f "$LOGS/first_stage.pth" ] || { echo "stage 1 did not finish"; exit 1; }
fi

# ---------- Stage 2 ----------
batch=$S2_BATCH
while [ "$(find "$LOGS" -maxdepth 1 -name "epoch_2nd_*.pth" | wc -l)" -lt "$S2_EPOCHS" ]; do
  common=("epochs_2nd=$S2_EPOCHS" "epochs=$S2_EPOCHS" "loss_params.joint_epoch=$S2_JOINT"
          "batch_size=$batch" "slmadv_params.max_len=$S2_SLM_MAXLEN")
  last=$(latest_valid "$LOGS/epoch_2nd_")
  if [ -n "$last" ]; then
    echo "[stage2] resuming from $(basename "$last") at batch $batch"
    runcfg "$OUT/run2.yml" "${common[@]}" "pretrained_model=$last" "second_stage_load_pretrained=true" "load_only_params=false"
  else
    runcfg "$OUT/run2.yml" "${common[@]}" "second_stage_load_pretrained=false"
  fi
  mark=$(wc -l < "$OUT/stage2.log" 2>/dev/null || echo 0)
  if PRAKRITI_STEPS_PER_EPOCH=$S2_STEPS $ACC train_second.py --config_path "$OUT/run2.yml" 2>&1 | tee -a "$OUT/stage2.log"; then
    continue
  fi
  # self-healing: out of memory -> smaller batch, resume from the last checkpoint
  if tail -n +"$((mark + 1))" "$OUT/stage2.log" | grep -q "OutOfMemoryError" && [ "$batch" -gt 2 ]; then
    batch=$((batch - 1))
    echo "[stage2] out of memory -> retrying at batch $batch"
    continue
  fi
  echo "[stage2] failed (not memory, or batch already 2)"; exit 1
done
n2=$(find "$LOGS" -maxdepth 1 -name "epoch_2nd_*.pth" | wc -l)
[ "$n2" -ge "$S2_EPOCHS" ] || { echo "stage 2 incomplete ($n2/$S2_EPOCHS)"; exit 1; }
echo "[ok] training complete"
