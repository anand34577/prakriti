#!/usr/bin/env bash
# Session 1 fine-tuning trial: short Stage 1 + Stage 2 on teacher clips.
# Measures step time, peak VRAM, loss trend; extracts voicepacks; renders samples.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${PRAKRITI_WORK:-$REPO/work}"
PY="$REPO/envs/train/bin/python"
KIK="$REPO/third_party/kikiri-tts"
OUT="$WORK/s1/train"
export PYTHONPATH="$REPO" PRAKRITI_WORK="$WORK"

cd "$KIK"
[ -f training/kokoro_base.pth ] || "$PY" scripts/prepare_training.py convert-weights
"$PY" scripts/prepare_training.py patch-styletts2 >/dev/null
"$PY" -m prakriti.s1.prep_train maya

# peak-VRAM sampler
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/vram.log"; sleep 5; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

cd "$KIK/StyleTTS2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACC="$REPO/envs/train/bin/accelerate launch --mixed_precision ${PRECISION:-bf16} --num_processes 1 --num_machines 1 --dynamo_backend no"
t0=$(date +%s)
[ -f "$OUT/logs/first_stage.pth" ] || $ACC train_first.py --config_path "$OUT/config.yml" 2>&1 | tee "$OUT/stage1.log" | grep -E "Epoch|Loss|Error|error" | tail -20
t1=$(date +%s)
[ -n "$(ls $OUT/logs/epoch_2nd_*.pth 2>/dev/null)" ] || $ACC train_second.py --config_path "$OUT/config.yml" 2>&1 | tee "$OUT/stage2.log" | grep -E "Epoch|Loss|Error|error" | tail -20
t2=$(date +%s)
echo "{\"stage1_s\": $((t1-t0)), \"stage2_s\": $((t2-t1)), \"peak_vram_mib\": $(sort -n $OUT/vram.log | tail -1)}" > "$OUT/timing.json"
cat "$OUT/timing.json"

CK=$(ls "$OUT"/logs/epoch_2nd_*.pth | tail -1)
mkdir -p "$OUT/voices" "$OUT/samples"
for d in "$OUT"/audio/*/; do
  v=$(basename "$d")
  [ -f "$OUT/voices/$v.pt" ] || "$PY" "$KIK/scripts/extract_voicepack.py" --model "$CK" --audio-dir "$d" --output "$OUT/voices/$v.pt" --device cuda | tail -1
done
"$REPO/envs/kokoro/bin/python" -m prakriti.s1.finetuned_samples "$CK"
