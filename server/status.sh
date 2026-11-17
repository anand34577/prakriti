#!/usr/bin/env bash
# One-screen progress view.  Live:  watch -n 10 bash ~/prakriti/server/status.sh
W="$HOME/prakriti/work"
echo "== GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader | awk -F', ' '{printf "  util %s | mem %s / %s | %s C | %s\n",$1,$2,$3,$4,$5}'
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/  /'
echo "== Running jobs (tmux)"
tmux ls 2>/dev/null | sed 's/^/  /' || echo "  none"
echo "== Session 1 progress"
for f in kokoro maya svara kokoro_ft; do
  [ -f "$W/s1/ledger_$f.jsonl" ] && printf "  %-10s clips generated: %s\n" "$f" "$(wc -l < "$W/s1/ledger_$f.jsonl")"
done
[ -f "$W/s1/scores.jsonl" ] && echo "  clips scored: $(wc -l < "$W/s1/scores.jsonl") (of ~1170)"
if [ -f "$W/s1/train/stage1.log" ]; then
  echo "== Fine-tuning trial"
  for s in 1 2; do
    L="$W/s1/train/stage$s.log"
    [ -f "$L" ] && echo "  stage $s: $(grep -aoE 'Epoch \[[0-9]+/[0-9]+\], Step \[[0-9]+/[0-9]+\][^\r]{0,120}' "$L" | tail -1)"
    [ -f "$L" ] && grep -aqE "OutOfMemory|Traceback" "$L" && echo "  stage $s: last run hit an error (see $L)"
  done
  [ -f "$W/s1/train/timing.json" ] && echo "  finished: $(cat "$W/s1/train/timing.json")"
fi
echo "== Disk"; df -h / | awk 'NR==2{print "  used "$3" of "$2" ("$5")"}'
