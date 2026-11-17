#!/usr/bin/env bash
# One-time server setup. Safe to re-run.
#   bash server/setup.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev ffmpeg sox espeak-ng tmux rsync git-lfs >/dev/null

if [ ! -d "$REPO/.venv" ]; then
  python3 -m venv "$REPO/.venv"
fi
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/requirements-server.txt"

mkdir -p "$REPO/work/logs"

# systemd unit: keeps the pipeline running when SSH drops, restarts after
# crashes, and starts again after a reboot. Finished work is never redone.
sed -e "s#__REPO__#$REPO#g" -e "s#__USER__#$USER_NAME#g" \
  "$REPO/server/prakriti-pipeline.service" | sudo tee /etc/systemd/system/prakriti-pipeline.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable prakriti-pipeline.service >/dev/null
echo "Setup done. Start with: sudo systemctl start prakriti-pipeline"
