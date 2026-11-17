"""Resumable stage runner.

    python -m prakriti.pipeline run      # run all stages, skipping finished ones
    python -m prakriti.pipeline status   # print progress (safe to run any time)
    python -m prakriti.pipeline reset <stage>

Each stage is a shell command that must itself be resumable (it uses
ItemLedger / CheckpointManager). The runner adds stage-level skipping,
retries, a heartbeat file and a lock so two copies never run at once.
On the server it runs under systemd, so it survives SSH disconnects,
crashes and reboots.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from prakriti.core.resumable import StageState, atomic_write_json

ROOT = Path(os.environ.get("PRAKRITI_WORK", "work")).resolve()
PIPELINE_FILE = Path(os.environ.get("PRAKRITI_PIPELINE", "configs/pipeline.yaml"))


def _load() -> list[dict]:
    with open(PIPELINE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["stages"]


def _status(**fields) -> None:
    path = ROOT / "status.json"
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    cur.update(fields, heartbeat=time.time())
    atomic_write_json(path, cur)


def _acquire_lock():
    """OS-level lock: released automatically if the process dies, so a crash
    never leaves a stale lock behind."""
    fh = open(ROOT / "pipeline.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("pipeline already running")
    return fh


def run() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock()
    state = StageState(ROOT)
    try:
        for st in _load():
            name = st["name"]
            if state.is_done(name):
                print(f"[skip] {name} already done", flush=True)
                continue
            retries = int(st.get("retries", 2))
            for attempt in range(1, retries + 2):
                print(f"[run] {name} attempt {attempt}: {st['cmd']}", flush=True)
                _status(stage=name, attempt=attempt, state="running")
                env = {**os.environ, "PRAKRITI_WORK": str(ROOT), "PRAKRITI_STAGE": name}
                cmd = st["cmd"]
                if cmd.startswith("python "):  # always use this venv's interpreter
                    cmd = f'"{sys.executable}" {cmd[len("python "):]}'
                rc = subprocess.call(cmd, shell=True, env=env)
                if rc == 0:
                    state.mark_done(name, {"attempts": attempt})
                    _status(stage=name, state="done")
                    break
                print(f"[fail] {name} exit {rc}", flush=True)
                time.sleep(min(60, 10 * attempt))
            else:
                _status(stage=name, state="failed")
                return 1  # systemd restarts us later; finished work is kept
        _status(stage=None, state="all_done")
        print("[ok] all stages done", flush=True)
        return 0
    finally:
        lock.close()


def status() -> int:
    state = StageState(ROOT)
    for st in _load():
        mark = "done" if state.is_done(st["name"]) else "todo"
        print(f"  [{mark}] {st['name']}")
    p = ROOT / "status.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        s["seconds_since_heartbeat"] = round(time.time() - s.pop("heartbeat", 0))
        print(json.dumps(s, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    r = sub.add_parser("reset")
    r.add_argument("stage")
    a = ap.parse_args()
    if a.cmd == "run":
        return run()
    if a.cmd == "status":
        return status()
    StageState(ROOT).reset(a.stage)
    print(f"reset {a.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
