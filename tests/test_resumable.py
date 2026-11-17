import os
import pickle
import subprocess
import sys
from pathlib import Path

from prakriti.core.resumable import CheckpointManager, ItemLedger

REPO = Path(__file__).resolve().parents[1]


def test_ledger_survives_torn_line(tmp_path):
    p = tmp_path / "l.jsonl"
    with ItemLedger(p) as led:
        led.add("a")
        led.add("b")
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"id": "c", "half')  # simulated crash mid-write
    led = ItemLedger(p)
    assert "a" in led and "b" in led and "c" not in led
    assert list(led.pending(["a", "b", "c"])) == ["c"]
    led.close()


def test_checkpoint_falls_back_on_corruption(tmp_path):
    cm = CheckpointManager(tmp_path, keep_last=2)
    for step in (10, 20, 30):
        cm.save(step, lambda fn, s=step: pickle.dump({"s": s}, open(fn, "wb")))
    assert [s for s, _ in cm.list()] == [20, 30]
    cm.list()[-1][1].write_bytes(b"garbage")
    step, obj = cm.load_latest(lambda fn: pickle.load(open(fn, "rb")))
    assert step == 20 and obj == {"s": 20}


def test_pipeline_resumes_after_kill(tmp_path):
    """Start the pipeline, kill it mid-run, restart: it must continue, not restart."""
    env = {**os.environ, "PRAKRITI_WORK": str(tmp_path), "SELFTEST_DELAY": "0.02",
           "PRAKRITI_PIPELINE": str(REPO / "configs" / "pipeline_selftest.yaml"),
           "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-m", "prakriti.pipeline", "run"]
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=(os.name != "nt"))
    ledger = tmp_path / "selftest" / "items.jsonl"
    import time
    deadline = time.time() + 60
    while not (ledger.exists() and len(ledger.read_text().splitlines()) >= 30):
        assert proc.poll() is None, proc.stdout.read().decode()
        assert time.time() < deadline, "pipeline made no progress"
        time.sleep(0.05)
    # Hard-kill the runner AND its child stage (what systemd/reboot does).
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()
    done_before = len(ledger.read_text().splitlines())
    assert done_before < 100

    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = ledger.read_text().splitlines()
    assert len(lines) == 100  # nothing redone, nothing missing
    assert (tmp_path / "state" / "selftest.done").exists()

    again = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert "[skip] selftest" in again.stdout
