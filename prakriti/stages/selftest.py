"""Proves resume works: processes items + fake training with checkpoints.

Kill it at any moment (Ctrl+C, kill -9, reboot) and run again: it continues
where it stopped instead of starting over.
"""
import os
import pickle
import time
from pathlib import Path

from prakriti.core.resumable import CheckpointManager, ItemLedger

work = Path(os.environ.get("PRAKRITI_WORK", "work")) / "selftest"
delay = float(os.environ.get("SELFTEST_DELAY", "0.05"))

with ItemLedger(work / "items.jsonl") as ledger:
    for item in ledger.pending(f"clip_{i:04d}" for i in range(100)):
        time.sleep(delay)
        ledger.add(item, ok=True)
    print(f"items done: {len(ledger)}/100")

ckpt = CheckpointManager(work / "ckpt", keep_last=2)


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


step, s = ckpt.load_latest(load)
s = s or {"loss": 10.0}
print(f"resuming training from step {step}")
while step < 200:
    step += 1
    s["loss"] *= 0.99
    time.sleep(delay)
    if step % 20 == 0:
        ckpt.save(step, lambda p: pickle.dump(s, open(p, "wb")))
print(f"training done at step {step}, loss {s['loss']:.3f}")
