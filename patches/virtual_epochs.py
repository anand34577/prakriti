"""Patches train_first.py / train_second.py for short 'virtual epochs'.

With PRAKRITI_STEPS_PER_EPOCH=N, each epoch stops after N batches of the
(reshuffled) loader and the LR scheduler is sized for N steps per epoch.
Checkpoints are written every epoch, so a crash costs at most one short
epoch instead of a multi-hour pass. Idempotent.
Run: python patches/virtual_epochs.py <StyleTTS2 dir>
"""
import re
import sys
from pathlib import Path

MARK = "# prakriti: virtual epochs"


def patch(path: Path):
    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print("already patched", path.name)
        return
    src = src.replace('"steps_per_epoch": len(train_dataloader),',
                      '"steps_per_epoch": int(os.environ.get("PRAKRITI_STEPS_PER_EPOCH", "0")) or len(train_dataloader),  '
                      + MARK, 1)

    def add_break(m):
        indent = m.group(1)
        return (f"{m.group(0)}\n{indent}    if int(os.environ.get('PRAKRITI_STEPS_PER_EPOCH', '0')) and "
                f"i >= int(os.environ['PRAKRITI_STEPS_PER_EPOCH']):  {MARK}\n{indent}        break")
    src, n = re.subn(r"^( *)for i, batch in enumerate\(train_dataloader\):", add_break, src, count=1, flags=re.M)
    assert n == 1, f"loop not found in {path}"
    # atomic checkpoint writes: a kill mid-save can never leave a corrupt file
    src = re.sub(r"^( *)torch\.save\(state, save_path\)$",
                 r"\1torch.save(state, save_path + '.tmp'); os.replace(save_path + '.tmp', save_path)  # prakriti: atomic",
                 src, flags=re.M)
    if not re.search(r"^import os$", src, re.M):
        src = "import os\n" + src
    path.write_text(src, encoding="utf-8")
    print("patched", path.name)


if __name__ == "__main__":
    d = Path(sys.argv[1])
    for f in ("train_first.py", "train_second.py"):
        patch(d / f)
