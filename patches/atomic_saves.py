"""Makes every checkpoint save in train_first.py / train_second.py atomic
(write .tmp, then rename). Repairs a previous broken attempt. Idempotent.
Run: python patches/atomic_saves.py <StyleTTS2 dir>
"""
import ast
import re
import sys
from pathlib import Path

GOOD = "torch.save(state, save_path + '.tmp'); os.replace(save_path + '.tmp', save_path)  # prakriti: atomic"

for name in ("train_first.py", "train_second.py"):
    p = Path(sys.argv[1]) / name
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"^( *)torch\.save\(state, save_path \+ \.tmp\).*$", lambda m: m.group(1) + GOOD, s, flags=re.M)
    s = re.sub(r"^( *)torch\.save\(state, save_path\)$", lambda m: m.group(1) + GOOD, s, flags=re.M)
    if not re.search(r"^import os$", s, re.M):
        s = "import os\n" + s
    ast.parse(s)  # refuse to write anything that is not valid Python
    p.write_text(s, encoding="utf-8")
    print(name, "atomic saves:", s.count("# prakriti: atomic"))
