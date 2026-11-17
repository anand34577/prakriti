"""After training: export model, build voice and voice x emotion style packs.

work/export/
  prakriti.pth        Kokoro-format weights (legacy weight-norm names, no 'module.')
  config.json      Kokoro config + non-verbal token ids
  voices/<voice>.pt and voices/<voice>__<emotion>.pt   [510, 1, 256] style packs
  voices.json      gender / source / emotions per voice
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from prakriti.core.kokoro_keys import to_legacy
from prakriti.s1.common import REPO
from prakriti.text.frontend import NONVERBAL_IDS

WORK = Path(os.environ.get("PRAKRITI_WORK", "work"))
S3 = WORK / "s3"
EXP = Path(os.environ.get("PRAKRITI_EXPORT", WORK / "export"))
KIKIRI = REPO / "third_party" / "kikiri-tts"
TRAIN_PY = REPO / "envs" / "train" / "bin" / "python"
KEYS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
GENDER = {"Rasa_F": "female", "Rasa_M": "male", "Ahana": "female", "Vaidehi": "female", "Malini": "female",
          "Charvi": "female", "Anirban": "male", "Nakul": "male", "Vasant": "male", "Siddhant": "male"}
MIN_REFS = 15


def latest_ckpt():
    cks = sorted((S3 / "logs").glob("epoch_2nd_*.pth"))
    if not cks:
        sys.exit("no stage-2 checkpoint")
    return cks[-1]


def pack(ckpt, audio_dir, out):
    subprocess.run([str(TRAIN_PY), str(KIKIRI / "scripts" / "extract_voicepack.py"), "--model", str(ckpt),
                    "--audio-dir", str(audio_dir), "--output", str(out), "--device", "cuda"],
                   check=True, capture_output=True)


def main(ckpt=None):
    ckpt = Path(ckpt) if ckpt else latest_ckpt()
    print("checkpoint:", ckpt.name)
    (EXP / "voices").mkdir(parents=True, exist_ok=True)
    net = torch.load(ckpt, map_location="cpu", weights_only=False)["net"]
    torch.save({k: to_legacy({n.removeprefix("module."): t for n, t in net[k].items()}) for k in KEYS},
               EXP / "prakriti.pth")
    cfg = json.load(open(KIKIRI / "training" / "config.json"))
    cfg["vocab"].update(NONVERBAL_IDS)
    json.dump(cfg, open(EXP / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    refs = json.load(open(S3 / "style_refs.json"))
    meta = {}
    for voice_dir in sorted((S3 / "audio").iterdir()):
        v = voice_dir.name
        out = EXP / "voices" / f"{v}.pt"
        if not out.exists():
            neutral = refs.get(f"{v}|neutral", [])
            if len(neutral) >= MIN_REFS:  # base voice = its most neutral-sounding clips
                with tempfile.TemporaryDirectory() as td:
                    for p in neutral:
                        os.symlink(S3 / "audio" / p, Path(td) / Path(p).name)
                    pack(ckpt, td, out)
            else:
                pack(ckpt, voice_dir, out)
        emos = []
        for key, paths in refs.items():
            rv, emo = key.split("|")
            if rv != v or emo == "neutral" or len(paths) < MIN_REFS:
                continue
            eout = EXP / "voices" / f"{v}__{emo}.pt"
            if not eout.exists():
                with tempfile.TemporaryDirectory() as td:
                    for p in paths:
                        os.symlink(S3 / "audio" / p, Path(td) / Path(p).name)
                    pack(ckpt, td, eout)
            emos.append(emo)
        meta[v] = {"gender": GENDER.get(v, "unknown"), "source": "real studio (Rasa, CC-BY-4.0)"
                   if v.startswith("Rasa") else "synthetic (Maya 2 Native, Apache-2.0)", "emotions": sorted(emos)}
        print(v, "emotions:", ", ".join(sorted(emos)) or "-", flush=True)
    json.dump(meta, open(EXP / "voices.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
