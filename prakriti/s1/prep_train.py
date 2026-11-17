"""Builds a StyleTTS2/kikiri training set from teacher clips.

Output (under work/s1/train/):
  audio/<voice>/<clip>.wav   24 kHz mono 16-bit, loudness-matched
  train_list.txt / val_list.txt   "<voice>/<clip>.wav|<ipa>|<int speaker id>"
  OOD_texts.txt                   IPA lines for style-diffusion sampling
  config.yml                      derived from kikiri's config, paths absolute
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import REPO, S1

KIKIRI = REPO / "third_party" / "kikiri-tts"
OUT = S1 / "train"


def loudnorm(a, target_rms=0.06):
    rms = float(np.sqrt(np.mean(a ** 2)) + 1e-9)
    a = a * (target_rms / rms)
    peak = float(np.max(np.abs(a)) + 1e-9)
    return a * (0.95 / peak) if peak > 0.95 else a


def main(teacher="maya", epochs1=3, epochs2=2, batch=8):
    from misaki import espeak
    g2p = {"hi": espeak.EspeakG2P(language="hi"), "en": espeak.EspeakG2P(language="en-us")}
    vocab = json.load(open(KIKIRI / "training" / "config.json"))["vocab"]

    recs = [r for r in ItemLedger(S1 / f"ledger_{teacher}.jsonl").records() if not r.get("failed")]
    voices = sorted({r["voice"] for r in recs})
    spk_id = {v: i for i, v in enumerate(voices)}
    lines, dropped = [], 0
    for r in recs:
        if r["audio_s"] < 1.0 or r["audio_s"] > 25:
            dropped += 1
            continue
        lang = "en" if r["lang"] == "en" else "hi"
        ipa, _ = g2p[lang](r["ref"])
        ipa = "".join(c for c in ipa if c in vocab)
        if len(ipa) < 5:
            dropped += 1
            continue
        a, sr = sf.read(S1 / "audio" / teacher / f"{r['id']}.wav", dtype="float32")
        assert sr == 24000, sr
        dst = OUT / "audio" / r["voice"] / f"{r['id']}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, loudnorm(a), 24000, subtype="PCM_16")
        lines.append(f"{r['voice']}/{r['id']}.wav|{ipa}|{spk_id[r['voice']]}")

    random.Random(42).shuffle(lines)
    n_val = max(8, len(lines) // 20)
    (OUT / "val_list.txt").write_text("\n".join(lines[:n_val]) + "\n", encoding="utf-8")
    (OUT / "train_list.txt").write_text("\n".join(lines[n_val:]) + "\n", encoding="utf-8")
    ood = [l.split("|")[1] for l in lines[:200]]
    (OUT / "OOD_texts.txt").write_text("\n".join(ood) + "\n", encoding="utf-8")
    json.dump(spk_id, open(OUT / "speakers.json", "w"), indent=2)

    cfg = yaml.safe_load(open(KIKIRI / "configs" / "config_german_ft.yml"))
    cfg.pop("training", None)  # dead nested block (kikiri pitfall)
    cfg.update(batch_size=batch, epochs_1st=epochs1, epochs_2nd=epochs2, epochs=epochs2,
               save_freq=1, max_len=400, log_dir=str(OUT / "logs"),
               pretrained_model=str(KIKIRI / "training" / "kokoro_base.pth"),
               first_stage_path="first_stage.pth")
    cfg["data_params"].update(train_data=str(OUT / "train_list.txt"), val_data=str(OUT / "val_list.txt"),
                              root_path=str(OUT / "audio"), OOD_data=str(OUT / "OOD_texts.txt"),
                              num_workers=4)
    yaml.safe_dump(cfg, open(OUT / "config.yml", "w"), allow_unicode=True, sort_keys=False)
    print(f"train {len(lines) - n_val}, val {n_val}, dropped {dropped}, speakers {spk_id}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
