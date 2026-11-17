"""Converts a StyleTTS2 training checkpoint to Kokoro format and renders the
eval sentences with each fitted voice, into work/s1/audio/kokoro_ft/ +
ledger_kokoro_ft.jsonl (so score.py picks it up)."""
import sys

import numpy as np
import torch

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import REPO, S1, load_jobs, out_dir, render, save_wav

KEYS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def convert(ckpt_path: str, dst):
    net = torch.load(ckpt_path, map_location="cpu", weights_only=False)["net"]
    from prakriti.core.kokoro_keys import to_legacy
    out = {k: to_legacy({n.removeprefix("module."): t for n, t in net[k].items()}) for k in KEYS}
    torch.save(out, dst)


def main(ckpt):
    from kokoro import KModel, KPipeline
    conv = S1 / "train" / "kokoro_ft.pth"
    convert(ckpt, conv)
    cfg = REPO / "third_party" / "kikiri-tts" / "training" / "config.json"
    model = KModel(config=str(cfg), model=str(conv)).to("cuda").eval()
    pipes = {"h": KPipeline(lang_code="h", model=model), "a": KPipeline(lang_code="a", model=model)}
    odir = out_dir("kokoro_ft")
    voices = sorted((S1 / "train" / "voices").glob("*.pt"))
    with ItemLedger(S1 / "ledger_kokoro_ft.jsonl") as led:
        for vp in voices:
            for j in load_jobs():
                key = f"{vp.stem}__{j.id}"
                if key in led:
                    continue
                text = render(j, {})
                parts = [a for _, _, a in pipes["a" if j.lang == "en" else "h"](text, voice=str(vp)) if a is not None]
                if not parts:
                    led.add(key, failed=True)
                    continue
                dur = save_wav(odir / f"{key}.wav", np.concatenate([np.asarray(p) for p in parts]), 24000)
                led.add(key, voice=vp.stem, job=j.id, lang=j.lang, emotion=j.emotion, tag=j.tag,
                        text=text, ref=text, audio_s=dur)
    print("fine-tuned samples:", len(ItemLedger(S1 / "ledger_kokoro_ft.jsonl")))


if __name__ == "__main__":
    main(sys.argv[1])
