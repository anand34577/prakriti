"""Renders the session-1 evaluation set with the trained Prakriti model (every
voice; emotion lines use the voice's emotion pack) into work/s1/audio/prakriti
+ ledger_prakriti.jsonl, so score.py / report.py compare it with stock Kokoro and
the teachers on identical sentences. Also measures speed on GPU and CPU.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1, load_jobs, out_dir, render, save_wav
from prakriti.tts import SR, Prakriti

EXP = Path(os.environ.get("PRAKRITI_EXPORT", Path(os.environ.get("PRAKRITI_WORK", "work")) / "export"))
NAME = os.environ.get("PRAKRITI_EVAL_NAME", "prakriti")
NV = {"laugh": "[laugh]", "sigh": "[sigh]", "gasp": "[gasp]", "hmm": "[hmm]"}


def main():
    tts = Prakriti(EXP, device="cuda")
    odir = out_dir(NAME)
    with ItemLedger(S1 / f"ledger_{NAME}.jsonl") as led:
        for v in tts.voices:
            for j in load_jobs():
                key = f"{v}__{j.id}"
                if key in led:
                    continue
                text = render(j, NV)
                a = tts.synthesize(text, voice=v, emotion=j.emotion if j.emotion in tts.emotions(v) else "neutral")
                if not len(a):
                    led.add(key, failed=True)
                    continue
                dur = save_wav(odir / f"{key}.wav", a, SR)
                led.add(key, voice=v, job=j.id, lang=j.lang, emotion=j.emotion, tag=j.tag,
                        text=text, ref=render(j, {}), audio_s=dur)
    if os.environ.get("SKIP_SPEED"):
        return
    # speed: same probe sentences as the stock-Kokoro baseline
    probe = [render(j, {}) for j in load_jobs() if j.lang in ("hi", "mix")][:10]
    speed = {}
    for dev, threads in (("cuda", None), ("cpu", 4), ("cpu", 2)):
        if threads:
            torch.set_num_threads(threads)
        t = Prakriti(EXP, device=dev)
        t.synthesize(probe[0], voice="Rasa_F")
        lat, audio = [], 0.0
        t0 = time.time()
        for p in probe:
            s = time.time()
            a = t.synthesize(p, voice="Rasa_F")
            lat.append(time.time() - s)
            audio += len(a) / SR
        wall = time.time() - t0
        speed[f"{dev}{threads or ''}"] = {"x_realtime": round(audio / wall, 2),
                                         "median_sentence_latency_s": round(float(np.median(lat)), 3)}
        print(dev, threads, speed[f"{dev}{threads or ''}"], flush=True)
    (S1 / "speed_prakriti.json").write_text(json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
