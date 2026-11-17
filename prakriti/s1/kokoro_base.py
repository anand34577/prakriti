"""Student baseline: stock Kokoro-82M. Quality samples on GPU + speed on GPU and CPU.

CPU speed is measured with 4 threads as a rough stand-in for an RK3588 /
RK3576 (real board numbers come from the board benchmark script later).
"""
import json
import time

import numpy as np
import torch

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1, load_jobs, out_dir, render, save_wav

HI_VOICES = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]
EN_VOICES = ["af_heart", "af_bella", "am_michael", "am_fenrir"]
SR = 24000


def synth(pipe, text, voice):
    parts = [a for _, _, a in pipe(text, voice=voice) if a is not None]
    return np.concatenate([np.asarray(p) for p in parts]) if parts else None


def main():
    from kokoro import KPipeline
    pipes = {"h": KPipeline(lang_code="h", device="cuda"), "a": KPipeline(lang_code="a", device="cuda")}
    odir = out_dir("kokoro")
    jobs = load_jobs()
    with ItemLedger(S1 / "ledger_kokoro.jsonl") as led:
        for j in jobs:
            code, voices = ("a", EN_VOICES) if j.lang == "en" else ("h", HI_VOICES)
            for v in voices:
                key = f"{v}__{j.id}"
                if key in led:
                    continue
                text = render(j, {})
                wav = synth(pipes[code], text, v)
                if wav is None:
                    led.add(key, failed=True)
                    continue
                dur = save_wav(odir / f"{key}.wav", wav, SR)
                led.add(key, voice=v, job=j.id, lang=j.lang, emotion=j.emotion, tag=j.tag,
                        text=text, ref=text, audio_s=dur)

    # Speed: same 10 sentences, warm, GPU then CPU (4 threads)
    probe = [render(j, {}) for j in jobs if j.lang in ("hi", "mix")][:10]
    speed = {}
    for dev, threads in (("cuda", None), ("cpu", 4), ("cpu", 2)):
        if threads:
            torch.set_num_threads(threads)
        p = KPipeline(lang_code="h", device=dev)
        synth(p, probe[0], "hf_alpha")  # warm-up
        t0, audio = time.time(), 0.0
        first_chunk = []
        for t in probe:
            ts = time.time()
            w = synth(p, t, "hf_alpha")
            first_chunk.append(time.time() - ts)
            audio += len(w) / SR
        wall = time.time() - t0
        speed[f"{dev}{threads or ''}"] = {"rtf": wall / audio, "x_realtime": audio / wall,
                                         "median_sentence_latency_s": float(np.median(first_chunk))}
        print(dev, threads, speed[f"{dev}{threads or ''}"], flush=True)
    (S1 / "speed_kokoro.json").write_text(json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
