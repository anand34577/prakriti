"""Measures what makes a voice sound 'aggressive' on neutral sentences:
speaking rate, loudness, pitch level and swing, brightness (spectral centroid),
and pitch jumps. Compares models with real human recordings as the reference.
Usage: python -m prakriti.s3.tone <ledger_name>...  (e.g. vaani vaani_pre kokoro maya)
"""
import json
import os
import statistics as st
import sys
from pathlib import Path

import librosa
import numpy as np

W = Path(os.environ.get("PRAKRITI_WORK", os.environ.get("VAANI_WORK", "work")))
S1 = W / "s1"


def features(path, words):
    y, sr = librosa.load(path, sr=16000)
    y = librosa.effects.trim(y, top_db=35)[0]
    dur = len(y) / sr
    f0 = librosa.yin(y, fmin=70, fmax=450, sr=sr, frame_length=1024)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    voiced = f0[(rms[: len(f0)] > 0.02 * rms.max())] if len(rms) >= len(f0) else f0
    semis = 12 * np.log2(voiced / np.median(voiced))
    return {
        "words_per_s": words / max(dur, 0.1),
        "loud_db": float(20 * np.log10(np.mean(rms) + 1e-9)),
        "f0_median_hz": float(np.median(voiced)),
        "f0_range_semitones": float(np.percentile(semis, 95) - np.percentile(semis, 5)),
        "brightness_hz": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
    }


def summarize(rows):
    keys = rows[0].keys()
    return {k: round(st.mean(r[k] for r in rows), 2) for k in keys}


def main(names):
    out = {}
    for name in names:
        recs = [json.loads(l) for l in open(S1 / f"ledger_{name}.jsonl", encoding="utf-8")]
        recs = [r for r in recs if not r.get("failed") and r.get("emotion") == "neutral" and not r.get("tag")
                and r.get("lang") in ("hi", "mix")]
        by_voice = {}
        for r in recs:
            f = features(S1 / "audio" / name / f"{r['id']}.wav", len(r["ref"].split()))
            by_voice.setdefault(r["voice"], []).append(f)
        out[name] = {"ALL": summarize([x for v in by_voice.values() for x in v]),
                     **{v: summarize(xs) for v, xs in by_voice.items()}}
    # real human reference: Rasa neutral conversational clips
    real = [json.loads(l) for l in open(W / "s2" / "ledger_real.jsonl", encoding="utf-8")]
    real = [r for r in real if r.get("style") == "CONV"][:150]
    for voice in ("Rasa_F", "Rasa_M"):
        xs = [features(W / "s2" / "audio" / voice / f"{r['id']}.wav", len(r["text"].split()))
              for r in real if r["voice"] == voice]
        if xs:
            out.setdefault("REAL_rasa", {})[voice] = summarize(xs)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (W / "tone.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
