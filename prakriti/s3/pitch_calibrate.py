"""Per-voice pitch calibration.

Each voice's target is the median pitch of its own neutral training recordings.
The model's output pitch is measured on neutral test sentences, and
pitch_scale = target / measured is written into voices.json. A second pass
verifies the calibrated output lands on the target.

Usage: python -m prakriti.s3.pitch_calibrate <model_dir> <s3_dir>
"""
import json
import statistics as st
import sys
from pathlib import Path

import librosa
import numpy as np

from prakriti.tts import SR, Prakriti

PROBE = [
    "नमस्ते, आज मौसम बहुत सुहाना है।",
    "क्या आप मुझे बता सकते हैं कि रेलवे स्टेशन कहाँ है?",
    "मुझे लगता है कि हमें थोड़ा और इंतज़ार करना चाहिए।",
    "कल मेरी office में एक important meeting है।",
    "आपका order confirm हो गया है, delivery कल तक हो जाएगी।",
    "दादी माँ हर रात हमें एक नई कहानी सुनाती थीं।",
]


def median_f0(y, sr):
    y = librosa.effects.trim(y, top_db=35)[0]
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=450, sr=sr, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


def main(model_dir, s3_dir):
    model_dir, s3 = Path(model_dir), Path(s3_dir)
    tts = Prakriti(model_dir)
    refs = json.loads((s3 / "style_refs.json").read_text())
    meta_path = model_dir / "voices.json"
    meta = json.loads(meta_path.read_text())
    report = {}
    for v in tts.voices:
        clips = refs.get(f"{v}|neutral", [])[:25]
        target = st.median(median_f0(*librosa.load(s3 / "audio" / c, sr=16000)) for c in clips)
        meta.setdefault(v, {})["pitch_scale"] = 1.0
        tts.meta = meta
        before = st.median(median_f0(tts.synthesize(t, voice=v), SR) for t in PROBE)
        scale = round(target / before, 3)
        meta[v]["pitch_scale"] = scale
        tts.meta = meta
        after = st.median(median_f0(tts.synthesize(t, voice=v), SR) for t in PROBE)
        report[v] = {"training_hz": round(target), "model_hz": round(before), "scale": scale,
                     "calibrated_hz": round(after), "semitones": round(12 * np.log2(scale), 1)}
        print(v, report[v], flush=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    (model_dir / "pitch_calibration.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
