"""Scores every generated clip: intelligibility (Whisper CER/WER),
emotion (emotion2vec+), naturalness (UTMOS, English-trained so only relative).
Resumable per clip. Writes work/s1/scores.jsonl.
"""
import json
import re
import unicodedata

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1

EMO_MAP = {"happy": "happy", "excited": "happy", "sad": "sad", "angry": "angry",
           "fear": "fearful", "surprised": "surprised", "calm": "neutral", "serious": "neutral"}


def roman(s: str) -> str:
    """Devanagari -> plain Latin, lowercase, no punctuation (script-neutral CER)."""
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    s = re.sub(r"[ऀ-ॿ]+", lambda m: transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.IAST), s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    # collapse spelling variation common in romanized Hindi
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("w", "v"), ("z", "j"), ("ph", "f")):
        s = s.replace(a, b)
    return " ".join(s.split())


def main():
    import jiwer
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    try:
        from funasr import AutoModel
        emo = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
    except Exception as e:
        print("emotion model unavailable:", e)
        emo = None
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda().eval()
    except Exception as e:
        print("UTMOS unavailable:", e)
        utmos = None

    with ItemLedger(S1 / "scores.jsonl") as out:
        for model in ("kokoro", "maya", "svara", "kokoro_ft", "prakriti", "prakriti_pre"):
            lp = S1 / f"ledger_{model}.jsonl"
            if not lp.exists():
                continue
            for rec in ItemLedger(lp).records():
                key = f"{model}/{rec['id']}"
                if key in out or rec.get("failed"):
                    continue
                wav_path = S1 / "audio" / model / f"{rec['id']}.wav"
                lang = "en" if rec["lang"] == "en" else "hi"
                segs, _ = asr.transcribe(str(wav_path), language=lang, beam_size=1)
                hyp = " ".join(s.text for s in segs).strip()
                ref = rec["ref"]
                if lang == "en":
                    norm = lambda s: re.sub(r"[^a-z' ]+", " ", s.lower())
                    err = jiwer.wer(norm(ref), norm(hyp))
                else:
                    err = jiwer.cer(roman(ref), roman(hyp)) if roman(hyp) else 1.0
                row = dict(model=model, voice=rec["voice"], job=rec["job"], lang=rec["lang"],
                           emotion=rec["emotion"], tag=rec["tag"], audio_s=rec["audio_s"],
                           hyp=hyp, err=round(float(err), 4))
                if emo is not None:
                    r = emo.generate(str(wav_path), granularity="utterance", extract_embedding=False)[0]
                    labels = [l.split("/")[-1] for l in r["labels"]]
                    probs = dict(zip(labels, map(float, r["scores"])))
                    row["emo_top"] = max(probs, key=probs.get)
                    want = EMO_MAP.get(rec["emotion"])
                    if want:
                        row["emo_p"] = round(probs.get(want, 0.0), 4)
                if utmos is not None:
                    a, sr = sf.read(wav_path, dtype="float32")
                    with torch.inference_mode():
                        row["mos"] = round(float(utmos(torch.from_numpy(a).cuda()[None], sr)), 3)
                out.add(key, **row)
    print("scored", len(ItemLedger(S1 / "scores.jsonl")), flush=True)


if __name__ == "__main__":
    main()
