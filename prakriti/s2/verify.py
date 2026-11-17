"""Session 2 quality gate for every generated clip (batched on GPU).

Checks: Whisper large-v3-turbo transcript vs text (CER/WER), emotion2vec+
emotion probabilities, and signal sanity (clipping, silence, speaking rate).
Writes verify.jsonl; keep/drop thresholds are applied later in build_train.py.
"""
import json
import os
import re

import numpy as np
import torch

from prakriti.core.resumable import ItemLedger
from prakriti.s1.score import EMO_MAP, roman

WORK = os.environ.get("PRAKRITI_WORK", "work")
S2 = os.path.join(WORK, "s2")
BATCH = int(os.environ.get("VERIFY_BATCH", "64"))


def _shard():
    i, n = os.environ.get("SHARD", "0/1").split("/")
    return int(i), int(n)


def in_shard(cid: str) -> bool:
    import hashlib
    i, n = _shard()
    return n == 1 or int(hashlib.md5(cid.encode()).hexdigest(), 16) % n == i


def shard_suffix() -> str:
    i, n = _shard()
    return "" if n == 1 else f".{i}"


def load_merged(directory, stem):
    """Merge <stem>.jsonl and <stem>.<k>.jsonl ledgers into {id: record}."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, f"{stem}.jsonl")) +
                    glob.glob(os.path.join(directory, f"{stem}.[0-9]*.jsonl"))):
        for r in ItemLedger(p).records():
            out[r["id"]] = r
    return out


def all_clips():
    """Teacher clips + real recordings, one record per clip."""
    recs = []
    for name in ("ledger_synth.jsonl", "ledger_real.jsonl"):
        p = os.path.join(S2, name)
        if os.path.exists(p):
            recs += ItemLedger(p).records()
    return recs


def signal_stats(a, sr):
    frame = sr // 50
    rms = np.sqrt(np.convolve(a ** 2, np.ones(frame) / frame, mode="valid")[::frame] + 1e-12)
    voiced = rms > 0.01
    return dict(peak=round(float(np.max(np.abs(a))), 3), clip_frac=round(float(np.mean(np.abs(a) > 0.99)), 5),
                voiced_frac=round(float(voiced.mean()), 3),
                lead_sil=round(int(np.argmax(voiced)) / 50, 2) if voiced.any() else 0.0,
                trail_sil=round(int(np.argmax(voiced[::-1])) / 50, 2) if voiced.any() else 0.0)


class ASR:
    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        mid = "openai/whisper-large-v3-turbo"
        self.proc = WhisperProcessor.from_pretrained(mid)
        self.model = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def __call__(self, wavs16, lang):
        feats = self.proc(wavs16, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        ids = self.model.generate(feats, language=lang, task="transcribe", max_new_tokens=200)
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


def main():
    import jiwer
    import soundfile as sf
    import torchaudio
    from funasr import AutoModel

    asr = ASR()
    emo = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, disable_pbar=True)
    recs = [r for r in all_clips() if in_shard(r["id"])]
    with ItemLedger(os.path.join(S2, f"verify{shard_suffix()}.jsonl"), fsync_every=BATCH) as out:
        todo = [r for r in recs if r["id"] not in out]
        print(f"verify: {len(todo)} of {len(recs)}", flush=True)
        for lang in ("hi", "en"):
            group = [r for r in todo if (r["lang"] == "en") == (lang == "en")]
            for b in range(0, len(group), BATCH):
                chunk = group[b:b + BATCH]
                paths = [os.path.join(S2, "audio", r["voice"], f"{r['id']}.wav") for r in chunk]
                audio = [sf.read(p, dtype="float32") for p in paths]
                w16 = [torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy() for a, sr in audio]
                hyps = asr(w16, lang)
                emos = emo.generate(w16, granularity="utterance", extract_embedding=False, batch_size=len(w16))
                for r, (a, sr), hyp, e in zip(chunk, audio, hyps, emos):
                    ref = re.sub(r"\[\w+\]", "", r["text"])
                    if lang == "en":
                        norm = lambda s: " ".join(re.sub(r"[^a-z' ]+", " ", s.lower()).split())
                        err = jiwer.wer(norm(ref), norm(hyp)) if norm(hyp) else 1.0
                    else:
                        err = jiwer.cer(roman(ref), roman(hyp)) if roman(hyp) else 1.0
                    probs = {l.split("/")[-1]: round(float(s), 4) for l, s in zip(e["labels"], e["scores"])}
                    out.add(r["id"], err=round(float(min(err, 1.0)), 4), hyp=hyp, emo=probs,
                            emo_want=EMO_MAP.get(r["emotion"]),
                            words_per_s=round(len(ref.split()) / max(r["audio_s"], 0.1), 2), **signal_stats(a, sr))
                if (b // BATCH) % 30 == 0:
                    print(f"  {lang}: {b + len(chunk)}/{len(group)}", flush=True)


if __name__ == "__main__":
    main()
