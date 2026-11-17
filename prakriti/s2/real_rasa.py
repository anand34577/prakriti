"""Imports Rasa Hindi (AI4Bharat, CC-BY-4.0): real expressive studio speech.

Each parquet shard is downloaded, decoded, resampled 48k -> 24k, edge-trimmed,
written to work/s2/audio/Rasa_<F|M>/<id>.wav, recorded in ledger_real.jsonl,
then the shard is deleted from the HF cache to save disk. Resumable per shard.
"""
import io
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

from prakriti.core.resumable import ItemLedger

WORK = os.environ.get("PRAKRITI_WORK", "work")
S2 = os.path.join(WORK, "s2")
REPO_ID = "ai4bharat/Rasa"
STYLE = {"SAD": "sad", "HAPPY": "happy", "ANGER": "angry", "FEAR": "fear", "SURPRISE": "surprised",
         "NEWS": "serious", "CONV": "neutral", "BOOK": "neutral", "WIKI": "neutral", "INDIC": "neutral",
         "PROPER NOUN": "neutral", "COMMAND": "serious", "DISGUST": None}  # disgust not in our emotion set


def trim(a, sr, thresh=0.008, pad=0.12):
    frame = int(sr * 0.01)
    rms = np.sqrt(np.convolve(a ** 2, np.ones(frame) / frame, mode="same"))
    idx = np.where(rms > thresh)[0]
    if not len(idx):
        return a
    s, e = max(0, idx[0] - int(pad * sr)), min(len(a), idx[-1] + int(pad * sr))
    return a[s:e]


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO_ID).siblings
                   if s.rfilename.startswith("Hindi/train-"))
    shard_led = ItemLedger(os.path.join(S2, "ledger_real_shards.jsonl"))
    with ItemLedger(os.path.join(S2, "ledger_real.jsonl"), fsync_every=200) as led:
        for f in files:
            if f in shard_led:
                continue
            p = hf_hub_download(REPO_ID, f, repo_type="dataset")
            pf = pq.ParquetFile(p)
            n = 0
            for batch in pf.iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    emo = STYLE.get(r["style"], "neutral")
                    if emo is None:
                        continue
                    voice = "Rasa_F" if r["gender"].lower().startswith("f") else "Rasa_M"
                    cid = "rasa_" + r["filename"]
                    if cid in led:
                        continue
                    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    if a.ndim > 1:
                        a = a.mean(1)
                    a = torchaudio.functional.resample(torch.from_numpy(a), sr, 24000).numpy()
                    a = trim(a, 24000)
                    d = os.path.join(S2, "audio", voice)
                    os.makedirs(d, exist_ok=True)
                    sf.write(os.path.join(d, f"{cid}.wav"), a, 24000, subtype="PCM_16")
                    led.add(cid, voice=voice, lang="hi", emotion=emo, tag=None, text=" ".join(r["text"].split()),
                            audio_s=round(len(a) / 24000, 3), source="rasa", style=r["style"])
                    n += 1
            shard_led.add(f, clips=n)
            shard_led.flush()
            os.remove(os.path.realpath(p))  # free disk; audio is now in our own wavs
            print(f"{f}: {n} clips, total {len(led)}", flush=True)
    shard_led.close()


if __name__ == "__main__":
    main()
