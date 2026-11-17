"""Session 2: Maya reads the corpus. Each sentence gets one voice, assigned so
every voice gets a balanced share of each (language, emotion) cell.
Output: work/s2/audio/<voice>/<id>.wav (24 kHz) + ledger_synth.jsonl. Resumable.
"""
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import save_wav

WORK = os.environ.get("PRAKRITI_WORK", "work")
S2 = os.path.join(WORK, "s2")
VOICES = ["Ahana", "Vaidehi", "Malini", "Charvi", "Anirban", "Nakul", "Vasant", "Siddhant"]
TAG_MAYA = {"laugh": "[laughs]", "sigh": "[sighs]", "gasp": "[gasps]", "hmm": "[hmm]"}
EMO_MAYA = {"sad": "[sad]", "excited": "[excited]", "happy": "[happy]", "angry": "[angry]",
            "whisper": "[whispers]", "fear": "[scared]", "surprised": "[surprised]"}
SEATS = int(os.environ.get("MAYA_SEATS", "64"))


def assign(rows):
    """Round-robin inside each (lang, emotion) cell -> balanced voices."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["lang"], r["emotion"], bool(r.get("tag")))].append(r)
    for i, (_, rs) in enumerate(sorted(cells.items())):
        for k, r in enumerate(rs):
            r["voice"] = VOICES[(k + i) % len(VOICES)]
    return rows


def maya_text(r):
    t = r["text"]
    for tag, m in TAG_MAYA.items():
        t = t.replace(f"[{tag}]", m)
    if r["emotion"] in EMO_MAYA:
        t = f"{EMO_MAYA[r['emotion']]} {t}"
    return t


def main():
    from maya3 import Maya3TTS
    rows = [json.loads(l) for l in open(os.path.join(S2, "corpus.jsonl"), encoding="utf-8")]
    rows = assign(rows)
    tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=SEATS)
    with ItemLedger(os.path.join(S2, "ledger_synth.jsonl"), fsync_every=64) as led:
        todo = [r for r in rows if r["id"] not in led]
        print(f"synth: {len(todo)} of {len(rows)} clips", flush=True)

        def one(r):
            d = os.path.join(S2, "audio", r["voice"])
            os.makedirs(d, exist_ok=True)
            wav = tts.generate(maya_text(r), lang="en" if r["lang"] == "en" else "hi",
                               voice=r["voice"], normalize=False, seed=int(r["id"][:8], 16) % 100000)
            dur = save_wav(__import__("pathlib").Path(d) / f"{r['id']}.wav", wav, 24000)
            return r, dur

        t0, audio = time.time(), 0.0
        with ThreadPoolExecutor(max_workers=SEATS) as ex:
            for n, (r, dur) in enumerate(ex.map(one, todo), 1):
                led.add(r["id"], voice=r["voice"], lang=r["lang"], emotion=r["emotion"], tag=r.get("tag"),
                        text=r["text"], audio_s=round(dur, 3))
                audio += dur
                if n % 2000 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} clips, {audio / 3600:.1f} h audio, {audio / el:.0f}x realtime", flush=True)
    tts.close()


if __name__ == "__main__":
    main()
