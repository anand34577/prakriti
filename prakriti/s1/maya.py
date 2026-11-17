"""Teacher test: Maya 2 Native. 8 voices x eval set, concurrent (GPU-batched)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1, load_jobs, out_dir, render, save_wav

VOICES = {  # 4 female + 4 male, all trained on Hindi and English
    "Ahana": "f", "Vaidehi": "f", "Malini": "f", "Charvi": "f",
    "Anirban": "m", "Nakul": "m", "Vasant": "m", "Siddhant": "m",
}
TAGS = {"laugh": "[laughs]", "sigh": "[sighs]", "whisper": "[whispers]",
        "gasp": "[gasps]", "hmm": "[hmm]", "cry": "[cries]"}
EMO_TAGS = {"sad": "[sad]", "excited": "[excited]", "happy": "[happy]", "angry": "[angry]",
            "whisper": "[whispers]", "fear": "[scared]", "surprised": "[surprised]"}
SR = 24000


def main():
    from maya3 import Maya3TTS
    tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=32)
    jobs = load_jobs()
    odir = out_dir("maya")
    work = []
    for v in VOICES:
        for j in jobs:
            text = render(j, TAGS)
            if j.emotion in EMO_TAGS:
                text = f"{EMO_TAGS[j.emotion]} {text}"
            work.append((f"{v}__{j.id}", v, j, text))

    with ItemLedger(S1 / "ledger_maya.jsonl") as led:
        todo = [w for w in work if w[0] not in led]
        print(f"maya: {len(todo)} of {len(work)} clips to generate", flush=True)

        def one(w):
            key, v, j, text = w
            t0 = time.time()
            wav = tts.generate(text, lang="en" if j.lang == "en" else "hi", voice=v,
                               normalize=False, seed=1234)
            dur = save_wav(odir / f"{key}.wav", wav, SR)
            return key, dict(voice=v, job=j.id, lang=j.lang, emotion=j.emotion, tag=j.tag,
                             text=text, ref=render(j, {}), audio_s=dur, gen_s=time.time() - t0)

        t0 = time.time()
        total_audio = 0.0
        with ThreadPoolExecutor(max_workers=32) as ex:
            for key, info in ex.map(one, todo):
                led.add(key, **info)
                total_audio += info["audio_s"]
        wall = time.time() - t0
    tts.close()
    if todo:
        (S1 / "speed_maya.json").write_text(json.dumps(
            {"clips": len(todo), "audio_s": total_audio, "wall_s": wall,
             "x_realtime": total_audio / max(wall, 1e-6)}, indent=2))
        print(f"maya: {total_audio/60:.1f} min audio in {wall/60:.1f} min "
              f"({total_audio/wall:.1f}x realtime, 32 concurrent)", flush=True)


if __name__ == "__main__":
    main()
