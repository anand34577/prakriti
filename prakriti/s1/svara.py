"""Teacher test: Svara-TTS v1 (Orpheus-style Llama-3B + SNAC 24 kHz).

Prompt/codec layout follows Kenpath/svara-tts-inference (encoder.py, codec.py).
Batched HF generate with left padding.
"""
import json
import time

import torch

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1, load_jobs, out_dir, render, save_wav

BOS, EOT, AUDIO = 128000, 128009, 156939
SOS, EOS_SPEECH, SOH, EOH, SOA, PAD = 128257, 128258, 128259, 128260, 128261, 128263
AUDIO_START = 128266
VOICES = {"Hindi (Female)": "hi", "Hindi (Male)": "hi",
          "English (Female)": "en", "English (Male)": "en"}
EMO = {"happy": "<happy>", "excited": "<happy>", "sad": "<sad>", "angry": "<anger>", "fear": "<fear>"}
TAGS = {"laugh": "<laugh>", "sigh": "<sigh>", "gasp": "<gasp>", "cry": "<sniffle>"}
BATCH = 16


def build(tok, speaker, text):
    ids = tok(f"{speaker}: {text}", add_special_tokens=False).input_ids
    return [BOS, SOH, AUDIO, *ids, EOH, EOT, SOA, SOS]


def decode(snac, toks):
    toks = [t for t in toks if t >= AUDIO_START]
    toks = toks[: len(toks) // 7 * 7]
    if not toks:
        return None
    c = torch.tensor(toks).view(-1, 7) - AUDIO_START - torch.arange(7) * 4096
    if (c < 0).any() or (c >= 4096).any():
        c = c.clamp(0, 4095)
    l0, l1, l2 = c[:, 0], c[:, [1, 4]].reshape(-1), c[:, [2, 3, 5, 6]].reshape(-1)
    with torch.inference_mode():
        audio = snac.decode([x.unsqueeze(0).cuda() for x in (l0, l1, l2)])
    return audio.squeeze().float().cpu().numpy()


def main():
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("kenpath/svara-tts-v1")
    model = AutoModelForCausalLM.from_pretrained("kenpath/svara-tts-v1", torch_dtype=torch.bfloat16).cuda().eval()
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    odir = out_dir("svara")

    work = []
    for spk, vlang in VOICES.items():
        for j in load_jobs():
            if (j.lang == "en") != (vlang == "en"):
                continue  # Hindi voices read hi/mix/rom, English voices read en
            text = render(j, TAGS)
            if j.emotion in EMO:
                text = f"{text} {EMO[j.emotion]}"  # Svara: style tag at the end
            key = f"{spk.replace(' ', '').replace('(', '_').replace(')', '')}__{j.id}"
            work.append((key, spk, j, text))

    with ItemLedger(S1 / "ledger_svara.jsonl") as led:
        todo = [w for w in work if w[0] not in led]
        print(f"svara: {len(todo)} of {len(work)} clips", flush=True)
        t0, total_audio = time.time(), 0.0
        for b in range(0, len(todo), BATCH):
            chunk = todo[b:b + BATCH]
            seqs = [build(tok, spk, text) for _, spk, _, text in chunk]
            L = max(map(len, seqs))
            inp = torch.tensor([[PAD] * (L - len(s)) + s for s in seqs]).cuda()
            att = (inp != PAD).long()
            tb = time.time()
            with torch.inference_mode():
                out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=1800,
                                     do_sample=True, temperature=0.6, top_p=0.9,
                                     repetition_penalty=1.1, eos_token_id=EOS_SPEECH, pad_token_id=PAD)
            per = (time.time() - tb) / len(chunk)
            for (key, spk, j, text), row in zip(chunk, out[:, L:].tolist()):
                if EOS_SPEECH in row:
                    row = row[: row.index(EOS_SPEECH)]
                wav = decode(snac, row)
                if wav is None:
                    led.add(key, failed=True, voice=spk, job=j.id)
                    continue
                dur = save_wav(odir / f"{key}.wav", wav, 24000)
                total_audio += dur
                led.add(key, voice=spk, job=j.id, lang=j.lang, emotion=j.emotion, tag=j.tag,
                        text=text, ref=render(j, {}), audio_s=dur, gen_s=per)
            print(f"  batch {b // BATCH + 1}: {len(chunk)} clips", flush=True)
        wall = time.time() - t0
    if todo:
        (S1 / "speed_svara.json").write_text(json.dumps(
            {"clips": len(todo), "audio_s": total_audio, "wall_s": wall,
             "x_realtime": total_audio / max(wall, 1e-6), "batch": BATCH}, indent=2))


if __name__ == "__main__":
    main()
