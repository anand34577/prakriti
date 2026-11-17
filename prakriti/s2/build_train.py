"""Builds the session-3 training set from verified clips.

Keep rules (strict, quality first):
  - transcript error <= 0.10 (Hindi/Hinglish CER) or <= 0.12 (English WER)
  - no clipping, enough voiced audio, plausible speaking rate, 1-18 s
Labels: audio-verified phonemes (pron_check), with non-verbal tokens.
Also writes the pronunciation lexicon (majority vote across all clips) and
emotion reference lists (best clips per voice x emotion) for style packs.
"""
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import REPO
from prakriti.text.frontend import NONVERBAL

WORK = os.environ.get("PRAKRITI_WORK", "work")
S2 = Path(WORK) / "s2"
OUT = Path(WORK) / "s3"
KIKIRI = REPO / "third_party" / "kikiri-tts"
VOICES = ["Rasa_F", "Rasa_M", "Ahana", "Vaidehi", "Malini", "Charvi", "Anirban", "Nakul", "Vasant", "Siddhant"]


def keep(r, v):
    # real studio speech: the text is a human transcript, so allow slightly more ASR disagreement
    lim = 0.12 if r["lang"] == "en" else (0.15 if r.get("source") == "rasa" else 0.10)
    return (v["err"] <= lim and v["clip_frac"] < 0.001 and v["voiced_frac"] > 0.35
            and 0.9 <= v["words_per_s"] <= 4.5 and 1.0 <= r["audio_s"] <= 12.0)  # longer clips exceed 48 GB in stage 1


def build_lexicon(votes_paths, min_votes=2, min_share=0.7):
    agg = defaultdict(Counter)
    for line in (l for p in votes_paths for l in open(p, encoding="utf-8")):
        v = json.loads(line)
        if v["margin"] >= 0.5:  # ignore near-ties
            agg[v["word"]][v["winner"]] += 1
    lex = {}
    for w, c in agg.items():
        best, n = c.most_common(1)[0]
        if n >= min_votes and n / sum(c.values()) >= min_share:
            lex[w] = best
    return lex


def loudnorm(a, target_rms=0.06):
    a = a * (target_rms / (float(np.sqrt(np.mean(a ** 2))) + 1e-9))
    peak = float(np.max(np.abs(a))) + 1e-9
    return a * (0.95 / peak) if peak > 0.95 else a


def label(ipa, text):
    for tag, tok in NONVERBAL.items():
        if f"[{tag}]" in text:
            ipa = ipa.replace(f"[{tag}]", tok)
    ipa = re.sub(r"\[\w+\]", "", ipa)
    return " ".join(unicodedata.normalize("NFD", ipa).split())


def main(stage1_epochs=2, stage2_epochs=1, batch=6):
    from prakriti.s2.verify import all_clips
    synth = {r["id"]: r for r in all_clips()}
    from prakriti.s2.verify import load_merged
    ver = load_merged(S2, "verify")
    labels = load_merged(S2 / "pron", "labels")
    vocab = json.load(open(KIKIRI / "training" / "config.json"))["vocab"]
    allowed = set(vocab) | set(NONVERBAL.values())

    lex = build_lexicon(sorted((S2 / "pron").glob("votes*.jsonl")))
    (REPO / "prakriti" / "text" / "lexicon.json").write_text(json.dumps(lex, ensure_ascii=False, indent=0), encoding="utf-8")

    spk = {v: i for i, v in enumerate(VOICES)}
    lines, stats, emo_rank = [], Counter(), defaultdict(list)
    for cid, r in synth.items():
        v = ver.get(cid)
        if v is None or cid not in labels:
            stats["unverified"] += 1
            continue
        if not keep(r, v):
            stats["dropped"] += 1
            continue
        ipa = "".join(ch for ch in label(labels[cid]["ipa"], r["text"]) if ch in allowed)
        if len(ipa) < 5 or len(ipa) > 500:
            stats["bad_label"] += 1
            continue
        a, sr = sf.read(S2 / "audio" / r["voice"] / f"{cid}.wav", dtype="float32")
        dst = OUT / "audio" / r["voice"] / f"{cid}.wav"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, loudnorm(a), sr, subtype="PCM_16")
        lines.append(f"{r['voice']}/{cid}.wav|{ipa}|{spk[r['voice']]}")
        stats["kept"] += 1
        stats["kept_s"] += r["audio_s"]
        want = v.get("emo_want")
        score = v["emo"].get(want, 0.0) if want else v["emo"].get("neutral", 0.0)
        emo_rank[(r["voice"], r["emotion"])].append((score, f"{r['voice']}/{cid}.wav"))

    random.Random(42).shuffle(lines)
    n_val = 200
    (OUT / "val_list.txt").write_text("\n".join(lines[:n_val]) + "\n", encoding="utf-8")
    (OUT / "train_list.txt").write_text("\n".join(lines[n_val:]) + "\n", encoding="utf-8")
    (OUT / "OOD_texts.txt").write_text("\n".join(l.split("|")[1] for l in lines[:500]) + "\n", encoding="utf-8")
    refs = {f"{v}|{e}": [p for _, p in sorted(xs, reverse=True)[:40]] for (v, e), xs in emo_rank.items()}
    json.dump(refs, open(OUT / "style_refs.json", "w"), indent=1)
    json.dump(spk, open(OUT / "speakers.json", "w"), indent=1)

    cfg = yaml.safe_load(open(KIKIRI / "configs" / "config_german_ft.yml"))
    cfg.pop("training", None)
    cfg.update(batch_size=batch, epochs_1st=stage1_epochs, epochs_2nd=stage2_epochs, epochs=stage2_epochs,
               save_freq=1, max_len=400, log_dir=str(OUT / "logs"),
               pretrained_model=str(KIKIRI / "training" / "kokoro_base.pth"), first_stage_path="first_stage.pth")
    cfg["data_params"].update(train_data=str(OUT / "train_list.txt"), val_data=str(OUT / "val_list.txt"),
                              root_path=str(OUT / "audio"), OOD_data=str(OUT / "OOD_texts.txt"), num_workers=5)
    yaml.safe_dump(cfg, open(OUT / "config.yml", "w"), allow_unicode=True, sort_keys=False)
    stats["kept_h"] = round(stats.pop("kept_s") / 3600, 2)
    stats["lexicon_words"] = len(lex)
    json.dump(stats, open(OUT / "build_stats.json", "w"), indent=1)
    print(dict(stats))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
