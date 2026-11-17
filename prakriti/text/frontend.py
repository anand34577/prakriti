"""Text -> phonemes, identical at training and inference time.

normalize (numbers, currency, time, acronyms) -> tokenize -> per word:
  1. audio-verified lexicon (built from teacher recordings)
  2. otherwise the default source: eSpeak with bug fixes (Devanagari) / eSpeak-hi reading (Latin)
Non-verbal markers [laugh] [sigh] [gasp] [hmm] [breath] become dedicated tokens
in otherwise-unused Kokoro vocabulary slots.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from prakriti.text.candidates import tokenize, word_candidates
from prakriti.text.normalize import normalize

NONVERBAL = {"laugh": "", "sigh": "", "gasp": "", "hmm": "", "breath": ""}
NONVERBAL_IDS = {"": 7, "": 8, "": 26, "": 27, "": 28}  # free Kokoro slots
_LEX = Path(__file__).with_name("lexicon.json")
LEXICON: dict[str, str] = json.loads(_LEX.read_text(encoding="utf-8")) if _LEX.exists() else {}
TAG = re.compile(r"\[(laugh|sigh|gasp|hmm|breath)s?\]")
PUNCT_MAP = {"।": ".", "॥": ".", "—": ",", "–": ","}


def word_ipa(word: str, choice: int | None = None) -> str:
    if choice is None and word in LEXICON:
        return LEXICON[word]
    c = word_candidates(word)
    if not c:
        return ""
    return c[min(choice or 0, len(c) - 1)]


def phonemize(text: str, lang_hint: str = "hi") -> str:
    text = normalize(text, default_lang="en" if lang_hint == "en" else "hi")
    parts = []
    for chunk in re.split(r"(\[\w+\])", text):
        m = TAG.fullmatch(chunk)
        if m:
            parts.append(NONVERBAL[m.group(1)])
            continue
        for tok in tokenize(chunk):
            if tok in PUNCT_MAP:
                parts.append(PUNCT_MAP[tok])
            elif re.fullmatch(r"[^\wऀ-ॿ]+", tok):
                parts.append(tok)
            else:
                parts.append(word_ipa(tok))
    out = " ".join(p for p in parts if p)
    out = re.sub(r" ([,.!?;:])", r"\1", out)
    return unicodedata.normalize("NFD", out)
