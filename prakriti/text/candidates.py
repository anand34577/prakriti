"""Candidate pronunciations per word, from eSpeak (with known-bug fixes) and our rules."""
from __future__ import annotations

import re
import unicodedata

from prakriti.text.hindi_g2p import DEVA_WORD, word_to_ipa

_G2P = {}


def _espeak(lang):
    if lang not in _G2P:
        from misaki import espeak
        _G2P[lang] = espeak.EspeakG2P(language=lang)
    return _G2P[lang]


def fix_espeak_hi(ipa: str) -> str:
    """Known eSpeak-hi bugs: ड़/ढ़ come out as 'r.' / 'r.h' (a stray pause)."""
    ipa = ipa.replace("r.h", "ɽʰ").replace("r.", "ɽ")
    return unicodedata.normalize("NFD", ipa)


def key(ipa: str) -> str:
    """Comparison key: ignore stress and the ʌ/ə stressed-schwa spelling."""
    return unicodedata.normalize("NFD", ipa).replace("ˈ", "").replace("ˌ", "").replace("ʌ", "ə")


def word_candidates(word: str) -> list[str]:
    """Distinct candidates, eSpeak first (the pretrained model's convention)."""
    if DEVA_WORD.fullmatch(word):
        cands = [fix_espeak_hi(_espeak("hi")(word)[0]), unicodedata.normalize("NFD", word_to_ipa(word))]
    elif re.fullmatch(r"[A-Za-z']+", word):
        cands = [_espeak("hi")(word)[0], _espeak("en-us")(word)[0]]
    else:
        return []
    out, seen = [], set()
    for c in cands:
        c = c.strip()
        if c and key(c) not in seen:
            seen.add(key(c))
            out.append(c)
    return out


DEVA_LETTERS = "ऀ-ॣ०-ॿ"  # excludes danda । and double danda ॥
TOKEN = re.compile(rf"[{DEVA_LETTERS}]+|[A-Za-z']+|[^\s{DEVA_LETTERS}A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)
