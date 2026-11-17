"""Rule-based Hindi (Devanagari) -> IPA in Kokoro / eSpeak symbol conventions.

Handles: aspirates, retroflexes, nukta sounds (क़ ख़ ग़ ज़ फ़ ड़ ढ़), schwa
deletion (final + medial V C _ C V rule, right to left), anusvara
(homorganic nasal after schwa, vowel nasalisation elsewhere), chandrabindu,
visarga, conjuncts, and stress (rightmost heavy syllable; a final syllable
counts only if superheavy). Exceptions come from a lexicon file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CONS = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    "च": "c", "छ": "cʰ", "ज": "ɟ", "झ": "ɟʰ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    "त": "t", "थ": "tʰ", "द": "d", "ध": "dʰ", "न": "n", "ऩ": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    "य": "j", "र": "ɾ", "ऱ": "ɾ", "ल": "l", "ळ": "l", "ऴ": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "h",
}
NUKTA = {"क": "q", "ख": "x", "ग": "ɣ", "ज": "z", "फ": "f", "ड": "ɽ", "ढ": "ɽʰ", "य": "j", "र": "ɾ", "न": "n"}
PRECOMPOSED = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य"}
VOWELS = {"अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː", "ऋ": "ɾɪ", "ॠ": "ɾiː",
          "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː", "ऑ": "ɔ", "ऍ": "æ", "ऎ": "eː", "ऒ": "oː"}
MATRAS = {"ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː", "ृ": "ɾɪ", "ॄ": "ɾiː",
          "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ॉ": "ɔ", "ॅ": "æ", "ॆ": "eː", "ॊ": "oː"}
VIRAMA, NUKTA_SIGN, ANUSVARA, CHANDRA, VISARGA = "्", "़", "ं", "ँ", "ः"
LONG = {"aː", "iː", "uː", "eː", "ɛː", "oː", "ɔː"}
HOMORGANIC = {}
for _n, _group in (("ŋ", "k kʰ ɡ ɡʰ q x ɣ"), ("ɲ", "c cʰ ɟ ɟʰ"), ("ɳ", "ʈ ʈʰ ɖ ɖʰ"),
                   ("n", "t tʰ d dʰ"), ("m", "p pʰ b bʰ f")):
    for _c in _group.split():
        HOMORGANIC[_c] = _n
KEEP_FINAL_AFTER = {"ɾ", "j", "ʋ"}  # पत्र, राज्य, तत्त्व keep final schwa

_LEX_PATH = Path(__file__).with_name("hindi_lexicon.json")
LEXICON: dict[str, str] = json.loads(_LEX_PATH.read_text(encoding="utf-8")) if _LEX_PATH.exists() else {}


def _units(word: str):
    """Parse into [kind, ipa] units. kind: C (consonant), V (vowel), N (nasal mark), H (visarga)."""
    for a, b in PRECOMPOSED.items():
        word = word.replace(a, b + NUKTA_SIGN)
    u = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in CONS:
            ipa = CONS[ch]
            if i + 1 < len(word) and word[i + 1] == NUKTA_SIGN:
                ipa = NUKTA.get(ch, ipa)
                i += 1
            u.append(["C", ipa])
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt in MATRAS:
                u.append(["V", MATRAS[nxt]])
                i += 1
            elif nxt == VIRAMA:
                i += 1
            else:
                u.append(["V", "ə", "inherent"])
        elif ch in VOWELS:
            u.append(["V", VOWELS[ch]])
        elif ch in (ANUSVARA, CHANDRA):
            u.append(["N", "anusvara" if ch == ANUSVARA else "chandra"])
        elif ch == VISARGA:
            u.append(["C", "h"])
        elif ch == "ॐ":
            u += [["V", "oː"], ["C", "m"]]
        i += 1
    return u


def _delete_schwas(u):
    vidx = [i for i, x in enumerate(u) if x[0] == "V"]
    # final inherent schwa
    if len(vidx) > 1 and len(u[vidx[-1]]) == 3 and vidx[-1] == len(u) - 1:
        prev = [x[1] for x in u[:vidx[-1]] if x[0] == "C"][-2:]
        cluster = len(u) >= 3 and u[-2][0] == "C" and u[-3][0] == "C"
        if not (cluster and prev and prev[-1] in KEEP_FINAL_AFTER):
            u[vidx[-1]][1] = None
    # medial: right to left, V C [ə] C V, never two adjacent deletions
    last_deleted = False
    for k in range(len(u) - 2, 1, -1):
        x = u[k]
        if x[0] != "V" or len(x) != 3 or x[1] is None:
            last_deleted = False
            continue
        if last_deleted:
            last_deleted = False
            continue
        left_c = u[k - 1][0] == "C"
        left_v = u[k - 2][0] == "V" and u[k - 2][1] is not None or u[k - 2][0] == "N"
        r = k + 1
        if r < len(u) and u[r][0] == "N":
            continue  # nasalised schwa is never deleted
        right_c = r < len(u) and u[r][0] == "C"
        right_v = r + 1 < len(u) and u[r + 1][0] == "V" and u[r + 1][1] is not None
        if left_c and left_v and right_c and right_v:
            x[1] = None
            last_deleted = True
    return u


def _segments(u):
    """Resolve nasal marks; return list of (kind, ipa) with deleted vowels removed."""
    out = []
    for i, x in enumerate(u):
        if x[0] == "V" and x[1] is None:
            continue
        if x[0] == "N":
            nxt = next((y for y in u[i + 1:] if not (y[0] == "V" and y[1] is None)), None)
            prev_v = next((y for y in reversed(out) if y[0] == "V"), None)
            if (x[1] == "anusvara" and prev_v and prev_v[1] == "ə" and nxt and nxt[0] == "C"
                    and nxt[1] in HOMORGANIC):
                out.append(("C", HOMORGANIC[nxt[1]]))
            elif x[1] == "anusvara" and prev_v and prev_v[1] == "ə" and nxt and nxt[0] == "C" and nxt[1] in ("j", "ɾ", "l", "ʋ", "ʃ", "s", "h"):
                out.append(("C", "n"))
            elif prev_v:
                k = len(out) - 1 - out[::-1].index(prev_v)
                v = prev_v[1]
                nas = v[0] + "̃" + v[1:]
                out[k] = ("V", nas)
            continue
        out.append((x[0], x[1]))
    return out


def _syllabify(seg):
    """Greedy onset-maximal syllables over (kind, ipa)."""
    vpos = [i for i, s in enumerate(seg) if s[0] == "V"]
    if not vpos:
        return [seg]
    bounds = [0]
    for a, b in zip(vpos, vpos[1:]):
        cons = b - a - 1
        bounds.append(a + 1 + (cons - 1 if cons >= 1 else 0))
    bounds.append(len(seg))
    return [seg[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def _stress(sylls):
    def weight(s, final):
        v = next(x[1] for x in s if x[0] == "V")
        vi = [x[0] for x in s].index("V")
        coda = len(s) - vi - 1
        long = v.replace("̃", "") in LONG
        if final:
            return 2 if (long and coda) or coda >= 2 else 0
        return 1 if long or coda else 0
    if len(sylls) <= 1:
        return 0
    w = [weight(s, i == len(sylls) - 1) for i, s in enumerate(sylls)]
    heavy = [i for i, x in enumerate(w) if x]
    return heavy[-1] if heavy else 0


def word_to_ipa(word: str, stress: bool = True) -> str:
    if word in LEXICON:
        return LEXICON[word]
    seg = _segments(_delete_schwas(_units(word)))
    if not seg:
        return ""
    sylls = _syllabify(seg)
    st = _stress(sylls) if stress else -1
    parts = []
    for i, s in enumerate(sylls):
        txt = "".join(x[1] for x in s)
        if i == st and len(sylls) > 1:
            vi = [x[0] for x in s].index("V")
            onset = "".join(x[1] for x in s[:vi])
            rest = "".join(x[1] for x in s[vi:]).replace("ə", "ʌ", 1)
            txt = onset + "ˈ" + rest
        parts.append(txt)
    ipa = "".join(parts)
    ipa = re.sub(r"iː$", "i", ipa)  # eSpeak convention: final ई is short
    return ipa


DEVA_WORD = re.compile(r"[ऀ-ॣ॰-ॿ]+")


def sentence_to_ipa(text: str) -> str:
    """Devanagari words -> IPA; everything else (punctuation, Latin) left as is."""
    return DEVA_WORD.sub(lambda m: word_to_ipa(m.group(0)), text)
