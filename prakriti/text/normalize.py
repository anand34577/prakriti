"""Text normalization for Hindi / Hinglish / English (v1).

Turns digits, currency, percentages and clock times into words, choosing the
spoken language from the surrounding words:
    "मेरे पास ₹1500 हैं"        -> "मेरे पास एक हज़ार पाँच सौ रुपये हैं"
    "I paid ₹1500"              -> "I paid one thousand five hundred rupees"
    "मीटिंग 5:30 बजे है"         -> "मीटिंग साढ़े पाँच बजे है"
"""
from __future__ import annotations

import re

HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानबे बानबे तिरानबे चौरानबे पंचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे"
).split()
assert len(HI_0_99) == 100

EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def hindi_number(n: int) -> str:
    if n < 0:
        return "माइनस " + hindi_number(-n)
    if n < 100:
        return HI_0_99[n]
    parts = []
    for value, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            parts.append(f"{hindi_number(n // value)} {word}")
            n %= value
    if n:
        parts.append(HI_0_99[n])
    return " ".join(parts)


def english_number(n: int) -> str:
    if n < 0:
        return "minus " + english_number(-n)
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return EN_TENS[t] + (f" {EN_ONES[o]}" if o else "")
    parts = []
    for value, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= value:
            parts.append(f"{english_number(n // value)} {word}")
            n %= value
    if n:
        parts.append(english_number(n))
    return " ".join(parts)


def _digits(s: str, lang: str) -> str:
    """Digit-by-digit reading (phone numbers, long codes)."""
    f = (lambda d: HI_0_99[int(d)]) if lang == "hi" else (lambda d: EN_ONES[int(d)])
    return " ".join(f(d) for d in s)


def say_number(s: str, lang: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        point = "दशमलव" if lang == "hi" else "point"
        return f"{say_number(whole or '0', lang)} {point} {_digits(frac, lang)}"
    if len(s) > 9 or (len(s) > 1 and s.startswith("0")):
        return _digits(s, lang)
    n = int(s)
    return hindi_number(n) if lang == "hi" else english_number(n)


def hindi_time(h: int, m: int) -> str:
    """Natural Hindi clock reading: सवा / साढ़े / पौने / डेढ़ / ढाई."""
    nxt = h % 12 + 1
    if m == 0:
        return hindi_number(h)
    if m == 15:
        return f"सवा {hindi_number(h)}"
    if m == 30:
        return {1: "डेढ़", 2: "ढाई"}.get(h, f"साढ़े {hindi_number(h)}")
    if m == 45:
        return f"पौने {hindi_number(nxt)}"
    return f"{hindi_number(h)} बजकर {hindi_number(m)} मिनट"


def english_time(h: int, m: int) -> str:
    if m == 0:
        return f"{english_number(h)} o'clock"
    mm = f"oh {english_number(m)}" if m < 10 else english_number(m)
    return f"{english_number(h)} {mm}"


def context_lang(text: str, start: int, end: int, default: str = "en") -> str:
    """Language of a span, judged by the nearest words on each side."""
    left, right = text[max(0, start - 40):start], text[end:end + 40]
    lw, rw = left.split()[-2:], right.split()[:2]
    score = 0
    for w in lw + rw:
        if DEVA.search(w):
            score += 1
        elif LATIN.search(w):
            score -= 1
    if score > 0:
        return "hi"
    if score < 0:
        return "en"
    return "hi" if DEVA.search(text) else default


NUM = r"\d[\d,]*(?:\.\d+)?"
_RULES = [
    ("time", re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")),
    ("rupee", re.compile(rf"(?:₹|\bRs\.?\s?|\bINR\s?)\s?({NUM})")),
    ("dollar", re.compile(rf"\$\s?({NUM})")),
    ("percent", re.compile(rf"({NUM})\s?%")),
    ("number", re.compile(NUM)),
]


SAID_AS_WORD = {"ISRO", "NASA", "UNESCO", "NATO", "AIIMS", "UNICEF", "SEBI", "NABARD", "LASER", "RADAR",
                "COVID", "OPEC", "BHIM", "GAIL", "NEET", "JEE", "SAARC", "ASEAN", "FIFA"}
ACRONYM = re.compile(r"\b[A-Z]{2,6}s?\b")


def spell_acronyms(text: str) -> str:
    """SBI -> S B I, HDFC -> H D F C; words like ISRO / NASA stay whole; 'ATMs' -> 'A T Ms'."""
    def f(m):
        w = m.group(0)
        core, plural = (w[:-1], "s") if w.endswith("s") and w[:-1].isupper() else (w, "")
        if core in SAID_AS_WORD or len(core) < 2:
            return w
        letters = list(core)
        if plural:
            letters[-1] += "s"
        return " ".join(letters)
    return ACRONYM.sub(f, text)


def normalize(text: str, default_lang: str = "en") -> str:
    text = spell_acronyms(text)
    for kind, rx in _RULES:
        out, pos = [], 0
        for m in rx.finditer(text):
            lang = context_lang(text, m.start(), m.end(), default_lang)
            if kind == "time":
                h, mi = int(m.group(1)), int(m.group(2))
                h12 = h % 12 or 12
                spoken = hindi_time(h12, mi) if lang == "hi" else english_time(h12, mi)
            else:
                n = say_number(m.group(1) if kind != "number" else m.group(0), lang)
                unit = {
                    "rupee": ("रुपये", "rupees"),
                    "dollar": ("डॉलर", "dollars"),
                    "percent": ("प्रतिशत", "percent"),
                }.get(kind)
                spoken = f"{n} {unit[0] if lang == 'hi' else unit[1]}" if unit else n
            out.append(text[pos:m.start()])
            out.append(spoken)
            pos = m.end()
        out.append(text[pos:])
        text = "".join(out)
    return re.sub(r"\s+", " ", text).strip()
