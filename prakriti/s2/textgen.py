"""Session 2: generate the training text corpus with an open LLM (Qwen3, Apache 2.0) on vLLM.

Every request asks for a batch of sentences for one (language, emotion, topic,
speaking-context) cell, so the corpus is balanced and diverse by design.
Output: work/s2/corpus.jsonl  {id, lang, emotion, tag, text}
Resumable: finished cells are recorded in a ledger.
"""
import hashlib
import json
import os
import random
import re

from prakriti.core.resumable import ItemLedger
from prakriti.text.normalize import normalize

WORK = os.environ.get("PRAKRITI_WORK", "work")
OUT = os.path.join(WORK, "s2")
MODEL = os.environ.get("TEXTGEN_MODEL", "Qwen/Qwen3-32B-AWQ")
QUOTED = re.compile(r'"((?:[^"\\\n]|\\.)+)"')
PER_CELL = 25

LANGS = {
    "hi": ("pure Hindi written in Devanagari script. Use everyday spoken Hindi, not overly Sanskritised.", 0.36),
    "mix": ("Hinglish: natural Indian code-mixed Hindi-English exactly as people type on WhatsApp. IMPORTANT: "
            "Hindi words in Devanagari script, but every English word written in Latin (English) script, never in "
            "Devanagari. Each sentence must contain at least one English word in Latin script. Examples: "
            "'कल मेरी office में meeting है', 'Sorry यार, मैं busy था', 'ये app बहुत slow चल रहा है'. "
            "Never write Hindi words in Latin script (no 'hai', 'karna', 'kya').", 0.40),
    "en": ("Indian English, natural and conversational, using Indian names and places where relevant.", 0.24),
}
EMOTIONS = {
    "neutral": 0.34, "happy": 0.07, "excited": 0.06, "sad": 0.07, "angry": 0.06, "calm": 0.06,
    "whisper": 0.04, "fear": 0.05, "surprised": 0.06, "empathetic": 0.07, "serious": 0.06,
}
EMO_DESC = {
    "neutral": "neutral, matter-of-fact", "happy": "happy and warm", "excited": "very excited and energetic",
    "sad": "sad, low, grieving or disappointed", "angry": "angry or frustrated", "calm": "calm, soothing",
    "whisper": "secretive, something one would whisper", "fear": "scared or worried",
    "surprised": "surprised or shocked", "empathetic": "apologetic and empathetic, like a kind assistant",
    "serious": "serious and formal, like an announcement",
}
CONTEXTS = [
    "a voice assistant answering a user", "two friends chatting", "family members at home",
    "a customer-care agent", "a storyteller narrating", "a news reader", "a teacher in class",
    "a shopkeeper and customer", "colleagues at work", "a doctor talking to a patient",
    "a railway or airport announcement", "someone on a phone call", "a cooking show host",
    "a cricket commentator", "a travel vlogger", "a parent talking to a child",
]
TOPICS = [
    "weather", "food and recipes", "cricket", "movies and songs", "trains and travel", "festivals",
    "studies and exams", "jobs and office", "health", "money and shopping", "technology and phones",
    "traffic", "relationships", "farming and village life", "city life", "reminders and alarms",
    "music", "news events", "banking and payments", "home chores", "pets", "history", "science facts",
    "jokes and fun", "sports", "weddings", "time and dates", "directions", "numbers and prices",
]
NONVERBAL = ["laugh", "sigh", "hmm", "gasp"]


def cells(n_sentences: int, seed: int = 7):
    rng = random.Random(seed)
    langs, lw = zip(*[(k, v[1]) for k, v in LANGS.items()])
    emos, ew = zip(*EMOTIONS.items())
    for i in range(n_sentences // PER_CELL):
        yield {"cell": i, "lang": rng.choices(langs, lw)[0], "emotion": rng.choices(emos, ew)[0],
               "context": rng.choice(CONTEXTS), "topic": rng.choice(TOPICS),
               "nonverbal": rng.choice(NONVERBAL) if rng.random() < 0.08 else None,
               "length": rng.choice(["short (3-8 words)", "medium (8-16 words)", "long (16-28 words)"])}


def prompt(c):
    nv = (f" In about a third of them, include the marker [{c['nonverbal']}] exactly where the speaker would "
          f"naturally {c['nonverbal']}." if c["nonverbal"] else "")
    return (f"Write {PER_CELL} different sentences in {LANGS[c['lang']][0]}\n"
            f"Speaker: {c['context']}. Topic: {c['topic']}. Emotion: {EMO_DESC[c['emotion']]} — the words "
            f"themselves should make the emotion obvious. Length: {c['length']}.{nv}\n"
            "Vary sentence structure, include questions and exclamations, write numbers as digits sometimes. "
            "No numbering, no translations, no quotes. Output a JSON array of strings only.")


DEVA = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")


def keep(text, lang):
    words = text.split()
    if not 2 <= len(words) <= 40:
        return False
    d, l = len(DEVA.findall(text)), len(LAT.findall(re.sub(r"\[\w+\]", "", text)))
    if lang == "hi":
        return d > 0 and l <= 0.15 * (d + l)
    if lang == "en":
        return d == 0 and l > 0
    if any(w.lower().strip(".,!?") in ROMAN_HINDI for w in words):
        return False  # romanized Hindi words would be read with English rules
    return d > 0 and l > 0  # mix must actually mix


ROMAN_HINDI = set("""hai hain tha thi the ho hoga hogi kar karna karo karega karegi karenge kiya kiye ki ka ke ko
se me mein mai main hum tum aap ye yeh wo woh kya kyu kyun kaise kahan kab nahi nahin na haan ji bhai yaar
raha rahi rahe gaya gayi gaye jaana jana aana ana tha diya liya bahut accha acha theek thik abhi phir aur
lekin par pe sab kuch koi kal aaj apna apni mera meri tera teri uska uski hamara tumhara wala wali""".split())


def main(n_sentences: int = 60000):
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    llm = LLM(MODEL, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
              max_model_len=4096, enable_prefix_caching=True)
    run(llm, list(cells(n_sentences)), prompt, "ledger_textgen.jsonl")
    if os.environ.get("COVER_TOP"):
        cover(llm, int(os.environ["COVER_TOP"]))


def _corpus():
    path = os.path.join(OUT, "corpus.jsonl")
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return path, rows


def run(llm, all_cells, prompt_fn, ledger_name, required=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=3000, repetition_penalty=1.05)
    corpus_path, rows = _corpus()
    seen = {r["text"] for r in rows}
    with ItemLedger(os.path.join(OUT, ledger_name)) as led, open(corpus_path, "a", encoding="utf-8") as fout:
        todo = [c for c in all_cells if str(c["cell"]) not in led]
        print(f"{ledger_name}: {len(todo)} of {len(all_cells)} cells to do", flush=True)
        for b in range(0, len(todo), 512):  # vLLM batches internally; chunk for checkpointing
            chunk = todo[b:b + 512]
            msgs = [[{"role": "user", "content": prompt_fn(c)}] for c in chunk]
            outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            for c, o in zip(chunk, outs):
                # every complete quoted string survives even if the array was cut off
                items = [json.loads(f'"{q}"') for q in QUOTED.findall(o.outputs[0].text)]
                kept = 0
                for s in items:
                    s = " ".join(s.split())
                    t = re.search(r"\[(laugh|sigh|hmm|gasp)\]", s)
                    spoken = normalize(s, default_lang="en" if c["lang"] == "en" else "hi")
                    lang = c["lang"]
                    if lang == "mix" and not keep(spoken, "mix") and keep(spoken, "hi"):
                        lang = "hi"  # English loanwords written in Devanagari: valid Hindi
                    if spoken in seen or not keep(spoken, lang):
                        continue
                    if required and not any(w in spoken.split() for w in c.get("words", [])):
                        continue
                    seen.add(spoken)
                    sid = hashlib.sha1(spoken.encode()).hexdigest()[:12]
                    fout.write(json.dumps({"id": sid, "lang": lang, "emotion": c["emotion"],
                                           "tag": t.group(1) if t else None, "context": c["context"],
                                           "text": spoken}, ensure_ascii=False) + "\n")
                    kept += 1
                fout.flush()
                os.fsync(fout.fileno())
                led.add(str(c["cell"]), kept=kept)
            print(f"  cells {b + len(chunk)}/{len(todo)}, corpus {len(seen)}", flush=True)


def missing_words(top: int) -> list[str]:
    words = [l.split("\t")[0] for l in open(os.path.join(OUT, "hindi_words.tsv"), encoding="utf-8")][:top]
    have = set()
    for r in _corpus()[1]:
        have.update(DEVA_TOKEN.findall(r["text"]))
    return [w for w in words if w not in have]


DEVA_TOKEN = re.compile(r"[ऀ-ॣॱ-ॿ]+")
PAIRS_PER_PROMPT = 10


def cover(llm, top: int, rounds: int = 3):
    """Sentences that use still-missing frequent words, 2 per sentence, until covered."""
    for rnd in range(rounds):
        miss = missing_words(top)
        print(f"coverage round {rnd + 1}: {len(miss)} of top {top} words missing", flush=True)
        if len(miss) < 50:
            break
        rng = random.Random(100 + rnd)
        rng.shuffle(miss)
        pairs = [miss[i:i + 2] for i in range(0, len(miss), 2)]
        cells_ = []
        for i in range(0, len(pairs), PAIRS_PER_PROMPT):
            group = pairs[i:i + PAIRS_PER_PROMPT]
            cells_.append({"cell": f"r{rnd}_{i}", "lang": "hi", "emotion": rng.choices(*zip(*EMOTIONS.items()))[0],
                           "context": rng.choice(CONTEXTS), "pairs": group, "words": sum(group, [])})

        def cover_prompt(c):
            lines = "\n".join(f"{k + 1}. {' + '.join(p)}" for k, p in enumerate(c["pairs"]))
            return ("For each numbered line, write ONE natural, everyday spoken Hindi sentence in Devanagari "
                    f"(6-18 words) that uses the given words exactly as written, correctly and meaningfully. "
                    f"Speaker: {c['context']}. Tone: {EMO_DESC[c['emotion']]}.\n{lines}\n"
                    "Output a JSON array of strings only, one sentence per line, same order.")
        run(llm, cells_, cover_prompt, "ledger_cover.jsonl", required=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
