"""Prints the final comparison table from work/s1/scores.jsonl."""
import collections
import json
import os
import statistics as st

W = os.environ.get("PRAKRITI_WORK", "work")
rows = [json.loads(l) for l in open(os.path.join(W, "s1", "scores.jsonl"), encoding="utf-8")]
MODELS = ("prakriti", "prakriti_pre", "kokoro", "maya", "svara")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


agg = collections.defaultdict(list)
for r in rows:
    if not r["tag"]:
        agg[(r["model"], r["lang"])].append(r)
print("model    lang   n    error   MOS")
for m in MODELS:
    for l in ("hi", "mix", "rom", "en"):
        v = agg.get((m, l))
        if v:
            print(f"{m:8} {l:5} {len(v):4}  {mean([x['err'] for x in v]):.3f}  {mean([x.get('mos') for x in v]):.2f}")

print("\nemotion recognisability (probability of the intended emotion):")
for m in MODELS:
    per = collections.defaultdict(list)
    for r in rows:
        if r["model"] == m and "emo_p" in r:
            per[r["emotion"]].append(r["emo_p"])
    print(f"{m:8}", "  ".join(f"{e}={mean(per[e]):.2f}" for e in ("happy", "excited", "sad", "angry", "fear", "surprised") if per.get(e)))

print("\nPrakriti per voice (Hindi + Hinglish):")
pv = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r["model"] in ("prakriti", "prakriti_pre") and r["lang"] in ("hi", "mix") and not r["tag"]:
        pv[r["voice"]][r["model"]].append(r)
print(f"  {'voice':9} {'final err/MOS':>16} {'pre-adv err/MOS':>18}")
for v, d in sorted(pv.items()):
    cells = [f"{mean([x['err'] for x in d[m]]):.3f} / {mean([x.get('mos') for x in d[m]]):.2f}" if d.get(m) else "-" for m in ("prakriti", "prakriti_pre")]
    print(f"  {v:9} {cells[0]:>16} {cells[1]:>18}")
