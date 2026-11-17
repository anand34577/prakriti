"""Calibrated (prakriti) vs uncalibrated (vaani) on the same evaluation set."""
import collections
import json
import os
import statistics as st

W = os.environ.get("PRAKRITI_WORK", "work")
rows = [json.loads(l) for l in open(os.path.join(W, "s1", "scores.jsonl"), encoding="utf-8")]


def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


print("model     lang   error   MOS")
for model in ("prakriti", "vaani", "kokoro", "maya"):
    for lang in ("hi", "mix", "rom", "en"):
        v = [r for r in rows if r["model"] == model and r["lang"] == lang and not r["tag"]]
        if v:
            print(f"{model:9} {lang:5} {m([x['err'] for x in v]):.3f}  {m([x.get('mos') for x in v]):.2f}")

print("\nemotion recognisability:")
for model in ("prakriti", "vaani"):
    per = collections.defaultdict(list)
    for r in rows:
        if r["model"] == model and "emo_p" in r:
            per[r["emotion"]].append(r["emo_p"])
    print(f"{model:9}", " ".join(f"{e}={m(per[e]):.2f}" for e in ("happy", "excited", "sad", "angry", "fear", "surprised") if per.get(e)))

print("\nper voice, Hindi + Hinglish (calibrated | before):")
pv = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r["model"] in ("prakriti", "vaani") and r["lang"] in ("hi", "mix") and not r["tag"]:
        pv[r["voice"]][r["model"]].append(r)
for v, d in sorted(pv.items()):
    a, b = d["prakriti"], d["vaani"]
    print(f"  {v:9} err {m([x['err'] for x in a]):.3f} MOS {m([x.get('mos') for x in a]):.2f} | "
          f"err {m([x['err'] for x in b]):.3f} MOS {m([x.get('mos') for x in b]):.2f}")

tone = json.load(open(os.path.join(W, "tone.json")))
print("\ncalibrated median pitch (Hz):", {v: round(f["f0_median_hz"]) for v, f in tone["prakriti"].items()})
