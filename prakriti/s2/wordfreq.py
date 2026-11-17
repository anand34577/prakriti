"""Hindi word-frequency list from Hindi Wikipedia (streamed, nothing stored but counts).

Output: work/s2/hindi_words.tsv   word<TAB>count   (Devanagari words only, sorted)
"""
import collections
import os
import re
import sys

WORK = os.environ.get("PRAKRITI_WORK", "work")
WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def main(max_docs=200000, top=40000):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    c = collections.Counter()
    for i, doc in enumerate(ds):
        if i >= max_docs:
            break
        c.update(w for w in WORD.findall(doc["text"]) if len(w) > 1 or w in "न व व")
        if i % 20000 == 0:
            print(f"  docs {i}, distinct words {len(c)}", flush=True)
    os.makedirs(os.path.join(WORK, "s2"), exist_ok=True)
    out = os.path.join(WORK, "s2", "hindi_words.tsv")
    total = sum(c.values())
    cum = 0
    with open(out, "w", encoding="utf-8") as f:
        for w, n in c.most_common(top):
            cum += n
            f.write(f"{w}\t{n}\n")
    print(f"wrote {out}: top {top} words cover {cum / total:.1%} of running text")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
