"""Manifest for pron_check: only clips that already pass the signal/ASR gate."""
import json
import os

from prakriti.core.resumable import ItemLedger
from prakriti.s2.verify import S2, all_clips, load_merged


def main():
    ver = load_merged(S2, "verify")
    n = 0
    with open(os.path.join(S2, "pron_manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in all_clips():
            v = ver.get(r["id"])
            lim = 0.12 if r["lang"] == "en" else (0.15 if r.get("source") == "rasa" else 0.10)
            if v is None or v["err"] > lim or v["clip_frac"] >= 0.001:
                continue
            f.write(json.dumps({"id": r["id"], "wav": os.path.join(S2, "audio", r["voice"], f"{r['id']}.wav"),
                                "text": r["text"]}, ensure_ascii=False) + "\n")
            n += 1
    print("pron manifest:", n)


if __name__ == "__main__":
    main()
