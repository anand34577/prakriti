"""Builds work/s1/report/report.html: one self-contained page with score
tables and side-by-side audio (Opus, base64) for listening comparisons."""
import base64
import html
import json
import statistics as st
import subprocess
from collections import defaultdict

from prakriti.core.resumable import ItemLedger
from prakriti.s1.common import S1

MODELS = {"prakriti": "Prakriti (final, 82M)", "kokoro": "Kokoro-82M (stock)", "kokoro_ft": "Kokoro-82M (trial fine-tune)",
          "maya": "Maya 2 Native (teacher, 2.5B)", "svara": "Svara-TTS v1 (teacher, 3B)"}
LANGS = {"hi": "Hindi", "mix": "Hinglish (mixed script)", "rom": "Hinglish (romanized)", "en": "English"}
LISTEN = ["hi03", "hi06", "mx01", "mx03", "mx07", "ro01", "en03",
          "emo_happy_1", "emo_sad_0", "emo_angry_1", "emo_whisper_0", "emo_empathetic_1",
          "nv_laugh_0", "nv_sigh_2"]
PREFERRED_VOICE = {"prakriti": ("Rasa_F", "Rasa_F"), "kokoro": ("hf_alpha", "af_heart"), "kokoro_ft": ("Ahana", "Ahana"),
                   "maya": ("Ahana", "Ahana"), "svara": ("Hindi (Female)", "English (Female)")}


def opus_b64(path):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-c:a", "libopus",
                          "-b:a", "32k", "-f", "ogg", "-"], capture_output=True, check=True).stdout
    return base64.b64encode(out).decode()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{x * 100:.1f}%" if pct else f"{x:.2f}"


def main():
    scores = ItemLedger(S1 / "scores.jsonl").records()
    by = defaultdict(list)
    for r in scores:
        by[(r["model"], r["lang"])].append(r)

    rows = []
    for m in MODELS:
        for l in LANGS:
            rs = [r for r in by.get((m, l), []) if not r["tag"]]
            if not rs:
                continue
            rows.append(f"<tr><td>{MODELS[m]}</td><td>{LANGS[l]}</td><td>{len(rs)}</td>"
                        f"<td>{fmt(mean([r['err'] for r in rs]), True)}</td>"
                        f"<td>{fmt(mean([r.get('mos') for r in rs]))}</td></tr>")

    emo_rows = []
    for m in MODELS:
        rs = [r for r in scores if r["model"] == m and "emo_p" in r]
        if not rs:
            continue
        per = defaultdict(list)
        for r in rs:
            per[r["emotion"]].append(r["emo_p"])
        cells = "".join(f"<td>{fmt(mean(per.get(e, [])), True)}</td>"
                        for e in ("happy", "excited", "sad", "angry", "fear", "surprised"))
        emo_rows.append(f"<tr><td>{MODELS[m]}</td>{cells}</tr>")

    speed = {}
    for m in ("prakriti", "kokoro", "maya", "svara"):
        p = S1 / f"speed_{m}.json"
        if p.exists():
            speed[m] = json.loads(p.read_text())
    timing = S1 / "train" / "timing.json"
    timing = json.loads(timing.read_text()) if timing.exists() else None

    listen = []
    ledgers = {m: {(r["voice"], r["job"]): r for r in ItemLedger(S1 / f"ledger_{m}.jsonl").records()
                   if not r.get("failed")}
               for m in MODELS if (S1 / f"ledger_{m}.jsonl").exists()}
    for job in LISTEN:
        cells, text = [], ""
        for m, led in ledgers.items():
            hv, ev = PREFERRED_VOICE[m]
            any_rec = next((r for (v, j), r in led.items() if j == job), None)
            if any_rec is None:
                cells.append("<td>–</td>")
                continue
            rec = led.get((ev if any_rec["lang"] == "en" else hv, job), any_rec)
            text = rec.get("ref") or text
            b64 = opus_b64(S1 / "audio" / m / f"{rec['id']}.wav")
            cells.append(f'<td><audio controls preload="none" src="data:audio/ogg;base64,{b64}"></audio>'
                         f'<div class="v">{html.escape(rec["voice"])}</div></td>')
        listen.append(f"<tr><td class='t'>{html.escape(text)}<div class='v'>{job}</div></td>{''.join(cells)}</tr>")

    heads = "".join(f"<th>{MODELS[m]}</th>" for m in ledgers)
    speed_html = "".join(f"<li><b>{MODELS[m]}</b>: {html.escape(json.dumps(v))}</li>" for m, v in speed.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prakriti Session 1</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e4e1da;--acc:#b4532a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161614;--fg:#ecebe6;--mut:#9a988f;--line:#2e2d29;--acc:#e08a5c}}}}
:root[data-theme=dark]{{--bg:#161614;--fg:#ecebe6;--mut:#9a988f;--line:#2e2d29;--acc:#e08a5c}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}}
h1{{font-size:24px}}h2{{font-size:18px;margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--mut);font-weight:600;font-size:13px}}.v{{color:var(--mut);font-size:12px}}.t{{min-width:220px}}audio{{width:200px;height:32px}}
</style></head><body>
<h1>Prakriti — Session 1 feasibility</h1>
<p class="v">Error = Whisper large-v3 CER (Hindi/Hinglish, script-neutral) or WER (English): lower is better. MOS = UTMOS (English-trained; compare models, not absolute).
Emotion = emotion2vec+ probability of the intended emotion.</p>
<h2>Intelligibility and naturalness</h2><div class="wrap"><table><tr><th>Model</th><th>Language</th><th>Clips</th><th>Error ↓</th><th>MOS ↑</th></tr>{''.join(rows)}</table></div>
<h2>Emotion recognisability</h2><div class="wrap"><table><tr><th>Model</th><th>happy</th><th>excited</th><th>sad</th><th>angry</th><th>fear</th><th>surprised</th></tr>{''.join(emo_rows)}</table></div>
<h2>Speed</h2><ul>{speed_html}</ul>
<h2>Fine-tuning trial</h2><p>{html.escape(json.dumps(timing)) if timing else 'not run'}</p>
<h2>Listen</h2><div class="wrap"><table><tr><th>Sentence</th>{heads}</tr>{''.join(listen)}</table></div>
</body></html>"""
    (S1 / "report").mkdir(exist_ok=True)
    (S1 / "report" / "report.html").write_text(page, encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
