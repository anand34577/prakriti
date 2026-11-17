"""Shared helpers for session-1 evaluation stages (no heavy imports here)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

WORK = Path(os.environ.get("PRAKRITI_WORK", "work"))
REPO = Path(__file__).resolve().parents[2]
S1 = WORK / "s1"


@dataclass
class Job:
    id: str
    lang: str          # hi | mix | rom | en
    text: str          # may contain {tag}
    emotion: str = "neutral"
    tag: str | None = None


def load_jobs() -> list[Job]:
    cfg = yaml.safe_load(open(REPO / "configs" / "s1_sentences.yaml", encoding="utf-8"))
    jobs = [Job(s["id"], s["lang"], s["text"]) for s in cfg["plain"]]
    for emo, items in cfg["emotions"].items():
        for i, s in enumerate(items):
            jobs.append(Job(f"emo_{emo}_{i}", s["lang"], s["text"], emotion=emo))
    for i, s in enumerate(cfg["nonverbal"]):
        jobs.append(Job(f"nv_{s['tag']}_{i}", s["lang"], s["text"], tag=s["tag"]))
    return jobs


def render(job: Job, tags: dict[str, str]) -> str:
    """Fill {tag} with a model's syntax, or drop it if the model lacks that tag."""
    t = tags.get(job.tag, "") if job.tag else ""
    return " ".join(job.text.replace("{tag}", t).split())


def out_dir(model: str) -> Path:
    d = S1 / "audio" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_wav(path: Path, audio, sr: int) -> float:
    import numpy as np
    import soundfile as sf
    audio = np.asarray(audio, dtype="float32").squeeze()
    tmp = path.with_name(f".{path.stem}.tmp.wav")
    sf.write(tmp, audio, sr)
    os.replace(tmp, path)
    return len(audio) / sr
