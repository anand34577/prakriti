"""Builds a Hugging Face model repo folder from a trained export.

    python tools/build_hf_release.py <export_dir> hf_release [--samples <dir_of_wavs>]

Output (ready for `git lfs` + `git push` or `hf upload`):
  README.md                 model card (from tools/model_card.md)
  LICENSE, NOTICE           Apache-2.0 + attributions
  .gitattributes            LFS rules for weights, voices, audio
  config.json               Kokoro config + Prakriti non-verbal token ids
  prakriti.safetensors      weights (safe format, used by the loader)
  prakriti.pth              same weights, Kokoro-compatible format
  voices/<voice>.pt, voices/<voice>__<emotion>.pt, voices.json
  lexicon.json              audio-verified pronunciations
  samples/*.wav             demo audio for the card (optional)
  pyproject.toml + prakriti/   inference package (pip install from this repo)
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODULES = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
INFERENCE_FILES = ["prakriti/__init__.py", "prakriti/tts.py", "prakriti/cli.py",
                   "prakriti/text/__init__.py", "prakriti/text/normalize.py", "prakriti/text/hindi_g2p.py",
                   "prakriti/text/candidates.py", "prakriti/text/frontend.py", "prakriti/text/lexicon.json"]
GITATTRIBUTES = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--samples", default=None)
    a = ap.parse_args()
    src, out = Path(a.export_dir), Path(a.out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "voices").mkdir(parents=True)

    # weights: strict check, then safetensors + Kokoro-compatible pth
    from safetensors.torch import save_file
    pth = next(p for p in (src / "prakriti.pth", src / "vaani.pth") if p.exists())
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    assert set(sd) == set(MODULES), f"unexpected modules {set(sd) ^ set(MODULES)}"
    flat = {f"{m}.{k}": v.contiguous() for m in MODULES for k, v in sd[m].items()}
    save_file(flat, str(out / "prakriti.safetensors"), metadata={"format": "pt", "model": "prakriti-v1"})
    torch.save(sd, out / "prakriti.pth")
    n_params = sum(v.numel() for v in flat.values())

    shutil.copy(src / "config.json", out / "config.json")
    for p in sorted((src / "voices").glob("*.pt")):
        shutil.copy(p, out / "voices" / p.name)
    shutil.copy(src / "voices.json", out / "voices.json")
    shutil.copy(REPO / "prakriti" / "text" / "lexicon.json", out / "lexicon.json")
    if a.samples:
        (out / "samples").mkdir()
        for p in sorted(Path(a.samples).glob("*.wav")):
            shutil.copy(p, out / "samples" / p.name)

    for f in INFERENCE_FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, out / f)
    shutil.copy(REPO / "pyproject.toml", out / "pyproject.toml")
    shutil.copy(REPO / "tools" / "model_card.md", out / "README.md")
    shutil.copy(REPO / "LICENSE", out / "LICENSE")
    shutil.copy(REPO / "NOTICE", out / "NOTICE")
    (out / ".gitattributes").write_text(GITATTRIBUTES)

    voices = json.loads((out / "voices.json").read_text())
    n_packs = len(list((out / "voices").glob("*.pt")))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"release ready: {out}  params {n_params / 1e6:.1f}M  voices {len(voices)}  style packs {n_packs}  "
          f"size {size:.0f} MB")


if __name__ == "__main__":
    main()
