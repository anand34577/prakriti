"""Prakriti TTS runtime: Hindi / Hinglish / English with voices and emotions.

    from prakriti.tts import Prakriti
    tts = Prakriti("export")                       # folder with prakriti.pth, config.json, voices/
    audio = tts.synthesize("कल मेरी office में meeting है।", voice="Rasa_F", emotion="happy")
    for chunk in tts.stream(long_text, voice="Anirban"):   # sentence by sentence, for real time
        play(chunk)

Emotion packs blend with the neutral voice: intensity 0 = neutral, 1 = full emotion.
Non-verbal markers in text: [laugh] [sigh] [gasp] [hmm] [breath].
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from prakriti.text.frontend import phonemize

SR = 24000
EMOTIONS = ("neutral", "happy", "excited", "sad", "angry", "calm", "whisper", "fear", "surprised",
            "empathetic", "serious")
_SENT = re.compile(r"(?<=[।.!?])\s+|\n+")
DEVA = re.compile(r"[ऀ-ॿ]")


DEFAULT_REPO = "anand-k-maurya/prakriti-v1"  # Hugging Face repo id of the release
WEIGHTS_ST = "prakriti.safetensors"
WEIGHTS_PTH = "prakriti.pth"
MODULES = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
def _is_adain_affine(root: torch.nn.Module, key: str) -> bool:
    """True if `key` is the affine weight/bias of an InstanceNorm sitting directly
    inside an AdaIN layer (a module with a style projection `fc`)."""
    path, leaf = key.rsplit(".", 1)
    if leaf not in ("weight", "bias") or not path.endswith("norm"):
        return False
    norm = root.get_submodule(path)
    parent = root.get_submodule(path.rsplit(".", 1)[0]) if "." in path else root
    return isinstance(norm, torch.nn.InstanceNorm1d) and hasattr(parent, "fc")


def load_weights(model, path: Path) -> None:
    """Strictly loads Prakriti weights (safetensors or Kokoro-style .pth) into a KModel.
    Fails loudly on any missing/unexpected tensor instead of silently skipping it."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        flat = load_file(str(path))
        groups: dict[str, dict] = {m: {} for m in MODULES}
        for k, v in flat.items():
            mod, name = k.split(".", 1)
            groups[mod][name] = v
    else:
        groups = torch.load(path, map_location="cpu", weights_only=True)
    for mod in MODULES:
        res = getattr(model, mod).load_state_dict(groups[mod], strict=False)
        # The only tolerated gap: affine params of AdaIN instance norms (prosody predictor
        # and decoder). Training (StyleTTS 2) uses affine-free norms, so Kokoro's identity
        # init (weight 1, bias 0) is exactly what the weights were trained with.
        root = getattr(model, mod)
        missing = [k for k in res.missing_keys if not _is_adain_affine(root, k)]
        if missing or res.unexpected_keys:
            raise RuntimeError(f"{mod}: missing {missing[:5]}, unexpected {res.unexpected_keys[:5]}")
        params = dict(root.named_parameters())
        for k in res.missing_keys:
            want = 1.0 if k.endswith("weight") else 0.0
            if not torch.all(params[k] == want):
                raise RuntimeError(f"{mod}.{k} is not at its identity value")


class Prakriti:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        import tempfile
        from kokoro import KModel
        d = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = d / WEIGHTS_ST if (d / WEIGHTS_ST).exists() else d / WEIGHTS_PTH
        if not weights.exists():
            raise FileNotFoundError(f"no {WEIGHTS_ST} or {WEIGHTS_PTH} in {d}")
        with tempfile.TemporaryDirectory() as td:  # build the network without Kokoro's own weights
            empty = Path(td) / "empty.pth"
            torch.save({}, empty)
            model = KModel(config=str(d / "config.json"), model=str(empty))
        load_weights(model, weights)
        self.model = model.to(self.device).eval()
        # Pitch control: scale the predicted F0 track before the decoder. The decoder
        # synthesises real voicing at that pitch (no resampling, formants unchanged).
        self._pitch = 1.0
        f0ntrain = self.model.predictor.F0Ntrain

        def scaled_f0ntrain(*a, **k):
            f0, n = f0ntrain(*a, **k)
            return f0 * self._pitch, n
        self.model.predictor.F0Ntrain = scaled_f0ntrain
        self.packs = {p.stem: torch.load(p, map_location="cpu", weights_only=True)
                      for p in sorted((d / "voices").glob("*.pt"))}
        meta = d / "voices.json"
        self.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        lex = d / "lexicon.json"
        if lex.exists():  # the audio-verified pronunciations shipped with this model
            from prakriti.text import frontend
            frontend.LEXICON.update(json.loads(lex.read_text(encoding="utf-8")))

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = DEFAULT_REPO, device: str | None = None,
                        revision: str | None = None) -> "Prakriti":
        """Local folder, or a Hugging Face repo id (downloaded once, then cached)."""
        if Path(repo_or_dir).is_dir():
            return cls(repo_or_dir, device)
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_or_dir, revision=revision,
                              allow_patterns=["config.json", WEIGHTS_ST, "voices/*", "voices.json", "lexicon.json"])
        return cls(d, device)

    @property
    def voices(self) -> list[str]:
        return sorted(k for k in self.packs if "__" not in k)

    def emotions(self, voice: str) -> list[str]:
        return ["neutral"] + sorted(k.split("__", 1)[1] for k in self.packs if k.startswith(voice + "__"))

    def _style(self, voice: str, emotion: str, intensity: float) -> torch.Tensor:
        if voice not in self.packs:
            raise ValueError(f"unknown voice {voice!r}; available: {self.voices}")
        base = self.packs[voice]
        emo = self.packs.get(f"{voice}__{emotion}")
        if emotion == "neutral" or emo is None:
            return base
        w = float(np.clip(intensity, 0.0, 1.5))
        return (1 - w) * base + w * emo

    @staticmethod
    def chunks(text: str, max_chars: int = 220) -> list[str]:
        out = []
        for s in _SENT.split(text.strip()):
            s = s.strip()
            while len(s) > max_chars:  # split long sentences at the last comma before the limit
                cut = max(s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
                cut = cut if cut > 40 else max_chars
                out.append(s[:cut + 1].strip())
                s = s[cut + 1:].strip()
            if s:
                out.append(s)
        return out

    @torch.inference_mode()
    def _say(self, text: str, style: torch.Tensor, speed: float) -> np.ndarray:
        ps = phonemize(text, lang_hint="hi" if DEVA.search(text) else "en")
        if not ps.strip():
            return np.zeros(0, dtype=np.float32)
        ps = ps[:500]
        ref = style[min(len(ps) - 1, style.shape[0] - 1)]
        return self.model(ps, ref, speed).numpy().astype(np.float32)

    def stream(self, text: str, voice: str = "Rasa_F", emotion: str = "neutral", intensity: float = 1.0,
               speed: float = 1.0, pitch: float = 1.0, pause_s: float = 0.12) -> Iterator[np.ndarray]:
        """pitch: extra multiplier on the voice's calibrated pitch (0.9 = a bit lower)."""
        style = self._style(voice, emotion, intensity)
        gap = np.zeros(int(SR * pause_s), dtype=np.float32)
        self._pitch = float(self.meta.get(voice, {}).get("pitch_scale", 1.0)) * float(pitch)
        for i, c in enumerate(self.chunks(text)):
            a = self._say(c, style, speed)
            yield np.concatenate([gap, a]) if i else a

    def synthesize(self, text: str, **kw) -> np.ndarray:
        parts = list(self.stream(text, **kw))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
