# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-17

First release of the Prakriti model and toolkit.

### Added
- Prakriti v1: an 82M-parameter Hindi / Hinglish / English TTS model, fine-tuned from Kokoro-82M.
- 10 voices: `Rasa_F` and `Rasa_M` from real studio speech, and 8 voices distilled from Maya 2 Native.
- Emotion style packs per voice (up to 10 emotions) with an intensity control.
- Per-voice pitch calibration so every voice speaks at the pitch of its own training recordings.
- Python API (`Prakriti.from_pretrained`, `synthesize`, `stream`) and a `prakriti` command-line tool.
- Hindi text front-end: numbers in the Indian system (lakh, crore), currency, clock times read the way
  people say them (साढ़े पाँच, पौने सात), acronyms, schwa deletion, nukta sounds.
- `lexicon.json`: 4,864 pronunciations chosen by matching candidate pronunciations against recorded audio.
- Resumable training pipeline under systemd with atomic checkpoints, crash-and-resume tests,
  and automatic batch reduction when the GPU runs out of memory.
- Release builder that writes a Hugging Face model repo (safetensors weights, voices, model card).

### Known limitations
- Emotion is moderate: sad and surprised come through; angry, fear and excited are subtle.
- `[laugh]` is weak (42 training examples).
- Romanized Hinglish ("kya haal hai") is read with English rules.

[1.0.0]: https://github.com/anand34577/prakriti/releases/tag/v1.0.0
