# Changelog

Newest first. The full list is in
[CHANGELOG.md](https://github.com/anand34577/prakriti/blob/main/CHANGELOG.md).

## 1.0.0 (2026-11-17)

First release.

- 10 voices (2 from real studio speech), emotion styles per voice, pitch calibrated to each voice's recordings
- Hindi, Hinglish and English, with Indian number, currency and time reading
- Python API with streaming, and the `prakriti` command
- 4,864-word pronunciation lexicon checked against recorded audio
- Full training pipeline that survives disconnects, crashes and reboots

Known limits: emotion is moderate, `[laugh]` is weak, romanized Hinglish is read as English.
