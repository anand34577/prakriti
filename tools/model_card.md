---
license: apache-2.0
language:
  - hi
  - en
library_name: kokoro
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
datasets:
  - ai4bharat/Rasa
tags:
  - text-to-speech
  - tts
  - hindi
  - hinglish
  - indian-english
  - code-switching
  - emotional-tts
  - styletts2
  - kokoro
---

# Prakriti v1 — Hindi · Hinglish · English TTS

**Prakriti** (प्रकृति, "nature") is a lightweight (82M-parameter) text-to-speech model for
**Hindi, Hinglish (code-mixed Hindi-English) and Indian English**, with **10 voices** and
**per-voice emotion styles**. It is a fine-tune of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
and keeps its speed: ~36x real time on a GPU and ~2.6x real time on 4 CPU threads.

- 10 voices: 2 from real studio recordings, 8 distilled from a large teacher model
- Emotion styles per voice (happy, sad, angry, fear, surprised, serious, and more) with an intensity control
- Hindi text front-end: numbers, currency, Indian time expressions (साढ़े पाँच, पौने सात), acronyms,
  rule-based schwa deletion, nukta sounds, and a pronunciation lexicon verified against recorded audio
- Sentence streaming for voice assistants; runs on CPU or GPU, Linux / Windows / macOS
- Apache-2.0

## Quick start

```bash
# espeak-ng is required by the phonemizer:
#   Ubuntu/Debian: sudo apt install espeak-ng      Windows: github.com/espeak-ng/espeak-ng/releases
pip install "git+https://huggingface.co/anand-k-maurya/prakriti-v1"
prakriti "कल मेरी office में meeting है।" -v Rasa_F -e happy -o hello.wav
prakriti --list-voices
```

```python
from prakriti.tts import Prakriti
import soundfile as sf

tts = Prakriti.from_pretrained("anand-k-maurya/prakriti-v1")    # downloads once, then cached
audio = tts.synthesize("नमस्ते! आपका order कल तक पहुँच जाएगा।", voice="Rasa_F", emotion="happy", intensity=0.8)
sf.write("out.wav", audio, 24000)

# Real time: speak sentence by sentence as soon as each is ready
for chunk in tts.stream(long_text, voice="Anirban"):
    play(chunk)          # your audio sink, 24 kHz mono float32
```

Non-verbal markers in text: `[laugh]`, `[sigh]`, `[gasp]`, `[hmm]`, `[breath]` (see limitations).

## Voices

| Voice | Gender | Source | Emotion styles |
|---|---|---|---|
| `Rasa_F` | female | real studio speech (Rasa) | angry, fear, happy, sad, serious, surprised |
| `Rasa_M` | male | real studio speech (Rasa) | angry, fear, happy, sad, serious, surprised |
| `Ahana`, `Vaidehi`, `Malini`, `Charvi` | female | distilled from Maya 2 Native | angry, calm, empathetic, excited, fear, happy, sad, serious, surprised, whisper |
| `Anirban`, `Nakul`, `Vasant`, `Siddhant` | male | distilled from Maya 2 Native | same as above |

`Rasa_F` and `Rasa_M` are the most natural-sounding voices (see evaluation).

## Evaluation

Same 65 test sentences for every system (Prakriti and Maya: all voices; Kokoro: its 4 Hindi and 4 English voices).

| System | Hindi error | Hinglish error | English error | Naturalness (UTMOS) | Size |
|---|---|---|---|---|---|
| **Prakriti v1** | 2.7% | 14.1% | 0.0% | 3.75 (Hindi), 4.11 (English) | 82M |
| Kokoro-82M (stock) | 2.6% | 11.3% | 0.0% | 4.26 (Hindi), 4.38 (English) | 82M |
| Maya 2 Native (teacher) | 3.1% | 11.0% | 0.9% | 3.45 (Hindi), 3.75 (English) | 2.5B |

The two studio voices score highest on naturalness: `Rasa_F` 4.15 and `Rasa_M` 4.12 on Hindi and Hinglish sentences (8.3% combined error for both), against 3.63 to 3.93 for the other eight.

| Device | Faster than real time | Median time per sentence |
|---|---|---|
| GPU (RTX A6000) | 36.6x | 0.11 s |
| CPU, 4 threads | 2.6x | 1.6 s |
| CPU, 2 threads | 1.6x | 2.5 s |

Error is Whisper large-v3's character error rate after script-neutral transliteration (Hindi, Hinglish) or word
error rate (English); lower is better, and Hinglish numbers include transliteration noise for every system.
Naturalness is UTMOS, an English-trained automatic predictor: use it to compare systems, not as an absolute score.
After pitch calibration every voice speaks within a few Hz of its own training recordings, with no change in
error or naturalness.

- *Error*: Whisper large-v3 character error rate after script-neutral transliteration (Hindi, Hinglish)
  or word error rate (English); lower is better. Hinglish numbers include transliteration noise for all systems.
- *MOS*: UTMOS (an English-trained automatic naturalness predictor). Use it to compare systems, not as an absolute score.
- Same 65 test sentences for every system; speed measured on one RTX A6000 and on 4 CPU threads of the same host.

## Training

- **Base**: Kokoro-82M, fine-tuned with the StyleTTS 2 two-stage recipe (adapted from kikiri-tts).
- **Data**: 54.7 h after filtering (41,592 clips, ≤ 12 s):
  - Rasa Hindi (real expressive studio speech, CC-BY-4.0)
  - 32k sentences synthesized with Maya 2 Native in 8 voices; the text was generated with Qwen3-32B to cover
    Hindi, Hinglish and English, 11 emotions, and 99.4% of the 20,000 most frequent Hindi words
- **Quality gate**: Whisper transcript match, signal checks (clipping, silence, speaking rate), emotion scoring.
- **Pronunciation labels**: for every word where eSpeak and Prakriti's Hindi rules disagree, the variant that
  matches the recorded audio (scored with a phoneme recognizer) is used; majority votes form `lexicon.json`.
- **Compute**: one RTX A6000 (48 GB).

## Limitations

- **Emotion is moderate.** Emotion styles are averaged style vectors; sadness and surprise come through, but
  anger, fear and excitement are subtle. Strong emotional acting needs further work (emotion-conditioned training).
- **Non-verbal sounds are weak.** `[laugh]` had only 42 training examples; expect a faint or absent laugh.
- **Romanized Hinglish** ("kya haal hai") is read with English rules; write Hindi words in Devanagari for best results.
- The 8 distilled voices sound less natural than the 2 studio voices; they inherit artifacts of the teacher model.
- Not evaluated on Rockchip NPUs yet; CPU real-time factor on ARM boards depends on the board.

## Responsible use

Do not use Prakriti to impersonate real people, to deceive, or to defeat voice authentication. Disclose
synthetic speech to listeners. The Rasa-derived voices are based on speakers recorded for the Rasa TTS
dataset; do not present generated speech as coming from those individuals.

## License and attribution

Apache-2.0. See `NOTICE` for attributions: Kokoro-82M (Apache-2.0), StyleTTS 2 (MIT), kikiri-tts (Apache-2.0),
Rasa by AI4Bharat (CC-BY-4.0), Maya 2 Native (Apache-2.0), Qwen3 (Apache-2.0).

## Files

| File | Purpose |
|---|---|
| `prakriti.safetensors` | weights (used by the loader) |
| `prakriti.pth` | same weights, Kokoro-compatible format |
| `config.json` | Kokoro config + non-verbal token ids |
| `voices/*.pt` | style packs: `<voice>.pt` and `<voice>__<emotion>.pt` |
| `voices.json` | voice metadata |
| `lexicon.json` | audio-verified pronunciations |
| `prakriti/`, `pyproject.toml` | inference package |
