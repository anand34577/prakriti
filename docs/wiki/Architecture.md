# Architecture

## From text to audio

```
text
 │  prakriti/text/normalize.py   numbers, ₹, times, acronyms → words
 ▼
words
 │  prakriti/text/frontend.py    per word: lexicon.json → else eSpeak (with fixes) / Hindi rules
 ▼
phonemes (IPA, Kokoro's 178-symbol set + 5 non-verbal tokens)
 │  Kokoro-82M network (StyleTTS 2 / iSTFTNet), weights = Prakriti
 │    style vector  ← voices/<voice>.pt blended with voices/<voice>__<emotion>.pt
 │    pitch track   × pitch_scale from voices.json
 ▼
24 kHz audio
```

## Main pieces

| Path | Role |
|---|---|
| `prakriti/tts.py` | `Prakriti` class: strict weight loading, voices, emotion blending, pitch, sentence streaming |
| `prakriti/cli.py` | The `prakriti` command |
| `prakriti/text/normalize.py` | Indian number system, currency, clock times (सवा, साढ़े, पौने), acronyms |
| `prakriti/text/hindi_g2p.py` | Rule-based Hindi letter-to-sound: schwa deletion, nukta, nasals, stress |
| `prakriti/text/candidates.py` | Candidate pronunciations per word (eSpeak with fixes, Hindi rules, English) |
| `prakriti/text/frontend.py` | Picks one pronunciation per word, adds non-verbal tokens |
| `prakriti/text/lexicon.json` | 4,864 words whose pronunciation was chosen by listening to recordings |
| `prakriti/pipeline.py` | Stage runner with done-markers, retries and a lock |
| `prakriti/core/resumable.py` | Atomic writes, item ledgers, checkpoint manager |
| `prakriti/s1`, `s2`, `s3` | Training stages: evaluation, data, training and export |
| `server/` | Server setup, systemd unit, status script |
| `patches/` | Fixes applied to the StyleTTS 2 training code |
| `tools/` | Release builder, model card, board benchmark |

## Voices and emotions

A voice is a 256-number style vector (128 for timbre, 128 for prosody), stored as a 510 × 1 × 256 tensor
like Kokoro's voice packs. Each voice's neutral vector is the average over its most neutral-sounding
training clips; each emotion vector is the average over the clips that an emotion classifier rated most
strongly for that emotion. At run time the two are blended: `(1 - intensity) × neutral + intensity × emotion`.

## Why the weights load "strictly"

Kokoro's own loader falls back to a loose mode when names don't match, and that silently skipped half the
weights during this project's first training attempt. Prakriti's loader accepts exactly one known
difference: the affine parameters of AdaIN normalization layers, which the training code doesn't use. It
checks those are at their identity values (weight 1, bias 0) and fails on anything else.

## Non-verbal tokens

`[laugh]`, `[sigh]`, `[gasp]`, `[hmm]` and `[breath]` map to five of the 63 unused slots in Kokoro's symbol
table (ids 7, 8, 26, 27, 28), so no architecture change was needed.
