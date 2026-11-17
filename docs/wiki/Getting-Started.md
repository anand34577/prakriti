# Getting started

This page takes you from an installed package to a few audio files you can play. If `prakriti
--list-voices` doesn't work yet, go back to [Installation](Installation).

## 1. Make your first file

```bash
prakriti "नमस्ते! आज मौसम बहुत सुहाना है।" -o namaste.wav
```

You should see:

```
wrote namaste.wav (2.8 s)
```

Open `namaste.wav` in any audio player. That's the default voice, `Rasa_F`.

## 2. Try another voice

```bash
prakriti "आपका order confirm हो गया है, delivery कल शाम तक हो जाएगी।" -v Anirban -o order.wav
```

Hindi words go in Devanagari and English words stay in English letters, the way people type on WhatsApp.
Prakriti reads each word in the right language.

## 3. Add an emotion

```bash
prakriti "अरे वाह, ये तो बहुत अच्छी ख़बर है!" -v Rasa_F -e happy -i 0.8 -o khabar.wav
```

`-e` picks the emotion and `-i` sets how strong it is, from `0` (neutral) to `1` (full). Values around
`0.6` to `0.8` usually sound the most natural. Not every voice has every emotion; `--list-voices` shows
which ones each voice has.

## 4. Numbers, money and time

Write them the way you normally would:

```bash
prakriti "कुल ₹1,50,000 लगेंगे और मीटिंग 5:30 बजे है।" -o paise.wav
```

Prakriti says "एक लाख पचास हज़ार रुपये" and "साढ़े पाँच बजे". In English sentences the same numbers are read
in English.

## 5. Use it from Python

```python
from prakriti.tts import Prakriti
import soundfile as sf

tts = Prakriti.from_pretrained()
audio = tts.synthesize("कल मेरी office में meeting है।", voice="Malini")
sf.write("meeting.wav", audio, 24000)
```

The audio is a NumPy array of float32 samples at 24,000 Hz, mono.

## What's next

- Every option and voice: [Configuration](Configuration)
- Streaming for a voice assistant, and fixing a word that sounds wrong: [Usage](Usage)
- Running it on a server or a board: [Deployment](Deployment)
