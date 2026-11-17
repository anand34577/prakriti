# FAQ

### Can I use it commercially?

The code and weights are Apache-2.0. The training data allows it too: Rasa is CC-BY-4.0 (attribution
required, which the NOTICE file provides) and Maya 2 Native and Qwen3 are Apache-2.0. Keep the NOTICE file
with any redistribution.

### Can it clone my voice?

No. Prakriti has a fixed set of 10 voices. Training a new voice needs recordings and the pipeline in
[Training](Training).

### Which voice should I use?

`Rasa_F` or `Rasa_M` for the most natural sound. Pick one of the other eight when you need an emotion the
Rasa voices don't have (calm, empathetic, excited, whisper).

### Does it work offline?

Yes, after the first download. See [Usage → Use it offline](Usage#use-it-offline).

### Does it support other Indian languages?

No, only Hindi, Hinglish and English. The text front-end is Hindi-specific.

### How is it different from stock Kokoro?

Kokoro has four Hindi voices and reads Hindi well, but it has no emotion styles, no Indian-English voices
tuned on Hinglish, and no Indian number reading. Prakriti adds those, plus the audio-checked lexicon and
two voices from real studio speech. On the same test sentences its Hindi accuracy is the same as Kokoro's.

### Why does the README say emotion is "moderate"?

Because that's what the measurements show: an emotion classifier recognises sadness and surprise in
Prakriti's output, but anger, fear and excitement only weakly. A stronger emotion model is planned.

### Will there be an NPU (Rockchip RKNN) build?

It's planned. Today it runs on the board's CPU cores.
