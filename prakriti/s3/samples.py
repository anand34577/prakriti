"""Renders listening samples for every voice: Hindi, Hinglish, English and a few emotions.
Usage: python -m prakriti.s3.samples <model_dir> <out_dir>
"""
import sys
from pathlib import Path

import soundfile as sf

from prakriti.tts import SR, Prakriti

LINES = {
    "hindi": ("neutral", "नमस्ते! आज मौसम बहुत सुहाना है, चलिए थोड़ी देर बाहर घूम आते हैं।"),
    "hinglish": ("neutral", "आपका order confirm हो गया है, delivery कल शाम तक हो जाएगी।"),
    "english": ("neutral", "Good morning! Your train to Pune leaves at seven thirty from platform two."),
    "happy": ("happy", "अरे वाह, ये तो बहुत ही अच्छी ख़बर है! मुझे तुम पर गर्व है।"),
    "sad": ("sad", "मुझे बहुत दुख है कि वो अब हमारे बीच नहीं रहे।"),
    "calm": ("calm", "कोई बात नहीं, आराम से करो, कोई जल्दी नहीं है।"),
}


def main(model_dir, out_dir):
    tts = Prakriti(model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for v in tts.voices:
        for name, (emo, text) in LINES.items():
            if emo != "neutral" and emo not in tts.emotions(v):
                continue
            sf.write(out / f"{v}_{name}.wav", tts.synthesize(text, voice=v, emotion=emo, intensity=0.8), SR)
    print("samples:", len(list(out.glob("*.wav"))))


if __name__ == "__main__":
    main(*sys.argv[1:])
