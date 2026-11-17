"""Command line:  prakriti "कल मेरी office में meeting है।" -v Rasa_F -e happy -o out.wav
                 prakriti --list-voices
"""
import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prakriti", description="Prakriti TTS: Hindi / Hinglish / English")
    ap.add_argument("text", nargs="?", help="text to speak (use - to read stdin)")
    ap.add_argument("-v", "--voice", default="Rasa_F")
    ap.add_argument("-e", "--emotion", default="neutral")
    ap.add_argument("-i", "--intensity", type=float, default=1.0, help="0 = neutral ... 1 = full emotion")
    ap.add_argument("-s", "--speed", type=float, default=1.0)
    ap.add_argument("-p", "--pitch", type=float, default=1.0, help="pitch multiplier on top of the voice's own")
    ap.add_argument("-o", "--output", default="prakriti.wav")
    ap.add_argument("-m", "--model", default=None, help="local folder or Hugging Face repo id")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args(argv)

    import os

    import soundfile as sf
    from prakriti.tts import DEFAULT_REPO, SR, Prakriti
    tts = Prakriti.from_pretrained(a.model or os.environ.get("PRAKRITI_MODEL") or DEFAULT_REPO, device=a.device)
    if a.list_voices:
        for v in tts.voices:
            info = tts.meta.get(v, {})
            print(f"{v:10} {info.get('gender', ''):7} emotions: {', '.join(tts.emotions(v))}")
        return 0
    text = sys.stdin.read() if a.text in (None, "-") else a.text
    if not text.strip():
        ap.error("no text given")
    audio = tts.synthesize(text, voice=a.voice, emotion=a.emotion, intensity=a.intensity, speed=a.speed,
                           pitch=a.pitch)
    sf.write(a.output, audio, SR)
    print(f"wrote {a.output} ({len(audio) / SR:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
