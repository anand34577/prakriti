"""Board benchmark: Kokoro-82M ONNX on CPU (RK3588 / RK3576 / any Linux or Windows).

Setup on the board (Debian/Ubuntu arm64):
    sudo apt install -y espeak-ng python3-venv
    python3 -m venv ~/prakriti-bench && . ~/prakriti-bench/bin/activate
    pip install kokoro-onnx soundfile onnxruntime
    python bench_board.py            # downloads models on first run (~330 MB)

Runs each model variant with 1/2/4 threads, pinned to the big cores
(RK3588: A76 cores 4-7, RK3576: A72 cores 4-7). Prints a table and writes
bench_<hostname>.json — send that file back.
"""
import json
import os
import platform
import socket
import statistics
import time
import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODELS = {"fp32": "kokoro-v1.0.onnx", "fp16": "kokoro-v1.0.fp16.onnx", "int8": "kokoro-v1.0.int8.onnx"}
VOICES = "voices-v1.0.bin"
SENTENCES = [
    ("hi", "नमस्ते, आज मौसम बहुत सुहाना है।"),
    ("hi", "कृपया ध्यान दें, गाड़ी प्लेटफ़ॉर्म नंबर तीन पर आ रही है।"),
    ("hi", "कल मेरी office में एक important meeting है।"),
    ("en-us", "Good morning! Here is your schedule for today."),
    ("en-us", "Please remember to carry an umbrella, it might rain later."),
]
DIR = Path.home() / ".cache" / "prakriti-bench"


def fetch(name):
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / name
    if not p.exists():
        print("downloading", name)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(BASE + name, tmp)
        tmp.rename(p)
    return p


def soc():
    try:
        return Path("/proc/device-tree/compatible").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return platform.processor() or platform.machine()


def pin_big_cores():
    if not hasattr(os, "sched_setaffinity"):
        return None
    n = os.cpu_count() or 1
    big = set(range(4, 8)) if n == 8 else set(range(n))
    os.sched_setaffinity(0, big)
    return sorted(big)


def main():
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    cores = pin_big_cores()
    voices = fetch(VOICES)
    results = {"host": socket.gethostname(), "soc": soc(), "machine": platform.machine(),
               "cpus": os.cpu_count(), "pinned_cores": cores, "runs": []}
    print(f"SoC: {results['soc']}  pinned: {cores}")
    print(f"{'model':6} {'thr':>3} {'x_realtime':>10} {'first_sentence_s':>16}")
    for tag, fname in MODELS.items():
        try:
            path = fetch(fname)
        except Exception as e:
            print(f"{tag}: download failed ({e})")
            continue
        for threads in (1, 2, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            k = Kokoro.from_session(sess, str(voices))
            k.create("warm up", voice="hf_alpha", lang="en-us")
            lat, audio, wall = [], 0.0, 0.0
            for lang, text in SENTENCES:
                v = "hf_alpha" if lang == "hi" else "af_heart"
                t0 = time.time()
                samples, sr = k.create(text, voice=v, lang=lang)
                dt = time.time() - t0
                lat.append(dt)
                wall += dt
                audio += len(samples) / sr
            r = {"model": tag, "threads": threads, "x_realtime": round(audio / wall, 2),
                 "median_sentence_s": round(statistics.median(lat), 3)}
            results["runs"].append(r)
            print(f"{tag:6} {threads:>3} {r['x_realtime']:>10} {r['median_sentence_s']:>16}")
    out = Path(f"bench_{results['host']}.json")
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out, "- x_realtime above 1.0 means faster than real time")


if __name__ == "__main__":
    main()
