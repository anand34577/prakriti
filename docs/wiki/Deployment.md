# Deployment

## Measured speed

On one RTX A6000 and on the same server's CPU (x86, PyTorch):

| Device | Faster than real time | Median time per sentence |
|---|---|---|
| GPU (CUDA) | 36.6x | 0.11 s |
| CPU, 4 threads | 2.6x | 1.6 s |
| CPU, 2 threads | 1.6x | 2.5 s |

"2.6x faster than real time" means 10 seconds of speech takes about 4 seconds to make.

## As a service on a Linux server

A small HTTP wrapper is enough for most setups. With FastAPI:

```python
# serve.py
import io
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from prakriti.tts import Prakriti

app = FastAPI()
tts = Prakriti.from_pretrained()

@app.get("/speak")
def speak(text: str, voice: str = "Rasa_F", emotion: str = "neutral"):
    buf = io.BytesIO()
    sf.write(buf, tts.synthesize(text, voice=voice, emotion=emotion), 24000, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")
```

```bash
pip install fastapi uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8080
```

Then `curl "http://localhost:8080/speak?text=नमस्ते" -o out.wav`. Put it behind a reverse proxy with
authentication before exposing it to a network.

## On CPU-only machines

Install the CPU build of PyTorch to save about 2 GB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/anand34577/prakriti.git"
```

Set threads explicitly; more threads than physical cores slows it down:

```python
import torch
torch.set_num_threads(4)
```

## Rockchip RK3588 / RK3576

Prakriti hasn't been benchmarked on these boards yet, and NPU (RKNN) export isn't done. What works today:

- It runs on the board's CPU cores with the steps above (arm64 PyTorch wheels exist for both).
- The repo has `tools/bench_board.py`, which measures Kokoro-82M (the same architecture and size as
  Prakriti) on the board's CPU through ONNX Runtime. Run it and share the JSON it writes; that tells us
  whether the RK3576 needs a smaller variant.

Pin the process to the big cores (cores 4 to 7 on both chips) for the best speed:

```bash
taskset -c 4-7 prakriti "नमस्ते" -o test.wav
```

## Pinning a version

Hugging Face repos can change. In production, pin a revision: copy a commit hash from the model page's
**Files and versions** tab.

```python
tts = Prakriti.from_pretrained("anand-k-maurya/prakriti-v1", revision="<commit hash>")
```
