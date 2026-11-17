# Installation

You need three things: Python, espeak-ng (the program that turns letters into sounds) and the Prakriti
package. Setup takes about five minutes, plus a one-time 360 MB model download.

## 1. Check Python

Prakriti works with Python 3.10, 3.11 and 3.12.

```bash
python --version
```

You should see `Python 3.10.x`, `3.11.x` or `3.12.x`. If you see 3.13 or newer, install 3.12 from
[python.org](https://www.python.org/downloads/) alongside it; PyTorch doesn't support every new version
right away.

## 2. Install espeak-ng

**Ubuntu / Debian / Raspberry Pi OS**

```bash
sudo apt install espeak-ng
```

**Windows**

Download the `.msi` installer from the
[espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases) and run it.

**macOS**

```bash
brew install espeak-ng
```

Check it worked:

```bash
espeak-ng --version
```

You should see a line starting with `eSpeak NG text-to-speech: 1.5`.

## 3. Create a virtual environment

A virtual environment is a private folder of Python packages, so Prakriti's dependencies don't clash with
other projects.

```bash
python -m venv prakriti-env
```

Activate it:

```bash
# Linux / macOS
source prakriti-env/bin/activate
# Windows PowerShell
prakriti-env\Scripts\Activate.ps1
```

Your prompt now starts with `(prakriti-env)`.

## 4. Install Prakriti

```bash
pip install "git+https://github.com/anand34577/prakriti.git"
```

This pulls in PyTorch, so it can take a few minutes. If you have an NVIDIA GPU and want to use it, install
the CUDA build of PyTorch first by following [pytorch.org](https://pytorch.org/get-started/locally/).

## 5. Check it

```bash
prakriti --list-voices
```

The first run downloads the model from Hugging Face (about 360 MB) and caches it. Then you should see ten
lines, one per voice:

```
Ahana      female  emotions: neutral, angry, calm, empathetic, excited, fear, happy, sad, serious, surprised, whisper
...
Vasant     male    emotions: neutral, angry, calm, empathetic, excited, fear, happy, sad, serious, surprised, whisper
```

Next: [Getting Started](Getting-Started).
