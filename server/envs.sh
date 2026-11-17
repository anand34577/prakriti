#!/usr/bin/env bash
# Builds everything the training pipeline needs on a fresh Ubuntu GPU server:
# one venv per tool (so dependency pins never clash), the StyleTTS 2 recipe with
# its helper models, and Prakriti's patches. Idempotent: finished parts are skipped.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENVS="$REPO/envs"; TP="$REPO/third_party"; mkdir -p "$ENVS" "$TP"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
TORCH_IDX=https://download.pytorch.org/whl/cu128
CU=(--extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

mk() {  # mk <name> <pip args...>
  local name=$1; shift
  [ -f "$ENVS/$name/.ok" ] && { echo "[skip] env $name"; return; }
  echo "[env] $name"
  uv venv -q --python 3.10 "$ENVS/$name"
  VIRTUAL_ENV="$ENVS/$name" uv pip install -q "$@"
  touch "$ENVS/$name/.ok"
}

# Teacher used for the synthetic voices: Maya 2 Native (its own engine, CUDA 12.8)
mk maya "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" soundfile "${CU[@]}"

# Teacher evaluated in session 1 (not used for training data)
mk svara torch torchaudio transformers accelerate snac soundfile "${CU[@]}"

# Text generation (Qwen3-32B-AWQ through vLLM)
mk llm vllm pyyaml

# Kokoro / Prakriti inference, export, pitch calibration
mk kokoro torch "kokoro>=0.9" "misaki[en]" soundfile onnxruntime pip librosa safetensors \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  "${CU[@]}"

# Evaluation: Whisper, emotion2vec, UTMOS, transliteration (+ CUDA 12 cuBLAS/cuDNN for faster-whisper)
mk eval torch torchaudio transformers faster-whisper funasr modelscope jiwer indic-transliteration \
  soundfile pyyaml datasets nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" "${CU[@]}"
if [ ! -f "$ENVS/eval/cuda12_libs" ]; then
  "$ENVS/eval/bin/python" -c "import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(list(a.__path__)[0] + ':' + list(b.__path__)[0])" \
    > "$ENVS/eval/cuda12_libs"
fi

# Training: StyleTTS 2 dependencies
mk train torch torchaudio soundfile munch pydub pyyaml librosa nltk matplotlib accelerate transformers \
  einops einops-exts tqdm loguru huggingface_hub cython numpy setuptools "misaki[en]>=0.9.4" pip \
  pandas ipython tensorboard "${CU[@]}"

# StyleTTS 2 recipe adapted for Kokoro (kikiri-tts), plus the helper models it expects
if [ ! -d "$TP/kikiri-tts" ]; then
  git clone -q --recurse-submodules https://github.com/semidark/kikiri-tts "$TP/kikiri-tts"
fi
if [ ! -d "$TP/StyleTTS2-upstream" ]; then
  git clone -q --depth 1 https://github.com/yl4579/StyleTTS2 "$TP/StyleTTS2-upstream"
fi
S="$TP/kikiri-tts/StyleTTS2"
cp -rn "$TP/StyleTTS2-upstream/Utils/"* "$S/Utils/"          # text aligner, pitch extractor, PL-BERT
if ! "$ENVS/train/bin/python" -c "import monotonic_align" 2>/dev/null; then
  [ -d "$S/monotonic_align" ] || git clone -q https://github.com/resemble-ai/monotonic_align.git "$S/monotonic_align"
  "$ENVS/train/bin/pip" install -q "$S/monotonic_align"
fi

# Kokoro-82M weights in the recipe's format, and the Kokoro symbol table
( cd "$TP/kikiri-tts"
  [ -f training/kokoro_base.pth ] || "$ENVS/train/bin/python" scripts/prepare_training.py convert-weights
  "$ENVS/train/bin/python" scripts/prepare_training.py patch-styletts2 >/dev/null )

# Prakriti's fixes to the training code (all idempotent)
"$ENVS/train/bin/python" "$REPO/patches/safe_load_checkpoint.py" "$S/models.py"
"$ENVS/train/bin/python" "$REPO/patches/virtual_epochs.py" "$S"
"$ENVS/train/bin/python" "$REPO/patches/atomic_saves.py" "$S"

echo "[ok] environments ready"
for e in "$ENVS"/*/; do
  "$e/bin/python" -c "import torch,sys;print(sys.argv[1], torch.__version__, torch.cuda.is_available())" "$(basename "$e")" 2>/dev/null || true
done
