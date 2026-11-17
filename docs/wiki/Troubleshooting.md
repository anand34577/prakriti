# Troubleshooting

Search this page for the exact error text you see.

## Using the model

### An error mentioning espeak, espeakng or phontab on the first run

espeak-ng is missing or can't be found. Install it as shown in [Installation](Installation#2-install-espeak-ng), then open a
new terminal so the PATH change takes effect.

### `FileNotFoundError: no prakriti.safetensors or prakriti.pth in ...`

The folder passed with `-m` (or in `PRAKRITI_MODEL`) isn't a Prakriti model folder. Point it at the folder
that contains `config.json`, `prakriti.safetensors` and `voices/`, or leave it unset to download from
Hugging Face.

### `ValueError: unknown voice 'rasa_f'`

Voice names are case-sensitive: use `Rasa_F`. `prakriti --list-voices` prints the exact names.

### `RuntimeError: decoder: missing [...]` or `... is not at its identity value`

The weights don't match the network. This happens with a model file from another project or a partial
download. Delete the cached copy and download again:

```bash
python -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir())"
```

Then remove the `prakriti` entry from `~/.cache/huggingface/hub/` and rerun.

### The voice sounds too high or too tense

Lower the pitch a little: `-p 0.95` on the command line or `pitch=0.95` in Python. If a voice sounds
strained at the default, please open an issue with the sentence and voice.

### An emotion has no effect

That voice may not have that emotion; the command falls back to neutral without an error. Check
`prakriti --list-voices`, and try `-i 1.0`. Anger, fear and excitement are subtle in v1.

### A Hindi word is pronounced wrongly

See [Usage → Fix a word that sounds wrong](Usage#fix-a-word-that-sounds-wrong).

### Romanized Hindi sounds English ("kya haal hai")

Romanized Hindi is read with English rules in v1. Write Hindi words in Devanagari.

### It's slow on CPU

Check that PyTorch isn't using more threads than you have physical cores (`torch.set_num_threads(4)`), and
that you loaded the model once rather than per sentence.

## Training pipeline

### `torch.OutOfMemoryError: CUDA out of memory` during stage 2

The adversarial phase needs about 44 GB at batch 4. `train.sh` lowers the batch automatically; if it reaches
batch 2 and still fails, check `nvidia-smi` for another process using the GPU (an Ollama server held 32 GB
during this project's run).

### `PytorchStreamReader failed reading zip archive: failed finding central directory`

A checkpoint was cut off mid-write by a crash. With the patches applied this can't happen for new
checkpoints, and `train.sh` sets aside any corrupt file and resumes from the previous one.

### `Cannot access gated repo ... you are not in the authorized list`

Your token is valid but the dataset terms aren't accepted for that account. Open the dataset page while
logged in as the token's owner and click **Agree and access**.

### `Token is required ... but no token found` after `hf auth login`

You logged in as a different Linux user (often root). Run `hf auth login` as the user the pipeline runs as.

### `Library libcublas.so.12 is not found`

faster-whisper needs the CUDA 12 math libraries. `server/envs.sh` installs them and writes their path to
`envs/eval/cuda12_libs`; the pipeline sets `LD_LIBRARY_PATH` from that file.

### vLLM: `Could not find nvcc`

Set `VLLM_USE_FLASHINFER_SAMPLER=0` (already set in `configs/pipeline.yaml`).
