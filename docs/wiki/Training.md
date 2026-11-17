# Training

This is how Prakriti v1 was trained, and how to rerun or extend it. You need a Linux server with one NVIDIA
GPU of 48 GB (an RTX A6000 was used), about 250 GB of disk, and a Hugging Face account with access to the
Rasa dataset. The whole run took about 15 GPU hours.

## What the pipeline does

| Stage | What happens | GPU time |
|---|---|---|
| `s2_textgen` | Qwen3-32B writes about 32,000 sentences in Hindi, Hinglish and English across 11 emotions, then fills in sentences until 99.4% of the 20,000 most frequent Hindi words appear | ~45 min |
| `s2_rasa` | Downloads Rasa Hindi (25,054 studio clips), resamples to 24 kHz, trims silence | CPU |
| `s2_synth` | Maya 2 Native reads every sentence in one of 8 voices, balanced by language and emotion | ~30 min |
| `s2_verify` | Whisper checks each clip says the right words; emotion2vec scores the emotion; clipping, silence and speaking-rate checks | ~40 min |
| `s2_pron` | For words where eSpeak and Prakriti's Hindi rules disagree, a phoneme recognizer listens to the clip and picks the version that was actually spoken | ~40 min |
| `s3_build` | Keeps clips that pass every check (41,592 clips, 54.7 h), writes the training lists | CPU |
| `s3_train` | StyleTTS 2 stage 1 then stage 2, starting from Kokoro-82M | ~8 h |
| `s4_*` | Export, style packs per voice and emotion, evaluation, report | ~30 min |

Every stage records what it finished, so a restart skips completed work.

## 1. Set up the server

```bash
git clone https://github.com/anand34577/prakriti.git
cd prakriti
bash server/setup.sh
bash server/envs.sh
```

`setup.sh` installs system packages and the systemd service. `envs.sh` builds one Python environment per
tool (the teacher, text generation, evaluation, training) so their dependencies don't clash, fetches the
StyleTTS 2 recipe and its helper models, converts the Kokoro-82M weights, and applies Prakriti's patches to
the training code. It takes about 25 minutes and is safe to run again.

## 2. Get dataset access

1. On huggingface.co, open the [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) page and click
   **Agree and access**.
2. Create a **read** token under Settings → Access Tokens.
3. On the server, as the same user that runs the pipeline (not root):
   ```bash
   envs/eval/bin/hf auth login
   ```

## What the patches change

`envs.sh` applies three patches from `patches/` to the StyleTTS 2 training code:

- `safe_load_checkpoint.py`: the original loader silently skipped Kokoro weights whose names didn't match,
  so training started from random weights. The patched loader matches the names and stops if anything is
  missing.
- `atomic_saves.py`: checkpoints are written to a temporary file and renamed, so a crash mid-save can't
  leave a corrupt file.
- `virtual_epochs.py`: splits training into roughly 30-minute sub-epochs, each ending in a checkpoint.

## 3. Run it

```bash
sudo systemctl start prakriti-pipeline
bash server/status.sh
```

The pipeline runs under systemd, so it keeps going if your SSH connection drops and restarts after a crash
or reboot. Check progress any time with `server/status.sh`, or follow the log:

```bash
tail -f work/logs/pipeline.log
```

## Training settings that matter

| Setting | Value used | Why |
|---|---|---|
| Stage 1 | 4 × 930 steps, batch 6 | Validation loss levelled off at 0.244 by the fourth sub-epoch |
| Stage 2 | 6 × 530 steps, batch 4 | The adversarial (SLM) phase needs about 44 GB; larger batches ran out of memory |
| Clip length | 1 to 12 s | Clips up to 18 s pushed stage 1 past 48 GB |
| Precision | fp32 | The vocoder's spectrogram step doesn't support complex bf16 |

If stage 2 runs out of memory, `prakriti/s3/train.sh` lowers the batch size by one and resumes from the
last checkpoint on its own.

## After training: pitch calibration

Training shifted every voice's pitch up by 1 to 6 semitones, which made male voices sound strained.
`prakriti/s3/pitch_calibrate.py` measures each voice against its own training recordings and stores a
`pitch_scale` in `voices.json`. Run it after every export.

## Building the release

```bash
python tools/build_hf_release.py work/export hf_release
```

That writes a folder ready to push to Hugging Face: safetensors weights, voices, lexicon, model card,
license and the inference package.
