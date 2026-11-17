# Configuration

Prakriti has no config file. You set everything per call, either as command-line options or as Python
arguments, plus one optional environment variable.

## Command-line options

| Option | Default | What it does |
|---|---|---|
| `text` | (required) | The text to speak. Use `-` to read it from standard input. |
| `-v`, `--voice` | `Rasa_F` | Which voice. See the table below or run `prakriti --list-voices`. |
| `-e`, `--emotion` | `neutral` | Emotion style. If the voice doesn't have it, neutral is used. |
| `-i`, `--intensity` | `1.0` | How strong the emotion is: `0` = neutral, `1` = full. Up to `1.5` is accepted. |
| `-s`, `--speed` | `1.0` | Speaking rate. `0.9` is a little slower, `1.1` a little faster. |
| `-p`, `--pitch` | `1.0` | Pitch multiplier on top of the voice's calibrated pitch. `0.95` is slightly lower. |
| `-o`, `--output` | `prakriti.wav` | Where to write the WAV file (24 kHz, mono). |
| `-m`, `--model` | `anand-k-maurya/prakriti-v1` | A local model folder or a Hugging Face repo id. |
| `--device` | auto | `cuda` for an NVIDIA GPU, `cpu` otherwise. Auto picks the GPU if PyTorch can see one. |
| `--list-voices` | | Print the voices and their emotions, then exit. |

## Python arguments

`Prakriti.synthesize(text, ...)` and `Prakriti.stream(text, ...)` take the same settings:

| Argument | Default | Same as |
|---|---|---|
| `voice` | `"Rasa_F"` | `-v` |
| `emotion` | `"neutral"` | `-e` |
| `intensity` | `1.0` | `-i` |
| `speed` | `1.0` | `-s` |
| `pitch` | `1.0` | `-p` |
| `pause_s` | `0.12` | Silence inserted between sentences, in seconds |

`Prakriti.from_pretrained(repo_or_dir, device=None, revision=None)` loads the model. `revision` pins a
specific version of the Hugging Face repo, which is a good idea in production.

## Environment variable

| Variable | What it does |
|---|---|
| `PRAKRITI_MODEL` | Model the CLI loads when you don't pass `-m`. Set it to a local folder to work fully offline. |

The training pipeline has a few more variables; they're described in [Training](Training) and in
`.env.example` in the repo.

## Voices

| Voice | Gender | Trained from | Emotions |
|---|---|---|---|
| `Rasa_F` | female | Real studio speech (Rasa dataset) | angry, fear, happy, sad, serious, surprised |
| `Rasa_M` | male | Real studio speech (Rasa dataset) | angry, fear, happy, sad, serious, surprised |
| `Ahana` | female | Maya 2 Native | angry, calm, empathetic, excited, fear, happy, sad, serious, surprised, whisper |
| `Vaidehi` | female | Maya 2 Native | same as Ahana |
| `Malini` | female | Maya 2 Native | same as Ahana |
| `Charvi` | female | Maya 2 Native | same as Ahana |
| `Anirban` | male | Maya 2 Native | same as Ahana |
| `Nakul` | male | Maya 2 Native | same as Ahana |
| `Vasant` | male | Maya 2 Native | same as Ahana |
| `Siddhant` | male | Maya 2 Native | same as Ahana |

The two Rasa voices sound the most natural because they come from real recordings. The other eight have
more emotions to choose from.

## How pitch calibration works

Each voice has a `pitch_scale` in `voices.json`. It was measured once so that the voice speaks at the median
pitch of its own training recordings (for example, `Rasa_M` around 150 Hz and `Rasa_F` around 210 Hz). The
`pitch` option multiplies on top of that, so `pitch=1.0` always means "as recorded".

## Non-verbal markers

You can put `[laugh]`, `[sigh]`, `[gasp]`, `[hmm]` or `[breath]` in the text. They're weak in this version,
especially `[laugh]`, so don't rely on them yet.
