# Contributing to Prakriti

Thanks for helping. Bug reports with a sentence that reads wrong are the most useful thing you can send:
paste the exact text, the voice, and what you expected to hear.

## Set up a development copy

```bash
git clone https://github.com/anand34577/prakriti.git
cd prakriti
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

You also need espeak-ng on your system (`sudo apt install espeak-ng`, or the Windows installer from
the espeak-ng releases page).

## Run the tests

```bash
python -m pytest -q tests
```

All tests run on CPU in about 15 seconds and don't download the model. CI runs the same command on
every pull request.

## Where things live

| Path | What it is |
|---|---|
| `prakriti/tts.py`, `prakriti/cli.py` | Runtime: loading the model, voices, emotions, pitch, streaming |
| `prakriti/text/` | Text front-end: normalization, Hindi letter-to-sound rules, lexicon |
| `prakriti/s1` … `prakriti/s3` | Training pipeline stages (data, verification, training, export) |
| `prakriti/pipeline.py`, `server/` | Resumable stage runner and the systemd setup for a GPU server |
| `patches/` | Changes applied to the StyleTTS 2 training code |
| `tools/` | Release builder, model card, board benchmark |

## Fixing a pronunciation

1. Check what the front-end produces:
   ```bash
   python -c "from prakriti.text.frontend import phonemize; print(phonemize('आपका वाक्य'))"
   ```
2. If one word is wrong, add it to `prakriti/text/lexicon.json` (word → IPA, in the same symbol set).
3. If a whole pattern is wrong (for example a schwa-deletion case), fix the rule in
   `prakriti/text/hindi_g2p.py` and add the word to `tests/test_hindi_g2p.py`.

## Pull requests

- Keep each pull request to one change, and add or update a test when you change behaviour.
- Update `CHANGELOG.md` under an `Unreleased` heading.
- Use short commit messages in the style `fix(text): read 1.5 lakh correctly`.
