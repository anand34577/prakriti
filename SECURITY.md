# Security policy

## Reporting a vulnerability

Please don't open a public issue for security problems. Use GitHub's private reporting instead:
**Security → Report a vulnerability** on this repository. We'll reply within a week.

Include what you found, how to reproduce it, and which version you used.

## What counts

- Code execution through a crafted model file, voice pack or lexicon
- The loader accepting weights it should reject
- Anything that leaks local files or credentials

Voice packs (`*.pt`) are loaded with `torch.load(..., weights_only=True)`, and model weights are read
from safetensors. If you find a way around either, we want to hear about it.

## Misuse of generated speech

Prakriti can produce realistic speech. Using it to impersonate real people, commit fraud or defeat
voice authentication is against the intended use described in the model card. If you see the model
used that way, report it to the platform where it happened.
