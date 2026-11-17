"""Audio-verified pronunciation labels.

For each teacher clip, every word with more than one candidate pronunciation
is resolved by CTC likelihood under a multilingual phoneme recognizer
(facebook/wav2vec2-xlsr-53-espeak-cv-ft, Apache 2.0): the variant the
teacher actually spoke wins. Outputs
  labels.jsonl   {id, ipa, n_ambiguous, n_changed}
  votes.jsonl    {word, winner, loser, margin}   (later aggregated into a lexicon)
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata

import numpy as np
import torch

from prakriti.core.resumable import ItemLedger
from prakriti.text.candidates import key, tokenize, word_candidates
from prakriti.text.frontend import NONVERBAL, TAG

MID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


class PhonemeScorer:
    def __init__(self, device="cuda"):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MID).to(device).eval().half()
        vocab = json.load(open(self._vocab_path()))
        self.vocab = {unicodedata.normalize("NFC", k): v for k, v in vocab.items()}
        self.maxlen = max(map(len, self.vocab))
        self.blank = self.model.config.pad_token_id
        self.device = device

    @staticmethod
    def _vocab_path():
        from huggingface_hub import hf_hub_download
        return hf_hub_download(MID, "vocab.json")

    def ids(self, ipa: str) -> list[int]:
        s = unicodedata.normalize("NFC", ipa.replace("ˈ", "").replace("ˌ", "").replace(" ", ""))
        out, i = [], 0
        while i < len(s):
            for L in range(min(self.maxlen, len(s) - i), 0, -1):
                t = s[i:i + L]
                if t in self.vocab:
                    out.append(self.vocab[t])
                    i += L
                    break
            else:
                i += 1  # symbol the recognizer lacks (punctuation etc.)
        return out

    @torch.inference_mode()
    def logprobs(self, wav16: np.ndarray) -> torch.Tensor:
        x = self.fe(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device).half()
        return self.model(x).logits.float().log_softmax(-1)[0]  # (T, V)

    def nll(self, lp: torch.Tensor, ipa_list: list[str]) -> list[float]:
        targets = [torch.tensor(self.ids(s) or [self.blank]) for s in ipa_list]
        T = lp.shape[0]
        lps = lp.unsqueeze(1).expand(T, len(targets), lp.shape[-1])
        flat = torch.cat(targets).to(lp.device)
        loss = torch.nn.functional.ctc_loss(
            lps, flat, input_lengths=torch.full((len(targets),), T, dtype=torch.long),
            target_lengths=torch.tensor([len(t) for t in targets]), blank=self.blank,
            reduction="none", zero_infinity=True)
        return loss.tolist()


def resolve(scorer: PhonemeScorer, lp, text: str):
    toks = tokenize(text)
    cands = [word_candidates(t) for t in toks]
    choice = [0] * len(toks)
    amb = [i for i, c in enumerate(cands) if len(c) > 1]

    def render(ch):
        return " ".join(c[k] if c else "" for c, k in zip(cands, ch))

    votes = []
    # coordinate ascent: each ambiguous word against the current best sentence
    for i in amb:
        trials = []
        for k in range(len(cands[i])):
            ch = list(choice)
            ch[i] = k
            trials.append(render(ch))
        losses = scorer.nll(lp, trials)
        best = int(np.argmin(losses))
        margin = sorted(losses)[1] - losses[best] if len(losses) > 1 else 0.0
        votes.append({"word": toks[i], "winner": cands[i][best],
                      "loser": [c for j, c in enumerate(cands[i]) if j != best], "margin": round(margin, 3)})
        choice[i] = best
    # final label keeps punctuation tokens (pauses/intonation) from the text
    parts = []
    for t, c, k in zip(toks, cands, choice):
        parts.append(c[k] if c else (t if not t.isspace() else ""))
    ipa = " ".join(p for p in parts if p).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ।", ".")
    return ipa, votes, sum(1 for i in amb if choice[i] != 0)


def main(manifest: str, out_dir: str):
    """manifest: jsonl with {id, wav, text}."""
    import soundfile as sf
    import torchaudio
    scorer = PhonemeScorer()
    os.makedirs(out_dir, exist_ok=True)
    from prakriti.s2.verify import in_shard, shard_suffix
    sfx = shard_suffix()
    vf = open(os.path.join(out_dir, f"votes{sfx}.jsonl"), "a", encoding="utf-8")
    with ItemLedger(os.path.join(out_dir, f"labels{sfx}.jsonl"), fsync_every=50) as led:
        for line in open(manifest, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in led or not in_shard(r["id"]):
                continue
            a, sr = sf.read(r["wav"], dtype="float32")
            wav16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
            lp = scorer.logprobs(wav16)
            text = TAG.sub(lambda m: NONVERBAL[m.group(1)], r["text"])  # [laugh] -> token, not a word
            ipa, votes, changed = resolve(scorer, lp, text)
            for v in votes:
                vf.write(json.dumps(v, ensure_ascii=False) + "\n")
            led.add(r["id"], ipa=ipa, n_ambiguous=len(votes), n_changed=changed)
    vf.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
