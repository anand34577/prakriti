"""Patches StyleTTS2/models.py: replaces load_checkpoint with a strict, name-aware loader.

Fixes two silent failures with strict=False loading of Kokoro weights:
  1. 'module.' prefix mismatch (modules are wrapped before loading)
  2. weight-norm names (weight_g/weight_v vs parametrizations.weight.original0/1)
and aborts if any pretrained module would be left partially random.
Run:  python patches/safe_load_checkpoint.py third_party/kikiri-tts/StyleTTS2/models.py
"""
import re
import sys

NEW = '''
def _match_keys(target_keys, src):
    """Map checkpoint keys onto the model's naming (prefix + weight-norm style)."""
    want_prefix = any(k.startswith("module.") for k in target_keys)
    out = {}
    for k, v in src.items():
        k = k.removeprefix("module.")
        if k.endswith(".weight_g"):
            k = k[:-9] + ".parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            k = k[:-9] + ".parametrizations.weight.original1"
        if not any(t.endswith(".parametrizations.weight.original0") for t in target_keys):
            k = k.replace(".parametrizations.weight.original0", ".weight_g").replace(
                ".parametrizations.weight.original1", ".weight_v")
        out[("module." + k) if want_prefix else k] = v
    return out


STRICT_MODULES = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            target = model[key].state_dict()
            src = _match_keys(set(target), params[key])
            missing = [k for k in target if k not in src]
            unexpected = [k for k in src if k not in target]
            shape_bad = [k for k in src if k in target and src[k].shape != target[k].shape]
            for k in shape_bad:
                src.pop(k)
            model[key].load_state_dict(src, strict=False)
            print(f"[load] {key}: {len(target) - len(missing) - len(shape_bad)}/{len(target)} tensors loaded"
                  f" (missing {len(missing)}, unexpected {len(unexpected)}, shape {len(shape_bad)})", flush=True)
            if key in STRICT_MODULES and (missing or shape_bad):
                raise RuntimeError(f"pretrained module {key} incomplete: {missing[:3]} {shape_bad[:3]}")
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1  # saved epoch is complete: resume with the next one
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        print(f"[resume] continuing at epoch {epoch}, iteration {iters}", flush=True)
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
'''


def main(path):
    src = open(path, encoding="utf-8").read()
    if "_match_keys" in src:
        print("already patched")
        return
    pat = re.compile(r"\ndef load_checkpoint\(.*?\n    return model, optimizer, epoch, iters\n", re.S)
    assert pat.search(src), "load_checkpoint not found"
    src = pat.sub(lambda m: NEW, src, count=1)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


if __name__ == "__main__":
    main(sys.argv[1])
