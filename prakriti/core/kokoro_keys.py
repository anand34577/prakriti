"""Weight-norm key conversion between Kokoro releases (old torch weight_norm:
`.weight_g` / `.weight_v`) and the training code (torch parametrizations:
`.parametrizations.weight.original0` / `.original1`).

Without this, load_state_dict(strict=False) silently skips every
weight-normalised layer and they train from random init.
"""
OLD_G, OLD_V = ".weight_g", ".weight_v"
NEW_G, NEW_V = ".parametrizations.weight.original0", ".parametrizations.weight.original1"


def to_parametrized(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(OLD_G):
            k = k[: -len(OLD_G)] + NEW_G
        elif k.endswith(OLD_V):
            k = k[: -len(OLD_V)] + NEW_V
        out[k] = v
    return out


def to_legacy(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(NEW_G):
            k = k[: -len(NEW_G)] + OLD_G
        elif k.endswith(NEW_V):
            k = k[: -len(NEW_V)] + OLD_V
        out[k] = v
    return out


def convert_base(src: str, dst: str, reference_ckpt: str | None = None) -> None:
    import torch
    net = torch.load(src, map_location="cpu", weights_only=False)["net"]
    net = {m: to_parametrized(sd) for m, sd in net.items()}
    if reference_ckpt:  # prove every key now lines up with what the trainer builds
        ref = torch.load(reference_ckpt, map_location="cpu", weights_only=False)["net"]
        for m in net:
            if m in ref:
                a, b = set(net[m]), set(ref[m])
                assert a == b, f"{m}: {len(a - b)} unmatched, e.g. {sorted(a - b)[:2]} / {sorted(b - a)[:2]}"
                for k in a:
                    assert net[m][k].shape == ref[m][k].shape, f"{m}.{k} shape {net[m][k].shape} vs {ref[m][k].shape}"
        print("all keys and shapes match the training model")
    torch.save({"net": net}, dst)


if __name__ == "__main__":
    import sys
    convert_base(*sys.argv[1:])
