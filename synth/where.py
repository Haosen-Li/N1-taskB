"""Where does the final position look in the last layer?  Attention mass (max-mass head)
binned by position relative to the record block, per member."""
import argparse
import json
from pathlib import Path

import torch

from task import make_batch
from zeroshot import TARGETS, load_model

BINS = [("before records", None), ("STORE/KEY tokens", None), ("VALUE tokens", None),
        ("1-8 after block", (1, 8)), ("9-32 after block", (9, 32)), ("33+ after block", (33, 10**6)),
        ("QUERY / self", None)]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    model, _ = load_model(a.ckpt, a.device)
    task = TARGETS[a.target]
    b = make_batch(128, torch.Generator(device=a.device).manual_seed(777), a.device, task)
    E, T = model.E, task.T
    acc = torch.zeros(E, len(BINS), device=a.device)
    for s in range(0, 128, 32):
        x = b["x"][s:s + 32]
        _, maps = model(x[None].expand(E, -1, -1), attn=True)
        last = maps[-1][:, :, :, -1, :]                         # [E,m,H,T]
        rs = b["vpos"][s:s + 32, 0] - 2                          # record block start
        re = b["vpos"][s:s + 32, -1]                            # last record token (the last value)
        pos = torch.arange(T, device=a.device)[None]
        rel = pos - re[:, None]
        inblock = (pos >= rs[:, None]) & (pos <= re[:, None])
        isval = torch.zeros_like(inblock)
        isval[torch.arange(len(x), device=a.device)[:, None], b["vpos"][s:s + 32]] = True
        masks = [pos < rs[:, None], inblock & ~isval, isval, (rel >= 1) & (rel <= 8) & (pos < T - 2),
                 (rel >= 9) & (rel <= 32) & (pos < T - 2), (rel >= 33) & (pos < T - 2), pos >= T - 2]
        for i, m in enumerate(masks):
            acc[:, i] += (last.mean(2) * m[None].float()).sum(-1).sum(1)
    acc /= 128
    res = {"ckpt": a.ckpt, "target": a.target, "bins": [n for n, _ in BINS], "mass_headmean": acc.tolist()}
    Path(a.out).write_text(json.dumps(res))
    print("member  " + " | ".join(n for n, _ in BINS))
    for e in range(E):
        print(f"{e:4d}    " + "   ".join(f"{v:.2f}" for v in acc[e].tolist()))


if __name__ == "__main__":
    main()
