"""How far back does the final position look in the LAST layer?  (experiment X5a)

Attention mass of the final position, binned by distance d = (T-1) - key position,
mean over heads and examples, per member.  For N=192 far the stored values sit at d = 186..203.

    python readout_distance.py --ckpt runs/X5_far_prev1_rope_early/step_100.pt --target N192_far
"""
import argparse
import json
from pathlib import Path

import torch

from task import make_batch
from zeroshot import TARGETS, load_model

BINS = [(0, 8), (9, 32), (33, 96), (97, 185), (186, 10**6)]


@torch.no_grad()
def profile(model, task, device, n=128):
    b = make_batch(n, torch.Generator(device=device).manual_seed(777), device, task)
    T = task.T
    d = (T - 1) - torch.arange(T, device=device)                       # distance of each key position
    acc = torch.zeros(model.E, len(BINS), device=device)
    for s in range(0, n, 32):
        _, maps = model(b["x"][s:s + 32][None].expand(model.E, -1, -1), attn=True)
        last = maps[-1][:, :, :, -1, :].mean(2)                         # [E,m,T] head-mean, final query
        for i, (lo, hi) in enumerate(BINS):
            acc[:, i] += (last * ((d >= lo) & (d <= hi)).float()).sum(-1).sum(1)
    return (acc / n).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--target", default="N192_far")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"bins": [f"{lo}-{hi if hi < 10**6 else ''}" for lo, hi in BINS], "runs": {}}
    for ck in a.ckpts:
        model, st = load_model(ck, a.device)
        prof = profile(model, TARGETS[a.target], a.device)
        res["runs"][ck] = prof.tolist()
        med = prof.median(0).values
        print(f"{ck:52s} median over members: " + "  ".join(f"d {b}: {v:.2f}" for b, v in zip(res["bins"], med)))
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
