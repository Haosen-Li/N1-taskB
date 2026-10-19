"""Causal test of HOW a model reads the stored values at the final position.

Only the final (prediction) position's attention in the LAST layer is edited:
  normal     unchanged
  no_values  may not attend to the 4 stored value positions
  only_values may attend only to the 4 stored value positions
Everything else (all other positions, all of layer 0) is untouched.
A model that reads values *directly* should collapse to chance (in-context ~0.41)
under no_values and keep its behaviour under only_values; a model that reads a
*diffuse summary* spread over other positions should show the opposite pattern.
"""
import argparse
import json
from pathlib import Path

import torch

from task import make_batch
from zeroshot import TARGETS, load_model


@torch.no_grad()
def run(model, task, n, device, chunk=64):
    g = torch.Generator(device=device).manual_seed(777)
    b = make_batch(n, g, device, task)
    E, T, L = model.E, task.T, model.cfg.layers
    kinds = ("normal", "no_values", "only_values", "no_values_and_next", "no_record_block", "no_after_block")
    res = {k: {"clean": torch.zeros(E, device=device), "in_context": torch.zeros(E, device=device)}
           for k in kinds}
    for s in range(0, n, chunk):
        x, vpos = b["x"][s:s + chunk], b["vpos"][s:s + chunk]
        m = len(x)
        rows = torch.arange(m, device=device)[:, None]
        ans, vals = b["answer"][s:s + chunk], b["values"][s:s + chunk]
        pos = torch.arange(T, device=device)[None]
        is_val = torch.zeros(m, T, dtype=torch.bool, device=device)
        is_val[rows, vpos] = True
        is_next = torch.zeros_like(is_val)
        is_next[rows, vpos + 1] = True                 # STORE of the next record / first token after block
        rs, re = vpos[:, :1] - 2, vpos[:, -1:] + 1     # record block [rs, re] incl. the token after it
        in_block = (pos >= rs) & (pos <= re)
        after = (pos > re) & (pos < T - 1)
        forbid = {"no_values": is_val, "no_values_and_next": is_val | is_next,
                  "no_record_block": in_block, "no_after_block": after}
        for kind in res:
            if kind == "normal":
                block = None
            else:
                blk = torch.zeros(m, T, T, dtype=torch.bool, device=device)
                if kind == "only_values":
                    keep = is_val.clone()
                    keep[:, -1] = True               # never leave a row empty
                    blk[:, -1] = ~keep
                else:
                    blk[:, -1] = forbid[kind]
                block = [None] * (L - 1) + [blk]
            pred = model(x[None].expand(E, -1, -1), block=block)[:, :, -1].argmax(-1)
            res[kind]["clean"] += (pred == ans).float().sum(1)
            res[kind]["in_context"] += (pred[..., None] == vals[None]).any(-1).float().sum(1)
    return {k: {m: (v / n).tolist() for m, v in d.items()} for k, d in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    model, st = load_model(a.ckpt, a.device)
    r = run(model, TARGETS[a.target], a.n, a.device)
    r.update(ckpt=a.ckpt, target=a.target, step=st["step"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r))
    f = lambda L: " ".join(f"{x:.2f}" for x in L)
    print(f"{a.ckpt} on {a.target}")
    for k in ("normal", "no_values", "only_values", "no_values_and_next", "no_record_block", "no_after_block"):
        print(f"  {k:19s} clean {f(r[k]['clean'])}")


if __name__ == "__main__":
    main()
