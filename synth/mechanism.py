"""What differs between behaviourally identical plateau states?

Per ensemble member, on a given task condition:
  prev_attn      layer-0, max over heads: attention from each stored VALUE position to the
                 token right before it (its KEY), averaged over records/examples.
  key_probe      linear decodability of that key's identity from the layer-0 residual
                 stream at the value position (closed-form ridge onto one-hot, 8 classes;
                 fit on 1024 examples x 4 records, scored on 512 held-out examples).
  readout_*      layer-1 attention from the final position (correct value / all 4 values / noise).
The probe is correlational: high decodability does not prove the network uses it.
"""
import argparse
import json
from pathlib import Path

import torch

from task import make_batch
from train import evaluate
from zeroshot import TARGETS, load_model


@torch.no_grad()
def collect(model, task, n, seed, device, chunk=64):
    g = torch.Generator(device=device).manual_seed(seed)
    b = make_batch(n, g, device, task)
    E = model.E
    feats, labels, prev, ro_c, ro_v, ro_n = [], [], [], [], [], []
    for s in range(0, n, chunk):
        x = b["x"][s:s + chunk]
        m = len(x)
        vpos = b["vpos"][s:s + chunk]                                  # [m,R]
        R = vpos.shape[1]
        logits, maps, states = model(x[None].expand(E, -1, -1), attn=True, hidden=True)
        rows = torch.arange(m, device=device)[:, None].expand(m, R)
        a0 = maps[0].permute(1, 3, 4, 0, 2)                            # [m,T,T,E,H]
        pa = a0[rows, vpos, vpos - 1]                                  # [m,R,E,H]
        prev.append(pa.mean((0, 1)))                                   # [E,H]
        h0 = states[0][:, rows, vpos]                                  # [E,m,R,d]
        feats.append(h0.reshape(E, m * R, -1))
        labels.append(b["keys"][s:s + chunk].reshape(-1) - 4)
        last = maps[1][:, :, :, -1, :]                                 # [E,m,H,T]
        src = b["src"][s:s + chunk]
        ro_c.append(last.permute(1, 3, 0, 2)[torch.arange(m, device=device), src].mean(0))  # [E,H]
        ro_v.append(last.gather(-1, vpos.view(1, m, 1, R).expand(E, m, last.shape[2], R)).sum(-1).mean(1))
        ro_n.append((last * b["noise_mask"][s:s + chunk].view(1, m, 1, -1)).sum(-1).mean(1))
    w = lambda L: torch.stack(L).mean(0)
    return dict(prev=w(prev), feats=torch.cat(feats, 1), labels=torch.cat(labels), ro_c=w(ro_c),
                ro_v=w(ro_v), ro_n=w(ro_n))


def ridge_probe(Xtr, ytr, Xte, yte, lam=1e-2):
    """Per-member closed-form ridge classifier. X [E,N,d]."""
    mu, sd = Xtr.mean(1, keepdim=True), Xtr.std(1, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ones = lambda X: torch.cat([X, torch.ones_like(X[..., :1])], -1)
    Xtr, Xte = ones(Xtr).double(), ones(Xte).double()
    Y = torch.nn.functional.one_hot(ytr, 8).double()[None].expand(Xtr.shape[0], -1, -1)
    A = Xtr.transpose(1, 2) @ Xtr + lam * Xtr.shape[1] * torch.eye(Xtr.shape[-1], device=Xtr.device, dtype=torch.double)
    W = torch.linalg.solve(A, Xtr.transpose(1, 2) @ Y)
    return ((Xte @ W).argmax(-1) == yte[None]).double().mean(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model, st = load_model(a.ckpt, a.device)
    task = TARGETS[a.target]
    tr = collect(model, task, 1024, 4242, a.device)
    te = collect(model, task, 512, 4343, a.device)
    probe = ridge_probe(tr["feats"], tr["labels"], te["feats"], te["labels"])
    val = make_batch(512, torch.Generator(device=a.device).manual_seed(777), a.device, task)
    ev = evaluate(model, task, val)
    res = dict(ckpt=a.ckpt, step=st["step"], target=a.target,
               prev_attn_maxhead=te["prev"].max(-1).values.tolist(), key_probe=probe.tolist(),
               readout_correct_maxhead=te["ro_c"].max(-1).values.tolist(),
               readout_allvalues_maxhead=te["ro_v"].max(-1).values.tolist(),
               readout_noise_mean=te["ro_n"].mean(-1).tolist(),
               clean=ev["clean"], in_context=ev["in_context"], passed=ev["pass"],
               query_change=ev["query_change_when_answer_differs"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    E = len(res["clean"])
    f = lambda k: " ".join(f"{x:.2f}" for x in res[k])
    print(f"{a.ckpt} on {a.target}\n  clean      {f('clean')}\n  prev_attn  {f('prev_attn_maxhead')}\n"
          f"  key_probe  {f('key_probe')}\n  ro_correct {f('readout_correct_maxhead')}\n"
          f"  ro_values  {f('readout_allvalues_maxhead')}\n  q_change   {f('query_change')}")


if __name__ == "__main__":
    main()
