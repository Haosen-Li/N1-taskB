"""Evaluate a trained ensemble, unchanged, on other task conditions (no training).

RoPE has no position parameters, so a model trained at T=31 can be run at T=207
without new weights; this asks whether the learned retrieval still works when the
context is longer / the source is farther / there are more confusable competitors.
"""
import argparse
import json
from pathlib import Path

import torch

from ensemble import Ensemble, ModelCfg
from task import TaskCfg, make_batch
from train import evaluate, attention_profile

HERE = Path(__file__).resolve().parent

TARGETS = {
    "N16_far": TaskCfg(N=16, post_lo=8, post_hi=16),
    "N16_near": TaskCfg(N=16, post_lo=0, post_hi=8),
    "N64_near": TaskCfg(N=64, post_lo=0, post_hi=8),
    "N64_far": TaskCfg(N=64, post_lo=56, post_hi=64),
    "N192_near": TaskCfg(N=192, post_lo=0, post_hi=8),
    "N192_far": TaskCfg(N=192, post_lo=184, post_hi=192),
    "N192_far_filler": TaskCfg(N=192, post_lo=184, post_hi=192, noise="filler"),
    "N192_far_mix16": TaskCfg(N=192, post_lo=184, post_hi=192, noise="mix", n_confusable=16),
    "N192_near_filler": TaskCfg(N=192, post_lo=0, post_hi=8, noise="filler"),
    "N192_mid": TaskCfg(N=192, post_lo=56, post_hi=64),
    "N512_near": TaskCfg(N=512, post_lo=0, post_hi=8),
    "N512_far": TaskCfg(N=512, post_lo=504, post_hi=512),
    "N512_far_filler": TaskCfg(N=512, post_lo=504, post_hi=512, noise="filler"),
}


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    m = st["meta"]["model"]
    cfg = ModelCfg(**{**m, "max_len": 1100})
    model = Ensemble(cfg, st["meta"]["seeds"], device)
    for k, v in st["params"].items():
        model.params[k].data.copy_(v)
    return model, st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    model, st = load_model(a.ckpt, a.device)
    res = {"ckpt": a.ckpt, "step": st["step"], "targets": {}}
    for name in a.targets.split(","):
        task = TARGETS[name]
        val = make_batch(a.n, torch.Generator(device=a.device).manual_seed(777), a.device, task)
        ev = evaluate(model, task, val, chunk=64 if task.T > 300 else 128)
        ev["attn"] = attention_profile(model, val, n=32)
        res["targets"][name] = ev
        E = len(ev["clean"])
        print(f"{Path(a.ckpt).parent.name}/{Path(a.ckpt).name} -> {name:16s} clean={sum(ev['clean'])/E:.3f} "
              f"paired={sum(ev['paired'])/E:.3f} query={sum(ev['query_new'])/E:.3f} "
              f"pass={sum(ev['pass'])}/{E}  min clean={min(ev['clean']):.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
