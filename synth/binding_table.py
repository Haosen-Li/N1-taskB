"""Is the missing piece in stuck models always the layer-0 binding?

For each run's final checkpoint, per member:
    status     passed / stuck (by the run's own first_pass)  and, if an ablation file exists,
               the route (pointer / relay / mixed, see routes.py)
    prev_attn  layer 0: attention from each stored VALUE to the token right before it (its KEY),
               best FREE head (head 0 is skipped in runs where it is hard-wired)
    key_probe  can a linear probe read the KEY's identity from the layer-0 output at the VALUE
               position?  (8 keys -> chance 0.125)  This also measures binding when head 0 is fixed.

    python binding_table.py                 -> results/binding_table.md, results/binding_table.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from analyze import load
from mechanism import collect, ridge_probe
from routes import classify
from zeroshot import TARGETS, load_model

HERE = Path(__file__).resolve().parent
RUNS = [("S1_N16_far", "N16_far"), ("S1_N64_far", "N64_far"), ("S1_N192_near", "N192_near"),
        ("S1_N192_far", "N192_far"), ("S2_N192_far_filler", "N192_far_filler"),
        ("S2_N192_far_mix16", "N192_far_mix16"), ("X2_N192_near_filler", "N192_near_filler"),
        ("X1_N192_far_prev1", "N192_far"), ("X1_N192_near_prev1", "N192_near"),
        ("X1_N192_far_filler_prev1", "N192_far_filler"), ("X4_N192_far_nope_prev1", "N192_far"),
        ("X6_N192_near_answeronly", "N192_near"), ("X6_N192_far_answeronly", "N192_far"),
        ("X7a_N192_near_prev1_answeronly", "N192_near"), ("X8_N192_mid", "N192_mid"),
        ("X8_N192_mid_answeronly", "N192_mid")]


@torch.no_grad()
def measure(run, target, device):
    model, _ = load_model(HERE / "runs" / run / "final.pt", device)
    tr = collect(model, TARGETS[target], 1024, 4242, device)
    te = collect(model, TARGETS[target], 512, 4343, device)
    prev = te["prev"]                                              # [E,H]
    if model.cfg.fixed_head_offset:
        prev = prev[:, 1:]                                         # ignore the hard-wired head
    return prev.max(-1).values.cpu().numpy(), ridge_probe(tr["feats"], tr["labels"], te["feats"], te["labels"]).cpu().numpy()


def main(device="cuda:0"):
    torch.backends.cuda.matmul.allow_tf32 = False
    table, rows = {}, []
    for run, target in RUNS:
        if not (HERE / "runs" / run / "final.pt").exists():
            continue
        r = load(run)
        prev, probe = measure(run, target, device)
        abl = HERE / "results" / f"ablate_{run}_final_on_{target}.json"
        a = json.load(open(abl)) if abl.exists() else None
        members = []
        for e, t in enumerate(r["t_pass"]):
            status = "stuck" if t is None else (classify(a, e) if a else "passed")
            members.append(dict(member=e, status=status, prev_attn=float(prev[e]), key_probe=float(probe[e])))
        table[run] = members
        for st in ["pointer", "relay", "mixed", "passed", "stuck"]:
            m = [x for x in members if x["status"] == st]
            if m:
                pa, kp = [x["prev_attn"] for x in m], [x["key_probe"] for x in m]
                rows.append(f"| {run} | {st} | {len(m)} | {np.median(pa):.2f} ({min(pa):.2f}–{max(pa):.2f}) | "
                            f"{np.median(kp):.2f} ({min(kp):.2f}–{max(kp):.2f}) |")
    lines = ["| 运行 | 状态 | 成员数 | 前一 token 注意力 中位（范围） | 键身份探针 中位（范围；机会 0.125） |",
             "|---|---|---:|---|---|"] + rows
    (HERE / "results" / "binding_table.md").write_text("\n".join(lines) + "\n")
    (HERE / "results" / "binding_table.json").write_text(json.dumps(table, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
