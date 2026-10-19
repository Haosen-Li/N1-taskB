"""Train E independent seeds on one task condition and log when each one learns.

Every `eval_every` steps, on a fixed validation set (seed 777, 512 examples), per member:
    clean        accuracy of the answer
    paired       correct on the clean input AND follows an edit of the queried value
    query_new    correct after the query asks for a different stored key
    in_context   prediction is one of the stored values (reads the records, maybe query-blind)
first_pass[e] = first evaluation step with clean>=.95, paired>=.90, query_new>=.95.

Every `attn_every` steps it also logs where attention goes (attention_profile).

Starting points (mutually exclusive in practice):
    default       fresh init from member seeds seed0 .. seed0+E-1
    --init_from   copy parameters (not optimizer) from a checkpoint: warm start / transfer
    --resume      continue a run exactly: parameters, Adam moments, data RNG, step counter
    --reinit l0   after loading, reset the listed parameter groups to the members' ORIGINAL
                  initial values (same seeds), e.g. to ask which layer holds a "trap"
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time

import torch

from ensemble import Ensemble, EnsembleAdamW, ModelCfg, init_params
from task import TaskCfg, make_batch, relevant_edit, query_edit, irrelevant_edit, VOCAB

HERE = Path(__file__).resolve().parent


def answer_nll(logits_last, ans):
    """-log p(answer) from the final-position logits."""
    return torch.logsumexp(logits_last, -1) - logits_last.gather(-1, ans[..., None]).squeeze(-1)


def slice_batch(b, s, e):
    return {k: (v[s:e] if torch.is_tensor(v) and v.ndim > 0 else v) for k, v in b.items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, task, val, window=None, chunk=128, thorough=False):
    """val: one batch from make_batch, shared by all members.  Returns {metric: [E floats]}."""
    E, n = model.E, len(val["x"])
    names = ["clean", "relevant_new", "paired", "query_new", "in_context", "nll",
             "irrelevant_ok", "irrelevant_same", "query_changed"]
    acc = {k: torch.zeros(E, device=val["x"].device) for k in names}
    n_answer_differs = 0
    predict = lambda x: model(x[None].expand(E, -1, -1), window=window)[:, :, -1]   # final-position logits [E,m,V]
    for s in range(0, n, chunk):
        b = slice_batch(val, s, s + chunk)
        logits = predict(b["x"])
        pred = logits.argmax(-1)                                         # [E,m]
        ok = pred == b["answer"]
        x_rel, new_rel = relevant_edit(b)                                # change the queried value
        pred_rel = predict(x_rel).argmax(-1)
        x_q, new_q = query_edit(b)                                       # ask for another key
        pred_q = predict(x_q).argmax(-1)
        acc["clean"] += ok.float().sum(1)
        acc["relevant_new"] += (pred_rel == new_rel).float().sum(1)
        acc["paired"] += (ok & (pred_rel == new_rel)).float().sum(1)
        acc["query_new"] += (pred_q == new_q).float().sum(1)
        acc["in_context"] += (pred[..., None] == b["values"][None]).any(-1).float().sum(1)
        acc["nll"] += answer_nll(logits, b["answer"][None].expand(E, -1)).sum(1)
        differs = new_q != b["answer"]                                   # the new query has a different answer
        acc["query_changed"] += ((pred_q != pred) & differs).float().sum(1)
        n_answer_differs += differs.sum().item()
        if thorough:
            pred_irr = predict(irrelevant_edit(b)).argmax(-1)
            acc["irrelevant_ok"] += (pred_irr == b["answer"]).float().sum(1)
            acc["irrelevant_same"] += (pred_irr == pred).float().sum(1)
    out = {k: (v / n).tolist() for k, v in acc.items() if k != "query_changed"}
    out["query_change_when_answer_differs"] = (acc["query_changed"] / max(n_answer_differs, 1)).tolist()
    if not thorough:
        out.pop("irrelevant_ok"), out.pop("irrelevant_same")
    out["pass"] = [c >= .95 and p >= .90 and q >= .95
                   for c, p, q in zip(out["clean"], out["paired"], out["query_new"])]
    return out


@torch.no_grad()
def attention_profile(model, val, n=64, window=None):
    """Where attention goes, per member.  One dict per layer for the FINAL position:
         correct_maxhead            mass on the queried value (best head)
         other_values_of_that_head  mean mass on each of the 3 other stored values, same head
         all_values_maxhead         mass on all 4 stored values (best head)
         confusable_noise_mean / noise_mean   mass on noise tokens (mean over heads)
    plus "l0_value_to_key" [E][H]: layer-0 attention from each stored VALUE to the token right
    before it (its KEY) -- the "binding" step.  A 2-layer solution needs some head near 1 here."""
    b = slice_batch(val, 0, n)
    E = model.E
    _, maps = model(b["x"][None].expand(E, -1, -1), attn=True, window=window)
    prof = []
    for w in maps:                                   # [E,n,H,T,T]
        last = w[:, :, :, -1, :]                     # attention of the final position [E,n,H,T]
        H = last.shape[2]
        corr = last.gather(-1, b["src"].view(1, n, 1, 1).expand(E, n, H, 1)).squeeze(-1)        # [E,n,H]
        R = b["vpos"].shape[1]
        allv = last.gather(-1, b["vpos"].view(1, n, 1, R).expand(E, n, H, R)).sum(-1)
        conf = (last * b["confusable"].view(1, n, 1, -1)).sum(-1)
        noise = (last * b["noise_mask"].view(1, n, 1, -1)).sum(-1)
        best = corr.mean(1).argmax(-1, keepdim=True)                     # best head per member
        prof.append({
            "correct_maxhead": corr.mean(1).max(-1).values.tolist(),
            "other_values_of_that_head": ((allv - corr).mean(1).gather(-1, best).squeeze(-1) / (R - 1)).tolist(),
            "all_values_maxhead": allv.mean(1).max(-1).values.tolist(),
            "confusable_noise_mean": conf.mean((1, 2)).tolist(),
            "noise_mean": noise.mean((1, 2)).tolist(),
        })
    a0 = maps[0]                                                         # [E,n,H,T,T]
    rows = torch.arange(n, device=a0.device)[:, None]
    from_values = a0.permute(1, 3, 0, 2, 4)[rows, b["vpos"]]             # [n,R,E,H,T]
    to_key = from_values.gather(-1, (b["vpos"] - 1)[:, :, None, None, None]
                                .expand(-1, -1, E, a0.shape[2], 1)).squeeze(-1)                # [n,R,E,H]
    prof[0]["l0_value_to_key"] = to_key.mean((0, 1)).tolist()
    return prof


# ----------------------------------------------------------------------------- setup
def build(args, device):
    task = TaskCfg(N=args.N, post_lo=args.post_lo, post_hi=args.post_hi, noise=args.noise,
                   n_confusable=args.n_confusable, layout=args.layout, n_records=args.n_records)
    slopes = [float(s) for s in args.slopes.split(",")] if args.slopes else [0.0] * args.heads
    mcfg = ModelCfg(vocab=VOCAB, d=args.d, heads=args.heads, layers=args.layers, pe=args.pe,
                    slopes=slopes, max_len=max(1100, task.T + 1), rope_theta=args.rope_theta,
                    fixed_head_offset=args.fixed_head_offset)
    seeds = list(range(args.seed0, args.seed0 + args.E))
    return task, mcfg, Ensemble(mcfg, seeds, device)


def save(path, model, opt, g, step, meta):
    torch.save({"params": {k: v.detach() for k, v in model.params.items()}, "opt": opt.state(),
                "rng": g.get_state(), "step": step, "meta": meta}, path)


def load_start(args, model, opt, g, device):
    """Apply --init_from / --resume / --reinit.  Returns the starting step."""
    start = 0
    if args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        for k, v in st["params"].items():
            assert v.shape == model.params[k].shape, (k, v.shape, model.params[k].shape)
            model.params[k].data.copy_(v)
    if args.resume:
        st = torch.load(args.resume, map_location=device, weights_only=False)
        for k, v in st["params"].items():
            model.params[k].data.copy_(v)
        opt.load(st["opt"])
        g.set_state(st["rng"].cpu())
        start = st["step"]
    if args.reinit:
        fresh = init_params(model.cfg, model.seeds)                      # the members' original init
        groups = args.reinit.split(",")
        hit = [k for k in fresh if any(k == gr or k.startswith(gr + ".") for gr in groups)]
        assert hit, f"--reinit {args.reinit} matched no parameter"
        for k in hit:
            model.P(k).data.copy_(fresh[k].to(device))
        print("re-initialized:", " ".join(hit), flush=True)
    return start


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    # task
    p.add_argument("--N", type=int, default=64, help="number of noise tokens (context length T = N + 15)")
    p.add_argument("--post_lo", type=int, default=64, help="min noise tokens between the records and the query")
    p.add_argument("--post_hi", type=int, default=64, help="max noise tokens between the records and the query")
    p.add_argument("--noise", default="value", help="value (confusable) | filler (inert) | mix")
    p.add_argument("--n_confusable", type=int, default=0, help="for --noise mix")
    p.add_argument("--layout", default="qk", help="qk | v1direct | v1delayed")
    p.add_argument("--n_records", type=int, default=4, help="number of STORE K V records (<= 8)")
    # model
    p.add_argument("--pe", default="rope", help="rope | none (NoPE) | learned")
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--slopes", default="")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--window", type=int, default=None, help="local attention window (structural blindness)")
    p.add_argument("--fixed_head_offset", type=int, default=0,
                   help="diagnostic: layer-0 head 0 attends only to t-k (1 = previous token)")
    # training
    p.add_argument("--E", type=int, default=16, help="number of independent seeds")
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--data_seed", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lm_weight", type=float, default=1.0, help="weight of next-token loss on all positions")
    p.add_argument("--answer_weight", type=float, default=2.0, help="extra weight on the answer position")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--attn_every", type=int, default=500)
    p.add_argument("--val_n", type=int, default=512)
    p.add_argument("--save_at", default="", help="comma list of steps to checkpoint")
    p.add_argument("--stop_after_all_pass", type=int, default=-1,
                   help="stop this many steps after every member passed (-1: never)")
    p.add_argument("--init_from", default=None, help="warm start parameters (not optimizer) from a checkpoint")
    p.add_argument("--resume", default=None, help="continue a run: params, optimizer and data RNG")
    p.add_argument("--reinit", default="", help="comma list of param groups to reset to the original init, "
                                                "e.g. l0 | l1 | emb,head")
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(HERE / "runs"))
    args = p.parse_args()

    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True          # same numerics for every condition
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "log.jsonl").exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    task, mcfg, model = build(args, device)
    opt = EnsembleAdamW(model.params.values(), lr=args.lr)
    g = torch.Generator(device=device).manual_seed(args.data_seed)       # training data stream
    start = load_start(args, model, opt, g, device)
    val = make_batch(args.val_n, torch.Generator(device=device).manual_seed(777), device, task)
    meta = {"args": vars(args), "task": asdict(task), "model": asdict(mcfg), "T": task.T,
            "params_per_member": sum(v[0].numel() for v in model.params.values()),
            "seeds": model.seeds, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if "cuda" in device else "cpu",
            "source_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16]
                              for f in ("ensemble.py", "task.py", "train.py")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    E = model.E
    save_at = {int(s) for s in args.save_at.split(",") if s}
    first_pass, all_pass_step = [None] * E, None
    run_loss, run_ans = torch.zeros(E, device=device), torch.zeros(E, device=device)
    t0 = time.time()
    log = (out / "log.jsonl").open("w", buffering=1)

    def do_eval(step):
        nonlocal all_pass_step
        ev = evaluate(model, task, val, window=args.window)
        for e in range(E):
            if first_pass[e] is None and ev["pass"][e]:
                first_pass[e] = step
        row = {"step": step, "time": round(time.time() - t0, 1), "val": ev, "first_pass": first_pass}
        if step > start:
            row["train_loss"] = (run_loss / args.eval_every).tolist()
            row["train_answer_nll"] = (run_ans / args.eval_every).tolist()
        if args.attn_every and step % args.attn_every == 0:
            row["attn"] = attention_profile(model, val, window=args.window)
        log.write(json.dumps(row) + "\n")
        npass = sum(f is not None for f in first_pass)
        print(f"[{args.name}] step {step} t={row['time']}s clean={sum(ev['clean'])/E:.3f} "
              f"ctx={sum(ev['in_context'])/E:.3f} passed={npass}/{E}", flush=True)
        if npass == E and all_pass_step is None:
            all_pass_step = step

    do_eval(start)
    use_compile = args.compile and args.window is None and not any(mcfg.slopes)
    fwd = torch.compile(model) if use_compile else model
    for step in range(start + 1, args.steps + 1):
        b = make_batch(E * args.batch, g, device, task)                  # member e gets rows e*B .. (e+1)*B-1
        x = b["x"].view(E, args.batch, -1)
        y = b["y"].view(E, args.batch, -1)                               # next token at every position
        opt.lr = args.lr * min(1.0, step / 50)                           # 50-step linear warmup, then constant
        for prm in opt.params:
            prm.grad = None
        logits = fwd(x, window=args.window)
        nll = torch.logsumexp(logits, -1) - logits.gather(-1, y[..., None]).squeeze(-1)    # [E,B,T]
        # loss = mean next-token NLL over all positions + answer_weight * NLL of the answer
        per_member = args.lm_weight * nll.mean((1, 2)) + args.answer_weight * nll[:, :, -1].mean(1)
        per_member.sum().backward()                                      # sum: members get separate gradients
        opt.clip(1.0)
        opt.step()
        with torch.no_grad():
            run_loss += per_member.detach()
            run_ans += nll[:, :, -1].mean(1).detach()
        if step in save_at:
            save(out / f"step_{step}.pt", model, opt, g, step, meta)
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)
            run_loss.zero_(), run_ans.zero_()
            if all_pass_step is not None and args.stop_after_all_pass >= 0 \
                    and step >= all_pass_step + args.stop_after_all_pass:
                break

    final = evaluate(model, task, val, window=args.window, thorough=True)
    save(out / "final.pt", model, opt, g, step, meta)
    summary = {"name": args.name, "steps_trained": step, "first_pass": first_pass,
               "n_pass": sum(f is not None for f in first_pass), "E": E, "final_val": final,
               "wall_seconds": round(time.time() - t0, 1), "meta": meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("name", "steps_trained", "first_pass", "n_pass")}), flush=True)


if __name__ == "__main__":
    main()
