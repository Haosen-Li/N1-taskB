"""In-context key-value retrieval in pretrained LMs, with distance and competitor
count varied independently at a FIXED total context length.

    [natural-text pre-filler] Here is a list of secret codes.
    The code for 4821 is river.            <- n lines (n-1 competitors)
    ...
    [natural-text post-filler of D tokens]  <- dependency distance
    The code for 4821 is                   -> " river"

Total length is identical for every (n, D) cell, so "longer distance" never
means "longer input".  Error decomposition at the last position:
  p_listed     probability on the n listed values (retrieved *something* from the list)
  p_unlisted   probability on the 64-n value words NOT in the list (guessing value-like words)
  select_acc   argmax restricted to the n listed values is correct (chance 1/n):
               query-conditioned selection among what was retrieved
  top1         full-vocabulary argmax is the correct value
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VALUES = ['apple', 'river', 'table', 'window', 'garden', 'pencil', 'mirror', 'candle', 'bottle', 'forest',
          'island', 'rabbit', 'silver', 'violin', 'hammer', 'jacket', 'ladder', 'lemon', 'market', 'needle',
          'orange', 'pepper', 'rocket', 'saddle', 'tiger', 'tunnel', 'wallet', 'anchor', 'basket', 'bridge',
          'camera', 'castle', 'cotton', 'desert', 'dragon', 'engine', 'feather', 'finger', 'flower', 'guitar',
          'helmet', 'magnet', 'monkey', 'museum', 'ocean', 'planet', 'pillow', 'pirate', 'puzzle', 'robot',
          'salmon', 'shadow', 'spider', 'spoon', 'statue', 'sugar', 'ticket', 'tomato', 'turtle', 'velvet',
          'wizard', 'bucket', 'cherry', 'coffee']
HERE = Path(__file__).resolve().parent


def filler_tokens(tok, cache_dir, wiki_path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = tok.name_or_path.replace("/", "_")[-60:]
    path = Path(cache_dir) / f"wiki_ids_{name}.pt"
    if path.exists():
        return torch.load(path)
    lines = [json.loads(l)["text"] for l in open(wiki_path)]
    text = "".join(l for l in lines if not l.strip().startswith("="))
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.int32)
    torch.save(ids, path)
    return ids


def build(tok, rng, n, D, L, wiki, P=None):
    keys = rng.sample(range(1000, 10000), n)
    # distinct values while possible; beyond 64 keys values must repeat (keys stay unique)
    vals = rng.sample(VALUES, n) if n <= len(VALUES) else [rng.choice(VALUES) for _ in range(n)]
    q = rng.randrange(n)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    header = enc("\n\nHere is a list of secret codes.\n")
    lines = [enc(f"The code for {k} is {v}.\n") for k, v in zip(keys, vals)]
    query = enc(f"\n\nThe code for {keys[q]} is")
    val_ids = [enc(" " + v)[0] for v in vals]
    body = sum(lines, [])
    if P is None:     # fixed total length L: the pre-filler absorbs whatever D and n do not use
        pre_len = L - len(header) - len(body) - D - len(query)
    else:             # fixed pre-filler P: total length grows with D and n
        pre_len = P
    if pre_len < 0:
        return None
    s1 = rng.randrange(len(wiki) - pre_len - D - 1)
    s2 = rng.randrange(len(wiki) - D - 1)
    pre, post = wiki[s1:s1 + pre_len].tolist(), wiki[s2:s2 + D].tolist()
    ids = pre + header + body + post + query
    L = len(ids)
    # position of the queried value token inside the list
    start = len(pre) + len(header) + sum(len(x) for x in lines[:q])
    vpos = start + lines[q].index(val_ids[q])
    return dict(ids=ids, target=val_ids[q], listed=val_ids, q=q, distance=L - 1 - vpos, vpos=vpos, L=L)


@torch.no_grad()
def score(model, examples, all_value_ids, device, bs):
    rows = []
    if len({len(e["ids"]) for e in examples}) > 1:
        bs = 1        # variable lengths (fixed-P mode): no padding, one sequence at a time
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        ids = torch.tensor([e["ids"] for e in chunk], device=device)
        out = model(input_ids=ids, logits_to_keep=1) if "qwen" in model.config.model_type else model(input_ids=ids)
        lp = out.logits[:, -1].float().log_softmax(-1)
        for e, l in zip(chunk, lp):
            p = l.exp()
            listed = torch.tensor(sorted(set(e["listed"])), device=device)   # unique tokens (values repeat when n>64)
            unlisted = torch.tensor([v for v in all_value_ids if v not in e["listed"]], device=device, dtype=torch.long)
            rows.append(dict(
                top1=int(l.argmax().item() == e["target"]),
                top1_other_listed=int(l.argmax().item() in e["listed"] and l.argmax().item() != e["target"]),
                select=int(listed[l[listed].argmax()].item() == e["target"]),
                logp_target=l[e["target"]].item(),
                p_listed=p[listed].sum().item(),
                p_unlisted=p[unlisted].sum().item() if len(unlisted) else 0.0,
                distance=e["distance"], L=e["L"]))
    return rows


def load(model_id, revision, device):
    kw = dict(torch_dtype=torch.float16 if "pythia" in model_id else torch.bfloat16)
    if "pythia" in model_id:
        kw["use_safetensors"] = False     # only the fp16 pytorch_model.bin was downloaded
    if revision:
        kw["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_id, **({"revision": revision} if revision else {}))
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", **kw).to(device).eval()
    return tok, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revisions", default="")
    p.add_argument("--L", type=int, default=1800)
    p.add_argument("--P", default="", help="comma list of fixed pre-filler lengths (fixed-P mode; L ignored)")
    p.add_argument("--ns", default="1,2,4,8,16,32")
    p.add_argument("--Ds", default="30,200,600,1400")
    p.add_argument("--per_cell", type=int, default=100)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--wiki", type=Path, default=HERE / "data" / "wikitext103_test.jsonl")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    revs = args.revisions.split(",") if args.revisions else [None]
    ns, Ds = [int(x) for x in args.ns.split(",")], [int(x) for x in args.Ds.split(",")]
    with out.open("a") as f:
        for rev in revs:
            t0 = time.time()
            tok, model = load(args.model, rev, args.device)
            wiki = filler_tokens(tok, HERE / "cache", args.wiki)
            all_vals = [tok(" " + v, add_special_tokens=False)["input_ids"][0] for v in VALUES]
            Ps = [int(x) for x in args.P.split(",")] if args.P else [None]
            for n, D, P in [(n, D, P) for n in ns for D in Ds for P in Ps]:
                if True:
                    rng = random.Random(f"{args.seed}-{n}-{D}" + ("" if P is None else f"-P{P}"))
                    ex = [build(tok, rng, n, D, args.L, wiki, P) for _ in range(args.per_cell)]
                    if any(e is None for e in ex):
                        continue
                    tc = time.time()
                    rows = score(model, ex, all_vals, args.device, args.bs)
                    agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
                    rec = dict(model=args.model, revision=rev, mode="fixedL" if P is None else "fixedP", P=P, n=n, D=D,
                               count=len(rows), **agg,
                               rows=rows)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{args.model}@{rev} P={P} n={n:3d} D={D:5d} top1={agg['top1']:.2f} select={agg['select']:.2f} "
                          f"p_listed={agg['p_listed']:.2f} p_unlisted={agg['p_unlisted']:.3f} ({time.time()-tc:.0f}s)", flush=True)
            print(f"done {args.model}@{rev} in {time.time()-t0:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
