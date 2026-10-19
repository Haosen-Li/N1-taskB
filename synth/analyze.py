"""Summaries of ensemble runs: time-to-learn distributions with censoring, plus simple exact tests.

t_pass[e]: first evaluation step meeting the acceptance rule (None = never, censored at the cap).
t_ctx[e]:  first evaluation step at which >=90% of predictions are one of the 4 stored values
           (the model reads the record region, whether or not it uses the query).

Two comparisons are used throughout:
  paired(a, b)   runs that share member seeds: for each seed, which run passed first.
                 Tested with an exact sign test (ties dropped).
  paired_pass_test(a, b)   paired pass counts at a shared horizon, exact McNemar test.
With 16 seeds, only large effects can be called reliable; the report says so where it matters.

    python analyze.py S1_N192_near S1_N192_far          # one line per run
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent


def load(name, root=HERE / "runs"):
    d = root / name
    rows = [json.loads(l) for l in open(d / "log.jsonl")]
    meta = json.load(open(d / "meta.json"))
    E = len(meta["seeds"])
    steps = np.array([r["step"] for r in rows])
    val = {k: np.array([r["val"][k] for r in rows]) for k in rows[0]["val"] if k != "pass"}
    passed = np.array([r["val"]["pass"] for r in rows])
    t_pass, t_ctx = [None] * E, [None] * E
    for i, s in enumerate(steps):
        for e in range(E):
            if t_pass[e] is None and passed[i, e]:
                t_pass[e] = int(s)
            if t_ctx[e] is None and val["in_context"][i, e] >= 0.9:
                t_ctx[e] = int(s)
    return dict(name=name, E=E, steps=steps, val=val, passed=passed, t_pass=t_pass, t_ctx=t_ctx,
                cap=int(steps[-1]), start=int(steps[0]), meta=meta, rows=rows)


def quantile(times, q):
    """q-quantile of first-pass times where None = censored at the cap.  None if not reached."""
    t = sorted(x for x in times if x is not None)
    k = int(np.ceil(q * len(times)))
    return t[k - 1] if k <= len(t) else None


def relative(r, key="t_pass"):
    """Times measured from the run's own start (warm starts / continuations start at step > 0)."""
    return [None if x is None else x - r["start"] for x in r[key]]


def describe(r):
    tp = relative(r)
    return dict(name=r["name"], E=r["E"], n_pass=sum(x is not None for x in tp), cap=r["cap"] - r["start"],
                q25=quantile(tp, .25), median=quantile(tp, .5), q75=quantile(tp, .75),
                ctx_median=quantile(relative(r, "t_ctx"), .5),
                n_pass_final=int(r["passed"][-1].sum()),
                final_clean_mean=float(r["val"]["clean"][-1].mean()))


def paired(a, b, members=None):
    """Compare first-pass times at the shorter observation horizon; censored pairs tie."""
    ta, tb = relative(a), relative(b)
    horizon = min(a["cap"] - a["start"], b["cap"] - b["start"])
    idx = range(len(ta)) if members is None else members
    fa = fb = tie = 0
    for e in idx:
        x = np.inf if ta[e] is None or ta[e] > horizon else ta[e]
        y = np.inf if tb[e] is None or tb[e] > horizon else tb[e]
        fa, fb, tie = fa + (x < y), fb + (y < x), tie + (x == y)
    return fa, fb, tie


def sign_test_p(n_a, n_b):
    """Two-sided exact sign test on the untied pairs."""
    return binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0


def paired_pass_test(times_a, times_b):
    """Exact paired binary test; inputs are first-pass times censored at the same cap."""
    if len(times_a) != len(times_b):
        raise ValueError("Pass outcomes must be paired.")
    only_a = sum(a is not None and b is None for a, b in zip(times_a, times_b))
    only_b = sum(b is not None and a is None for a, b in zip(times_a, times_b))
    return dict(only_a=only_a, only_b=only_b, p=sign_test_p(only_a, only_b))


def fmt(x):
    return ">cap" if x is None else str(x)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        try:
            d = describe(load(n))
        except FileNotFoundError:
            continue
        print(f"{d['name']:30s} pass {d['n_pass']:2d}/{d['E']:2d} (cap {d['cap']:5d})  t_pass q25/med/q75 = "
              f"{fmt(d['q25'])}/{fmt(d['median'])}/{fmt(d['q75'])}   t_ctx med {fmt(d['ctx_median'])}   "
              f"final clean {d['final_clean_mean']:.3f}")
