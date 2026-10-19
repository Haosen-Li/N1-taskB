"""Print a few examples of a task condition in readable form.

    python show_task.py --N 6 --post_lo 2 --post_hi 2
    python show_task.py --N 192 --post_lo 184 --post_hi 192 --noise filler --n 1

k* = keys, v* = values, _* = inert filler.  [brackets] mark the queried record's value.
"""
import argparse

import torch

from task import TaskCfg, make_batch, decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--post_lo", type=int, default=2)
    p.add_argument("--post_hi", type=int, default=2)
    p.add_argument("--noise", default="value")
    p.add_argument("--n_confusable", type=int, default=0)
    p.add_argument("--layout", default="qk")
    p.add_argument("--n_records", type=int, default=4)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    c = TaskCfg(N=a.N, post_lo=a.post_lo, post_hi=a.post_hi, noise=a.noise, n_confusable=a.n_confusable,
                layout=a.layout, n_records=a.n_records)
    b = make_batch(a.n, torch.Generator().manual_seed(a.seed), "cpu", c)
    print(f"T = {c.T} tokens; noise = {c.noise}; records start after {c.N - c.post_hi}..{c.N - c.post_lo} noise tokens")
    for i in range(a.n):
        toks = decode(b["x"][i]).split()
        s = int(b["src"][i])
        toks[s] = f"[{toks[s]}]"
        print(f"\n#{i}  distance {int(b['distance'][i])} (from the final position back to the queried value)")
        print("   " + " ".join(toks) + f"   ->  {decode(b['answer'][i:i+1])}")


if __name__ == "__main__":
    main()
