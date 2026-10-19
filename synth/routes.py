"""Classify every member that passed, using the last-layer / final-position attention ablations
(ablate.py).  Fixed rule, written before looking at the v2 results:

  pointer : blocking the record block            -> clean < .30, and blocking the positions
            after the block keeps clean >= .80      (the answer is read directly from the values)
  relay   : the opposite                            (the answer is read from intermediate positions)
  mixed   : anything else

Reads every results/ablate_<run>_final_on_<target>.json and writes results/routes.{md,json}.
"""
import json
import re
from pathlib import Path

from analyze import load

HERE = Path(__file__).resolve().parent


def classify(a, e):
    rb, ab = a["no_record_block"]["clean"][e], a["no_after_block"]["clean"][e]
    if rb < .30 and ab >= .80:
        return "pointer"
    if ab < .30 and rb >= .80:
        return "relay"
    return "mixed"


def main():
    table, per_member = [], {}
    for f in sorted((HERE / "results").glob("ablate_*_final_on_*.json")):
        run = re.match(r"ablate_(.+)_final_on_(.+)\.json", f.name).group(1)
        if not (HERE / "runs" / run).exists():
            continue
        a = json.load(open(f))
        r = load(run)
        passed = [e for e, t in enumerate(r["t_pass"]) if t is not None]
        labels = {e: classify(a, e) for e in passed}
        per_member[run] = {"t_pass": r["t_pass"], "route": labels}
        c = lambda k: sum(v == k for v in labels.values())
        table.append((run, len(passed), c("pointer"), c("relay"), c("mixed")))
    lines = ["| 条件 | 学会 | 指针 | 中继 | 混合 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} |" for t in table]
    (HERE / "results" / "routes.md").write_text("\n".join(lines) + "\n")
    (HERE / "results" / "routes.json").write_text(json.dumps(per_member, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
