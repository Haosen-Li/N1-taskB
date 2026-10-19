"""Figures and tables for the pretrained-model experiments (R1, R1b, R1c, R2, R3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)


def rows(name):
    d = {}
    f = HERE / "results" / name
    if not f.exists():
        return d
    for line in open(f):
        r = json.loads(line)
        r.pop("rows", None)
        d[(r["model"].split("/")[-1], r.get("revision"), r.get("P"), r["n"], r["D"])] = r   # last record wins
    return d


def main():
    r1, r1b, r1c = rows("R1_pythia_final.jsonl"), rows("R1b_pythia_fixedP200.jsonl"), rows("R1c_pythia_fixedD30.jsonl")
    r2, r3 = rows("R2_pythia_dynamics.jsonl"), rows("R3_qwen3_4b_L16k.jsonl")
    sizes = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
    F = "step143000"
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    # (a) final checkpoints: selection vs number of listed pairs
    a = ax[0, 0]
    for i, s in enumerate(sizes):
        for D, ls in [(30, "-"), (1400, "--")]:
            xs = [n for n in [1, 2, 4, 8, 16, 32] if (s, F, None, n, D) in r1]
            if xs:
                a.plot(xs, [r1[(s, F, None, n, D)]["select"] for n in xs], ls, c=f"C{i}", marker="o", ms=3,
                       label=s if D == 30 else None)
    a.plot([1, 2, 4, 8, 16, 32], [1, .5, .25, .125, 1 / 16, 1 / 32], ":", c="gray", label="chance 1/n")
    a.set_xscale("log", base=2); a.set_xlabel("pairs in the list n"); a.set_ylabel("select (argmax among listed values)")
    a.set_title("R1 Pythia final, L=1800 fixed (solid D=30, dashed D=1400)"); a.legend(fontsize=7)

    # (b) distance vs list position
    a = ax[0, 1]
    for s, c in [("pythia-410m", "C2"), ("pythia-1b", "C3"), ("pythia-1.4b", "C4")]:
        for n, mk in [(4, "o"), (16, "s")]:
            Ds = [D for D in [30, 200, 600, 1400] if (s, F, 200, n, D) in r1b]
            a.plot(Ds, [r1b[(s, F, 200, n, D)]["select"] for D in Ds], "-", c=c, marker=mk, ms=4,
                   label=f"{s} n={n}: vary distance D (P=200)")
            Ps = [P for P in [50, 400, 1000, 1600] if (s, F, P, n, 30) in r1c]
            a.plot(Ps, [r1c[(s, F, P, n, 30)]["select"] for P in Ps], ":", c=c, marker=mk, ms=4, mfc="none",
                   label=f"{s} n={n}: vary list position P (D=30)")
    a.set_xlabel("D (solid) or pre-filler P (dotted), tokens"); a.set_ylabel("select")
    a.set_title("R1b/R1c: distance, not list position"); a.legend(fontsize=6)

    # (c) Qwen3-4B at 16k tokens
    a = ax[0, 2]
    for D, c in [(30, "C0"), (1000, "C1"), (4000, "C2"), (12000, "C3")]:
        ns = [n for n in [4, 16, 64, 256, 512, 1024] if ("Qwen3-4B-Instruct-2507", None, None, n, D) in r3]
        a.plot(ns, [r3[("Qwen3-4B-Instruct-2507", None, None, n, D)]["top1"] for n in ns], marker="o", c=c,
               label=f"D={D}")
    a.set_xscale("log", base=2); a.set_ylim(0, 1.05); a.set_xlabel("pairs in the list n"); a.set_ylabel("top-1 accuracy")
    a.set_title("R3 Qwen3-4B-Instruct, L=16000 fixed (40/cell)"); a.legend(fontsize=8)

    # (d-f) training dynamics
    steps = [0, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000, 16000, 33000, 66000, 143000]
    for a, s in zip(ax[1], ["pythia-160m", "pythia-410m", "pythia-70m"]):
        for (n, D), c in [((4, 30), "C0"), ((4, 1400), "C1"), ((16, 30), "C2"), ((16, 1400), "C3")]:
            xs = [t for t in steps if (s, f"step{t}", None, n, D) in r2]
            a.plot([max(t, 64) for t in xs], [r2[(s, f"step{t}", None, n, D)]["select"] for t in xs], "-", c=c,
                   marker="o", ms=3, label=f"select n={n} D={D}")
            a.plot([max(t, 64) for t in xs], [r2[(s, f"step{t}", None, n, D)]["p_listed"] for t in xs], ":", c=c,
                   label=f"p(listed) n={n} D={D}")
            a.axhline(1 / n, c=c, lw=.5, alpha=.5)
        a.set_xscale("log"); a.set_ylim(0, 1.02); a.set_xlabel("Pythia training step (step 0 drawn at 64)")
        a.set_title(f"R2 {s}: retrieval (dotted) vs selection (solid)"); a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "pretrained.png", dpi=130)

    lines = ["## R1 Pythia 最终 checkpoint，固定 L=1800（每格 100 条）：select / top1", "",
             "| 模型 | n | D=30 | D=200 | D=600 | D=1400 |", "|---|---:|---|---|---|---|"]
    for s in sizes:
        for n in [1, 2, 4, 8, 16, 32]:
            cells = [r1.get((s, F, None, n, D)) for D in [30, 200, 600, 1400]]
            if any(cells):
                lines.append(f"| {s} | {n} | " + " | ".join(f"{c['select']:.2f} / {c['top1']:.2f}" if c else "-"
                                                          for c in cells) + " |")
    lines += ["", "## R1b（前置文本 P=200 固定，D 变）与 R1c（D=30 固定，P 变）：select", "",
              "| 模型 | n | D=30,200,600,1400 (P=200) | P=50,400,1000,1600 (D=30) |", "|---|---:|---|---|"]
    for s in sizes[1:]:
        for n in [4, 16]:
            b = " ".join(f"{r1b[(s, F, 200, n, D)]['select']:.2f}" for D in [30, 200, 600, 1400] if (s, F, 200, n, D) in r1b)
            c = " ".join(f"{r1c[(s, F, P, n, 30)]['select']:.2f}" for P in [50, 400, 1000, 1600] if (s, F, P, n, 30) in r1c)
            lines.append(f"| {s} | {n} | {b} | {c} |")
    lines += ["", "## R3 Qwen3-4B-Instruct，固定 L=16000（每格 40 条）：top1", "",
              "| n | D=30 | D=1000 | D=4000 | D=12000 |", "|---:|---|---|---|---|"]
    for n in [4, 16, 64, 256, 512, 1024]:
        lines.append(f"| {n} | " + " | ".join(f"{r3[('Qwen3-4B-Instruct-2507', None, None, n, D)]['top1']:.2f}"
                                              if ('Qwen3-4B-Instruct-2507', None, None, n, D) in r3 else "超长"
                                              for D in [30, 1000, 4000, 12000]) + " |")
    lines += ["", "## R2 Pythia 训练动态（每格 100 条）：p_listed / select", ""]
    for s in ["pythia-70m", "pythia-160m", "pythia-410m"]:
        lines += [f"**{s}**", "", "| step | n=4 D=30 | n=4 D=1400 | n=16 D=30 | n=16 D=1400 |", "|---:|---|---|---|---|"]
        for t in steps:
            cells = [r2.get((s, f"step{t}", None, n, D)) for n, D in [(4, 30), (4, 1400), (16, 30), (16, 1400)]]
            lines.append(f"| {t} | " + " | ".join(f"{c['p_listed']:.2f} / {c['select']:.2f}" if c else "-" for c in cells) + " |")
        lines.append("")
    (HERE / "results" / "summary_pretrained.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
