"""Summary tables (results/summary_synth.{json,md}) and figures (../figures/*.png).

    figures/synth_survival.png   v2: time-to-learn curves for S1-S5, F2
    figures/synth_zeroshot.png   v2: a model solved at N=16, evaluated unchanged on longer contexts
    figures/synth_v3.png         v3: which component is the bottleneck, and what causes the far penalty
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load, describe, paired, sign_test_p

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

GROUPS = {
    "S1 distance x competitors": ["S1_N16_near", "S1_N16_far", "S1_N64_near", "S1_N64_far", "S1_N192_near", "S1_N192_far"],
    "S2 confusability (N=192 far)": ["S1_N192_far", "S2_N192_far_filler", "S2_N192_far_mix16"],
    "S3 warm start into N=192 far": ["S1_N192_far", "F1_warm_step100_to_N192_far", "F1_warm_step200_to_N192_far",
                                     "S3_warm_N16step300_to_N192_far", "S3_warm_N16final_to_N192_far"],
    "S5 depth / width (N=64 far)": ["S1_N64_far", "S5_N64_far_L4", "S5_N64_far_d128h8"],
    "S4 v1 configuration (32 seeds)": ["S4_v1direct", "S4_v1delayed"],
    "F2 reverse transfer into N=16 far": ["S1_N16_far", "F2_N192stuck_to_N16_far"],
}
V3_RUNS = ["X1_N16_far_prev1", "X1_N64_far_prev1", "X1_N192_near_prev1", "X1_N192_far_prev1", "X1_N192_far_prev2",
           "X1_N192_far_filler_prev1", "X2_N192_near_filler", "X3_N192far_reinit_l0_to_N16",
           "X3_N192far_reinit_l1_to_N16", "X3_filler_reinit_l0_to_N16", "X3_filler_reinit_l1_to_N16",
           "X3_filler_noreinit_to_N16", "X4_N16_far_nope_prev1", "X4_N192_near_nope_prev1", "X4_N192_far_nope_prev1",
           "X5_far_prev1_rope_answeronly", "X5_far_prev1_nope_answeronly", "X6_N16_far_answeronly",
           "X6_N192_near_answeronly", "X6_N192_far_answeronly", "X7a_N192_near_prev1_answeronly",
           "X7b_rep_N192_far", "X7b_rep_N192_far_answeronly", "X8_N16_far_L1", "X8_N16_far_R8",
           "X8_N192_mid", "X8_N192_mid_answeronly"]
PAIRS = [("S1_N16_near", "S1_N16_far"), ("S1_N64_near", "S1_N64_far"), ("S1_N192_near", "S1_N192_far"),
         ("S1_N16_far", "S1_N192_far"), ("S1_N16_near", "S1_N192_near"),
         ("S2_N192_far_filler", "S1_N192_far"), ("S2_N192_far_mix16", "S1_N192_far"),
         ("S2_N192_far_mix16", "S2_N192_far_filler"), ("X2_N192_near_filler", "S1_N192_near"),
         ("S3_warm_N16step300_to_N192_far", "S1_N192_far"), ("F1_warm_step100_to_N192_far", "S1_N192_far"),
         ("S5_N64_far_L4", "S1_N64_far"), ("S5_N64_far_d128h8", "S1_N64_far"),
         ("X1_N192_near_prev1", "S1_N192_near"), ("X1_N192_far_prev1", "S1_N192_far"),
         ("X1_N192_near_prev1", "X1_N192_far_prev1"), ("X1_N192_far_prev2", "X1_N192_far_prev1"),
         ("X4_N192_far_nope_prev1", "X1_N192_far_prev1"), ("X4_N192_near_nope_prev1", "X4_N192_far_nope_prev1"),
         ("X5_far_prev1_rope_answeronly", "X1_N192_far_prev1"),
         ("X5_far_prev1_nope_answeronly", "X4_N192_far_nope_prev1"),
         ("X6_N192_far_answeronly", "S1_N192_far"), ("X6_N192_near_answeronly", "S1_N192_near"),
         ("X6_N192_near_answeronly", "X6_N192_far_answeronly"), ("X6_N16_far_answeronly", "S1_N16_far"),
         ("X7a_N192_near_prev1_answeronly", "X1_N192_near_prev1"),
         ("X7b_rep_N192_far_answeronly", "X7b_rep_N192_far"),
         ("X8_N192_mid", "S1_N192_near"), ("X8_N192_mid", "S1_N192_far"),
         ("X8_N192_mid_answeronly", "X8_N192_mid"), ("X8_N16_far_R8", "S1_N16_far")]
LABEL = {
    "S1_N16_near": "N=16 near", "S1_N16_far": "N=16 far", "S1_N64_near": "N=64 near", "S1_N64_far": "N=64 far",
    "S1_N192_near": "N=192 near", "S1_N192_far": "N=192 far (scratch)",
    "S2_N192_far_filler": "all inert filler", "S2_N192_far_mix16": "16 confusable + 176 filler",
    "F1_warm_step100_to_N192_far": "warm: N=16 @100 steps", "F1_warm_step200_to_N192_far": "warm: N=16 @200 steps",
    "S3_warm_N16step300_to_N192_far": "warm: N=16 @300 (bag)", "S3_warm_N16final_to_N192_far": "warm: N=16 solved",
    "S5_N64_far_L4": "4 layers", "S5_N64_far_d128h8": "width 128, 8 heads",
    "S4_v1direct": "v1 direct", "S4_v1delayed": "v1 delayed",
    "F2_N192stuck_to_N16_far": "from N=192-far @12k",
}


def merged_v1(name):
    """S4 run (0-12k) followed by its F4 continuation (12k-24k)."""
    a = load(name)
    cont = {"S4_v1direct": "F4_v1direct_to24k", "S4_v1delayed": "F4_v1delayed_to24k"}[name]
    try:
        b = load(cont)
    except FileNotFoundError:
        return a
    tp = [x if x is not None else y for x, y in zip(a["t_pass"], b["t_pass"])]
    a = dict(a)
    a["t_pass"], a["cap"] = tp, b["cap"]
    a["val"] = {k: np.concatenate([a["val"][k], b["val"][k][1:]]) for k in a["val"]}
    a["passed"] = np.concatenate([a["passed"], b["passed"][1:]])
    a["steps"] = np.concatenate([a["steps"], b["steps"][1:]])
    return a


def get(name):
    if name in ("S4_v1direct", "S4_v1delayed"):
        return merged_v1(name)
    return load(name)


def survival(ax, r, label, **kw):
    tp = [np.inf if x is None else x - r["start"] for x in r["t_pass"]]
    cap = r["cap"] - r["start"]
    xs = np.linspace(0, cap, 400)
    ys = [np.mean([t > x for t in tp]) for x in xs]
    ax.step(xs, ys, where="post", label=f"{label} ({sum(np.isfinite(tp))}/{len(tp)})", **kw)


def main():
    runs, summary = {}, {}
    for names in list(GROUPS.values()) + [V3_RUNS]:
        for n in names:
            if n not in runs:
                try:
                    runs[n] = get(n)
                    summary[n] = describe(runs[n])
                except FileNotFoundError:
                    pass
    stuck = [e for e, t in enumerate(runs["S1_N192_far"]["t_pass"]) if t is None]
    sub = lambda r, members: {**r, "t_pass": [r["t_pass"][e] for e in members]}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, (title, names) in zip(axes.flat, GROUPS.items()):
        if title.startswith("F2"):
            title = f"F2: the {len(stuck)} seeds stuck at N=192 far, trained on N=16 far"
            survival(ax, sub(runs["S1_N16_far"], stuck), "same seeds, from scratch")
            if "F2_N192stuck_to_N16_far" in runs:
                survival(ax, sub(runs["F2_N192stuck_to_N16_far"], stuck), "same seeds, from their 12k N=192 state")
            names = []
        for n in names:
            if n in runs:
                survival(ax, runs[n], LABEL.get(n, n))
        ax.set_title(title)
        ax.set_xlabel("training steps in this condition")
        ax.set_ylabel("fraction of seeds not yet passed")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "synth_survival.png", dpi=130)

    zs = HERE / "results" / "zeroshot_S1_N16_far_final.json"
    if zs.exists():
        z = json.load(open(zs))["targets"]
        order = ["N16_far", "N16_near", "N64_near", "N64_far", "N192_near", "N192_far", "N192_far_filler",
                 "N512_near", "N512_far"]
        order = [o for o in order if o in z]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        x = np.arange(len(order))
        for k, lab in [("clean", "clean accuracy"), ("paired", "paired counterfactual"), ("query_new", "query edit")]:
            ax[0].plot(x, [np.mean(z[o][k]) for o in order], marker="o", label=lab)
        ax[0].set_xticks(x, order, rotation=30); ax[0].set_ylim(0, 1.02)
        ax[0].set_title("16 seeds solved at N=16 far, evaluated unchanged"); ax[0].legend(fontsize=8)
        a1 = [z[o]["attn"][1] for o in order]
        ax[1].plot(x, [np.mean(a["correct_maxhead"]) for a in a1], marker="o", label="correct value (max head)")
        ax[1].plot(x, [np.mean(a["other_values_of_that_head"]) for a in a1], marker="o",
                   label="each other stored value (same head)")
        ax[1].plot(x, [np.mean(a["noise_mean"]) for a in a1], marker="o", label="all noise tokens (head mean)")
        ax[1].set_xticks(x, order, rotation=30); ax[1].set_ylim(0, 1.02)
        ax[1].set_title("layer-1 attention from the final position"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "synth_zeroshot.png", dpi=130)

    pairs = {}
    for a, b in PAIRS:
        if a in runs and b in runs:
            fa, fb, tie = paired(runs[a], runs[b])
            pairs[f"{a} vs {b}"] = dict(first_faster=fa, second_faster=fb, ties_or_both_censored=tie,
                                        sign_test_p=round(sign_test_p(fa, fb), 4))
    out = dict(conditions=summary, paired_same_init=pairs)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary_synth.json").write_text(json.dumps(out, indent=1))
    fmt = lambda x: "未通过" if x is None else str(x)
    lines = ["| 条件 | seeds | 曾达标 | 最终达标 | 上限步数 | t_pass 25%/中位/75% | t_ctx 中位 | 最终 clean 均值 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for n, d in summary.items():
        lines.append(f"| {n} | {d['E']} | {d['n_pass']} | {d['n_pass_final']} | {d['cap']} | {fmt(d['q25'])} / {fmt(d['median'])} / "
                     f"{fmt(d['q75'])} | {fmt(d['ctx_median'])} | {d['final_clean_mean']:.3f} |")
    lines += ["", "同初始化配对（截断到两次运行中较短的观察时长；两者都未通过算平局）：", "",
              "| 比较 | 前者更快 | 后者更快 | 平局/都未通过 | 符号检验 p |", "|---|---:|---:|---:|---:|"]
    for k, v in pairs.items():
        lines.append(f"| {k} | {v['first_faster']} | {v['second_faster']} | {v['ties_or_both_censored']} | "
                     f"{v['sign_test_p']:.3f} |")
    (HERE / "results" / "summary_synth.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    v3_figure(runs)


def v3_figure(runs):
    """Six panels: give binding / remove RoPE / remove the LM loss / readout distance / trap / binding probe."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def curves(ax, title, items):
        for name, label, style in items:
            if name in runs:
                survival(ax, runs[name], label, ls=style[0], c=style[1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training steps")
        ax.set_ylabel("fraction of seeds not yet passed")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)

    curves(axes[0, 0], "(a) Hand the model a previous-token head (N=192)", [
        ("S1_N192_near", "near, learned", ("--", "C0")), ("X1_N192_near_prev1", "near, given", ("-", "C0")),
        ("S1_N192_far", "far, learned", ("--", "C3")), ("X1_N192_far_prev1", "far, given", ("-", "C3"))])
    curves(axes[0, 1], "(b) Same, without position encoding (NoPE)", [
        ("X1_N192_near_prev1", "near, RoPE", ("--", "C0")), ("X4_N192_near_nope_prev1", "near, NoPE", ("-", "C0")),
        ("X1_N192_far_prev1", "far, RoPE", ("--", "C3")), ("X4_N192_far_nope_prev1", "far, NoPE", ("-", "C3"))])
    curves(axes[0, 2], "(c) Train the answer only (no next-token loss)", [
        ("S1_N192_near", "near, +LM loss", ("--", "C0")), ("X6_N192_near_answeronly", "near, answer only", ("-", "C0")),
        ("S1_N192_far", "far, +LM loss", ("--", "C3")), ("X6_N192_far_answeronly", "far, answer only", ("-", "C3")),
        ("X5_far_prev1_rope_answeronly", "far, head given, answer only", ("-", "C2"))])

    ax = axes[1, 0]
    prof = {}
    for f in ("readout_distance_X5.json", "readout_distance_X6.json"):
        p = HERE / "results" / f
        if p.exists():
            d = json.load(open(p))
            bins = d["bins"]
            for ck, v in d["runs"].items():
                prof[ck] = np.median(np.array(v), 0)
    show = [("runs/X5_far_prev1_rope_early/step_300.pt", "RoPE + LM loss"),
            ("runs/X5_far_prev1_nope_early/step_300.pt", "NoPE + LM loss"),
            ("runs/X6_far_prev1_rope_answeronly_early/step_300.pt", "RoPE, answer only")]
    show = [(k, l) for k, l in show if k in prof]
    if show:
        T = 207
        width = np.array([9, 24, 64, 89, 21]) / T                      # uniform-attention baseline per bin
        x = np.arange(len(width))
        for i, (k, lab) in enumerate(show):
            ax.bar(x + (i - 1) * 0.25, prof[k], 0.25, label=lab)
        ax.plot(x, width, "k_", ms=25, mew=2, label="uniform attention")
        ax.set_xticks(x, [f"d {b}" for b in bins])
        ax.set_title("(d) N=192 far, step 300: final-position attention by distance\n(stored values at d=186-203)",
                     fontsize=11)
        ax.set_ylabel("attention mass (median over 16 seeds)")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    stuck_v = [e for e, t in enumerate(runs["S1_N192_far"]["t_pass"]) if t is None]
    stuck_f = [e for e, t in enumerate(runs["S2_N192_far_filler"]["t_pass"]) if t is None]
    arms = [("F2_N192stuck_to_N16_far", "X3_filler_noreinit_to_N16", "no reset"),
            ("X3_N192far_reinit_l1_to_N16", "X3_filler_reinit_l1_to_N16", "reset layer 1"),
            ("X3_N192far_reinit_l0_to_N16", "X3_filler_reinit_l0_to_N16", "reset layer 0"),
            ("S1_N16_far", "S1_N16_far", "fresh init")]
    xs = np.arange(len(arms))
    for j, (members, which, lab) in enumerate([(stuck_v, 0, "stuck at N192 far (value)"),
                                              (stuck_f, 1, "stuck at N192 far (filler)")]):
        frac = []
        for arm in arms:
            r = runs.get(arm[which])
            frac.append(np.mean([r["t_pass"][e] is not None and r["t_pass"][e] - r["start"] <= 3000 for e in members])
                        if r else np.nan)
        ax.bar(xs + (j - .5) * 0.35, frac, 0.35, label=f"{lab} ({len(members)} seeds)")
    ax.set_xticks(xs, [a[2] for a in arms])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("passed on N=16 far within 3000 steps")
    ax.set_title("(e) Where is the trap?  Stuck 12k weights, one layer reset to its own init", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    bt = HERE / "results" / "binding_table.json"
    if bt.exists():
        t = json.load(open(bt))
        names = [n for n in ["S1_N192_near", "S1_N192_far", "S2_N192_far_filler", "S2_N192_far_mix16",
                             "X2_N192_near_filler", "X1_N192_far_prev1"] if n in t]
        colors = {"pointer": "C2", "relay": "C1", "mixed": "C7", "stuck": "C3"}
        for i, n in enumerate(names):
            for m in t[n]:
                ax.scatter(i + np.random.RandomState(m["member"]).uniform(-.2, .2), m["key_probe"], s=18,
                           c=colors.get(m["status"], "k"))
        for k, c in colors.items():
            ax.scatter([], [], c=c, label=k)
        ax.axhline(0.125, c="k", ls=":", lw=1)
        ax.set_xticks(range(len(names)), [n.replace("_", "\n", 1) for n in names], fontsize=8)
        ax.set_ylabel("key identity decodable at the value position (chance .125)")
        ax.set_title("(f) Key probes at value positions (historical first-pass groups)", fontsize=11)
        ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "synth_v3.png", dpi=120)


if __name__ == "__main__":
    main()
