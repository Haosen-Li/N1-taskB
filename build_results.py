"""Concatenate generated tables into RESULTS.md.  Run first (in synth/): binding_table.py,
routes.py, figures.py; and pretrained/figures.py.  Nothing here is typed by hand."""
from pathlib import Path
import sys

H = Path(__file__).resolve().parent
R = H / "synth/results"
sys.path.insert(0, str(H / "synth"))
from analyze import load, paired_pass_test

answer_only = load("X6_N192_far_answeronly")["t_pass"] + load("X7b_rep_N192_far_answeronly")["t_pass"]
with_lm = load("S1_N192_far")["t_pass"] + load("X7b_rep_N192_far")["t_pass"]
pooled = paired_pass_test(answer_only, with_lm)
parts = [
    "# Task B：全部结果表（自动生成，勿手改）",
    "",
    "## 合成实验：每个条件的学会时间（未通过 = 在上限处删失）",
    "",
    "t_pass：首次同时满足 clean≥95%、相关值反事实≥90%、换查询≥95% 的步数（每 100 步评估一次，验证集 512 条）。"
    "t_ctx：预测落在存储值之内的比例首次≥90% 的步数。S4_* 合并了 F4 的续训（12k→24k）。warm-start、F2、X3 的步数从迁移时刻起算。"
    "“上限步数”是该条件实际训练到的步数：上限一般为 12k（S4 为 24k；warm-start、F1、F2、X3 为 3k）；"
    "部分运行在全部 seed 首次达标后提前停止，具体配置见 synth/runs/*/meta.json。"
    "条件名的含义见 [决策日志](docs/DECISIONS.md)（S/F 为 v2，X1–X8 为 v3）。"
    "曾达标与最终达标分别报告；首次达标不保证持续达标。探索性 p 值未作多重检验校正。",
    "",
    (R / "summary_synth.md").read_text(),
    "## 主结论的配对通过率检验",
    "",
    f"N192 far 从头训练，两组独立 seed 合并：只训练答案 {sum(t is not None for t in answer_only)}/32，"
    f"带 LM 损失 {sum(t is not None for t in with_lm)}/32；仅前者通过 {pooled['only_a']} 对，"
    f"仅后者通过 {pooled['only_b']} 对。精确 McNemar 检验 p={pooled['p']:.6f}。",
    "",
    "## 合成实验：历史曾达标成员的路线诊断",
    "",
    "以下为最终 checkpoint 的消融，但成员按历史 t_pass 分组；包括最终回落的成员，不能视作稳定通过模型的机制统计。"
    "固定判定规则见 synth/routes.py。",
    "",
    (R / "routes.md").read_text(),
    "## 合成实验：键身份探针与前一 token 注意力",
    "",
    "原始诊断保留历史分组。探针可解码性不等于网络实际使用；低探针分数也不能排除其他编码或成功路线。",
    "",
    (R / "binding_table.md").read_text(),
    "# 真实模型",
    "",
    "`select`：在列表中出现的值 token 里取 argmax 是否正确（机会水平 1/n）。`top1`：全词表 argmax 正确。"
    "`p_listed`：列表中值 token 的总概率（n>64 时值重复，存档中的 p_listed 作废，只看 top1）。"
    "每格 100 条（Qwen 40 条），二项 95% 区间约 ±0.1（Qwen 约 ±0.15）。",
    "",
    (H / "pretrained/results/summary_pretrained.md").read_text(),
]
(H / "RESULTS.md").write_text("\n".join(parts))
print("wrote RESULTS.md")
