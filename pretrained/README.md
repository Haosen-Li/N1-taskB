# 补充：预训练模型的键值检索

Pythia 与 Qwen 的评测只用于现象对照，不用于推断小模型中的因果机制。原始逐样本结果在 [results](results/)，汇总见 [summary_pretrained.md](results/summary_pretrained.md)。

- Pythia：观察训练过程中“概率集中到列表值”和“按查询选择”是否先后出现；分别固定总长、列表前文本长度、依赖距离。
- Qwen3-4B：在 16k token 内测试较多键值记录。
- n>64 时值会重复，历史结果中的 `p_listed` 有重复计数问题，仅使用这些条件的 `top1`。当前评估代码已对值 token 去重。

## 重建已有图表

在仓库根目录运行，只需主 requirements：

```bash
python pretrained/figures.py
```

## 重新评测（可选，需要下载模型和数据）

```bash
python -m pip install -r pretrained/requirements.txt
python pretrained/download.py

python pretrained/kv_eval.py --model EleutherAI/pythia-160m \
  --revisions step143000 --L 1800 --ns 4,16 --Ds 30,200,600,1400 \
  --per_cell 100 --device cuda:0 --out artifacts/pythia.jsonl

python pretrained/kv_eval.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --L 16000 --ns 4,16,64 --Ds 30,1000,4000,12000 \
  --per_cell 40 --device cuda:0 --out artifacts/qwen.jsonl
```

Pythia 中间 checkpoint 用 `--revisions step0,step128,...,step143000` 指定。固定前置长度使用 `--P 200` 并改变 `--Ds`；固定距离使用 `--Ds 30 --P 50,400,1000,1600`。完整模型、条件与每格样本数均记录在原始 JSONL 中。

`--wiki` 可指定 WikiText JSONL 路径；模型缓存遵循 Hugging Face 默认配置或 `HF_HOME`。Qwen 评测不套 chat 模板。16k 及大模型条件需要足够 GPU 显存。
