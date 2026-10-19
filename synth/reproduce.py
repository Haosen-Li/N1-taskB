"""Replay saved experiment configurations on one device, including dependencies.

This reads the submitted metadata but writes to artifacts/runs by default.
Use --dry-run to inspect commands without starting training.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RECORDED = HERE / "runs"


def configurations():
    return {p.parent.name: json.loads(p.read_text())["args"]
            for p in sorted(RECORDED.glob("*/meta.json"))}


def checkpoint_ref(value):
    path = Path(value)
    return path.parent.name, path.name


def ordered_runs(names, configs):
    order, active, done = [], set(), set()

    def visit(name):
        if name not in configs:
            raise ValueError(f"Unknown run: {name}. Use --list.")
        if name in active:
            raise ValueError(f"Cyclic checkpoint dependency: {name}")
        if name in done:
            return
        active.add(name)
        for key in ("init_from", "resume"):
            if configs[name].get(key):
                visit(checkpoint_ref(configs[name][key])[0])
        active.remove(name)
        done.add(name)
        order.append(name)

    for name in names:
        visit(name)
    return order


def command(name, config, out, device):
    args = dict(config, name=name, out=str(out), device=device)
    for key in ("init_from", "resume"):
        if args.get(key):
            run, filename = checkpoint_ref(args[key])
            args[key] = str(out / run / filename)
    cmd = [sys.executable, str(HERE / "train.py")]
    for key, value in args.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list available experiment names")
    mode.add_argument("--runs", nargs="+", help="run names; prerequisites are included automatically")
    mode.add_argument("--all", action="store_true", help="retrain all recorded configurations")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=HERE.parent / "artifacts" / "runs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    configs = configurations()
    if a.list:
        print("\n".join(configs))
        return
    out = a.out.resolve()
    if out == RECORDED or RECORDED in out.parents:
        p.error("Choose a new output directory; synth/runs contains the submitted evidence.")
    try:
        names = ordered_runs(list(configs) if a.all else a.runs, configs)
    except ValueError as exc:
        p.error(str(exc))
    for name in names:
        cmd = command(name, configs[name], out, a.device)
        print(shlex.join(cmd), flush=True)
        if a.dry_run:
            continue
        run = out / name
        if (run / "final.pt").is_file() and (run / "summary.json").is_file():
            print(f"Already complete: {name}", flush=True)
            continue
        if (run / "log.jsonl").exists():
            p.error(f"Incomplete run at {run}; choose a fresh --out directory.")
        for key in ("init_from", "resume"):
            if configs[name].get(key):
                dep, filename = checkpoint_ref(configs[name][key])
                if not (out / dep / filename).is_file():
                    p.error(f"Missing dependency checkpoint: {out / dep / filename}")
        subprocess.run(cmd, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
