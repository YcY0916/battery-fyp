"""Run two run_shard.py processes at once on Kaggle, one per GPU, and wait for both.

Called by notebooks/kaggle_run_shard.ipynb (the logic lives here, in the repository, so that a
fix reaches Kaggle through git clone without uploading the notebook again):

    python src/kaggle_two_gpus.py --mode train --exp paper_0to40_v2 \
        --shard0 shards/step2_kaggle_gpu0.txt --shard1 shards/step2_kaggle_gpu1.txt \
        --out /kaggle/working/out --logs /kaggle/working --deadline-h 11
    python src/kaggle_two_gpus.py --mode probe --out /kaggle/working/out --logs /kaggle/working

Process k runs with CUDA_VISIBLE_DEVICES=k; its console output goes to <logs>/console_gpu<k>.log.
The processes are checked every 10 s: a process that ends is reported at once (with the last lines
of its log); a progress line is printed every 5 minutes. Exit status: 0 if both processes exit
with 0, otherwise 1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too

REPO_ROOT = Path(__file__).resolve().parents[1]
POLL_S = 10
PROGRESS_S = 300
EXIT_MEANING = "0 = all runs complete; 3 = some runs not finished (deadline); other = error, stop and ask"


def tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def last_line(path: Path) -> str:
    lines = [ln for ln in tail(path, 5).splitlines() if ln.strip()]
    return lines[-1][:100] if lines else "(no output yet)"


def main() -> None:
    p = argparse.ArgumentParser(description="Two run_shard.py processes, one per GPU (Kaggle)")
    p.add_argument("--mode", choices=["train", "probe"], required=True)
    p.add_argument("--exp", default="paper_0to40_v2")
    p.add_argument("--shard0", default="shards/step2_kaggle_gpu0.txt")
    p.add_argument("--shard1", default="shards/step2_kaggle_gpu1.txt")
    p.add_argument("--out", type=Path, required=True, help="out root for results/ and models/")
    p.add_argument("--logs", type=Path, required=True, help="folder for console_gpu0.log / console_gpu1.log")
    p.add_argument("--deadline-h", type=float, default=11.0)
    args = p.parse_args()

    args.logs.mkdir(parents=True, exist_ok=True)
    procs = {}
    t0 = time.time()
    for gpu, shard in ((0, args.shard0), (1, args.shard1)):
        if args.mode == "probe":
            cmd = [sys.executable, "src/run_shard.py", "--exp", "timing_v2", "--shard-file", "shards/probe.txt",
                   "--max-epochs", "3", "--note", "kaggle two processes", "--out-root", str(args.out / f"probe_gpu{gpu}")]
        else:
            cmd = [sys.executable, "src/run_shard.py", "--exp", args.exp, "--shard-file", shard,
                   "--out-root", str(args.out)]
        cmd += ["--platform", "kaggle", "--skip-existing", "--deadline-h", str(args.deadline_h)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        log_path = args.logs / f"console_gpu{gpu}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"GPU {gpu}: $ {' '.join(cmd)}", flush=True)
        procs[gpu] = {"proc": subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file,
                                               stderr=subprocess.STDOUT),
                      "file": log_file, "log": log_path, "rc": None}

    last_progress = time.time()
    while any(v["rc"] is None for v in procs.values()):
        time.sleep(POLL_S)
        for gpu, v in procs.items():
            if v["rc"] is None and v["proc"].poll() is not None:
                v["rc"] = v["proc"].returncode
                v["file"].close()
                print(f"[{(time.time() - t0) / 3600:5.2f} h] GPU {gpu} finished with exit code {v['rc']} "
                      f"({EXIT_MEANING}). Last lines:\n{tail(v['log'], 15)}", flush=True)
        if time.time() - last_progress >= PROGRESS_S and any(v["rc"] is None for v in procs.values()):
            last_progress = time.time()
            print(f"[{(time.time() - t0) / 3600:5.2f} h] " + " | ".join(
                f"GPU{g}: {'running' if v['rc'] is None else 'exit ' + str(v['rc'])}: {last_line(v['log'])}"
                for g, v in procs.items()), flush=True)

    print(f"\nBoth processes finished after {(time.time() - t0) / 3600:.2f} h:")
    for gpu, v in procs.items():
        print(f"  GPU {gpu} exit code {v['rc']}")
    if args.mode == "probe":
        subprocess.run([sys.executable, "src/probe_summary.py", "--exp", "timing_v2", "--source",
                        str(args.out / "probe_gpu0"), str(args.out / "probe_gpu1")], cwd=REPO_ROOT)
    sys.exit(0 if all(v["rc"] == 0 for v in procs.values()) else 1)


if __name__ == "__main__":
    main()
