"""Step 3.5: CPU inference time of every step-2 config, measured on one machine in one go.

Loads the seed-42 weights of all step-2 configs (models/<name>/<config>_seed42.pt) and, on the
CPU with torch.set_num_threads(1), times a forward pass of one window (batch = 1): warm-up first,
then 1000 timed passes; reports mean and sample std in milliseconds, the number of trainable
parameters and the CPU model. Do not run any training (or other heavy job) on the machine
while this runs.

The input is a fixed random window: the run time of an LSTM does not depend on the values, so no
data file is read.

Writes results/<name>/inference_benchmark.csv and then reruns merge_results.py so that
sweep_results.csv and multiseed_results.csv get the inference-time columns.

Usage (from the repository root, after all training has finished and the weights are copied here):
    python src/benchmark_inference.py
    python src/benchmark_inference.py --smoke
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402
import merge_results as M  # noqa: E402
from lstm_model import LSTMSOC, count_parameters  # noqa: E402

WARMUP = 100
ITERS = 1000


def benchmark(model: torch.nn.Module, window: int, iters: int, warmup: int) -> np.ndarray:
    x = torch.randn(1, window, 3, generator=torch.Generator().manual_seed(0))
    times = np.empty(iters, dtype=np.float64)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for i in range(iters):
            t0 = time.perf_counter_ns()
            model(x)
            times[i] = (time.perf_counter_ns() - t0) / 1e6
    return times


def main() -> None:
    p = argparse.ArgumentParser(description="CPU inference time (batch 1, 1 thread) of the step-2 configs")
    C.add_common_args(p)
    p.add_argument("--allow_missing", action="store_true", help="skip configs whose weights are missing")
    args = p.parse_args()
    ctx = C.resolve(args)
    paths = ctx.paths

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cpu = C.cpu_model()
    host = socket.gethostname()
    print("Reminder: do not run training or other heavy jobs on this machine during the measurement.")
    print(f"CPU: {cpu} ({host}), torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"batch 1, {WARMUP} warm-up + {ITERS} timed passes per config")

    missing = [c for c in ctx.config_names
               if not paths.model_file(C.run_name(c, C.SWEEP_SEED)).exists()]
    if missing and not args.allow_missing:
        raise SystemExit("Missing seed-42 weights (copy them from the other machine into "
                         f"{paths.rel(paths.models)}/): {', '.join(missing)}")
    rows = []
    for cname in ctx.config_names:
        if cname in missing:
            continue
        ckpt = torch.load(paths.model_file(C.run_name(cname, C.SWEEP_SEED)), map_location="cpu",
                          weights_only=True)
        cfg = ckpt["config"]
        model = LSTMSOC(cfg["hidden_size"], cfg["num_layers"], cfg["input_size"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        t = benchmark(model, cfg["window"], ITERS, WARMUP)
        rows.append({"config": cname, "window": cfg["window"], "hidden_size": cfg["hidden_size"],
                     "num_layers": cfg["num_layers"], "n_params": count_parameters(model),
                     "inference_ms_mean": float(t.mean()), "inference_ms_std": float(t.std(ddof=1)),
                     "inference_ms_median": float(np.median(t)), "n_iters": ITERS, "warmup": WARMUP,
                     "threads": 1, "batch_size": 1, "cpu_model": cpu, "hostname": host,
                     "torch_version": torch.__version__,
                     "measured_at": datetime.now().isoformat(timespec="seconds")})
        print(f"  {cname:14s} {t.mean():8.3f} +- {t.std(ddof=1):6.3f} ms   params {rows[-1]['n_params']:>7,d}")
    df = pd.DataFrame(rows)
    paths.results.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.bench_csv, index=False, float_format="%.5f")
    print(f"Wrote {paths.rel(paths.bench_csv)}")

    print("\nAdding the inference times to sweep_results.csv / multiseed_results.csv ...")
    try:
        M.merge(ctx, allow_missing=args.allow_missing, log=print)
    except M.MergeError as e:
        print(f"WARNING: tables not updated ({e}).\nFix the problem and run src/merge_results.py; "
              "it adds the inference times automatically.")


if __name__ == "__main__":
    main()
