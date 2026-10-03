"""CPU inference time of the step-3 configs on the laptop (same method as v1 benchmark_inference.py).

    python src/benchmark_v2.py --exp paper_0to40_v2

For every config in shards/step3_candidates.txt (or --configs) the seed-42 weights are loaded and a
forward pass of one window (batch 1) is timed on the CPU with torch.set_num_threads(1):
100 warm-up passes, then 1000 timed passes; mean and sample std in ms. The input is a fixed random
window (the run time of an LSTM does not depend on the values), so no data file is read.

The numbers show the relative cost of the configs on this laptop CPU only; they are not the
absolute time on the MCU. Writes results/<exp>/inference_benchmark.csv; then run
src/merge_v2.py --stage step3 again so the step-3 tables get the times.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import v2_common as V  # noqa: E402
from benchmark_inference import ITERS, WARMUP, benchmark  # noqa: E402
from lstm_model import LSTMSOC, count_parameters  # noqa: E402
from paper_common import cpu_model  # noqa: E402

CHECKLIST = """Before the measurement, on this laptop:
  1. close every other program (browser, IDE, chat apps, Anaconda Navigator, ...);
  2. pause Google Drive and OneDrive sync;
  3. plug in the charger;
  4. Windows power mode: Best performance;
  5. make sure no training runs on this laptop (no other Anaconda Prompt busy);
  6. do not touch the laptop until the script prints 'Wrote'."""


def main() -> None:
    p = argparse.ArgumentParser(description="CPU inference time (1 thread, batch 1) of the step-3 configs")
    p.add_argument("--exp", default=V.FORMAL_EXP)
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--configs", default=None, help="comma-separated configs (default: step-3 candidates)")
    p.add_argument("--yes", action="store_true", help="do not wait for Enter (smoke test only)")
    args = p.parse_args()
    paths = V.Paths(args.out_root, args.exp)
    configs = [c.strip() for c in args.configs.split(",")] if args.configs else V.read_step3_candidates()

    print(CHECKLIST)
    if not args.yes:
        input("Press Enter when all six points are done ... ")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cpu, host = cpu_model(), socket.gethostname()
    print(f"CPU: {cpu} ({host}), torch {torch.__version__}, threads {torch.get_num_threads()}, batch 1, "
          f"{WARMUP} warm-up + {ITERS} timed passes per config")
    rows = []
    for c in configs:
        run = V.run_name(c, V.STEP2_SEED)
        if V.run_status(paths, run)[0] != "complete":
            raise SystemExit(f"{run} is not complete in {paths.rel(paths.runs)}")
        ck = torch.load(paths.model_file(run), map_location="cpu", weights_only=True)
        cfg = ck["config"]
        model = LSTMSOC(cfg["hidden_size"], cfg["num_layers"], cfg["input_size"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        t = benchmark(model, cfg["window"], ITERS, WARMUP)
        rows.append({"config": c, "L": cfg["num_layers"], "W": cfg["window"], "H": cfg["hidden_size"],
                     "n_params": count_parameters(model), "inference_ms_mean": float(t.mean()),
                     "inference_ms_std": float(t.std(ddof=1)), "inference_ms_median": float(np.median(t)),
                     "n_iters": ITERS, "warmup": WARMUP, "threads": 1, "batch_size": 1, "cpu_model": cpu,
                     "hostname": host, "torch_version": torch.__version__, "measured_at": V.now_iso()})
        print(f"  {c:14s} {t.mean():8.3f} ± {t.std(ddof=1):6.3f} ms   params {rows[-1]['n_params']:>7,d}")
        time.sleep(1.0)
    V.write_text_atomic(paths.bench_csv, pd.DataFrame(rows).to_csv(index=False, float_format="%.5f",
                                                                    lineterminator="\n"))
    print(f"Wrote {paths.rel(paths.bench_csv)}")
    print("Note: laptop CPU, 1 thread, batch 1; relative trend only, not the absolute time on the MCU.")


if __name__ == "__main__":
    main()
