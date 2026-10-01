"""Run the whole paper pipeline in --smoke mode, step by step (25 degC, 2 configs, 2 seeds, 2 epochs).

Simulates two machines with --shard 1/2 and --shard 2/2, then merges, selects, benchmarks,
evaluates (with --unseen) and plots. Also checks that rerunning a finished shard trains nothing
and that merge_results.py rejects a duplicated run file. Output goes to results/paper_0to40_smoke/
and models/paper_0to40_smoke/ (not committed).

Usage (from the repository root):
    python src/smoke_paper.py --fresh          # delete the old smoke output first
    python src/smoke_paper.py --fresh --out_root D:/tmp/smoke --data_root D:/battery-fyp
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

SRC = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end smoke test of the paper pipeline")
    p.add_argument("--fresh", action="store_true", help="delete the previous smoke output first")
    p.add_argument("--out_root", type=Path, default=C.REPO_ROOT)
    p.add_argument("--data_root", type=Path, default=C.REPO_ROOT)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    paths = C.PaperPaths(args.out_root.resolve(), C.DEFAULT_NAME + "_smoke")
    if args.fresh:
        for d in (paths.results, paths.models):
            if d.exists():
                shutil.rmtree(d)
    common = ["--smoke", "--out_root", str(args.out_root), "--data_root", str(args.data_root)]
    dev = ["--device", args.device]

    def run(script: str, *extra: str, expect_fail: bool = False) -> None:
        cmd = [sys.executable, str(SRC / script), *extra]
        print(f"\n{'=' * 78}\n$ {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        if expect_fail:
            if rc == 0:
                raise SystemExit(f"SMOKE FAILED: {script} should have failed")
            print(f"(failed as expected, exit code {rc})")
        elif rc != 0:
            raise SystemExit(f"SMOKE FAILED at {script} (exit code {rc})")
        print(f"[{time.perf_counter() - t0:.1f} s]")

    run("lstm_data.py", "--paper", "--smoke", "--data_root", str(args.data_root))
    run("archive_old_runs.py", "--smoke")
    run("sweep_paper.py", *common, *dev, "--timing")
    run("sweep_paper.py", *common, *dev, "--shard", "1/2")          # "machine 1"
    run("sweep_paper.py", *common, *dev, "--shard", "2/2")          # "machine 2"
    run("sweep_paper.py", *common, *dev, "--shard", "1/2")          # resume: nothing to train
    run("merge_results.py", *common)
    run("sweep_paper.py", *common, *dev, "--stage", "multiseed", "--shard", "1/2")
    run("sweep_paper.py", *common, *dev, "--stage", "multiseed", "--shard", "2/2")
    run("merge_results.py", *common)
    run("merge_results.py", *common, "--eval_start", "359")         # recompute from val_preds/*.npz

    dup = paths.runs / f"{C.run_name(C.config_name(*C.SMOKE_CONFIGS[0]), C.SWEEP_SEED)} (1).json"
    shutil.copy(paths.run_json(C.run_name(C.config_name(*C.SMOKE_CONFIGS[0]), C.SWEEP_SEED)), dup)
    try:
        run("merge_results.py", *common, expect_fail=True)          # duplicate file must be rejected
    finally:
        dup.unlink()

    run("select_config.py", *common)
    chosen = C.read_json(paths.selected)["recommended_config"]
    run("benchmark_inference.py", *common)
    run("evaluate_paper.py", *common, *dev, "--config", chosen, "--unseen")
    run("plot_paper_figs.py", *common)

    expected = [paths.sweep_csv, paths.multiseed_csv, paths.candidates, paths.selected, paths.bench_csv,
                paths.test / chosen / "test_table.tex", paths.test / chosen / "unseen_temp_results.csv",
                paths.notes] + [paths.figs / f"{s}.{e}" for s in
                                ("fig1_window_vs_rmse", "fig2_params_vs_rmse", "fig3_temp_heatmap",
                                 "fig4_us06_traces") for e in ("png", "pdf")]
    missing = [paths.rel(f) for f in expected if not f.exists()]
    if missing:
        raise SystemExit(f"SMOKE FAILED: missing outputs {missing}")
    print(f"\nSMOKE OK: every step ran; outputs in {paths.results} and {paths.models}")


if __name__ == "__main__":
    main()
