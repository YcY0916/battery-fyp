"""Check and merge all v2 runs into tables (laptop, after collect_runs.py).

    python src/merge_v2.py --exp paper_0to40_v2 --stage step2
    python src/merge_v2.py --exp paper_0to40_v2 --stage step3
    python src/merge_v2.py --exp v2_smoke --stage step2 --allow-missing

Checks (any failure stops the script with a list):
    - every folder in runs/ and models/ has a valid run name, is complete and its hashes match;
    - no missing runs: step2 = the 30 configs with seed 42; step3 = the 6 configs of
      shards/step3_candidates.txt with seeds 42-46 (only with --stage step3);
    - no unexpected runs (a config or seed outside the protocol);
    - every run has the same training code fingerprint, normalisation sha256, data sha256 summary
      and hyperparameters; metrics.json agrees with val_predictions.npz.
Duplicates cannot exist inside one folder (one folder per run); runs trained twice on different
platforms are caught by collect_runs.py.

Writes results/<exp>/tables/:
    all_runs.csv            one row per run, with the platform / GPU / versions it ran on
    runs_by_platform.csv    which runs each platform trained
    step2_ranking.csv       step 2, sorted by overall validation RMSE (rows >= 359)
    step2_summary.txt       the same as copyable text
    step3_summary.csv       (step3) mean ± sample std (ddof = 1) over seeds 42-46, per-seed overall
                            values, parameters, CPU inference time, initial delay W / f_s
    step3_per_seed.csv      (step3) one row per run
    step3_summary.txt       (step3) copyable text, configs in fixed (L, W, H) order
    window_saturation.csv   (step3) adjacent windows among candidates with the same L and H
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_analysis as A  # noqa: E402
import v2_common as V  # noqa: E402


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    V.write_text_atomic(path, df.to_csv(index=False, float_format="%.6g", lineterminator="\n"))


def main() -> None:
    p = argparse.ArgumentParser(description="Check and merge v2 runs into tables")
    p.add_argument("--exp", default=V.FORMAL_EXP)
    p.add_argument("--stage", choices=["step2", "step3"], required=True)
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--allow-missing", action="store_true", help="partial tables (never for the paper)")
    p.add_argument("--candidates-file", type=Path, default=V.STEP3_CANDIDATES_FILE)
    args = p.parse_args()
    paths = V.Paths(args.out_root, args.exp)
    log = Log(paths.logs / "merge.log")
    log("=" * 78)
    log(f"{V.now_iso()}  merge {args.exp}  stage {args.stage}")

    runs, incomplete, problems = A.scan_runs(paths)
    if incomplete:
        problems.append("incomplete run folders (rerun their shard with --skip-existing, or delete "
                        "them if the run was moved to another platform): " + ", ".join(incomplete))
    grid = V.step2_configs()
    want = {V.run_name(c, V.STEP2_SEED) for c in grid}
    cands = []
    if args.stage == "step3":
        cands = V.read_step3_candidates(args.candidates_file)
        if len(cands) != V.N_STEP3_CONFIGS and not args.allow_missing:
            problems.append(f"{args.candidates_file.name} lists {len(cands)} configs, expected {V.N_STEP3_CONFIGS}")
        want |= {V.run_name(c, s) for c in cands for s in V.STEP3_SEEDS}
    allowed = {V.run_name(c, V.STEP2_SEED) for c in grid} | \
              {V.run_name(c, s) for c in (cands or grid) for s in V.STEP3_SEEDS}
    unexpected = sorted(set(runs) - allowed)
    if unexpected:
        problems.append("runs outside the protocol: " + ", ".join(unexpected))
    for r, rec in runs.items():
        exp_stage = "step2" if rec["meta"]["seed"] == V.STEP2_SEED else "step3"
        if args.exp == V.FORMAL_EXP and rec["meta"]["stage"] != exp_stage:
            problems.append(f"{r}: stage {rec['meta']['stage']} but seed {rec['meta']['seed']}")
    missing = sorted(want - set(runs), key=lambda r: (V.config_sort_key(V.parse_run(r)[0]), V.parse_run(r)[1]))
    if missing and not args.allow_missing:
        problems.append(f"missing runs ({len(missing)}): " + ", ".join(missing))
    if problems:
        log("CHECK FAILED:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    A.check_consistency(runs, log)
    A.recheck_metrics(paths, runs)
    log(f"All {len(runs)} runs complete, hashes verified, metrics recomputed from npz (rows >= {V.EVAL_START})")
    if missing:
        log(f"WARNING (--allow-missing): {len(missing)} runs missing: {', '.join(missing)}")

    # ---- all runs and platforms ----
    rows = [A.run_row(rec) for rec in runs.values()]
    all_df = pd.DataFrame(rows)
    all_df["_k"] = all_df["config"].map(V.config_sort_key)
    all_df = all_df.sort_values(["_k", "seed"]).drop(columns="_k").reset_index(drop=True)
    save_csv(all_df, paths.tables / "all_runs.csv")
    plat = all_df.groupby(["platform", "gpu_name", "hostname"], sort=True).agg(
        n_runs=("run", "size"), runs=("run", lambda v: " ".join(v)),
        torch=("torch_version", lambda v: " ".join(sorted(set(v)))),
        cuda=("cuda_version", lambda v: " ".join(sorted(set(map(str, v)))))).reset_index()
    save_csv(plat, paths.tables / "runs_by_platform.csv")
    log("\nRuns per platform (results of the same seed differ slightly between GPUs / torch versions):")
    for _, r in plat.iterrows():
        log(f"  {r['platform']:7s} {r['gpu_name']} ({r['hostname']}, torch {r['torch']}, CUDA {r['cuda']}): "
            f"{r['n_runs']} runs: {r['runs']}")

    # ---- step 2 ----
    s2 = all_df[all_df["seed"] == V.STEP2_SEED].copy()
    s2 = A.add_worst(s2)
    s2 = s2.sort_values(["val_rmse_Overall"], kind="stable").reset_index(drop=True)
    s2.insert(0, "rank", np.arange(1, len(s2) + 1))
    s2["gap_to_rank1"] = s2["val_rmse_Overall"] - s2["val_rmse_Overall"].iloc[0]
    keep = ["rank", "config", "L", "W", "H", "n_params", "latency_s", "val_rmse_Overall", "gap_to_rank1"] + \
           [f"val_rmse_{f}" for f in V.FOLDERS] + ["worst_temp", "worst_temp_rmse", "val_mae_Overall"] + \
           [f"val_mae_{f}" for f in V.FOLDERS] + ["best_epoch", "epochs_trained", "sec_per_epoch",
                                                   "train_time_s", "platform", "gpu_name"]
    s2 = s2[keep]
    save_csv(s2, paths.tables / "step2_ranking.csv")
    plat_count = Counter(s2["platform"])
    text = A.step2_text(s2, plat_count, len(grid), args.exp)
    V.write_text_atomic(paths.tables / "step2_summary.txt", text)
    log("\n" + text)
    log(f"Wrote {paths.rel(paths.tables / 'step2_ranking.csv')} and step2_summary.txt")

    if args.stage != "step3":
        return
    # ---- step 3 ----
    bench = pd.read_csv(paths.bench_csv) if paths.bench_csv.exists() else None
    summary, per_seed = A.step3_tables(runs, cands, bench)
    save_csv(summary, paths.tables / "step3_summary.csv")
    save_csv(per_seed, paths.tables / "step3_per_seed.csv")
    sat = A.window_saturation(summary)
    if not sat.empty:
        save_csv(sat, paths.tables / "window_saturation.csv")
    lines = ["STEP 3 SUMMARY (copy everything between the two lines of '=')", "=" * 100,
             f"validation RMSE in SOC %, rows >= {V.EVAL_START}; mean ± sample std (ddof = 1) over seeds "
             f"{', '.join(map(str, V.ALL_SEEDS))}; configs in fixed (L, W, H) order",
             "CPU inference: laptop CPU, 1 thread, batch 1 (relative trend only, not the time on the MCU)", ""]
    head = ["config", "n"] + [A.COL_LABEL[c] for c in A.COLS] + ["params", "W/fs(s)", "CPU ms"]
    tab = [head]
    for _, r in summary.iterrows():
        tab.append([r["config"], str(int(r["n_seeds"]))] + [A.pm(r[f"rmse_{c}_mean"], r[f"rmse_{c}_std"]) for c in A.COLS]
                   + [f"{int(r['n_params']):,}", f"{r['latency_s']:.0f}",
                      "-" if pd.isna(r["cpu_infer_ms_mean"]) else f"{r['cpu_infer_ms_mean']:.3f}"])
    lines += A.fmt_table(tab, "lr" + "r" * len(A.COLS) + "rrr")
    lines += ["", "Overall RMSE per seed:"]
    for _, r in summary.iterrows():
        lines.append(f"  {r['config']:14s} " + "  ".join(
            f"s{s} {r[f'rmse_Overall_seed{s}']:.3f}" for s in V.ALL_SEEDS if f"rmse_Overall_seed{s}" in r
            and not pd.isna(r[f"rmse_Overall_seed{s}"])))
    if not sat.empty:
        lines += ["", "Adjacent windows (same L and H), overall column: improvement vs pooled seed std"]
        for _, r in sat[sat["column"] == "Overall"].iterrows():
            lines.append(f"  L{r['L']}_H{r['H']}: W{r['W_from']} -> W{r['W_to']}: {r['abs_improvement']:+.3f} "
                         f"({r['rel_improvement_pct']:+.1f} %), pooled seed std {r['pooled_seed_std']:.3f}, "
                         f"ratio {r['improvement_over_pooled_std']:.2f}")
    lines.append("=" * 100)
    text3 = "\n".join(lines) + "\n"
    V.write_text_atomic(paths.tables / "step3_summary.txt", text3)
    log("\n" + text3)
    if bench is None:
        log("Note: inference_benchmark.csv not found yet; run src/benchmark_v2.py, then merge again.")
    log(f"Wrote step3_summary.csv, step3_per_seed.csv, step3_summary.txt"
        + (", window_saturation.csv" if not sat.empty else ""))


if __name__ == "__main__":
    main()
