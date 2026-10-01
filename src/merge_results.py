"""Merge the per-run JSON files of steps 2 and 3 into the result tables.

Reads results/<name>/runs/<config>_seed<seed>.json (copy the files of every machine into this
folder first) and writes:
    results/<name>/sweep_results.csv      step 2: one row per config (seed 42), sorted by val RMSE
    results/<name>/candidates.json        step 3 candidates: top 4 by val RMSE + every L=1 H at the best W
    results/<name>/multiseed_results.csv  step 3: mean +- std (sample std, ddof = 1) over the seeds
    results/<name>/all_runs.csv           one row per run

Checks (the script stops with an error and lists the runs concerned):
    - unexpected file names in runs/ (e.g. "W60_H32_L1_seed42 (1).json" from a second copy),
      two files describing the same run, a file whose content does not match its name;
    - missing runs (all 18 step-2 runs; every seed of every candidate once step 3 has started);
    - all runs must share the same git commit, normalisation-file hash and training settings.
If the best config has W = 300 (edge of the grid) a warning says that W may not be saturated.
If inference_benchmark.csv exists (benchmark_inference.py) its CPU times are added to both tables.
Training times come from different machines and must not be compared across machines.

Usage (from the repository root):
    python src/merge_results.py
    python src/merge_results.py --allow_missing      # partial tables while runs are still missing
    python src/merge_results.py --eval_start 359     # recompute val metrics from val_preds/*.npz on
                                                     # rows >= 359 (writes *_start359.csv, no retraining)
    python src/merge_results.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

INFERENCE_COLS = ["inference_ms_mean", "inference_ms_std", "inference_cpu"]


class MergeError(SystemExit):
    pass


def load_runs(paths: C.PaperPaths) -> tuple[dict, list[str]]:
    runs, problems, where = {}, [], {}
    for f in sorted(paths.runs.glob("*.json")):
        m = C.RUN_RE.match(f.stem)
        if m is None:
            problems.append(f"unexpected file name (duplicate copy?): {paths.rel(f)}")
            continue
        r = C.read_json(f)
        if r.get("run") != f.stem:
            problems.append(f"{paths.rel(f)} contains run {r.get('run')!r}, not {f.stem!r}")
            continue
        key = (r["config"], int(r["seed"]))
        if key in runs:
            problems.append(f"duplicate run {f.stem}: {where[key]} and {paths.rel(f)}")
            continue
        runs[key], where[key] = r, paths.rel(f)
    return runs, problems


def check_consistency(runs: dict, paths: C.PaperPaths, log) -> None:
    def groups(fn):
        g = defaultdict(list)
        for (cfg, seed), r in runs.items():
            g[fn(r)].append(C.run_name(cfg, seed))
        return g

    errors = []
    for label, fn in (("git commit", lambda r: r["environment"]["git_commit"]),
                      ("normalisation file sha256", lambda r: r["environment"]["norm_stats_sha256"]),
                      ("training settings", lambda r: json.dumps(r["settings"], sort_keys=True))):
        g = groups(fn)
        if len(g) > 1:
            lines = [f"  {str(v)[:70]}: {', '.join(sorted(rs))}" for v, rs in g.items()]
            errors.append(f"runs differ in {label}:\n" + "\n".join(lines))
    if paths.norm_stats.exists() and runs:
        sha = C.file_sha256(paths.norm_stats)
        bad = sorted(C.run_name(*k) for k, r in runs.items() if r["environment"]["norm_stats_sha256"] != sha)
        if bad:
            errors.append(f"{paths.rel(paths.norm_stats)} (sha256 {sha[:12]}) is not the file used by: "
                          + ", ".join(bad))
    if errors:
        raise MergeError("Consistency check failed:\n" + "\n".join(errors))
    dirty = sorted(C.run_name(*k) for k, r in runs.items() if r["environment"].get("git_dirty"))
    if dirty:
        log(f"WARNING: trained with uncommitted code changes: {', '.join(dirty)}")
    if runs:
        r0 = next(iter(runs.values()))["environment"]
        log(f"Consistency OK: {len(runs)} runs, git commit {r0['git_commit'][:12]}, "
            f"normalisation sha256 {r0['norm_stats_sha256'][:12]}, identical settings")


def run_metrics(r: dict, paths: C.PaperPaths, eval_start: int | None) -> dict:
    """Validation metrics of one run: from its JSON, or recomputed from its npz at eval_start."""
    if eval_start is None or eval_start == r["val_eval_start_index"]:
        return {"val_rmse_pct": r["val_rmse_pct"], "val_mae_pct": r["val_mae_pct"],
                "per_temp": {k: v["rmse_pct"] for k, v in r["val_per_temperature"].items()},
                "n_val_points": r["n_val_points"]}
    npz = paths.out_root / r["files"]["val_predictions"]
    if not npz.exists():
        raise MergeError(f"--eval_start {eval_start}: missing {paths.rel(npz)}")
    m = C.metrics_from_npz(npz, eval_start)
    return {"val_rmse_pct": m["val_rmse_pct"], "val_mae_pct": m["val_mae_pct"],
            "per_temp": {k: v["rmse_pct"] for k, v in m["per_temp"].items()},
            "n_val_points": m["n_val_points"]}


def read_inference(paths: C.PaperPaths) -> pd.DataFrame | None:
    if not paths.bench_csv.exists():
        return None
    b = pd.read_csv(paths.bench_csv)
    return b[["config", "inference_ms_mean", "inference_ms_std", "cpu_model"]].rename(
        columns={"cpu_model": "inference_cpu"})


def add_inference(df: pd.DataFrame, inf: pd.DataFrame | None) -> pd.DataFrame:
    if inf is None or df.empty:
        return df
    return df.drop(columns=[c for c in INFERENCE_COLS if c in df.columns]).merge(inf, on="config", how="left")


def pick_candidates(sweep: pd.DataFrame, grid: list[str]) -> dict:
    best = sweep.iloc[0]
    top = list(sweep["config"].iloc[:C.TOP_K])
    best_w = [c for c in grid if C.parse_config(c)[0] == int(best["window"]) and C.parse_config(c)[2] == 1]
    cands = [c for c in sweep["config"] if c in set(top) | set(best_w)]  # ordered by val RMSE
    return {"candidates": cands, "best_config": best["config"], "best_window": int(best["window"]),
            "top_k": top, "best_window_l1": best_w,
            "rule": f"top {C.TOP_K} by step-2 validation RMSE (seed {C.SWEEP_SEED}) "
                    "union every L=1 hidden size at the best window"}


def merge(ctx: C.Context, allow_missing: bool = False, eval_start: int | None = None, log=print) -> dict:
    paths = ctx.paths
    runs, problems = load_runs(paths)
    if problems:
        raise MergeError("Problems in " + paths.rel(paths.runs) + ":\n  " + "\n  ".join(problems))
    if not runs:
        raise MergeError(f"No run JSON files in {paths.rel(paths.runs)}")
    check_consistency(runs, paths, log)
    if eval_start is not None:
        max_w = max(r["window"] for r in runs.values())
        if eval_start < max_w - 1:
            raise MergeError(f"--eval_start {eval_start} is below W-1 = {max_w - 1} of the longest window")
    suffix = "" if eval_start is None or eval_start == C.EVAL_START else f"_start{eval_start}"
    start_used = C.EVAL_START if eval_start is None else eval_start
    inf = read_inference(paths)
    grid, folders, seeds = ctx.config_names, ctx.folders, ctx.seeds
    out = {"runs": runs}

    # ---- all runs ----
    rows = []
    for (cfg, seed), r in sorted(runs.items()):
        m = run_metrics(r, paths, eval_start)
        rows.append({"run": C.run_name(cfg, seed), "config": cfg, "seed": seed, "stage": r["stage"],
                     "val_rmse_pct": m["val_rmse_pct"], "val_mae_pct": m["val_mae_pct"],
                     **{f"val_rmse_pct_{f}": m["per_temp"].get(f) for f in folders},
                     "best_epoch": r["best_epoch"], "epochs_trained": r["epochs_trained"],
                     "n_params": r["n_params"], "train_time_s": r["train_time_s"],
                     "train_host": r["environment"]["hostname"], "train_device": r["environment"]["gpu"],
                     "git_commit": r["environment"]["git_commit"][:12],
                     "finished_at": r["finished_at"]})
    all_df = pd.DataFrame(rows)
    all_df.to_csv(paths.results / f"all_runs{suffix}.csv", index=False, float_format="%.4f")

    # ---- step 2 ----
    missing2 = [c for c in grid if (c, C.SWEEP_SEED) not in runs]
    if missing2 and not allow_missing:
        raise MergeError(f"Missing step-2 runs (seed {C.SWEEP_SEED}): {', '.join(missing2)}\n"
                         "Copy them from the other machine or use --allow_missing for a partial table")
    srows = []
    for c in grid:
        r = runs.get((c, C.SWEEP_SEED))
        if r is None:
            continue
        m = run_metrics(r, paths, eval_start)
        srows.append({"config": c, "window": r["window"], "hidden_size": r["hidden_size"],
                      "num_layers": r["num_layers"], "val_rmse_pct": m["val_rmse_pct"],
                      "val_mae_pct": m["val_mae_pct"],
                      **{f"val_rmse_pct_{f}": m["per_temp"].get(f) for f in folders},
                      "n_val_points": m["n_val_points"], "best_epoch": r["best_epoch"],
                      "epochs_trained": r["epochs_trained"], "stopped_early": r["stopped_early"],
                      "n_params": r["n_params"], "train_time_s": r["train_time_s"],
                      "sec_per_epoch": r["sec_per_epoch"],
                      "train_host": r["environment"]["hostname"], "train_device": r["environment"]["gpu"]})
    sweep = pd.DataFrame(srows).sort_values("val_rmse_pct", kind="stable").reset_index(drop=True)
    sweep.insert(0, "rank", np.arange(1, len(sweep) + 1))
    sweep = add_inference(sweep, inf)
    sweep_path = paths.results / f"sweep_results{suffix}.csv"
    sweep.to_csv(sweep_path, index=False, float_format="%.4f")
    out["sweep"] = sweep
    log(f"\nStep 2: {len(sweep)}/{len(grid)} configs (validation rows >= {start_used}; SOC percentage points)")
    show = ["rank", "config", "val_rmse_pct", "val_mae_pct"] + [f"val_rmse_pct_{f}" for f in folders] + \
           ["best_epoch", "epochs_trained", "n_params"] + (["inference_ms_mean"] if inf is not None else [])
    log(sweep[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"Wrote {paths.rel(sweep_path)}")
    if missing2:
        log(f"WARNING: partial table, missing {', '.join(missing2)}; candidates.json not written")
        return out

    best_w = int(sweep["window"].iloc[0])
    if best_w == max(C.parse_config(c)[0] for c in grid):
        log("\n" + "!" * 78 + f"\nWARNING: the best config {sweep['config'].iloc[0]} uses W = {best_w}, the "
            "edge of the grid.\nThe validation error may still be falling with longer windows (W not saturated).\n"
            + "!" * 78)

    # ---- candidates for step 3 ----
    cand = pick_candidates(sweep, grid)
    if suffix:
        log(f"Candidates at eval start {start_used} (not written): {', '.join(cand['candidates'])}")
        cand_list = C.read_json(paths.candidates)["candidates"] if paths.candidates.exists() else cand["candidates"]
    else:
        if paths.candidates.exists():
            old = C.read_json(paths.candidates)["candidates"]
            if old != cand["candidates"]:
                log(f"WARNING: candidates changed from {old} to {cand['candidates']}")
        r0 = next(iter(runs.values()))["environment"]
        C.write_json(paths.candidates, {**cand, "seeds": seeds, "git_commit": r0["git_commit"],
                                        "norm_stats_sha256": r0["norm_stats_sha256"],
                                        "created_at": datetime.now().isoformat(timespec="seconds")})
        log(f"\nStep-3 candidates ({cand['rule']}):\n  top {C.TOP_K}: {', '.join(cand['top_k'])}\n"
            f"  L=1 at W={cand['best_window']}: {', '.join(cand['best_window_l1'])}\n"
            f"  -> {', '.join(cand['candidates'])}   (wrote {paths.rel(paths.candidates)})")
        cand_list = cand["candidates"]
    out["candidates"] = cand_list

    # ---- step 3 ----
    extra = sorted(C.run_name(c, s) for (c, s) in runs
                   if s != C.SWEEP_SEED and (c not in cand_list or s not in seeds))
    if extra:
        log(f"WARNING: runs that are not part of step 3 (ignored): {', '.join(extra)}")
    started = any((c, s) in runs for c in cand_list for s in seeds if s != C.SWEEP_SEED)
    if not started:
        log(f"\nStep 3 not started yet (no runs with seeds {[s for s in seeds if s != C.SWEEP_SEED]}); "
            f"{paths.rel(paths.multiseed_csv)} not written")
        return out
    missing3 = [C.run_name(c, s) for c in cand_list for s in seeds if (c, s) not in runs]
    if missing3 and not allow_missing:
        raise MergeError(f"Missing step-3 runs: {', '.join(missing3)}\n"
                         "Copy them from the other machine or use --allow_missing for a partial table")
    mrows = []
    for c in cand_list:
        rs = [runs[(c, s)] for s in seeds if (c, s) in runs]
        ms = [run_metrics(r, paths, eval_start) for r in rs]
        row = {"config": c, "window": rs[0]["window"], "hidden_size": rs[0]["hidden_size"],
               "num_layers": rs[0]["num_layers"], "n_seeds": len(rs),
               "seeds": " ".join(str(r["seed"]) for r in rs),
               "val_rmse_pct_mean": np.mean([m["val_rmse_pct"] for m in ms]),
               "val_rmse_pct_std": C.sample_std(m["val_rmse_pct"] for m in ms),
               "val_mae_pct_mean": np.mean([m["val_mae_pct"] for m in ms]),
               "val_mae_pct_std": C.sample_std(m["val_mae_pct"] for m in ms)}
        for f in folders:
            v = [m["per_temp"][f] for m in ms if f in m["per_temp"]]
            row[f"val_rmse_pct_{f}_mean"] = np.mean(v) if v else np.nan
            row[f"val_rmse_pct_{f}_std"] = C.sample_std(v)
        hosts = sorted({r["environment"]["hostname"] for r in rs})
        gpus = sorted({r["environment"]["gpu"] for r in rs})
        if len(hosts) > 1:
            log(f"WARNING: seeds of {c} were trained on different machines {hosts}; "
                "their training times are not comparable")
        row.update({"n_params": rs[0]["n_params"],
                    "best_epoch_mean": np.mean([r["best_epoch"] for r in rs]),
                    "epochs_trained_mean": np.mean([r["epochs_trained"] for r in rs]),
                    "train_time_s_mean": np.mean([r["train_time_s"] for r in rs]),
                    "train_time_s_std": C.sample_std(r["train_time_s"] for r in rs),
                    "train_host": "; ".join(hosts), "train_device": "; ".join(gpus)})
        mrows.append(row)
    multi = pd.DataFrame(mrows).sort_values("val_rmse_pct_mean", kind="stable").reset_index(drop=True)
    multi.insert(0, "rank", np.arange(1, len(multi) + 1))
    multi = add_inference(multi, inf)
    multi_path = paths.results / f"multiseed_results{suffix}.csv"
    multi.to_csv(multi_path, index=False, float_format="%.4f")
    out["multiseed"] = multi
    log(f"\nStep 3: {len(multi)} candidates x {len(seeds)} seeds (mean +- sample std; validation rows >= {start_used})")
    show = ["rank", "config", "n_seeds", "val_rmse_pct_mean", "val_rmse_pct_std", "val_mae_pct_mean",
            "val_mae_pct_std", "n_params", "train_host"] + (["inference_ms_mean"] if inf is not None else [])
    log(multi[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("Note: train_time_s_* depends on the machine in train_host/train_device; "
        "do not compare training times across machines.")
    if missing3:
        log(f"WARNING: partial table, missing {', '.join(missing3)}")
    log(f"Wrote {paths.rel(multi_path)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge the run JSON files into sweep_results.csv / multiseed_results.csv")
    C.add_common_args(p)
    p.add_argument("--allow_missing", action="store_true", help="write partial tables when runs are missing")
    p.add_argument("--eval_start", type=int, default=None,
                   help=f"recompute validation metrics from val_preds/*.npz on rows >= this index "
                        f"(default: {C.EVAL_START}, the value used in training)")
    args = p.parse_args()
    ctx = C.resolve(args)
    log = C.Tee(ctx.paths.logs / "merge.log")
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  merge {ctx.paths.name}  ({ctx.paths.results})")
    try:
        merge(ctx, args.allow_missing, args.eval_start, log)
    except MergeError as e:
        log(f"ERROR: {e}")
        log.close()
        raise SystemExit(1)
    log.close()


if __name__ == "__main__":
    main()
