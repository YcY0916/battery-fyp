"""Recommend the final configuration from multiseed_results.csv (validation set only).

Rule:
    1. best      = candidate with the lowest mean validation RMSE over the seeds;
    2. threshold = mean of best + (sample) std of best;
    3. among the candidates whose mean is <= threshold, pick the one with the fewest trainable
       parameters (ties: lower mean RMSE).

This only writes a recommendation (results/<name>/selected_config.json). The test set is
evaluated later, by hand, with src/evaluate_paper.py --config <the config you decide on>.

Usage (from the repository root):
    python src/select_config.py
    python src/select_config.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402


def select(df: pd.DataFrame) -> dict:
    df = df.sort_values("val_rmse_pct_mean", kind="stable").reset_index(drop=True)
    best = df.iloc[0]
    threshold = float(best["val_rmse_pct_mean"] + best["val_rmse_pct_std"])
    eligible = df[df["val_rmse_pct_mean"] <= threshold + 1e-12]
    chosen = eligible.sort_values(["n_params", "val_rmse_pct_mean"], kind="stable").iloc[0]
    return {"best": best, "threshold": threshold, "eligible": eligible, "chosen": chosen, "table": df}


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend the final config (lowest params within best mean + std)")
    C.add_common_args(p)
    args = p.parse_args()
    ctx = C.resolve(args)
    paths = ctx.paths
    if not paths.multiseed_csv.exists():
        raise SystemExit(f"{paths.rel(paths.multiseed_csv)} not found; run src/merge_results.py after step 3")
    df = pd.read_csv(paths.multiseed_csv)
    incomplete = df[df["n_seeds"] < len(ctx.seeds)]
    if not incomplete.empty:
        raise SystemExit(f"Candidates with fewer than {len(ctx.seeds)} seeds: {', '.join(incomplete['config'])}; "
                         "finish step 3 and rerun src/merge_results.py")
    s = select(df)
    best, chosen, thr = s["best"], s["chosen"], s["threshold"]

    print(f"Candidates (validation RMSE over seeds {ctx.seeds}, mean +- sample std, SOC percentage points):")
    for _, r in s["table"].iterrows():
        flag = "eligible" if r["val_rmse_pct_mean"] <= thr + 1e-12 else "above threshold"
        print(f"  {r['config']:14s} {r['val_rmse_pct_mean']:.3f} +- {r['val_rmse_pct_std']:.3f}   "
              f"params {int(r['n_params']):>7,d}   {flag}")
    print(f"\n1. Best mean validation RMSE: {best['config']} "
          f"({best['val_rmse_pct_mean']:.3f} +- {best['val_rmse_pct_std']:.3f})")
    print(f"2. Threshold = {best['val_rmse_pct_mean']:.3f} + {best['val_rmse_pct_std']:.3f} = {thr:.3f}")
    print(f"3. Eligible (mean <= {thr:.3f}): " + ", ".join(
        f"{r['config']} ({int(r['n_params']):,} params)" for _, r in s["eligible"].iterrows()))
    print(f"   -> fewest parameters: {chosen['config']} ({int(chosen['n_params']):,} params, "
          f"val RMSE {chosen['val_rmse_pct_mean']:.3f} +- {chosen['val_rmse_pct_std']:.3f})")
    max_w = max(C.parse_config(c)[0] for c in ctx.config_names)
    if int(best["window"]) == max_w or int(chosen["window"]) == max_w:
        print(f"WARNING: W = {max_w} is the edge of the grid; the error may still fall with longer windows.")

    record = {
        "recommended_config": chosen["config"],
        "rule": "best = lowest mean val RMSE; threshold = best mean + best sample std; "
                "pick the fewest trainable parameters among candidates with mean <= threshold",
        "best_config": best["config"], "best_mean": float(best["val_rmse_pct_mean"]),
        "best_std": float(best["val_rmse_pct_std"]), "threshold": thr,
        "eligible": list(s["eligible"]["config"]),
        "candidates": [{"config": r["config"], "val_rmse_pct_mean": float(r["val_rmse_pct_mean"]),
                        "val_rmse_pct_std": float(r["val_rmse_pct_std"]), "n_params": int(r["n_params"])}
                       for _, r in s["table"].iterrows()],
        "seeds": ctx.seeds,
        "note": "Recommendation only. Evaluate the test set by hand: "
                f"python src/evaluate_paper.py --config {chosen['config']}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    C.write_json(paths.selected, record)
    print(f"\nWrote {paths.rel(paths.selected)}")
    print("The test set has not been touched. When you have decided, run e.g.:\n"
          f"  python src/evaluate_paper.py --config {chosen['config']}"
          + (" --smoke" if ctx.smoke else ""))


if __name__ == "__main__":
    main()
