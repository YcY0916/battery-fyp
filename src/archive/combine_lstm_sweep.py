"""Merge the original 18 configs and the 4 extended configs into one 22-config table (no retraining).

The validation errors of the original 18 configs (results/lstm_sweep, models/lstm_sweep) were computed on the points from in-sequence row 119;
the 4 extended configs (results/lstm_sweep_ext, models/lstm_sweep_ext) include W=240 and must be compared from row 239.
This script loads all 22 saved models and recomputes, on the validation set only and with one evaluation start (default 239),
the validation RMSE / MAE and the RMSE per temperature, then merges and sorts them.

- Reads the validation set only; the test-set guard is the same as in sweep_lstm.py (reading a test file raises an error).
- Changes no file in results/lstm_sweep/; the old 18-config table is kept as it is.
- Also recomputes each config at the evaluation start it was trained with and compares with the recorded validation RMSE, to check that the models were loaded correctly.

Usage (from the repository root):
    python src/combine_lstm_sweep.py
    python src/combine_lstm_sweep.py --smoke     # merge the two smoke folders, only checks that the code runs

Outputs:
    results/lstm_sweep_ext/sweep_results_22.csv   summary of the 22 configs, sorted by validation RMSE
    results/lstm_sweep_ext/combine_log.txt        run log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
from lstm_model import LSTMSOC  # noqa: E402
from sweep_lstm import (DEFAULT_NAME, EXT_GRID_H, EXT_GRID_L, EXT_GRID_W,  # noqa: E402
                        EXT_NAME, EXT_SMOKE_NAME, GRID_H, GRID_L, GRID_W, SMOKE_NAME,
                        Logger, config_name, metrics_pct, pick_device, predict)

ROOT = D.ROOT
OUT_FILE = "sweep_results_22.csv"
CHECK_TOL_PCT = 0.01  # warn when the recomputed validation RMSE differs from the recorded one by more than this


def per_temperature(pred, y, ds):
    out = {}
    folder_of = np.array([s.info.folder for s in ds.seqs])[ds.seq_of]
    for folder in D.TEMP_FOLDERS:
        m = folder_of == folder
        if m.any():
            out[folder] = metrics_pct(pred[m], y[m])[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge the 18 + 4 LSTM configs (no retraining, validation errors recomputed only)")
    p.add_argument("--eval-start", type=int, default=D.EVAL_START_INDEX,
                   help=f"common validation evaluation start (in-sequence row index), default {D.EVAL_START_INDEX}")
    p.add_argument("--smoke", action="store_true",
                   help=f"merge {SMOKE_NAME} and {EXT_SMOKE_NAME} (whatever configs exist), only checks the code")
    p.add_argument("--allow-missing", action="store_true",
                   help="write the table even when some configs have not finished (only finished configs)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    base_name, ext_name = (SMOKE_NAME, EXT_SMOKE_NAME) if args.smoke else (DEFAULT_NAME, EXT_NAME)
    sources = {
        base_name: [(w, h, l) for w in GRID_W for h in GRID_H for l in GRID_L],
        ext_name: [(w, h, l) for w in EXT_GRID_W for h in EXT_GRID_H for l in EXT_GRID_L],
    }
    out_dir = ROOT / "results" / ext_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_FILE
    assert out_path.resolve() != (ROOT / "results" / base_name / "sweep_results.csv").resolve()
    log = Logger(out_dir / "combine_log.txt")
    D.block_test_files()  # switch on the test-set guard first
    device = pick_device(args.device)
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  merging {base_name} + {ext_name}, "
        f"common evaluation start {args.eval_start}, device {device}")

    # ---- find the finished configs ----
    found, missing = [], []
    for name, configs in sources.items():
        for w, h, l in configs:
            cname = config_name(w, h, l)
            run_json = ROOT / "results" / name / "runs" / f"{cname}.json"
            ckpt = ROOT / "models" / name / f"{cname}.pt"
            if run_json.exists() and ckpt.exists():
                found.append((name, cname, run_json, ckpt))
            elif not args.smoke:
                missing.append(f"{name}/{cname}")
    if missing and not args.allow_missing:
        raise SystemExit("These configs have not finished (or their result JSON / weights are missing):\n  " + "\n  ".join(missing)
                         + "\nFinish them first, or add --allow-missing to merge only the finished configs")
    if not found:
        raise SystemExit("No finished config found")
    log(f"Found {len(found)} configs" + (f", {len(missing)} missing: {missing}" if missing else ""))

    runs = {}
    for name, cname, run_json, ckpt_path in found:
        with open(run_json, "r", encoding="utf-8") as f:
            r = json.load(f)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        runs[(name, cname)] = (r, ckpt)

    # ---- consistency: same settings apart from the evaluation start; same normalisation; the start covers the longest window ----
    ref_settings = {k: v for k, v in next(iter(runs.values()))[0]["settings"].items()
                    if k != "eval_start_index"}
    ref_stats = next(iter(runs.values()))[1]["norm_stats"]
    for (name, cname), (r, ckpt) in runs.items():
        s = {k: v for k, v in r["settings"].items() if k != "eval_start_index"}
        if s != ref_settings:
            raise SystemExit(f"{name}/{cname} has different training settings: {s} vs {ref_settings}")
        if list(ckpt["features"]) != D.FEATURES:
            raise SystemExit(f"{name}/{cname} has input features {ckpt['features']}")
        st = ckpt["norm_stats"]
        if not (np.allclose(st["mean"], ref_stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(st["std"], ref_stats["std"], rtol=1e-9, atol=0)):
            raise SystemExit(f"{name}/{cname} has different normalisation stats")
    for name in sources:
        sp = ROOT / "results" / name / "norm_stats.json"
        if sp.exists():
            st = D.load_norm_stats(sp)
            if not (np.allclose(st["mean"], ref_stats["mean"], rtol=1e-9, atol=0)
                    and np.allclose(st["std"], ref_stats["std"], rtol=1e-9, atol=0)):
                raise SystemExit(f"{sp.relative_to(ROOT)} differs from the normalisation stats in the model")
    for (name, cname), (r, _) in runs.items():
        if name == ext_name and r["settings"]["eval_start_index"] != args.eval_start:
            raise SystemExit(f"{name}/{cname} was trained with evaluation start {r['settings']['eval_start_index']}, "
                             f"not {args.eval_start}")
    max_w = max(r["window"] for r, _ in runs.values())
    if args.eval_start < max_w - 1:
        raise SystemExit(f"--eval-start {args.eval_start} is below {max_w - 1} (longest window {max_w})")
    log("Consistency check passed: training settings (apart from the evaluation start) and normalisation stats are all identical")

    # ---- read the validation set only ----
    folders = ref_settings["folders"]
    splits = D.build_splits(folders)
    try:
        D.read_raw(splits["test"][0])
        raise AssertionError("test-set guard is not active")
    except D.TestSetAccessError:
        log(f"Test-set guard on: the {len(splits['test'])} test files cannot be read (self-check passed)")
    log("Loading validation set ...")
    val_seqs = D.load_sequences(splits["val"], log)

    # ---- recompute config by config ----
    rows = []
    quiet = lambda *_: None  # noqa: E731
    for (name, cname), (r, ckpt) in runs.items():
        cfg = ckpt["config"]
        model = LSTMSOC(cfg["hidden_size"], cfg["num_layers"], cfg["input_size"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        w = cfg["window"]

        ds = D.WindowDataset(val_seqs, ref_stats, w, args.eval_start, quiet).to(device)
        y = ds.y[ds.ends].cpu().numpy()
        pred = predict(model, ds).cpu().numpy()
        rmse, mae = metrics_pct(pred, y)
        per_t = per_temperature(pred, y, ds)

        trained_start = r["settings"]["eval_start_index"]
        if trained_start == args.eval_start:
            check_rmse = rmse
        else:
            ds0 = D.WindowDataset(val_seqs, ref_stats, w, trained_start, quiet).to(device)
            check_rmse = metrics_pct(predict(model, ds0).cpu().numpy(), ds0.y[ds0.ends].cpu().numpy())[0]
        diff = check_rmse - r["val_rmse_pct"]
        flag = "" if abs(diff) <= CHECK_TOL_PCT else "  <-- differs from the recorded value, please check"
        log(f"  {name}/{cname}: validation RMSE from row {args.eval_start} {rmse:.3f}%  MAE {mae:.3f}%  "
            f"| recomputed from row {trained_start} {check_rmse:.3f}% vs recorded {r['val_rmse_pct']:.3f}%{flag}")

        row = {"config": cname, "window": w, "hidden_size": cfg["hidden_size"],
               "num_layers": cfg["num_layers"], "sweep": name,
               "val_rmse_pct": rmse, "val_mae_pct": mae}
        for folder in folders:
            row[f"val_rmse_pct_{folder}"] = per_t.get(folder)
        row.update({
            "early_stop_eval_start": trained_start,
            "val_rmse_pct_recorded": r["val_rmse_pct"],
            "recheck_diff_pct": diff,
            "best_epoch": r["best_epoch"], "epochs_trained": r["epochs_trained"],
            "stopped_early": r["stopped_early"], "train_time_s": r["train_time_s"],
            "n_params": r["n_params"], "n_train_windows": r["n_train_windows"],
            "n_val_points": len(ds), "device_name": r["device_name"],
        })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("val_rmse_pct", kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(out_path, index=False, float_format="%.4f")

    show = ["rank", "config", "val_rmse_pct", "val_mae_pct", "val_rmse_pct_recorded",
            "early_stop_eval_start", "best_epoch", "epochs_trained", "n_params"]
    log(f"\nSummary of {len(df)} configs (validation points: in-sequence row {args.eval_start} onwards, {df['n_val_points'].iloc[0]:,} points; in %):")
    log(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\nNote: val_rmse_pct_recorded is the validation RMSE recorded during training (evaluation start in early_stop_eval_start); "
        "it cannot be compared with val_rmse_pct directly.")
    log(f"Wrote {out_path.relative_to(ROOT).as_posix()} ({base_name}/sweep_results.csv unchanged)")
    log.close()


if __name__ == "__main__":
    main()
