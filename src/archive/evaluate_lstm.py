"""Evaluate one chosen LSTM config on the test set.

Do not run this script during the hyperparameter comparison. Run it once by hand after the config has been chosen
from results/lstm_sweep_ext/sweep_results_22.csv (validation set only), for example:
    python src/evaluate_lstm.py --window 240 --hidden 128 --layers 1

The weights are looked up in models/lstm_sweep_ext/ and models/lstm_sweep/ (or given with --name).
By default only points with in-sequence row index >= 239 are scored (as in the extended validation set); --eval-start changes this.

Outputs in results/<name>/test_eval/<config>_start<evaluation start>/:
    rmse_table.csv / mae_table.csv   RMSE / MAE per temperature x cycle (SOC percentage points)
    per_file_metrics.csv             metrics of every test file
    summary.json                     RMSE / MAE over all test points together, and the evaluation settings
    predictions/<file>.csv           true and predicted SOC of every test file, second by second
    plots/<file>.png                 "true SOC vs predicted SOC" curve of every test file
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
from sweep_lstm import (COLOR_TRAIN, COLOR_VAL, DEFAULT_NAME, EXT_NAME,  # noqa: E402
                        config_name, metrics_pct, pick_device, predict)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = D.ROOT
TEMP_LABEL = {"n20degC": "-20degC", "n10degC": "-10degC", "0degC": "0degC",
              "10degC": "10degC", "25degC": "25degC", "40degC": "40degC"}


def plot_file(sec, y_true, y_pred, title, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=120, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    ax1.plot(sec, y_true * 100, color=COLOR_TRAIN, lw=2, label="true SOC")
    ax1.plot(sec, y_pred * 100, color=COLOR_VAL, lw=1.2, label="predicted SOC")
    ax1.set_ylabel("SOC (%)")
    ax1.set_title(title, fontsize=11)
    ax1.legend(frameon=False, loc="upper right")
    ax2.plot(sec, (y_pred - y_true) * 100, color="0.35", lw=1)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_ylabel("error (pp)")
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        ax.grid(True, color="0.9", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the chosen LSTM config on the test set")
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--name", default=None,
                   help=f"output name of the comparison (models/<name>/); default: look in {EXT_NAME}, then {DEFAULT_NAME}")
    p.add_argument("--eval-start", type=int, default=D.EVAL_START_INDEX,
                   help=f"score only points with in-sequence row index >= this value (default {D.EVAL_START_INDEX}, as in the extended validation set)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    cname = config_name(args.window, args.hidden, args.layers)
    names = [args.name] if args.name else [EXT_NAME, DEFAULT_NAME]
    found = [n for n in names if (ROOT / "models" / n / f"{cname}.pt").exists()]
    if not found:
        raise SystemExit(f"{cname}.pt not found in models/{{{','.join(names)}}}/; train this config with sweep_lstm.py first")
    if len(found) > 1:
        raise SystemExit(f"{cname}.pt exists in both {found}; choose one with --name")
    args.name = found[0]
    ckpt_path = ROOT / "models" / args.name / f"{cname}.pt"
    device = pick_device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg, stats, settings = ckpt["config"], ckpt["norm_stats"], ckpt["settings"]
    if list(ckpt["features"]) != D.FEATURES:
        raise SystemExit(f"input features of the weights {ckpt['features']} differ from {D.FEATURES}")
    stats_path = ROOT / "results" / args.name / "norm_stats.json"
    if stats_path.exists():
        saved = D.load_norm_stats(stats_path)
        if not (np.allclose(saved["mean"], stats["mean"]) and np.allclose(saved["std"], stats["std"])):
            raise SystemExit(f"the normalisation stats in the weights differ from {stats_path.relative_to(ROOT)}")

    model = LSTMSOC(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                    input_size=cfg["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    window = cfg["window"]
    eval_start = max(args.eval_start, window - 1)

    # the folder name includes the evaluation start, so re-evaluating with another start never overwrites earlier results
    out = ROOT / "results" / args.name / "test_eval" / f"{cname}_start{eval_start}"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    folders = settings["folders"]
    splits = D.build_splits(folders)
    print(f"Config {cname} ({args.name}, best epoch {ckpt['best_epoch']}, validation RMSE during training "
          f"{ckpt['val_rmse_pct']:.3f}%, evaluation start {settings['eval_start_index']}), "
          f"device {device}, scoring rows >= {eval_start}")
    print(f"Reading the {len(splits['test'])} test files ...")
    test_seqs = D.load_sequences(splits["test"])

    rows, all_err = [], []
    for seq in test_seqs:
        info = seq.info
        ds = D.WindowDataset([seq], stats, window, eval_start).to(device)
        pred = predict(model, ds).cpu().numpy().astype(np.float64)
        loc = ds.local_index
        y_true, sec = seq.y[loc], seq.sec[loc]
        rmse, mae = metrics_pct(pred, y_true)
        all_err.append((pred - y_true) * 100)
        rows.append({"file": info.id, "temperature": TEMP_LABEL[info.folder], "temp_c": info.temp_c,
                     "cycle": info.test_name, "n_points": len(loc),
                     "rmse_pct": rmse, "mae_pct": mae,
                     "max_abs_err_pct": float(np.max(np.abs(pred - y_true)) * 100)})
        stem = info.id.replace("/", "_")
        pd.DataFrame({"time_s": sec, "soc_true": y_true, "soc_pred": pred}).to_csv(
            out / "predictions" / f"{stem}.csv", index=False, float_format="%.6f")
        plot_file(sec, y_true, pred,
                  f"{TEMP_LABEL[info.folder]}  {info.test_name}  ({info.id.split('/')[1]})   "
                  f"{cname}   RMSE {rmse:.2f}%  MAE {mae:.2f}%",
                  out / "plots" / f"{stem}.png")
        print(f"  {info.id:24s} RMSE {rmse:6.3f}%  MAE {mae:6.3f}%")

    per_file = pd.DataFrame(rows)
    per_file.to_csv(out / "per_file_metrics.csv", index=False, float_format="%.4f")

    # temperature x cycle table; the "all" row / column recomputes the metric over the pooled test points (not an average of cells)
    temps = [TEMP_LABEL[f] for f in D.TEMP_FOLDERS if f in folders]
    err_by_file = dict(zip(per_file["file"], all_err))
    tables = {}
    for metric in ("rmse", "mae"):
        def agg(files):
            if len(files) == 0:
                return np.nan
            e = np.concatenate([err_by_file[f] for f in files])
            return float(np.sqrt(np.mean(e ** 2))) if metric == "rmse" else float(np.mean(np.abs(e)))
        tab = pd.DataFrame(index=temps + ["all"], columns=D.DRIVE_CYCLES + ["all"], dtype=float)
        for t in temps + ["all"]:
            for c in D.DRIVE_CYCLES + ["all"]:
                m = np.ones(len(per_file), bool)
                if t != "all":
                    m &= (per_file["temperature"] == t).to_numpy()
                if c != "all":
                    m &= (per_file["cycle"] == c).to_numpy()
                tab.loc[t, c] = agg(per_file.loc[m, "file"].tolist())
        tab.index.name = "temperature"
        tab.to_csv(out / f"{metric}_table.csv", float_format="%.4f")
        tables[metric] = tab
        print(f"\nTest {metric.upper()} (SOC percentage points; '-' means no such file):")
        print(tab.to_string(float_format=lambda v: f"{v:.3f}", na_rep="-"))

    e = np.concatenate(all_err)
    summary = {"config": cname, "model_file": ckpt_path.relative_to(ROOT).as_posix(),
               "best_epoch": ckpt["best_epoch"], "val_rmse_pct": ckpt["val_rmse_pct"],
               "val_eval_start_index": settings["eval_start_index"],
               "test_rmse_pct": float(np.sqrt(np.mean(e ** 2))),
               "test_mae_pct": float(np.mean(np.abs(e))),
               "n_test_points": int(len(e)), "n_test_files": len(per_file),
               "eval_start_index": eval_start, "device": str(device),
               "evaluated_at": datetime.now().isoformat(timespec="seconds")}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll test points: RMSE {summary['test_rmse_pct']:.3f}%, MAE {summary['test_mae_pct']:.3f}%"
          f" ({summary['n_test_points']:,} points)")
    print(f"Results written to {out.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
