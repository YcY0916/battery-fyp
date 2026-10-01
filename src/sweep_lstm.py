"""LSTM hyperparameter sweep of the old 6-temperature experiment.

- Base grid (18 configs): W x H x L = {30, 60, 120} x {32, 64, 128} x {1, 2}, output name lstm_sweep
- Extended grid (4 configs, --ext): W x H x L = {180, 240} x {32, 128} x {1}, output name lstm_sweep_ext

Trains on the training set and selects on the validation set only; the test set is never read
(reading a test file raises an error).
Resumable: finished configs (result JSON in runs/ and weights in models/) are skipped.

Validation error and early stopping only use points with in-sequence index >= --eval-start
(default 239, the first point the longest window 240 can cover). The original 18 configs were
run with 119; to resume those 18 in the lstm_sweep folder you must pass --eval-start 119,
otherwise the script stops because the settings differ.

fit() and val_metrics() are also the training loop and validation metrics of the 0-40 degC
paper pipeline (src/sweep_paper.py).

Usage (from the repository root):
    python src/sweep_lstm.py --ext              # the 4 extended configs (evaluation start 239)
    python src/sweep_lstm.py --ext --smoke      # 25 degC only, W180/W240, 2 epochs each
    python src/sweep_lstm.py --eval-start 119   # the original 18 configs (same settings as back then)
    python src/sweep_lstm.py --smoke            # 25 degC only, 2 configs, 2 epochs each

The merged 18 + 4 table was made by src/archive/combine_lstm_sweep.py (archived).

Outputs (default --name lstm_sweep):
    results/lstm_sweep/sweep_results.csv         summary, sorted by validation RMSE
    results/lstm_sweep/norm_stats.json           normalisation stats (training set only)
    results/lstm_sweep/sweep_log.txt             run log
    results/lstm_sweep/runs/<config>.json        full result of each config (also the "finished" marker)
    results/lstm_sweep/runs/<config>_history.csv loss of every epoch
    results/lstm_sweep/curves/<config>.png       training curves
    models/lstm_sweep/<config>.pt                weights of the epoch with the lowest validation loss
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
from lstm_model import LSTMSOC, count_parameters  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = D.ROOT

GRID_W = [30, 60, 120]
GRID_H = [32, 64, 128]
GRID_L = [1, 2]
# Extended grid: the best config was at the W=120 edge, so longer windows were added
EXT_GRID_W = [180, 240]
EXT_GRID_H = [32, 128]
EXT_GRID_L = [1]

SEED = 42
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15
EVAL_BATCH_SIZE = 2048  # only affects memory use during evaluation, not the results

DEFAULT_NAME = "lstm_sweep"
EXT_NAME = "lstm_sweep_ext"
SMOKE_NAME = "lstm_sweep_smoke"
EXT_SMOKE_NAME = "lstm_sweep_ext_smoke"
SMOKE_FOLDERS = ["25degC"]
SMOKE_CONFIGS = [(30, 32, 1), (60, 64, 2)]
EXT_SMOKE_CONFIGS = [(180, 32, 1), (240, 32, 1)]
SMOKE_EPOCHS = 2

COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"


# ---------------------------------------------------------------------------
class Logger:
    """Print to the screen and append to a log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def config_name(w: int, h: int, l: int) -> str:
    return f"W{w}_H{h}_L{l}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was given, but torch.cuda.is_available() is False")
    return torch.device(arg)


def write_json(path: Path, obj) -> None:
    """Write a temporary file, then rename it, so an interruption never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


@torch.no_grad()
def predict(model: torch.nn.Module, ds: D.WindowDataset) -> torch.Tensor:
    model.eval()
    return torch.cat([model(xb) for xb, _ in ds.iter_batches(EVAL_BATCH_SIZE)])


def metrics_pct(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """RMSE and MAE in SOC percentage points (error x 100)."""
    err = (pred.astype(np.float64) - y.astype(np.float64)) * 100.0
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


# ---------------------------------------------------------------------------
def fit(model: torch.nn.Module, train_ds: D.WindowDataset, val_ds: D.WindowDataset, name: str,
        max_epochs: int, device, log, seed: int = SEED, lr: float = LR,
        batch_size: int = BATCH_SIZE, patience: int = PATIENCE) -> dict:
    """Generic training loop (any model mapping (batch, W, 3) -> (batch,)).

    Adam + MSE; the training windows are shuffled every epoch (with a separate Generator(seed)).
    The validation loss is computed after every epoch; training stops when it has not reached a
    new minimum for `patience` epochs. Early stopping only looks at val_ds, never at the test set.
    On return the model holds the weights of the epoch with the lowest validation loss.
    """
    y_val = val_ds.y[val_ds.ends]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)  # shuffles the training windows every epoch

    best_loss, best_epoch, best_state, bad = math.inf, 0, None, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        te = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for xb, yb in train_ds.iter_batches(batch_size, shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += loss.detach().double() * len(yb)
        train_loss = float(loss_sum) / len(train_ds)
        val_loss = float(torch.mean((predict(model, val_ds) - y_val).double() ** 2))
        epoch_time = time.perf_counter() - te

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_rmse_pct": math.sqrt(val_loss) * 100, "epoch_time_s": epoch_time})
        log(f"  [{name}] epoch {epoch:3d}  train {train_loss:.3e}  val {val_loss:.3e}  "
            f"val RMSE {math.sqrt(val_loss) * 100:6.3f}%  {epoch_time:6.1f}s"
            + ("  *" if improved else f"  ({bad}/{patience})"))
        if bad >= patience:
            log(f"  [{name}] no new best validation loss for {patience} epochs, stopping")
            break
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)
    return {"best_state": best_state, "best_epoch": best_epoch, "best_loss": best_loss,
            "history": history, "train_time": train_time}


def val_metrics(model: torch.nn.Module, val_ds: D.WindowDataset) -> tuple[float, float, dict]:
    """Validation RMSE and MAE (SOC percentage points), overall and per temperature."""
    pred = predict(model, val_ds).cpu().numpy()
    y = val_ds.y[val_ds.ends].cpu().numpy()
    rmse, mae = metrics_pct(pred, y)
    per_temp = {}
    for folder in D.TEMP_FOLDERS:
        mask = np.array([val_ds.seqs[k].info.folder == folder for k in val_ds.seq_of])
        if mask.any():
            r, m = metrics_pct(pred[mask], y[mask])
            per_temp[folder] = {"rmse_pct": r, "mae_pct": m, "n_points": int(mask.sum())}
    return rmse, mae, per_temp


def train_one(w, h, l, train_seqs, val_seqs, stats, settings, dirs, device, log) -> dict:
    name = config_name(w, h, l)
    set_seed(SEED)
    train_ds = D.WindowDataset(train_seqs, stats, w, None, log).to(device)
    val_ds = D.WindowDataset(val_seqs, stats, w, settings["eval_start_index"], log).to(device)
    model = LSTMSOC(hidden_size=h, num_layers=l).to(device)
    n_params = count_parameters(model)
    log(f"  train windows {len(train_ds):,}, validation points {len(val_ds):,}, parameters {n_params:,}")

    max_epochs = settings["max_epochs"]
    fitted = fit(model, train_ds, val_ds, name, max_epochs, device, log)
    best_state, best_epoch, best_loss = fitted["best_state"], fitted["best_epoch"], fitted["best_loss"]
    history, train_time = fitted["history"], fitted["train_time"]
    epochs_trained = len(history)

    # Validation metrics with the best-epoch weights (fit returns with them loaded)
    rmse, mae, per_temp = val_metrics(model, val_ds)

    # Save the weights with everything needed to rebuild and evaluate the model
    ckpt_path = dirs["models"] / f"{name}.pt"
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": best_state,
                "config": {"window": w, "hidden_size": h, "num_layers": l, "input_size": 3},
                "features": D.FEATURES, "norm_stats": stats, "settings": settings,
                "best_epoch": best_epoch, "val_rmse_pct": rmse}, tmp)
    tmp.replace(ckpt_path)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(dirs["runs"] / f"{name}_history.csv", index=False)
    plot_curve(hist_df, best_epoch, name, dirs["curves"] / f"{name}.png")

    result = {
        "config": name, "window": w, "hidden_size": h, "num_layers": l,
        "val_rmse_pct": rmse, "val_mae_pct": mae,
        "val_per_temperature": per_temp,
        "best_epoch": best_epoch, "epochs_trained": epochs_trained,
        "stopped_early": epochs_trained < max_epochs,
        "best_val_loss": best_loss,
        "train_time_s": train_time, "sec_per_epoch": train_time / epochs_trained,
        "n_params": n_params, "n_train_windows": len(train_ds), "n_val_points": len(val_ds),
        "skipped_short_train_sequences": [s for s, _ in train_ds.skipped],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": ckpt_path.relative_to(ROOT).as_posix(),
        "settings": settings,
    }
    # The result JSON is written last: if it exists, this config has finished
    write_json(dirs["runs"] / f"{name}.json", result)
    log(f"  [{name}] done: best epoch {best_epoch}/{epochs_trained}, val RMSE {rmse:.3f}%, "
        f"MAE {mae:.3f}%, {train_time / 60:.1f} min")
    return result


def plot_curve(hist: pd.DataFrame, best_epoch: int, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    ax.plot(hist["epoch"], hist["train_loss"], color=COLOR_TRAIN, lw=2, label="train MSE")
    ax.plot(hist["epoch"], hist["val_loss"], color=COLOR_VAL, lw=2, label="val MSE")
    ax.axvline(best_epoch, color="0.45", lw=1, ls="--")
    best_val = float(hist.loc[hist["epoch"] == best_epoch, "val_loss"].iloc[0])
    right = best_epoch > (hist["epoch"].min() + hist["epoch"].max()) / 2
    ax.annotate(f"best epoch {best_epoch}\nval RMSE {math.sqrt(best_val) * 100:.2f}%",
                xy=(best_epoch, best_val), xytext=(-8 if right else 8, 14),
                textcoords="offset points", ha="right" if right else "left",
                fontsize=9, color="0.2")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (SOC as 0-1)")
    ax.set_title(name.replace("_", "  "), fontsize=11)
    ax.grid(True, which="major", color="0.9", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(dirs, folders, log) -> pd.DataFrame | None:
    rows = []
    for p in sorted(dirs["runs"].glob("W*_H*_L*.json")):
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        row = {k: r[k] for k in ("config", "window", "hidden_size", "num_layers",
                                  "val_rmse_pct", "val_mae_pct")}
        for folder in folders:
            row[f"val_rmse_pct_{folder}"] = r["val_per_temperature"].get(folder, {}).get("rmse_pct")
        row.update({k: r[k] for k in ("best_epoch", "epochs_trained", "stopped_early",
                                       "train_time_s", "n_params", "n_train_windows",
                                       "n_val_points", "device_name")})
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("val_rmse_pct", kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(dirs["results"] / "sweep_results.csv", index=False, float_format="%.4f")
    return df


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="LSTM hyperparameter sweep (old 6-temperature experiment)")
    p.add_argument("--ext", action="store_true",
                   help=f"run the extended grid W{EXT_GRID_W} x H{EXT_GRID_H} x L{EXT_GRID_L}, output name {EXT_NAME}")
    p.add_argument("--smoke", action="store_true",
                   help="quick check: 25 degC only, 2 configs, 2 epochs each, output to *_smoke folders")
    p.add_argument("--windows", nargs="+", type=int, default=None, help="only these W")
    p.add_argument("--hidden", nargs="+", type=int, default=None, help="only these H")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="only these L")
    p.add_argument("--eval-start", type=int, default=D.EVAL_START_INDEX,
                   help="evaluation start (in-sequence row index) of the validation error and early "
                        f"stopping, default {D.EVAL_START_INDEX}; the original 18 configs used {D.EVAL_START_INDEX_ORIGINAL}")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="debugging: override the maximum number of epochs (needs --name with another output name)")
    p.add_argument("--name", default=None,
                   help=f"output folder name, default {DEFAULT_NAME} ({EXT_NAME} with --ext)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    grid_w, grid_h, grid_l = (EXT_GRID_W, EXT_GRID_H, EXT_GRID_L) if args.ext else (GRID_W, GRID_H, GRID_L)
    formal_name = EXT_NAME if args.ext else DEFAULT_NAME
    if args.smoke:
        folders, max_epochs = SMOKE_FOLDERS, SMOKE_EPOCHS
        configs = EXT_SMOKE_CONFIGS if args.ext else SMOKE_CONFIGS
        name = args.name or (EXT_SMOKE_NAME if args.ext else SMOKE_NAME)
    else:
        folders, max_epochs = list(D.TEMP_FOLDERS), args.max_epochs or MAX_EPOCHS
        ws, hs, ls = args.windows or grid_w, args.hidden or grid_h, args.layers or grid_l
        bad = [v for v in ws if v not in grid_w] + \
              [v for v in hs if v not in grid_h] + [v for v in ls if v not in grid_l]
        if bad:
            raise SystemExit(f"values not in the {'extended' if args.ext else 'base'} grid: {bad}"
                             + ("" if args.ext else " (W=180, 240 need --ext)"))
        configs = [(w, h, l) for w in ws for h in hs for l in ls]
        name = args.name or formal_name
    if name in (DEFAULT_NAME, EXT_NAME) and max_epochs != MAX_EPOCHS:
        raise SystemExit("--max-epochs needs --name with another output folder, so that it does not mix with the real results")
    max_w = max(c[0] for c in configs)
    if args.eval_start < max_w - 1:
        raise SystemExit(f"--eval-start {args.eval_start} is below W-1 of the longest window {max_w}, i.e. {max_w - 1}; "
                         "the window lengths would not be compared on the same points")

    dirs = {"results": ROOT / "results" / name, "runs": ROOT / "results" / name / "runs",
            "curves": ROOT / "results" / name / "curves", "models": ROOT / "models" / name}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log = Logger(dirs["results"] / "sweep_log.txt")

    settings = {"folders": folders, "features": D.FEATURES, "target": D.TARGET,
                "resample": "1 s, floor(time_s); V/I/T mean, soc last",
                "eval_start_index": args.eval_start, "seed": SEED, "lr": LR,
                "batch_size": BATCH_SIZE, "max_epochs": max_epochs, "patience": PATIENCE,
                "optimizer": "Adam", "loss": "MSE", "dropout": 0.0}

    device = pick_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  sweep '{name}'  smoke={args.smoke}")
    log(f"torch {torch.__version__}, device {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else
           f" (CPU threads {torch.get_num_threads()})"))
    log(f"{len(configs)} configs: {', '.join(config_name(*c) for c in configs)}")
    log(f"Evaluation start of the validation error and early stopping: in-sequence row {args.eval_start} onwards")

    # ---- split, and switch on the test-set guard ----
    D.block_test_files()
    splits = D.build_splits(folders)
    try:  # self-check: the guard really works
        D.read_raw(splits["test"][0])
        raise AssertionError("test-set guard is not active")
    except D.TestSetAccessError:
        log(f"Test-set guard on: the {len(splits['test'])} test files cannot be read (self-check passed)")
    for split in ("train", "val"):
        log(f"{split} ({len(splits[split])} sequences): " + ", ".join(s.id for s in splits[split]))

    log("Loading training set ...")
    train_seqs = D.load_sequences(splits["train"], log)
    log("Loading validation set ...")
    val_seqs = D.load_sequences(splits["val"], log)

    stats = D.compute_norm_stats(train_seqs)
    stats_path = dirs["results"] / "norm_stats.json"
    if stats_path.exists():
        old = D.load_norm_stats(stats_path)
        if old["computed_from"] != stats["computed_from"] or not (
                np.allclose(old["mean"], stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(old["std"], stats["std"], rtol=1e-9, atol=0)):
            raise SystemExit(f"{stats_path} differs from the normalisation stats computed now (did the data change?); "
                             "delete the old results in this folder and rerun")
    else:
        D.save_norm_stats(stats, stats_path)
    log("Normalisation stats (training set): " + ", ".join(
        f"{f} mean={m:.4f} std={s:.4f}" for f, m, s in zip(D.FEATURES, stats["mean"], stats["std"])))

    # ---- before training, check that the results already in this folder used the same settings ----
    for run_json in sorted(dirs["runs"].glob("W*_H*_L*.json")):
        with open(run_json, "r", encoding="utf-8") as f:
            done_settings = json.load(f)["settings"]
        if done_settings != settings:
            diff = {k: (done_settings.get(k), settings.get(k))
                    for k in set(done_settings) | set(settings)
                    if done_settings.get(k) != settings.get(k)}
            hint = ""
            if set(diff) == {"eval_start_index"}:
                hint = (f". The results in {name} were run with --eval-start {done_settings['eval_start_index']}"
                        "; add this option to resume")
            raise SystemExit(f"{run_json.relative_to(ROOT).as_posix()} was run with different settings"
                             f" (existing vs now: {diff}){hint}")

    # ---- train config by config, skipping finished ones ----
    for i, (w, h, l) in enumerate(configs, 1):
        cname = config_name(w, h, l)
        run_json = dirs["runs"] / f"{cname}.json"
        if run_json.exists() and (dirs["models"] / f"{cname}.pt").exists():
            with open(run_json, "r", encoding="utf-8") as f:
                done = json.load(f)
            log(f"[{i}/{len(configs)}] {cname} already finished (val RMSE {done['val_rmse_pct']:.3f}%), skipped")
            continue
        log(f"[{i}/{len(configs)}] {cname} training")
        train_one(w, h, l, train_seqs, val_seqs, stats, settings, dirs, device, log)
        write_summary(dirs, folders, log)

    df = write_summary(dirs, folders, log)
    if df is not None:
        show = ["rank", "config", "val_rmse_pct", "val_mae_pct", "best_epoch",
                "epochs_trained", "train_time_s", "n_params"]
        log("\nSummary (sorted by validation RMSE, in %):")
        log(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        log(f"\nWrote {(dirs['results'] / 'sweep_results.csv').relative_to(ROOT).as_posix()}")
    log.close()


if __name__ == "__main__":
    main()
