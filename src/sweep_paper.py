"""Steps 2 and 3 of the 0-40 degC paper experiment: LSTM hyperparameter sweep and multi-seed runs.

Step 2 (--stage sweep, default): 18 configs, seed 42
    W in {60, 120, 180, 240, 300} x H in {32, 64, 128}, L = 1   (15 configs)
    W = 240, L = 2, H in {32, 64, 128}                         (3 configs)
Step 3 (--stage multiseed): every candidate in candidates.json (written by merge_results.py
    after step 2) with seeds 43-46; seed 42 is never retrained, it is reused from step 2.

Training is unchanged from the old sweep (sweep_lstm.fit): MSE loss, Adam lr 0.001, batch 256,
dropout 0, at most 200 epochs, stop after 15 epochs without a new best validation loss, keep
the weights of the best epoch. Validation metrics and the early-stopping loss use only rows
with in-sequence index >= 299 (300 s), so every W is scored on the same points.

The test set is never read: the test-set guard is switched on before any data is loaded and
any attempt to read a test file stops the script.

Several machines (--shard k/n): a config goes to shard sha256(config name) % n + 1, so a config
and all of its seeds always run on the same machine in step 2 and step 3. --configs picks
configs by hand and overrides --shard. Each run (config + seed) runs from start to end on one
machine; finished runs (their JSON exists) are skipped, an interrupted run restarts from epoch 1.

Usage (from the repository root):
    python src/sweep_paper.py --shard 1/2                     # step 2, this machine's half
    python src/sweep_paper.py --stage multiseed --shard 1/2   # step 3 (after merge_results.py)
    python src/sweep_paper.py --dry_run --shard 1/2           # only list the runs of this shard
    python src/sweep_paper.py --init_norm                     # only create / check norm_stats.json
    python src/sweep_paper.py --timing                        # 1 epoch of W300_H128_L1 + time estimates
    python src/sweep_paper.py --smoke --shard 1/2             # quick check (25 degC, 2 configs, 2 epochs)
    Colab: add --out_root /content/drive/MyDrive/battery-fyp (and --data_root if needed)

Outputs (under --out_root; <run> = <config>_seed<seed>):
    results/<name>/runs/<run>.json            result of one run (also the "finished" marker)
    results/<name>/runs/<run>_history.csv     loss of every epoch
    results/<name>/val_preds/<run>.npz        validation predictions for every point from W-1
    results/<name>/sweep/curves/<run>.png     training curves (seed 42); seeds 43-46 in multiseed/curves/
    results/<name>/norm_stats.json            normalisation stats (training set only; read if present)
    results/<name>/logs/<stage>_<host>.log    log of this machine
    results/<name>/timing/timing_<host>.json  --timing measurements of this machine
    models/<name>/<run>.pt                    weights of the best epoch
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
import paper_common as C  # noqa: E402
import sweep_lstm as SL  # noqa: E402
from lstm_model import LSTMSOC, count_parameters  # noqa: E402

QUIET = lambda *_: None  # noqa: E731

# --timing
OLD_RUN_DIRS = ["results/lstm_sweep/runs", "results/lstm_sweep_ext/runs",
                "archive/2026-09_6temp/results/lstm_sweep/runs",
                "archive/2026-09_6temp/results/lstm_sweep_ext/runs"]
ASSUMED_CANDIDATES = ["W240_H32_L1", "W240_H64_L1", "W240_H128_L1", "W240_H32_L2", "W240_H64_L2"]
TIMING_REF = (300, 128, 1)


@dataclass
class Data:
    train: list
    val: list


# ---------------------------------------------------------------------------
# Which runs this machine does
# ---------------------------------------------------------------------------
def plan_runs(ctx: C.Context, stage: str, configs_arg, shard_arg, log) -> list[tuple[str, int]]:
    grid = ctx.config_names
    candidates = None
    if ctx.paths.candidates.exists():
        candidates = C.read_json(ctx.paths.candidates)["candidates"]

    if stage == "sweep":
        configs, seeds = list(grid), [C.SWEEP_SEED]
    else:
        seeds = [s for s in ctx.seeds if s != C.SWEEP_SEED]
        if configs_arg:
            configs = list(configs_arg)
        elif candidates is None:
            raise SystemExit(
                f"{ctx.paths.rel(ctx.paths.candidates)} not found. Run src/merge_results.py after "
                "step 2 (it writes candidates.json); on the other machine copy that file into "
                f"{ctx.paths.rel(ctx.paths.results)}/ or pass the configs with --configs.")
        else:
            configs = list(candidates)

    if configs_arg:
        bad = [c for c in configs_arg if c not in grid]
        if bad:
            raise SystemExit(f"--configs not in the grid: {bad}; grid: {grid}")
        if stage == "multiseed" and candidates is not None:
            extra = [c for c in configs_arg if c not in candidates]
            if extra:
                log(f"WARNING: --configs {extra} are not multi-seed candidates ({candidates})")
        configs = [c for c in grid if c in configs_arg]
        if shard_arg:
            log("--configs given: --shard is ignored")
    elif shard_arg:
        ks, n = C.parse_shard(shard_arg)
        configs = [c for c in configs if C.shard_index(c, n) in ks]
        log(f"Shard {shard_arg}: {len(configs)} configs -> {', '.join(configs) or '(none)'}")
    return [(c, s) for c in configs for s in seeds]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def train_run(cname: str, seed: int, data: Data, stats: dict, norm_sha: str, settings: dict,
              paths: C.PaperPaths, device: torch.device, env: dict, log) -> dict:
    w, h, l = C.parse_config(cname)
    rname = C.run_name(cname, seed)
    SL.set_seed(seed)
    train_ds = D.WindowDataset(data.train, stats, w, None, log).to(device)
    val_ds = D.WindowDataset(data.val, stats, w, C.EVAL_START, log).to(device)
    model = LSTMSOC(hidden_size=h, num_layers=l).to(device)
    n_params = count_parameters(model)
    log(f"  train windows {len(train_ds):,}, validation points {len(val_ds):,}, "
        f"trainable parameters {n_params:,}")

    fitted = SL.fit(model, train_ds, val_ds, rname, settings["max_epochs"], device, log, seed=seed)
    history, best_epoch = fitted["history"], fitted["best_epoch"]
    epochs_trained, train_time = len(history), fitted["train_time"]
    rmse, mae, per_temp = SL.val_metrics(model, val_ds)  # model holds the best-epoch weights

    for d in (paths.models, paths.runs, paths.val_preds, paths.sweep_curves, paths.multiseed_curves):
        d.mkdir(parents=True, exist_ok=True)
    ckpt_path = paths.model_file(rname)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": fitted["best_state"],
                "config": {"window": w, "hidden_size": h, "num_layers": l, "input_size": 3},
                "config_name": cname, "seed": seed, "features": D.FEATURES,
                "norm_stats": stats, "norm_stats_sha256": norm_sha, "settings": settings,
                "best_epoch": best_epoch, "val_rmse_pct": rmse}, tmp)
    os.replace(tmp, ckpt_path)

    hist_path = paths.runs / f"{rname}_history.csv"
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(hist_path, index=False)
    curve_path = (paths.sweep_curves if seed == C.SWEEP_SEED else paths.multiseed_curves) / f"{rname}.png"
    SL.plot_curve(hist_df, best_epoch, rname, curve_path)
    npz_path = paths.val_preds / f"{rname}.npz"
    C.save_val_predictions(npz_path, model, data.val, stats, w, device)

    result = {
        "run": rname, "config": cname, "window": w, "hidden_size": h, "num_layers": l, "seed": seed,
        "stage": "sweep" if seed == C.SWEEP_SEED else "multiseed",
        "val_rmse_pct": rmse, "val_mae_pct": mae, "val_per_temperature": per_temp,
        "val_eval_start_index": C.EVAL_START,
        "best_epoch": best_epoch, "epochs_trained": epochs_trained,
        "stopped_early": epochs_trained < settings["max_epochs"],
        "best_val_loss": fitted["best_loss"],
        "train_time_s": train_time, "sec_per_epoch": train_time / epochs_trained,
        "n_params": n_params, "n_train_windows": len(train_ds), "n_val_points": len(val_ds),
        "skipped_short_train_sequences": [s for s, _ in train_ds.skipped],
        "environment": env,
        "files": {"model": paths.rel(ckpt_path), "history": paths.rel(hist_path),
                  "curve": paths.rel(curve_path), "val_predictions": paths.rel(npz_path)},
        "settings": settings,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    C.write_json(paths.run_json(rname), result)  # written last: it marks the run as finished
    log(f"  [{rname}] done: best epoch {best_epoch}/{epochs_trained}, val RMSE {rmse:.3f}%, "
        f"MAE {mae:.3f}%, {train_time / 60:.1f} min")
    return result


# ---------------------------------------------------------------------------
# --timing
# ---------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_config(cfg, data, stats, device, steps, warmup) -> dict:
    w, h, l = cfg
    SL.set_seed(C.SWEEP_SEED)
    train_ds = D.WindowDataset(data.train, stats, w, None, QUIET).to(device)
    val_ds = D.WindowDataset(data.val, stats, w, C.EVAL_START, QUIET).to(device)
    model = LSTMSOC(hidden_size=h, num_layers=l).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=SL.LR)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(train_ds), (warmup + steps, SL.BATCH_SIZE), generator=g)
    model.train()
    t0 = time.perf_counter()
    for i in range(warmup + steps):
        if i == warmup:
            _sync(device)
            t0 = time.perf_counter()
        xb, yb = train_ds.get_batch(idx[i])
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()
    _sync(device)
    step_s = (time.perf_counter() - t0) / steps
    SL.predict(model, val_ds)  # warm-up of the evaluation path
    _sync(device)
    t0 = time.perf_counter()
    SL.predict(model, val_ds)
    _sync(device)
    val_s = time.perf_counter() - t0
    n_batches = math.ceil(len(train_ds) / SL.BATCH_SIZE)
    return {"step_s": step_s, "val_s": val_s, "n_batches": n_batches,
            "estimate_s": n_batches * step_s + val_s}


def typical_epochs(out_root: Path, override: float | None) -> tuple[float | None, str]:
    if override is not None:
        return float(override), "--epochs"
    values, seen = [], set()
    for root in dict.fromkeys([out_root.resolve(), C.REPO_ROOT.resolve()]):
        for rel in OLD_RUN_DIRS:
            for p in sorted((root / rel).glob("W*_H*_L*.json")):
                if p.resolve() in seen or not C.CONFIG_RE.match(p.stem):
                    continue
                seen.add(p.resolve())
                try:
                    values.append(int(C.read_json(p)["epochs_trained"]))
                except (KeyError, ValueError, OSError):
                    pass
    if values:
        return float(np.mean(values)), f"mean epochs_trained of {len(values)} old sweep runs"
    return None, "no old sweep runs found in " + ", ".join(OLD_RUN_DIRS)


def fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:6.1f} h"


def shard_loads(per_epoch: dict, cands: list[str], n: int, extra_seeds: int) -> list[dict]:
    rows = []
    for k in range(1, n + 1):
        cfgs = [c for c in per_epoch if C.shard_index(c, n) == k]
        rows.append({"shard": f"{k}/{n}", "configs": cfgs,
                     "step2": sum(per_epoch[c] for c in cfgs),
                     "step3": extra_seeds * sum(per_epoch[c] for c in cfgs if c in cands)})
    return rows


def best_two_machine_plan(a: dict, b: dict, cands: list[str], extra_seeds: int):
    """Choose n and the shards of machine A so that the slower of the two finishes first."""
    best = None
    names = list(a["per_epoch_s"])
    for n in range(2, 7):
        for r in range(1, n):
            for sub in itertools.combinations(range(1, n + 1), r):
                load = {"A": 0.0, "B": 0.0}
                for c in names:
                    who = "A" if C.shard_index(c, n) in sub else "B"
                    t = (a if who == "A" else b)["per_epoch_s"][c]
                    load[who] += t * (1 + (extra_seeds if c in cands else 0))
                span = max(load.values())
                if best is None or span < best[0] - 1e-9:
                    best = (span, n, sub, load)
    return best


def run_timing(ctx: C.Context, data: Data, args, device, log) -> None:
    paths = ctx.paths
    stats, _ = C.load_or_create_norm_stats(paths.norm_stats, data.train, ctx.temps, log, save=False)
    ref = max(ctx.configs) if ctx.smoke else TIMING_REF
    ref_name = SL.config_name(*ref)
    log(f"\n[timing] 1 full epoch of {ref_name} on {C.device_label(device)} ...")
    SL.set_seed(C.SWEEP_SEED)
    train_ds = D.WindowDataset(data.train, stats, ref[0], None, QUIET).to(device)
    val_ds = D.WindowDataset(data.val, stats, ref[0], C.EVAL_START, QUIET).to(device)
    model = LSTMSOC(hidden_size=ref[1], num_layers=ref[2]).to(device)
    fitted = SL.fit(model, train_ds, val_ds, ref_name, 1, device, log)
    ref_epoch = fitted["history"][0]["epoch_time_s"]
    del train_ds, val_ds, model

    steps, warmup = (30, 5) if device.type == "cuda" else (6, 2)
    log(f"[timing] {steps} training steps (+{warmup} warm-up) and 1 validation pass per config ...")
    raw = {}
    for cfg in ctx.configs:
        raw[SL.config_name(*cfg)] = time_config(cfg, data, stats, device, steps, warmup)
    scale = ref_epoch / raw[ref_name]["estimate_s"]
    per_epoch = {c: r["estimate_s"] * scale for c, r in raw.items()}
    log(f"[timing] measured epoch {ref_epoch:.1f} s vs step-based estimate "
        f"{raw[ref_name]['estimate_s']:.1f} s -> calibration factor {scale:.3f}")
    log(f"\n{'config':16s}{'s/epoch':>10s}{'batches':>9s}")
    for c in per_epoch:
        log(f"{c:16s}{per_epoch[c]:>10.1f}{raw[c]['n_batches']:>9d}")

    if paths.candidates.exists():
        cands, cand_src = C.read_json(paths.candidates)["candidates"], "candidates.json"
    else:
        cands = [c for c in ASSUMED_CANDIDATES if c in per_epoch] or list(per_epoch)
        cand_src = ("ASSUMED (candidates.json not found yet): " + ", ".join(cands)
                    + " - rerun --timing after step 2 for the real candidates")
    extra_seeds = len([s for s in ctx.seeds if s != C.SWEEP_SEED])
    ep_typ, ep_src = typical_epochs(paths.out_root, args.epochs)
    ep_worst = SL.MAX_EPOCHS if not ctx.smoke else C.SMOKE_EPOCHS
    s2 = sum(per_epoch.values())
    s3 = extra_seeds * sum(per_epoch[c] for c in cands)
    log(f"\nStep 3 candidates: {cand_src}")
    log(f"Estimated time on this machine ({socket.gethostname()}, {C.device_label(device)}):")
    log(f"  worst case ({ep_worst} epochs per run):  step 2 {fmt_h(s2 * ep_worst)}   "
        f"step 3 {fmt_h(s3 * ep_worst)}   total {fmt_h((s2 + s3) * ep_worst)}")
    if ep_typ is None:
        log(f"  typical case: {ep_src}; pass --epochs N (typical epochs per run) to get it")
    else:
        log(f"  typical case ({ep_typ:.0f} epochs per run, {ep_src}):  step 2 {fmt_h(s2 * ep_typ)}   "
            f"step 3 {fmt_h(s3 * ep_typ)}   total {fmt_h((s2 + s3) * ep_typ)}")
    ep_show = ep_typ if ep_typ is not None else ep_worst
    log(f"\nWork per shard on this machine ({ep_show:.0f} epochs per run):")
    for n in (2, 3, 4):
        for row in shard_loads(per_epoch, cands, n, extra_seeds):
            log(f"  --shard {row['shard']}: step 2 {fmt_h(row['step2'] * ep_show)}, "
                f"step 3 {fmt_h(row['step3'] * ep_show)}  ({len(row['configs'])} configs: "
                f"{', '.join(row['configs'])})")

    paths.timing.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    record = {"hostname": host, "gpu": C.device_label(device), "device": str(device),
              "cpu_model": C.cpu_model(), "torch_version": torch.__version__,
              "reference_config": ref_name, "reference_epoch_s": ref_epoch,
              "calibration_factor": scale, "per_epoch_s": per_epoch,
              "raw": raw, "epochs_typical": ep_typ, "epochs_typical_source": ep_src,
              "measured_at": datetime.now().isoformat(timespec="seconds")}
    out = paths.timing / f"timing_{host}.json"
    C.write_json(out, record)
    log(f"\nSaved {paths.rel(out)}")

    others = [C.read_json(p) for p in sorted(paths.timing.glob("timing_*.json")) if p != out]
    others = [o for o in others if set(o.get("per_epoch_s", {})) == set(per_epoch)]
    if not others:
        log("To get a two-machine --shard recommendation, copy the timing_<host>.json of the other "
            f"machine into {paths.rel(paths.timing)}/ and rerun --timing (or run it there with this file).")
    for o in others:
        span, n, sub, load = best_two_machine_plan(record, o, cands, extra_seeds)
        rest = [k for k in range(1, n + 1) if k not in sub]
        log(f"\nTwo-machine plan with {o['hostname']} ({o['gpu']}), steps 2 + 3, {ep_show:.0f} epochs/run:")
        log(f"  {host}: --shard {','.join(map(str, sub))}/{n}  -> {fmt_h(load['A'] * ep_show)}")
        log(f"  {o['hostname']}: --shard {','.join(map(str, rest))}/{n}  -> {fmt_h(load['B'] * ep_show)}")
        log(f"  both finish after about {fmt_h(span * ep_show)}")


# ---------------------------------------------------------------------------
def log_window_counts(data: Data, folders: list[str], windows: list[int], log) -> None:
    log(f"Windows per temperature (train: every point from W-1; validation: rows >= {C.EVAL_START}):")
    log(f"  {'split':6s}{'temperature':>12s}{'files':>7s}{'rows':>9s}" + "".join(f"{'W=' + str(w):>9s}" for w in windows))
    for split, seqs in (("train", data.train), ("val", data.val)):
        for folder in folders:
            part = [s for s in seqs if s.info.folder == folder]
            counts = []
            for w in windows:
                if split == "train":
                    counts.append(sum(max(0, len(s) - w + 1) for s in part))
                else:
                    counts.append(sum(max(0, len(s) - max(w - 1, C.EVAL_START)) for s in part if len(s) >= w))
            log(f"  {split:6s}{C.temp_label(folder):>12s}{len(part):>7d}{sum(len(s) for s in part):>9,d}"
                + "".join(f"{c:>9,d}" for c in counts))


def main() -> None:
    p = argparse.ArgumentParser(description="Paper steps 2-3: LSTM sweep (seed 42) and multi-seed runs")
    C.add_common_args(p)
    p.add_argument("--stage", choices=["sweep", "multiseed"], default="sweep")
    p.add_argument("--shard", default=None, help="k/n (or k1,k2/n): run only the configs of these shards")
    p.add_argument("--configs", nargs="+", default=None, help="run exactly these configs (overrides --shard)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dry_run", action="store_true", help="list the runs of this machine and exit")
    p.add_argument("--timing", action="store_true",
                   help=f"train {SL.config_name(*TIMING_REF)} for 1 epoch, time every config, estimate steps 2-3")
    p.add_argument("--init_norm", action="store_true",
                   help="only create (or check) norm_stats.json from the training set, print its sha256 and exit")
    p.add_argument("--epochs", type=float, default=None,
                   help="--timing: typical epochs per run (default: mean of the old sweep runs)")
    args = p.parse_args()

    ctx = C.resolve(args)
    paths = ctx.paths
    device = SL.pick_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    host = socket.gethostname()
    task = "timing" if args.timing else "init_norm" if args.init_norm else args.stage
    log = C.Tee(paths.logs / f"{task}_{host}.log")
    max_epochs = C.SMOKE_EPOCHS if ctx.smoke else SL.MAX_EPOCHS
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {paths.name}  stage={args.stage}  smoke={ctx.smoke}  "
        f"host={host}  device={C.device_label(device)}  torch {torch.__version__}")
    log(f"Temperatures {ctx.temps} -> {ctx.folders}; data {D.DATA_DIR}; output {paths.out_root}")
    if ctx.smoke:
        n2 = {C.shard_index(c, 2) for c in ctx.config_names}
        assert n2 == {1, 2}, "smoke configs should fall into different shards for --shard k/2"

    plan = None
    if not (args.timing or args.init_norm):
        plan = plan_runs(ctx, args.stage, args.configs, args.shard, log)
        todo = [r for r in plan if not paths.run_json(C.run_name(*r)).exists()]
        log(f"Runs for this machine: {len(plan)} ({len(plan) - len(todo)} already finished, {len(todo)} to train)")
        for cname, seed in plan:
            done = paths.run_json(C.run_name(cname, seed)).exists()
            log(f"  {C.run_name(cname, seed):24s} {'finished' if done else 'to train'}")
        if args.dry_run:
            log.close()
            return
        if not todo:
            log("Nothing to train.")
            log.close()
            return

    # ---- data: switch on the test-set guard first ----
    D.block_test_files()
    splits = D.build_splits(ctx.folders)
    try:
        D.read_raw(splits["test"][0])
        raise AssertionError("test-set guard is not active")
    except D.TestSetAccessError:
        log(f"Test-set guard on: the {len(splits['test'])} test files cannot be read (self-check passed)")
    for split in ("train", "val", "test"):
        log(f"{split} ({len(splits[split])} sequences{', not read' if split == 'test' else ''}): "
            + ", ".join(s.id for s in splits[split]))
    log("Loading training set ...")
    train_seqs = D.load_sequences(splits["train"], log)
    log("Loading validation set ...")
    val_seqs = D.load_sequences(splits["val"], log)
    data = Data(train_seqs, val_seqs)
    log_window_counts(data, ctx.folders, sorted({c[0] for c in ctx.configs}), log)

    if args.timing:
        run_timing(ctx, data, args, device, log)
        log.close()
        return

    stats, norm_sha = C.load_or_create_norm_stats(paths.norm_stats, train_seqs, ctx.temps, log)
    log("Normalisation (training set): " + ", ".join(
        f"{f} mean={m:.4f} std={s:.4f}" for f, m, s in zip(D.FEATURES, stats["mean"], stats["std"])))
    if args.init_norm:
        log(f"{paths.rel(paths.norm_stats)}  sha256 {norm_sha}")
        log.close()
        return
    settings = {"experiment": paths.name, "temperatures_C": ctx.temps, "folders": ctx.folders,
                "features": D.FEATURES, "target": D.TARGET,
                "resample": "1 s, floor(time_s); V/I/T mean, soc last",
                "eval_start_index": C.EVAL_START, "lr": SL.LR, "batch_size": SL.BATCH_SIZE,
                "max_epochs": max_epochs, "patience": SL.PATIENCE, "optimizer": "Adam",
                "loss": "MSE", "dropout": 0.0, "smoke": ctx.smoke}
    env = C.environment_info(device, norm_sha)
    log(f"git commit {env['git_commit']}" + ("  (uncommitted changes!)" if env["git_dirty"] else ""))

    # ---- runs already in this folder must use the same settings and normalisation file ----
    for f in sorted(paths.runs.glob("*.json")):
        r = C.read_json(f)
        if r.get("settings") != settings:
            diff = {k: (r.get("settings", {}).get(k), settings.get(k))
                    for k in set(r.get("settings", {})) | set(settings)
                    if r.get("settings", {}).get(k) != settings.get(k)}
            raise SystemExit(f"{paths.rel(f)} was trained with different settings (existing vs now): {diff}")
        if r["environment"]["norm_stats_sha256"] != norm_sha:
            raise SystemExit(f"{paths.rel(f)} used a different normalisation file "
                             f"({r['environment']['norm_stats_sha256'][:12]} vs {norm_sha[:12]})")

    finished = []
    for i, (cname, seed) in enumerate(plan, 1):
        rname = C.run_name(cname, seed)
        if paths.run_json(rname).exists():
            r = C.read_json(paths.run_json(rname))
            log(f"[{i}/{len(plan)}] {rname} already finished (val RMSE {r['val_rmse_pct']:.3f}%), skipped")
            continue
        log(f"[{i}/{len(plan)}] {rname} training")
        finished.append(train_run(cname, seed, data, stats, norm_sha, settings, paths, device, env, log))

    log(f"\nFinished {len(finished)} runs on {host} in this session:")
    for r in finished:
        log(f"  {r['run']:24s} val RMSE {r['val_rmse_pct']:.3f}%  MAE {r['val_mae_pct']:.3f}%  "
            f"best {r['best_epoch']}/{r['epochs_trained']}  {r['train_time_s'] / 60:.1f} min")
    log("Next: copy every results/<name>/runs/*.json (and val_preds/, models/) to one machine and run "
        "src/merge_results.py")
    log.close()


if __name__ == "__main__":
    main()
