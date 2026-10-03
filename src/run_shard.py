"""Unified training entry of the v2 experiment: runs a list of (config, seed) runs on one GPU.

Pick the runs either directly or with a shard file:
    python src/run_shard.py --exp paper_0to40_v2 --stage step2 --configs L1_W360_H64,L2_W240_H32 --seeds 42 --platform kaggle
    python src/run_shard.py --exp paper_0to40_v2 --shard-file shards/step2_kaggle_gpu0.txt --platform kaggle --skip-existing

Shard file format (one run per line, '#' starts a comment):
    stage: step2
    L1_W360_H64 42
    L2_W240_H32 42,43

--skip-existing: complete runs (all files present and matching the hashes in COMPLETE.json)
are skipped; incomplete runs are deleted and trained again from epoch 1. Rerunning the same
command after any interruption therefore continues the shard. Without --skip-existing the script
refuses to touch a run folder that already exists.

Before any training the pre-flight checks must pass (otherwise the script exits):
    1. code: HEAD = tag v2-protocol, or the training files do not differ from the tag;
    2. data: every train / val file and manifest.json matches protocol/v2/data_sha256.json,
       every test file is present (test files are never opened here);
    3. normalisation: protocol/v2/norm_stats.json has the v1 sha256 and is reproduced exactly
       from the training set; it is copied to results/<exp>/norm_stats.json;
    4. GPU: CUDA is available.
For experiments other than paper_0to40_v2 (smoke test, timing probe) a failed code check only
warns, --max-epochs may be set and --allow-cpu is accepted.

Each run writes (files first to a temporary name, then renamed; COMPLETE.json last):
    results/<exp>/runs/<run>/run_meta.json         environment, versions, hashes, hyperparameters, timing
    results/<exp>/runs/<run>/metrics.json          validation RMSE / MAE, overall and per temperature, rows >= 359
    results/<exp>/runs/<run>/train_log.csv         one row per epoch
    results/<exp>/runs/<run>/val_predictions.npz   validation predictions for every point from W-1
    results/<exp>/runs/<run>/run.log               console output of this run
    results/<exp>/runs/<run>/curve.png             training curve
    results/<exp>/runs/<run>/COMPLETE.json         sha256 of every file above and of the weights
    models/<exp>/<run>/model.pt                    weights of the best epoch
The test-set guard is on for the whole script: reading a test file raises an error.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import lstm_data as D  # noqa: E402
import sweep_lstm as SL  # noqa: E402
import v2_common as V  # noqa: E402
from lstm_model import LSTMSOC, count_parameters  # noqa: E402


class DeadlineReached(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.extra = None  # per-run log file

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
        if self.extra is not None:
            self.extra.write(msg + "\n")
            self.extra.flush()


# ---------------------------------------------------------------------------
# Which runs
# ---------------------------------------------------------------------------
def parse_shard_file(path: Path) -> tuple[str | None, list[tuple[str, int]]]:
    stage, runs = None, []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("stage:"):
            stage = line.split(":", 1)[1].strip()
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"{path}:{i}: expected '<config> <seed>[,<seed>...]', got {raw!r}")
        V.parse_config(parts[0])
        for s in parts[1].split(","):
            runs.append((parts[0], int(s)))
    return stage, runs


def plan(args) -> tuple[str, list[tuple[str, int]]]:
    stage = args.stage
    if args.shard_file:
        fstage, runs = parse_shard_file(Path(args.shard_file))
        if fstage and stage and fstage != stage:
            raise SystemExit(f"--stage {stage} but the shard file says stage: {fstage}")
        stage = stage or fstage
        if args.configs or args.seeds:
            raise SystemExit("Use either --shard-file or --configs/--seeds, not both")
    else:
        if not args.configs or not args.seeds:
            raise SystemExit("Give --shard-file, or both --configs and --seeds")
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        for c in configs:
            V.parse_config(c)
        runs = [(c, s) for c in configs for s in seeds]
    if stage not in V.STAGES:
        raise SystemExit(f"--stage must be one of {V.STAGES} (or 'stage:' in the shard file)")
    dup = sorted({V.run_name(*r) for r in runs if runs.count(r) > 1})
    if dup:
        raise SystemExit(f"Runs listed twice: {', '.join(dup)}")

    formal = args.exp == V.FORMAL_EXP
    if formal:
        if stage not in V.FORMAL_STAGES:
            raise SystemExit(f"{V.FORMAL_EXP} only takes stages {V.FORMAL_STAGES}; use another --exp for {stage}")
        grid = V.step2_configs()
        for c, s in runs:
            if c not in grid:
                raise SystemExit(f"{c} is not one of the 30 step-2 configs")
            if stage == "step2" and s != V.STEP2_SEED:
                raise SystemExit(f"step2 uses seed {V.STEP2_SEED} only, got {V.run_name(c, s)}")
            if stage == "step3" and s not in V.STEP3_SEEDS:
                raise SystemExit(f"step3 uses seeds {V.STEP3_SEEDS}, got {V.run_name(c, s)}")
        if stage == "step3":
            cands = V.read_step3_candidates()
            extra = sorted({c for c, _ in runs if c not in cands})
            if extra:
                raise SystemExit(f"step3 configs not in {V.STEP3_CANDIDATES_FILE.name}: {extra}")
    else:
        if stage in V.FORMAL_STAGES:
            raise SystemExit(f"Stage {stage} only runs under --exp {V.FORMAL_EXP}")
        if args.exp.startswith("paper_0to40"):
            raise SystemExit("Temporary experiments must not start with 'paper_0to40'")
    return stage, runs


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def train_run(config: str, seed: int, stage: str, data: dict, stats: dict, norm_sha: str,
              hp: dict, paths: V.Paths, device: torch.device, pre: dict, args, log: Logger,
              deadline: float | None) -> dict:
    l, w, h = V.parse_config(config)
    run = V.run_name(config, seed)
    rdir, mdir = paths.run_dir(run), paths.model_dir(run)
    V.clear_run(paths, run)
    rdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    log.extra = open(rdir / "run.log", "w", encoding="utf-8")
    started = V.now_iso()

    def epoch_log(msg: str = "") -> None:
        log(msg)
        if deadline is not None and time.time() > deadline:
            raise DeadlineReached(f"deadline reached during {run}")
        if args.test_abort_epoch and f"epoch {args.test_abort_epoch:3d}" in msg:
            raise DeadlineReached(f"simulated interruption after epoch {args.test_abort_epoch} of {run}")

    try:
        log(f"[{run}] stage {stage}, L={l} W={w} H={h} seed={seed}, device {pre['environment']['gpu_name']}")
        SL.set_seed(seed)
        train_ds = D.WindowDataset(data["train"], stats, w, None, log).to(device)
        val_ds = D.WindowDataset(data["val"], stats, w, V.EVAL_START, log).to(device)
        model = LSTMSOC(hidden_size=h, num_layers=l).to(device)
        n_par = count_parameters(model)
        log(f"[{run}] train windows {len(train_ds):,} (ends from row W-1), validation points "
            f"{len(val_ds):,} (rows >= {V.EVAL_START}), trainable parameters {n_par:,}")

        fitted = SL.fit(model, train_ds, val_ds, run, hp["max_epochs"], device, epoch_log, seed=seed)
        history, best_epoch = fitted["history"], fitted["best_epoch"]
        epochs, train_time = len(history), fitted["train_time"]
        finished = V.now_iso()

        # weights of the best epoch (fit returns with them loaded)
        tmp = mdir / f".model.pt.{os.getpid()}.tmp"
        torch.save({"state_dict": fitted["best_state"],
                    "config": {"window": w, "hidden_size": h, "num_layers": l, "input_size": 3},
                    "config_name": config, "seed": seed, "features": list(D.FEATURES),
                    "norm_stats": stats, "norm_stats_sha256": norm_sha, "exp": paths.exp,
                    "best_epoch": best_epoch}, tmp)
        os.replace(tmp, paths.model_file(run))

        hist = pd.DataFrame(history)
        hist["is_best"] = hist["epoch"] == best_epoch
        V.write_text_atomic(rdir / "train_log.csv", hist.to_csv(index=False, lineterminator="\n"))
        V.save_val_predictions(rdir / "val_predictions.npz", model, data["val"], stats, w, device)
        met = V.metrics_from_npz(rdir / "val_predictions.npz", V.EVAL_START)
        met.update({"run": run, "config": config, "seed": seed, "split": "validation",
                    "units": "SOC percentage points"})
        V.write_json(rdir / "metrics.json", met)
        tmp_png = rdir / f".curve.{os.getpid()}.tmp.png"
        SL.plot_curve(hist, best_epoch, run, tmp_png)
        os.replace(tmp_png, rdir / "curve.png")

        meta = {
            "run": run, "config": config, "num_layers": l, "window": w, "hidden_size": h, "seed": seed,
            "exp": paths.exp, "stage": stage, "note": args.note,
            **pre["environment"], **pre["code"], **pre["data"],
            "norm_stats_sha256": norm_sha, "hyperparameters": hp,
            "n_params": n_par, "n_train_windows": len(train_ds), "n_val_points": len(val_ds),
            "skipped_short_train_sequences": [s for s, _ in train_ds.skipped],
            "start_time": started, "end_time": finished,
            "epochs_trained": epochs, "best_epoch": best_epoch,
            "stopped_early": epochs < hp["max_epochs"], "best_val_loss": fitted["best_loss"],
            "train_time_s": train_time, "sec_per_epoch": train_time / epochs,
            "val_rmse_pct": met["overall"]["rmse_pct"], "val_mae_pct": met["overall"]["mae_pct"],
            "command": " ".join(sys.argv),
        }
        V.write_json(rdir / "run_meta.json", meta)
        log(f"[{run}] done: best epoch {best_epoch}/{epochs}, val RMSE {met['overall']['rmse_pct']:.3f}%, "
            f"MAE {met['overall']['mae_pct']:.3f}%, {train_time / 60:.1f} min, {train_time / epochs:.1f} s/epoch")
        log.extra.close()
        log.extra = None
        V.write_complete_marker(paths, run)  # written last: marks the run as complete
        return meta
    finally:
        if log.extra is not None:
            log.extra.close()
            log.extra = None


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="v2 training entry: run a shard of (config, seed) runs")
    p.add_argument("--exp", required=True, help=f"experiment name ({V.FORMAL_EXP} for the formal runs)")
    p.add_argument("--stage", choices=V.STAGES, default=None)
    p.add_argument("--configs", default=None, help="comma-separated configs, e.g. L1_W360_H64,L2_W240_H32")
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 42 or 43,44,45,46")
    p.add_argument("--shard-file", default=None, help="file with one '<config> <seed>' per line")
    p.add_argument("--platform", required=True, choices=["laptop", "colab", "kaggle", "other"])
    p.add_argument("--skip-existing", action="store_true",
                   help="skip complete runs, retrain incomplete ones")
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT,
                   help="folder that receives results/<exp> and models/<exp> (default: repository root)")
    p.add_argument("--data-root", type=Path, default=V.REPO_ROOT,
                   help="folder that contains data/processed/kollmeyer_clean (default: repository root)")
    p.add_argument("--max-epochs", type=int, default=None, help="only for non-formal experiments")
    p.add_argument("--allow-cpu", action="store_true", help="only for non-formal experiments")
    p.add_argument("--deadline-h", type=float, default=None,
                   help="stop (and discard the unfinished run) this many hours after start")
    p.add_argument("--note", default="", help="free text stored in run_meta.json (e.g. 'two processes')")
    p.add_argument("--dry-run", action="store_true", help="only list the runs and their status")
    p.add_argument("--test-abort-epoch", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    t_start = time.time()
    deadline = t_start + args.deadline_h * 3600 if args.deadline_h else None
    formal = args.exp == V.FORMAL_EXP
    if formal and (args.max_epochs is not None or args.allow_cpu or args.test_abort_epoch):
        raise SystemExit(f"--max-epochs and --allow-cpu are not allowed for {V.FORMAL_EXP}")
    stage, runs = plan(args)
    D.set_data_root(args.data_root)
    paths = V.Paths(args.out_root, args.exp)
    host = socket.gethostname()
    gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-") or "none"
    log = Logger(paths.logs / f"{stage}_{args.platform}_{host}_gpu{gpu_tag}_{datetime.now():%Y%m%d_%H%M%S}.log")
    log("=" * 78)
    log(f"{V.now_iso()}  run_shard  exp={args.exp}  stage={stage}  platform={args.platform}  host={host}")
    log(f"Output: {paths.results} and {paths.models}")

    status = {V.run_name(c, s): V.run_status(paths, V.run_name(c, s)) for c, s in runs}
    for r, (st, why) in status.items():
        log(f"  {r:26s} {st}{' (' + why + ')' if why else ''}")
    n_done = sum(1 for st, _ in status.values() if st == "complete")
    log(f"{len(runs)} runs listed: {n_done} complete, {len(runs) - n_done} to train")
    if args.dry_run:
        return
    existing = [r for r, (st, _) in status.items() if st != "absent"]
    if existing and not args.skip_existing:
        raise SystemExit(f"Run folders already exist ({', '.join(existing)}). Add --skip-existing to "
                         "skip complete runs and retrain incomplete ones.")
    todo = [(c, s) for c, s in runs if status[V.run_name(c, s)][0] != "complete"]
    if not todo:
        log("Nothing to train.")
        return

    # ---- pre-flight checks (nothing is trained unless all pass) ----
    log("Pre-flight checks:")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    code = V.check_code_version(formal, log)
    code.update(V.code_fingerprint())
    data_info = V.check_data_files(log)
    device = V.check_device(args.allow_cpu and not formal, log)
    D.block_test_files()
    splits = D.build_splits(V.FOLDERS)
    try:
        D.read_raw(splits["test"][0])
        raise AssertionError("test-set guard is not active")
    except D.TestSetAccessError:
        log(f"Test-set guard on: the {len(splits['test'])} test files cannot be read (self-check passed)")
    data = {"train": D.load_sequences(splits["train"], log), "val": D.load_sequences(splits["val"], log)}
    stats, norm_sha = V.check_norm_stats(data["train"], paths, log)
    hp = V.hyperparameters(args.max_epochs)
    env = V.environment(device, args.platform)
    pre = {"environment": env, "code": code, "data": data_info}
    log(f"Hyperparameters: lr {hp['lr']}, batch {hp['batch_size']}, max epochs {hp['max_epochs']}, "
        f"patience {hp['patience']}, eval start {hp['eval_start_index']}")
    log("All pre-flight checks passed.")

    done, failed = [], []
    for i, (c, s) in enumerate(todo, 1):
        run = V.run_name(c, s)
        if deadline is not None and time.time() > deadline:
            log(f"Deadline reached; {run} and later runs not started")
            failed.append(run)
            continue
        if status[run][0] == "incomplete":
            log(f"[{i}/{len(todo)}] {run}: incomplete folder from an earlier attempt ({status[run][1]}), retraining")
        else:
            log(f"[{i}/{len(todo)}] {run}: training")
        try:
            done.append(train_run(c, s, stage, data, stats, norm_sha, hp, paths, device, pre, args, log, deadline))
        except DeadlineReached as e:
            log(f"{e}; the unfinished run is discarded and will be retrained next time")
            V.clear_run(paths, run)
            failed.append(run)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log(f"\nSummary ({host}, {env['gpu_name']}): {len(done)} trained now, "
        f"{n_done} were already complete, {len(failed)} not finished")
    for m in done:
        log(f"  {m['run']:26s} val RMSE {m['val_rmse_pct']:.3f}%  best {m['best_epoch']}/{m['epochs_trained']}  "
            f"{m['sec_per_epoch']:.1f} s/epoch  {m['train_time_s'] / 60:.1f} min")
    if failed:
        log(f"  not finished: {', '.join(failed)} (rerun the same command with --skip-existing)")
    log(f"Elapsed {(time.time() - t_start) / 3600:.2f} h")
    sys.exit(3 if failed else 0)


if __name__ == "__main__":
    main()
