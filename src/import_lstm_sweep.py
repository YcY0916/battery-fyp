"""把已经跑完的 22 组 LSTM 结果导入统一的搜索表 results/tables/sweep_lstm.csv（不重新训练）。

LSTM 的 22 组（results/lstm_sweep_ext/sweep_results_22.csv，由 combine_lstm_sweep.py 生成）的验证误差
都是在"序列内第 239 行及之后"的点上算的，与新模型的搜索相同，可以直接放进同一张表。完整网格
W{30,60,120,180,240} x H{32,64,128} x L{1,2} 剩下的 8 组，导入后再用 sweep_models.py --model lstm 补跑。

第 1 步：核对（用新代码重新训练一组旧配置，默认是 22 组里验证 RMSE 最低的那组）
    python src/import_lstm_sweep.py --verify
    python src/import_lstm_sweep.py --verify W30_H32_L1     # 也可以指定别的组（窗口小的更快）
  - 走新的统一流程（models.SPECS["lstm"] + sweep_lstm.fit），设置与当初完全相同，包括当初早停用的
    评估起点（最初 18 组是 119，扩展 4 组是 239）；
  - 对照当初记录的验证 RMSE 和最佳轮，再和旧权重逐个参数比较；
  - 结果写到 results/lstm_verify/ 和 models/lstm_verify/，不碰 results/tables/，也不改任何旧结果。

第 2 步：导入
    python src/import_lstm_sweep.py
  - 只读 sweep_results_22.csv 和旧权重（models/lstm_sweep*/，不修改、不复制）；
  - 每组用旧权重补测推理时间（batch = 1，GPU 和 CPU 各一次）；
  - 没有通过第 1 步的核对时拒绝导入（确实要跳过可以加 --skip-verify-check）；
  - 可以重复运行：sweep_lstm.csv 里已有的配置跳过。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
from models import SPECS  # noqa: E402
from sweep_lstm import (BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, SEED, Logger,  # noqa: E402
                        pick_device, val_metrics, write_json)
from sweep_models import (LSTM_NORM_STATS, append_row, check_table_settings,  # noqa: E402
                          device_name, load_train_val, make_settings, measure_latencies,
                          model_from_checkpoint, output_paths, read_table, rel, save_checkpoint,
                          setup_torch, train_config)

ROOT = D.ROOT
TABLE_22 = ROOT / "results" / "lstm_sweep_ext" / "sweep_results_22.csv"
VERIFY_DIR = ROOT / "results" / "lstm_verify"
VERIFY_MODELS = ROOT / "models" / "lstm_verify"
DEFAULT_TOL = 0.05  # 重跑的验证 RMSE 与记录值相差不超过 0.05 个百分点算"基本一致"


def load_table22(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"找不到 {rel(path)}，请先按 results/lstm_sweep/README.md 跑完 combine_lstm_sweep.py")
    df = pd.read_csv(path)
    need = {"config", "window", "hidden_size", "num_layers", "sweep", "val_rmse_pct", "val_mae_pct",
            "early_stop_eval_start", "val_rmse_pct_recorded", "best_epoch", "epochs_trained",
            "stopped_early", "train_time_s", "n_params", "n_train_windows", "n_val_points", "device_name"}
    if need - set(df.columns):
        raise SystemExit(f"{rel(path)} 缺少列：{sorted(need - set(df.columns))}")
    if df["config"].duplicated().any():
        raise SystemExit(f"{rel(path)} 里有重复的配置")
    return df.sort_values("val_rmse_pct", kind="stable").reset_index(drop=True)


def old_ckpt_path(row, models_root: Path) -> Path:
    return models_root / str(row["sweep"]) / f"{row['config']}.pt"


def _same_stats(a: dict, b: dict) -> bool:
    return (np.allclose(a["mean"], b["mean"], rtol=1e-9, atol=0)
            and np.allclose(a["std"], b["std"], rtol=1e-9, atol=0))


def _check_old_ckpt(ckpt: dict, row, path: Path, max_epochs: int = MAX_EPOCHS) -> None:
    cfg, s = ckpt["config"], ckpt["settings"]
    if (int(cfg["window"]), int(cfg["hidden_size"]), int(cfg["num_layers"])) != (
            int(row["window"]), int(row["hidden_size"]), int(row["num_layers"])):
        raise SystemExit(f"{rel(path)} 的配置 {cfg} 与表里的 {row['config']} 不一致")
    if list(ckpt["features"]) != D.FEATURES:
        raise SystemExit(f"{rel(path)} 的输入特征是 {ckpt['features']}")
    want = {"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED, "max_epochs": max_epochs, "patience": PATIENCE}
    diff = {k: (s.get(k), v) for k, v in want.items() if s.get(k) != v}
    if diff:
        raise SystemExit(f"{rel(path)} 的训练设置与统一流程不同（旧 vs 现在）：{diff}")
    if int(s["eval_start_index"]) != int(row["early_stop_eval_start"]):
        raise SystemExit(f"{rel(path)} 的早停评估起点 {s['eval_start_index']} 与表里的 "
                         f"{row['early_stop_eval_start']} 不一致")


# ---------------------------------------------------------------------------
# 第 1 步：核对
# ---------------------------------------------------------------------------
def verify_config(row, ckpt: dict, train_seqs, val_seqs, stats: dict, device, log,
                  out_dir: Path, models_dir: Path, common_eval_start: int, tol: float) -> dict:
    spec = SPECS["lstm"]
    cname = str(row["config"])
    w = int(row["window"])
    hp = {"hidden_size": int(row["hidden_size"]), "num_layers": int(row["num_layers"])}
    old = ckpt["settings"]
    if not _same_stats(stats, ckpt["norm_stats"]):
        raise SystemExit("本次计算的标准化参数与旧权重里保存的不同，数据流程不一致，停止")
    log("标准化参数与旧权重里保存的完全相同")

    # 设置与当初完全相同（包括当初早停用的评估起点）
    settings = make_settings("lstm", old["lr"], old["max_epochs"], old["folders"],
                             int(old["eval_start_index"]), seed=old["seed"])
    log(f"用新代码重跑 {cname}（seed {old['seed']}，lr {old['lr']}，早停评估起点 {old['eval_start_index']}）")
    res = train_config(spec, w, hp, train_seqs, val_seqs, stats, settings, device, log,
                       name=f"verify {cname}")

    # 在统一的评估起点（239）上也算一次，与 22 组表里的 val_rmse_pct 对照
    ds = D.WindowDataset(val_seqs, stats, w, common_eval_start, lambda *_: None).to(device)
    rmse_c, mae_c, _ = val_metrics(res["model"], ds)

    old_sd, new_sd = ckpt["state_dict"], res["best_state"]
    max_w_diff = max(float((new_sd[k].double() - old_sd[k].double()).abs().max()) for k in old_sd)
    diff = res["val_rmse_pct"] - float(row["val_rmse_pct_recorded"])
    record = {
        "config": cname, "sweep": str(row["sweep"]), "passed": abs(diff) <= tol, "tolerance_pct": tol,
        "early_stop_eval_start": int(old["eval_start_index"]),
        "val_rmse_pct_recorded": float(row["val_rmse_pct_recorded"]),
        "val_rmse_pct_rerun": res["val_rmse_pct"], "val_rmse_diff_pct": diff,
        "best_epoch_recorded": int(row["best_epoch"]), "best_epoch_rerun": res["best_epoch"],
        "epochs_trained_recorded": int(row["epochs_trained"]), "epochs_trained_rerun": res["epochs_trained"],
        "common_eval_start": common_eval_start,
        "val_rmse_pct_common_table": float(row["val_rmse_pct"]), "val_rmse_pct_common_rerun": rmse_c,
        "n_val_points_common": len(ds), "n_val_points_common_table": int(row["n_val_points"]),
        "max_abs_weight_diff": max_w_diff, "train_time_s": res["train_time"],
        "device_name": device_name(device), "torch_version": torch.__version__,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(models_dir / f"{cname}.pt", spec, w, hp, res, stats, settings)
    pd.DataFrame(res["history"]).to_csv(out_dir / f"{cname}_history.csv", index=False)
    write_json(out_dir / f"{cname}.json", record)

    log("\n核对结果（验证 RMSE 单位：SOC 百分点）：")
    log(f"  早停起点 {record['early_stop_eval_start']} 上的验证 RMSE：记录 {record['val_rmse_pct_recorded']:.4f}  "
        f"重跑 {record['val_rmse_pct_rerun']:.4f}  差 {diff:+.4f}")
    log(f"  起点 {common_eval_start} 上的验证 RMSE：    22 组表 {record['val_rmse_pct_common_table']:.4f}  "
        f"重跑 {rmse_c:.4f}  差 {rmse_c - record['val_rmse_pct_common_table']:+.4f}")
    log(f"  最佳轮：记录 {record['best_epoch_recorded']}  重跑 {record['best_epoch_rerun']}；"
        f"总轮数：记录 {record['epochs_trained_recorded']}  重跑 {record['epochs_trained_rerun']}")
    log(f"  与旧权重的最大参数差：{max_w_diff:.3e}（0 表示逐位相同）")
    if record["n_val_points_common"] != record["n_val_points_common_table"]:
        log(f"  注意：起点 {common_eval_start} 的验证点数 {len(ds)} 与 22 组表的 "
            f"{record['n_val_points_common_table']} 不同，22 组表可能不是用这个起点生成的")
    log(("  结论：通过（差值在 ±{:.2f} 以内），可以运行 python src/import_lstm_sweep.py 导入".format(tol))
        if record["passed"] else
        f"  结论：未通过（差值超过 ±{tol:.2f}），请先查明原因（PyTorch/CUDA 版本、显卡是否与当初相同？）再导入")
    return record


# ---------------------------------------------------------------------------
# 第 2 步：导入
# ---------------------------------------------------------------------------
def passed_verification(verify_dir: Path, table: pd.DataFrame) -> dict | None:
    for p in sorted(verify_dir.glob("W*_H*_L*.json")):
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        if r.get("passed") and r["n_val_points_common"] == int(table["n_val_points"].iloc[0]):
            return r
    return None


def import_rows(table: pd.DataFrame, models_root: Path, out_csv: Path, verify_dir: Path, device,
                log, eval_start: int, skip_verify_check: bool = False,
                latency_kw: dict | None = None, max_epochs: int = MAX_EPOCHS,
                windows: list[int] | None = None) -> int:
    spec = SPECS["lstm"]
    if not skip_verify_check:
        v = passed_verification(verify_dir, table)
        if v is None:
            raise SystemExit(f"{rel(verify_dir)} 里没有通过的核对结果。请先运行 "
                             "python src/import_lstm_sweep.py --verify（或加 --skip-verify-check 跳过这项检查）")
        log(f"核对已通过：{v['config']} 重跑验证 RMSE {v['val_rmse_pct_rerun']:.4f} vs 记录 "
            f"{v['val_rmse_pct_recorded']:.4f}（{v['finished_at']}）")
    if table["n_val_points"].nunique() != 1:
        raise SystemExit("22 组表里各组的验证点数不同，不是在同一批点上比较的")

    settings = make_settings("lstm", LR, max_epochs, list(D.TEMP_FOLDERS), eval_start)
    existing = read_table(out_csv, spec)
    done = set()
    if existing is not None:
        check_table_settings(existing, settings, out_csv)
        done = set(existing["config"])

    ref_stats = None
    for ref in LSTM_NORM_STATS:
        if ref.exists():
            ref_stats = D.load_norm_stats(ref)
            break

    order = {spec.config_name(w, hp): i for i, (w, hp) in enumerate(spec.grid(windows))}
    unknown = [c for c in table["config"] if c not in order]
    if unknown:
        raise SystemExit(f"22 组表里有不在完整网格里的配置：{unknown}")
    n_new = 0
    for _, row in table.sort_values("config", key=lambda s: s.map(order)).iterrows():
        cname = str(row["config"])
        if cname in done:
            log(f"  {cname} 已在 {rel(out_csv)} 里，跳过")
            continue
        path = old_ckpt_path(row, models_root)
        if not path.exists():
            raise SystemExit(f"找不到旧权重 {rel(path)}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        _check_old_ckpt(ckpt, row, path, max_epochs)
        if ref_stats is None:
            ref_stats = ckpt["norm_stats"]
        elif not _same_stats(ref_stats, ckpt["norm_stats"]):
            raise SystemExit(f"{rel(path)} 的标准化参数与其他组不同")
        _, _, model = model_from_checkpoint(ckpt)
        lat = measure_latencies(model, int(row["window"]), **(latency_kw or {}))

        finished_at = ""
        run_json = models_root.parent / "results" / str(row["sweep"]) / "runs" / f"{cname}.json"
        if run_json.exists():
            with open(run_json, "r", encoding="utf-8") as f:
                finished_at = json.load(f).get("finished_at", "")
        out = {"model": "lstm", "config": cname, "window": int(row["window"]),
               "hidden_size": int(row["hidden_size"]), "num_layers": int(row["num_layers"]),
               "lr": LR, "batch_size": BATCH_SIZE, "seed": SEED, "max_epochs": max_epochs,
               "patience": PATIENCE, "loss": "MSE", "eval_start": eval_start,
               "early_stop_eval_start": int(row["early_stop_eval_start"]),
               "val_rmse_pct": float(row["val_rmse_pct"]), "val_mae_pct": float(row["val_mae_pct"]),
               **{f"val_rmse_pct_{f}": float(row.get(f"val_rmse_pct_{f}", math.nan)) for f in D.TEMP_FOLDERS},
               "best_epoch": int(row["best_epoch"]), "epochs_trained": int(row["epochs_trained"]),
               "stopped_early": bool(row["stopped_early"]), "train_time_s": float(row["train_time_s"]),
               "n_params": int(row["n_params"]), **lat,
               "n_train_windows": int(row["n_train_windows"]), "n_val_points": int(row["n_val_points"]),
               "device_name": str(row["device_name"]), "source": f"imported:{row['sweep']}",
               "model_file": rel(path), "finished_at": finished_at}
        append_row(out_csv, spec, out)
        n_new += 1
        gpu_txt = "-" if math.isnan(lat["infer_ms_gpu"]) else f"{lat['infer_ms_gpu']:.3f} ms"
        log(f"  导入 {cname}（{row['sweep']}）：验证 RMSE {out['val_rmse_pct']:.3f}%，"
            f"推理 GPU {gpu_txt} / CPU {lat['infer_ms_cpu']:.3f} ms")
    return n_new


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="核对并导入已有的 22 组 LSTM 结果（不重新训练）")
    p.add_argument("--verify", nargs="?", const="best", default=None, metavar="CONFIG",
                   help="第 1 步：用新代码重跑一组旧配置并对照记录值（不写配置名则用验证 RMSE 最低的那组）")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help=f"核对的容差（SOC 百分点），默认 {DEFAULT_TOL}")
    p.add_argument("--skip-verify-check", action="store_true", help="导入时不检查是否已通过核对")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    table = load_table22(TABLE_22)
    device = pick_device(args.device)
    setup_torch(device)
    eval_start = D.EVAL_START_INDEX

    if args.verify is not None:
        cname = table["config"].iloc[0] if args.verify == "best" else args.verify
        if cname not in set(table["config"]):
            raise SystemExit(f"{cname} 不在 22 组表里（可选：{', '.join(table['config'])}）")
        row = table.loc[table["config"] == cname].iloc[0]
        VERIFY_DIR.mkdir(parents=True, exist_ok=True)
        log = Logger(VERIFY_DIR / "verify_log.txt")
        log("=" * 78)
        log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  核对 {cname}（{row['sweep']}），设备 {device}"
            + (f"（{torch.cuda.get_device_name(device)}）" if device.type == "cuda" else ""))
        path = old_ckpt_path(row, ROOT / "models")
        if not path.exists():
            raise SystemExit(f"找不到旧权重 {rel(path)}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        _check_old_ckpt(ckpt, row, path)
        train_seqs, val_seqs, stats = load_train_val(list(ckpt["settings"]["folders"]), log)
        verify_config(row, ckpt, train_seqs, val_seqs, stats, device, log,
                      VERIFY_DIR, VERIFY_MODELS, eval_start, args.tol)
        log.close()
        return

    out_csv = output_paths("lstm")["csv"]
    log_dir = output_paths("lstm")["search"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(log_dir / "import_log.txt")
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  导入 {rel(TABLE_22)}（{len(table)} 组）-> {rel(out_csv)}")
    n = import_rows(table, ROOT / "models", out_csv, VERIFY_DIR, device, log, eval_start,
                    skip_verify_check=args.skip_verify_check)
    df = read_table(out_csv, SPECS["lstm"])
    log(f"本次导入 {n} 组，{rel(out_csv)} 现有 {0 if df is None else len(df)} 组")
    missing = [SPECS["lstm"].config_name(*c) for c in SPECS["lstm"].grid()
               if df is None or SPECS["lstm"].config_name(*c) not in set(df["config"])]
    if missing:
        log(f"完整网格还差 {len(missing)} 组：{', '.join(missing)}")
        log("下一步：python src/sweep_models.py --model lstm")
    log.close()


if __name__ == "__main__":
    main()
