"""最终评估：四个模型各取搜索阶段验证 RMSE 最低的配置，用 5 个种子重新训练，再在测试集上比较。

**只有手动运行本脚本时才会读取测试集。** 流程：
1. 选配置（只看验证集）：从 results/tables/sweep_<模型>.csv 里取验证 RMSE 最低的一组。
   如果用 --lr 另外跑过（sweep_<模型>_lr*.csv），这些表也一起参与比较，学习率算作被选出的超参数之一；
   想固定用某一张表，用 --pick 模型=路径。默认要求 sweep_<模型>.csv 已经跑完完整网格。
2. 重新训练：每个最佳配置用 seed 42、43、44、45、46 各训练一次。流程与搜索完全相同
   （sweep_models.train_config），早停只看验证集；这一阶段测试集保护是打开的。
3. 测试：全部训练结束后才关闭保护、读取测试集。每个测试文件只评估序列内第 239 行及之后的点
   （与验证集、evaluate_lstm.py 相同）。按驾驶工况（UDDS / HWFET / LA92 / US06，把 6 个温度的测试点合在一起）
   和全部测试点分别计算 RMSE、MAE（SOC 百分点）。
4. 汇总：每个模型 5 个种子的均值 ± 标准差（样本标准差，ddof = 1），加上参数量和推理时间
   （batch = 1，GPU、CPU 各测一次，用 seed 42 的权重；四个模型在同一次运行里连续测量）。

用法（在仓库根目录）：
    python src/final_eval.py
    python src/final_eval.py --models gru cnn                     # 只评估部分模型
    python src/final_eval.py --pick transformer=results/tables/sweep_transformer.csv

中断后重跑同一条命令即可：已经训练好的（模型, 种子）直接读权重，不重新训练。

输出：
    results/tables/final_summary.csv            每个模型一行：各指标的均值、标准差（数值），参数量，推理时间
    results/tables/final_summary_formatted.csv  同上，写成 "均值 ± 标准差"，方便贴进论文
    results/tables/final_runs.csv               每个（模型, 种子）一行
    results/final_eval/per_file_metrics.csv     每个（模型, 种子, 测试文件）一行
    results/final_eval/final_log.txt            运行日志
    results/final_eval/runs/<模型>_<运行名>.json / _history.csv   每次训练的记录
    models/final/<模型>/<配置>_seed<种子>.pt     每次训练的最佳权重
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
from models import MODEL_NAMES, SPECS  # noqa: E402
from sweep_lstm import LR, MAX_EPOCHS, Logger, metrics_pct, pick_device, predict, write_json  # noqa: E402
from sweep_models import (check_norm_stats, device_name, load_train_val, make_settings,  # noqa: E402
                          measure_latencies, model_from_checkpoint, read_table, rel,
                          save_checkpoint, setup_torch, train_config)

ROOT = D.ROOT
SEEDS = [42, 43, 44, 45, 46]
SCOPES = ["all", *D.DRIVE_CYCLES]  # "all" = 全部测试点合在一起


def final_paths(root: Path = ROOT) -> dict:
    out = root / "results" / "final_eval"
    return {"tables": root / "results" / "tables", "out": out, "runs": out / "runs",
            "models": root / "models" / "final"}


def allow_test_reads() -> None:
    """训练全部结束之后才调用：关闭 lstm_data 的测试集保护，之后才能读测试文件。"""
    D._TEST_GUARD_ON = False


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def hp_text(window: int, hp: dict) -> str:
    return ", ".join([f"window={window}", *(f"{k}={v}" for k, v in hp.items())])


# ---------------------------------------------------------------------------
# 1. 选配置（只看验证集）
# ---------------------------------------------------------------------------
def candidate_tables(model: str, tables_dir: Path) -> list[Path]:
    main = tables_dir / f"sweep_{model}.csv"
    return [p for p in [main, *sorted(tables_dir.glob(f"sweep_{model}_lr*.csv"))] if p.exists()]


def select_best(model: str, tables: list[Path], tables_dir: Path, allow_incomplete: bool, log) -> dict:
    spec = SPECS[model]
    frames = []
    for p in tables:
        df = read_table(p, spec)
        if df is not None:
            frames.append(df.assign(source_table=rel(p)))
    if not frames:
        raise SystemExit(f"{model} 没有搜索结果，请先运行 python src/sweep_models.py --model {model}")
    main = tables_dir / f"sweep_{model}.csv"
    if main in tables:
        done = set(read_table(main, spec)["config"])
        missing = [spec.config_name(*c) for c in spec.grid() if spec.config_name(*c) not in done]
        if missing:
            msg = f"{rel(main)} 还差 {len(missing)} 组没跑完：{', '.join(missing)}"
            if not allow_incomplete:
                raise SystemExit(msg + "\n先跑完，或加 --allow-incomplete 只在已完成的组里选")
            log("注意：" + msg)
    df = pd.concat(frames, ignore_index=True)
    for col in ("eval_start", "batch_size", "max_epochs", "patience", "seed"):
        if df[col].nunique() != 1:
            raise SystemExit(f"{model} 的搜索表里 {col} 不统一：{sorted(df[col].unique())}，不能放在一起选")
    best = df.sort_values("val_rmse_pct", kind="stable").iloc[0]
    sel = {"model": model, "config": str(best["config"]), "window": int(best["window"]),
           "hp": {n: int(best[n]) for n in spec.hp_names}, "lr": float(best["lr"]),
           "eval_start": int(best["eval_start"]),
           "search_val_rmse_pct": float(best["val_rmse_pct"]),
           "search_val_mae_pct": float(best["val_mae_pct"]),
           "source_table": str(best["source_table"]), "n_candidates": len(df)}
    log(f"  {model:12s} {sel['config']:16s} lr {sel['lr']:g}  验证 RMSE {sel['search_val_rmse_pct']:.3f}%  "
        f"（{sel['n_candidates']} 组中最低，来自 {sel['source_table']}）")
    return sel


def run_name(sel: dict, seed: int) -> str:
    lr_part = "" if sel["lr"] == LR else "_lr{:g}".format(sel["lr"])
    return f"{sel['config']}{lr_part}_seed{seed}"


# ---------------------------------------------------------------------------
# 2. 重新训练（测试集保护打开）
# ---------------------------------------------------------------------------
def train_phase(selections: list[dict], seeds: list[int], train_seqs, val_seqs, stats: dict,
                folders: list[str], max_epochs: int, paths: dict, device, log) -> list[dict]:
    paths["runs"].mkdir(parents=True, exist_ok=True)
    runs, total, k = [], len(selections) * len(seeds), 0
    for sel in selections:
        spec = SPECS[sel["model"]]
        for seed in seeds:
            k += 1
            rn = run_name(sel, seed)
            ckpt_path = paths["models"] / sel["model"] / f"{rn}.pt"
            json_path = paths["runs"] / f"{sel['model']}_{rn}.json"
            settings = make_settings(sel["model"], sel["lr"], max_epochs, folders, sel["eval_start"], seed=seed)
            if json_path.exists() and ckpt_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if rec["settings"] != settings:
                    raise SystemExit(f"{rel(json_path)} 是用不同设置训练的，请删除它和对应的权重后重跑")
                log(f"[{k}/{total}] {sel['model']} {rn} 已训练（验证 RMSE {rec['val_rmse_pct']:.3f}%），跳过")
                runs.append(rec)
                continue
            log(f"[{k}/{total}] {sel['model']} {rn} 开始训练")
            res = train_config(spec, sel["window"], sel["hp"], train_seqs, val_seqs, stats, settings,
                               device, log, name=f"{sel['model']} {rn}")
            save_checkpoint(ckpt_path, spec, sel["window"], sel["hp"], res, stats, settings)
            pd.DataFrame(res["history"]).to_csv(paths["runs"] / f"{sel['model']}_{rn}_history.csv", index=False)
            rec = {"model": sel["model"], "config": sel["config"], "run": rn, "seed": seed, "lr": sel["lr"],
                   "window": sel["window"], "hp": sel["hp"],
                   "val_rmse_pct": res["val_rmse_pct"], "val_mae_pct": res["val_mae_pct"],
                   "best_epoch": res["best_epoch"], "epochs_trained": res["epochs_trained"],
                   "stopped_early": res["stopped_early"], "train_time_s": res["train_time"],
                   "n_params": res["n_params"], "n_val_points": res["n_val_points"],
                   "model_file": rel(ckpt_path), "device_name": device_name(device),
                   "torch_version": torch.__version__, "settings": settings,
                   "finished_at": datetime.now().isoformat(timespec="seconds")}
            write_json(json_path, rec)
            log(f"  [{sel['model']} {rn}] 完成：最佳轮 {res['best_epoch']}/{res['epochs_trained']}，"
                f"验证 RMSE {res['val_rmse_pct']:.3f}%，用时 {res['train_time'] / 60:.1f} min")
            runs.append(rec)
    return runs


# ---------------------------------------------------------------------------
# 3. 测试
# ---------------------------------------------------------------------------
def test_phase(runs: list[dict], test_seqs, stats: dict, device, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    quiet = lambda *_: None  # noqa: E731
    run_rows, file_rows = [], []
    for rec in runs:
        ckpt = torch.load(_abs(rec["model_file"]), map_location="cpu", weights_only=True)
        if not (np.allclose(ckpt["norm_stats"]["mean"], stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(ckpt["norm_stats"]["std"], stats["std"], rtol=1e-9, atol=0)):
            raise SystemExit(f"{rec['model_file']} 里的标准化参数与本次计算的不同")
        _, _, model = model_from_checkpoint(ckpt)
        model.to(device)
        window = int(ckpt["config"]["window"])
        start = max(int(rec["settings"]["eval_start_index"]), window - 1)

        errs = []  # (工况, 误差数组，SOC 百分点)
        for seq in test_seqs:
            ds = D.WindowDataset([seq], stats, window, start, quiet).to(device)
            pred = predict(model, ds).cpu().numpy().astype(np.float64)
            y = seq.y[ds.local_index]
            rmse, mae = metrics_pct(pred, y)
            errs.append((seq.info.test_name, (pred - y) * 100.0))
            file_rows.append({"model": rec["model"], "config": rec["config"], "lr": rec["lr"],
                              "seed": rec["seed"], "file": seq.info.id, "temperature": seq.info.folder,
                              "temp_c": seq.info.temp_c, "cycle": seq.info.test_name,
                              "n_points": len(y), "rmse_pct": rmse, "mae_pct": mae})
        row = {"model": rec["model"], "config": rec["config"], "lr": rec["lr"], "seed": rec["seed"],
               "window": rec["window"], "hyperparams": hp_text(rec["window"], rec["hp"]),
               "val_rmse_pct": rec["val_rmse_pct"], "val_mae_pct": rec["val_mae_pct"],
               "best_epoch": rec["best_epoch"], "epochs_trained": rec["epochs_trained"],
               "train_time_s": rec["train_time_s"], "n_params": rec["n_params"]}
        for scope in SCOPES:
            parts = [e for c, e in errs if scope == "all" or c == scope]
            e = np.concatenate(parts) if parts else np.array([])
            row[f"test_rmse_pct_{scope}"] = float(np.sqrt(np.mean(e ** 2))) if len(e) else math.nan
            row[f"test_mae_pct_{scope}"] = float(np.mean(np.abs(e))) if len(e) else math.nan
            row[f"n_test_points_{scope}"] = int(len(e))
        run_rows.append(row)
        log(f"  {rec['model']:12s} {rec['run']:26s} 测试 RMSE {row['test_rmse_pct_all']:.3f}%  "
            f"MAE {row['test_mae_pct_all']:.3f}%  （"
            + "，".join(f"{c} {row[f'test_rmse_pct_{c}']:.3f}" for c in D.DRIVE_CYCLES) + "）")
    return pd.DataFrame(run_rows), pd.DataFrame(file_rows)


# ---------------------------------------------------------------------------
# 4. 汇总
# ---------------------------------------------------------------------------
def summarize(selections: list[dict], run_df: pd.DataFrame, latencies: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fmt_rows = [], []
    for sel in selections:
        g = run_df[(run_df["model"] == sel["model"]) & (run_df["config"] == sel["config"])
                   & np.isclose(run_df["lr"], sel["lr"])]
        std = (lambda s: float(s.std(ddof=1)) if len(s) > 1 else math.nan)
        row = {"model": sel["model"], "config": sel["config"], "hyperparams": hp_text(sel["window"], sel["hp"]),
               "lr": sel["lr"], "search_val_rmse_pct": sel["search_val_rmse_pct"],
               "source_table": sel["source_table"], "n_seeds": len(g),
               "seeds": " ".join(str(s) for s in g["seed"]),
               "n_params": int(g["n_params"].iloc[0]), **latencies[sel["model"]]}
        fmt = {"model": sel["model"], "config": sel["config"], "hyperparams": row["hyperparams"],
               "lr": f"{sel['lr']:g}", "n_params": row["n_params"],
               "infer_ms_gpu": "-" if math.isnan(row["infer_ms_gpu"]) else f"{row['infer_ms_gpu']:.3f}",
               "infer_ms_cpu": f"{row['infer_ms_cpu']:.3f}"}
        for metric in ("rmse", "mae"):
            for scope in SCOPES:
                s = g[f"test_{metric}_pct_{scope}"]
                m, sd = float(s.mean()), std(s)
                row[f"test_{metric}_pct_{scope}_mean"] = m
                row[f"test_{metric}_pct_{scope}_std"] = sd
                fmt[f"test_{metric}_{scope} (%)"] = f"{m:.3f} ± {sd:.3f}" if not math.isnan(sd) else f"{m:.3f}"
        for col in ("val_rmse_pct", "best_epoch", "train_time_s"):
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = std(g[col])
        rows.append(row)
        fmt_rows.append(fmt)
    return pd.DataFrame(rows), pd.DataFrame(fmt_rows)


def measure_selected_latencies(selections: list[dict], runs: list[dict], log, **kw) -> dict:
    out = {}
    for sel in selections:
        rec = next(r for r in runs if r["model"] == sel["model"])  # 第一个种子（默认 42）的权重
        ckpt = torch.load(_abs(rec["model_file"]), map_location="cpu", weights_only=True)
        _, _, model = model_from_checkpoint(ckpt)
        out[sel["model"]] = measure_latencies(model, sel["window"], **kw)
        lat = out[sel["model"]]
        log(f"  {sel['model']:12s} 推理（batch 1）GPU "
            + ("-" if math.isnan(lat["infer_ms_gpu"]) else f"{lat['infer_ms_gpu']:.3f} ms")
            + f"，CPU {lat['infer_ms_cpu']:.3f} ms（{lat['cpu_threads']} 线程）")
    return out


def write_outputs(summary: pd.DataFrame, formatted: pd.DataFrame, run_df: pd.DataFrame,
                  file_df: pd.DataFrame, paths: dict) -> dict:
    paths["tables"].mkdir(parents=True, exist_ok=True)
    paths["out"].mkdir(parents=True, exist_ok=True)
    files = {"summary": paths["tables"] / "final_summary.csv",
             "formatted": paths["tables"] / "final_summary_formatted.csv",
             "runs": paths["tables"] / "final_runs.csv",
             "per_file": paths["out"] / "per_file_metrics.csv"}
    summary.to_csv(files["summary"], index=False, float_format="%.6g")
    formatted.to_csv(files["formatted"], index=False)
    run_df.to_csv(files["runs"], index=False, float_format="%.6g")
    file_df.to_csv(files["per_file"], index=False, float_format="%.6g")
    return files


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="四个模型的最终评估（多种子重训 + 测试集）")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--pick", action="append", default=[], metavar="MODEL=CSV",
                   help="指定某个模型只从这张搜索表里选，例如 transformer=results/tables/sweep_transformer.csv")
    p.add_argument("--allow-incomplete", action="store_true", help="搜索表没跑完完整网格时也继续")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    paths = final_paths()
    paths["out"].mkdir(parents=True, exist_ok=True)
    log = Logger(paths["out"] / "final_log.txt")
    device = pick_device(args.device)
    setup_torch(device)
    folders = list(D.TEMP_FOLDERS)
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  最终评估：模型 {args.models}，种子 {args.seeds}，设备 {device}"
        + (f"（{torch.cuda.get_device_name(device)}）" if device.type == "cuda" else ""))

    picks = {}
    for item in args.pick:
        m, _, path = item.partition("=")
        if m not in MODEL_NAMES or not path:
            raise SystemExit(f"--pick 的格式是 模型=CSV路径，收到 {item}")
        picks[m] = [_abs(path)]
    log("选定的配置（只看验证集）：")
    selections = [select_best(m, picks.get(m) or candidate_tables(m, paths["tables"]),
                              paths["tables"], args.allow_incomplete, log) for m in args.models]
    if len({s["eval_start"] for s in selections}) != 1:
        raise SystemExit("各模型搜索时的验证评估起点不同，不能公平比较")

    # ---- 训练：测试集保护打开 ----
    train_seqs, val_seqs, stats = load_train_val(folders, log)
    check_norm_stats(stats, paths["out"] / "norm_stats.json", log)
    runs = train_phase(selections, args.seeds, train_seqs, val_seqs, stats, folders, MAX_EPOCHS,
                       paths, device, log)

    # ---- 测试：全部训练结束之后才读测试集 ----
    allow_test_reads()
    test_infos = D.build_splits(folders)["test"]
    log(f"\n全部训练完成。读取测试集 {len(test_infos)} 个文件 ...")
    test_seqs = D.load_sequences(test_infos, log)
    run_df, file_df = test_phase(runs, test_seqs, stats, device, log)

    log("\n推理时间：")
    latencies = measure_selected_latencies(selections, runs, log)
    summary, formatted = summarize(selections, run_df, latencies)
    files = write_outputs(summary, formatted, run_df, file_df, paths)

    log(f"\n汇总（测试集，{len(args.seeds)} 个种子的均值 ± 标准差，SOC 百分点）：")
    log(formatted.to_string(index=False))
    log("\n已写入：" + "，".join(rel(f) for f in files.values()))
    log.close()


if __name__ == "__main__":
    main()
