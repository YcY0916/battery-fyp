"""四个模型（lstm / gru / cnn / transformer）统一的超参数搜索。

数据流程与 sweep_lstm.py 完全相同（直接调用同一套函数）：lstm_data 的读取、划分、1 s 重采样、
只用训练集计算的标准化、窗口 Dataset；训练循环是 sweep_lstm.fit，验证指标是 sweep_lstm.val_metrics。
只用训练集训练、验证集早停和选模型；测试集保护打开，读到测试文件会直接报错。

固定设置（与 LSTM 相同）：Adam，lr 0.001（可用 --lr 覆盖），batch 256，MSE loss，最多 200 轮，
早停 patience 15，seed 42；验证误差和早停只在"序列内行号 >= 239"的点上计算（与 LSTM 的 22 组表相同）。
网格见 src/models.py。

用法（在仓库根目录）：
    python src/sweep_models.py --model gru
    python src/sweep_models.py --model cnn
    python src/sweep_models.py --model transformer
    python src/sweep_models.py --model transformer --lr 0.0005    # 另存为 *_lr0.0005，不和 0.001 的结果混在一起
    python src/sweep_models.py --model lstm     # 先用 import_lstm_sweep.py 导入已有的 22 组，这里只补跑其余 8 组
    python src/sweep_models.py --model gru --windows 30 60          # 只跑部分窗口（可以分几次跑）

每跑完一组就往 CSV 追加一行。中断后输入同一条命令即可续跑：CSV 里已有的配置直接跳过，
中断时正在跑的那一组从第 1 轮重新开始。

输出（<tag> 是模型名；--lr 不是 0.001 时为 <模型名>_lr<lr>，例如 transformer_lr0.0005）：
    results/tables/sweep_<tag>.csv            每组一行：超参数、验证 RMSE/MAE、最佳轮、训练用时、参数量、推理时间
    models/<tag>/<配置>.pt                    验证 loss 最低那一轮的权重（附带配置、标准化参数和训练设置）
    results/search/<tag>/sweep_log.txt        运行日志
    results/search/<tag>/norm_stats.json      标准化参数（只用训练集计算）
    results/search/<tag>/history/<配置>.csv   每轮的 loss
    results/search/<tag>/curves/<配置>.png    训练曲线
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
from models import INPUT_SIZE, MODEL_NAMES, SPECS, ModelSpec, count_parameters  # noqa: E402
from sweep_lstm import (BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, SEED, Logger,  # noqa: E402
                        fit, pick_device, plot_curve, set_seed, val_metrics)

ROOT = D.ROOT
TABLES_DIR = ROOT / "results" / "tables"
# LSTM 对比实验保存的标准化参数：新模型算出来的必须与它完全相同
LSTM_NORM_STATS = [ROOT / "results" / "lstm_sweep_ext" / "norm_stats.json",
                   ROOT / "results" / "lstm_sweep" / "norm_stats.json"]

LATENCY_WARMUP = 50   # 推理计时前的预热次数
LATENCY_ITERS = 300   # 计时次数，取平均


# ---------------------------------------------------------------------------
# 路径、设置、CSV
# ---------------------------------------------------------------------------
def run_tag(model: str, lr: float) -> str:
    return model if lr == LR else f"{model}_lr{lr:g}"


def output_paths(tag: str, root: Path = ROOT) -> dict:
    search = root / "results" / "search" / tag
    return {"csv": root / "results" / "tables" / f"sweep_{tag}.csv",
            "models": root / "models" / tag,
            "search": search, "history": search / "history", "curves": search / "curves"}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def make_settings(model: str, lr: float, max_epochs: int, folders: list[str],
                  eval_start: int, seed: int = SEED) -> dict:
    return {"model": model, "folders": list(folders), "features": D.FEATURES, "target": D.TARGET,
            "resample": "1 s, floor(time_s); V/I/T mean, soc last",
            "eval_start_index": eval_start, "seed": seed, "lr": lr,
            "batch_size": BATCH_SIZE, "max_epochs": max_epochs, "patience": PATIENCE,
            "optimizer": "Adam", "loss": "MSE", "model_fixed": SPECS[model].fixed}


def csv_columns(spec: ModelSpec) -> list[str]:
    return ["model", "config", "window", *spec.hp_names,
            "lr", "batch_size", "seed", "max_epochs", "patience", "loss",
            "eval_start", "early_stop_eval_start",
            "val_rmse_pct", "val_mae_pct", *[f"val_rmse_pct_{f}" for f in D.TEMP_FOLDERS],
            "best_epoch", "epochs_trained", "stopped_early", "train_time_s", "n_params",
            "infer_ms_gpu", "infer_ms_cpu", "infer_gpu_name", "cpu_threads",
            "n_train_windows", "n_val_points", "device_name", "source", "model_file", "finished_at"]


def read_table(path: Path, spec: ModelSpec) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if list(df.columns) != csv_columns(spec):
        raise SystemExit(f"{rel(path)} 的列与 {spec.name} 的搜索表不一致，请检查是不是放错了文件")
    if (df["model"] != spec.name).any():
        raise SystemExit(f"{rel(path)} 里有不是 {spec.name} 的行")
    return df


def append_row(path: Path, spec: ModelSpec, row: dict) -> None:
    cols = csv_columns(spec)
    if set(row) != set(cols):
        raise ValueError(f"CSV 行的字段不对：多 {set(row) - set(cols)}，少 {set(cols) - set(row)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row], columns=cols).to_csv(path, mode="a", header=header, index=False,
                                             float_format="%.6g")


def check_table_settings(df: pd.DataFrame, settings: dict, path: Path) -> None:
    """CSV 里已有的行必须和本次用同样的设置跑（否则不能放在一张表里比较）。"""
    want = {"lr": settings["lr"], "batch_size": settings["batch_size"], "seed": settings["seed"],
            "max_epochs": settings["max_epochs"], "patience": settings["patience"],
            "eval_start": settings["eval_start_index"]}
    for col, v in want.items():
        bad = df.loc[~np.isclose(df[col].astype(float), float(v), rtol=1e-9, atol=0), "config"]
        if len(bad):
            raise SystemExit(f"{rel(path)} 里 {list(bad)} 的 {col} 与本次的 {v} 不同，"
                             "不能续跑到同一张表（改了设置请换 --lr 或 --tag）")


def check_norm_stats(stats: dict, own_path: Path, log, extra_refs: list[Path] = LSTM_NORM_STATS) -> None:
    """与本目录之前保存的、以及 LSTM 实验保存的标准化参数逐一比对，保证数据流程完全相同。"""
    for ref in [own_path, *extra_refs]:
        if not ref.exists():
            continue
        old = D.load_norm_stats(ref)
        same = (old["computed_from"] == stats["computed_from"]
                and np.allclose(old["mean"], stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(old["std"], stats["std"], rtol=1e-9, atol=0))
        if not same:
            raise SystemExit(f"本次计算的标准化参数与 {rel(ref)} 不一致（数据或划分变了？）")
        log(f"标准化参数与 {rel(ref)} 一致")
    if not own_path.exists():
        D.save_norm_stats(stats, own_path)


# ---------------------------------------------------------------------------
# 训练一组、保存、推理计时
# ---------------------------------------------------------------------------
def train_config(spec: ModelSpec, window: int, hp: dict, train_seqs, val_seqs, stats: dict,
                 settings: dict, device, log, name: str | None = None) -> dict:
    """与 sweep_lstm.train_one 相同的步骤和随机数顺序：设种子 -> 切窗口 -> 建模型 -> fit -> 验证指标。"""
    name = name or spec.config_name(window, hp)
    set_seed(settings["seed"])
    train_ds = D.WindowDataset(train_seqs, stats, window, None, log).to(device)
    val_ds = D.WindowDataset(val_seqs, stats, window, settings["eval_start_index"], log).to(device)
    model = spec.build(hp).to(device)
    n_params = count_parameters(model)
    log(f"  训练窗口 {len(train_ds):,}，验证点 {len(val_ds):,}，参数量 {n_params:,}")
    fitted = fit(model, train_ds, val_ds, name, settings["max_epochs"], device, log,
                 seed=settings["seed"], lr=settings["lr"], batch_size=settings["batch_size"],
                 patience=settings["patience"])
    rmse, mae, per_temp = val_metrics(model, val_ds)
    return {**fitted, "model": model, "val_rmse_pct": rmse, "val_mae_pct": mae,
            "val_per_temperature": per_temp, "n_params": n_params,
            "n_train_windows": len(train_ds), "n_val_points": len(val_ds),
            "epochs_trained": len(fitted["history"]),
            "stopped_early": len(fitted["history"]) < settings["max_epochs"]}


def save_checkpoint(path: Path, spec: ModelSpec, window: int, hp: dict, result: dict,
                    stats: dict, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"state_dict": result["best_state"], "model": spec.name,
                "config": {"window": window, **hp, "input_size": INPUT_SIZE},
                "features": D.FEATURES, "norm_stats": stats, "settings": settings,
                "best_epoch": result["best_epoch"], "val_rmse_pct": result["val_rmse_pct"]}, tmp)
    tmp.replace(path)


def model_from_checkpoint(ckpt: dict) -> tuple[ModelSpec, dict, torch.nn.Module]:
    """重建模型并载入权重。LSTM 旧权重（sweep_lstm.py 保存的）没有 "model" 字段，按 lstm 处理。"""
    spec = SPECS[ckpt.get("model", "lstm")]
    hp = {k: int(ckpt["config"][k]) for k in spec.hp_names}
    model = spec.build(hp)
    model.load_state_dict(ckpt["state_dict"])
    return spec, hp, model


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, window: int, device: torch.device,
                       warmup: int = LATENCY_WARMUP, iters: int = LATENCY_ITERS) -> float:
    """单个样本（batch = 1）的平均推理时间，毫秒：先预热 warmup 次，再连续计时 iters 次取平均。"""
    m = copy.deepcopy(model).to(device).eval()
    # 用独立的 Generator 生成输入，不影响全局随机数状态
    x = torch.randn(1, window, INPUT_SIZE, generator=torch.Generator().manual_seed(0)).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        m(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        m(x)
    sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def measure_latencies(model: torch.nn.Module, window: int, **kw) -> dict:
    """GPU（有的话）和 CPU 各测一次。"""
    has_gpu = torch.cuda.is_available()
    return {"infer_ms_gpu": measure_latency_ms(model, window, torch.device("cuda"), **kw) if has_gpu else math.nan,
            "infer_ms_cpu": measure_latency_ms(model, window, torch.device("cpu"), **kw),
            "infer_gpu_name": torch.cuda.get_device_name(0) if has_gpu else "",
            "cpu_threads": torch.get_num_threads()}


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def run_search(spec: ModelSpec, configs: list[tuple[int, dict]], train_seqs, val_seqs, stats: dict,
               settings: dict, paths: dict, device, log, latency_kw: dict | None = None) -> pd.DataFrame | None:
    """逐组训练；CSV 里已有的配置跳过；每组跑完立刻往 CSV 追加一行。"""
    for key in ("models", "history", "curves"):
        paths[key].mkdir(parents=True, exist_ok=True)
    df = read_table(paths["csv"], spec)
    done: dict[str, float] = {}
    if df is not None:
        check_table_settings(df, settings, paths["csv"])
        done = dict(zip(df["config"], df["val_rmse_pct"]))
        log(f"{rel(paths['csv'])} 里已有 {len(done)} 组，将跳过")

    for i, (w, hp) in enumerate(configs, 1):
        cname = spec.config_name(w, hp)
        if cname in done:
            log(f"[{i}/{len(configs)}] {cname} 已完成（验证 RMSE {done[cname]:.3f}%），跳过")
            continue
        log(f"[{i}/{len(configs)}] {cname} 开始训练")
        res = train_config(spec, w, hp, train_seqs, val_seqs, stats, settings, device, log, name=cname)

        ckpt_path = paths["models"] / f"{cname}.pt"
        save_checkpoint(ckpt_path, spec, w, hp, res, stats, settings)
        hist = pd.DataFrame(res["history"])
        hist.to_csv(paths["history"] / f"{cname}.csv", index=False)
        plot_curve(hist, res["best_epoch"], f"{spec.name}_{cname}", paths["curves"] / f"{cname}.png")
        lat = measure_latencies(res["model"], w, **(latency_kw or {}))

        row = {"model": spec.name, "config": cname, "window": w, **hp,
               "lr": settings["lr"], "batch_size": settings["batch_size"], "seed": settings["seed"],
               "max_epochs": settings["max_epochs"], "patience": settings["patience"],
               "loss": settings["loss"], "eval_start": settings["eval_start_index"],
               "early_stop_eval_start": settings["eval_start_index"],
               "val_rmse_pct": res["val_rmse_pct"], "val_mae_pct": res["val_mae_pct"],
               **{f"val_rmse_pct_{f}": res["val_per_temperature"].get(f, {}).get("rmse_pct", math.nan)
                  for f in D.TEMP_FOLDERS},
               "best_epoch": res["best_epoch"], "epochs_trained": res["epochs_trained"],
               "stopped_early": res["stopped_early"], "train_time_s": res["train_time"],
               "n_params": res["n_params"], **lat,
               "n_train_windows": res["n_train_windows"], "n_val_points": res["n_val_points"],
               "device_name": device_name(device), "source": "sweep_models",
               "model_file": rel(ckpt_path), "finished_at": datetime.now().isoformat(timespec="seconds")}
        append_row(paths["csv"], spec, row)
        gpu_txt = "-" if math.isnan(lat["infer_ms_gpu"]) else f"{lat['infer_ms_gpu']:.3f} ms"
        log(f"  [{cname}] 完成：最佳轮 {res['best_epoch']}/{res['epochs_trained']}，"
            f"验证 RMSE {res['val_rmse_pct']:.3f}%，MAE {res['val_mae_pct']:.3f}%，"
            f"用时 {res['train_time'] / 60:.1f} min，推理 GPU {gpu_txt} / CPU {lat['infer_ms_cpu']:.3f} ms")
    return read_table(paths["csv"], spec)


def load_train_val(folders: list[str], log) -> tuple[list, list, dict]:
    """打开测试集保护（并自检），只读训练集和验证集，用训练集计算标准化参数。"""
    D.block_test_files()
    splits = D.build_splits(folders)
    try:
        D.read_raw(splits["test"][0])
        raise AssertionError("测试集保护没有生效")
    except D.TestSetAccessError:
        log(f"测试集保护已开启：{len(splits['test'])} 个测试文件不会被读取（已自检）")
    for split in ("train", "val"):
        log(f"{split}（{len(splits[split])} 条）：" + "，".join(s.id for s in splits[split]))
    log("读取训练集 ...")
    train_seqs = D.load_sequences(splits["train"], log)
    log("读取验证集 ...")
    val_seqs = D.load_sequences(splits["val"], log)
    stats = D.compute_norm_stats(train_seqs)
    log("标准化参数（训练集）：" + "，".join(
        f"{f} mean={m:.4f} std={s:.4f}" for f, m, s in zip(D.FEATURES, stats["mean"], stats["std"])))
    return train_seqs, val_seqs, stats


def setup_torch(device: torch.device) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="lstm / gru / cnn / transformer 统一的超参数搜索（只用验证集）")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--lr", type=float, default=LR, help=f"学习率，默认 {LR}；改了会另存为 <模型>_lr<lr>")
    p.add_argument("--windows", nargs="+", type=int, default=None, help="只跑这些窗口长度")
    p.add_argument("--only", nargs="+", default=None, help="只跑这些配置，例如 W240_H64_L2")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="调试用：覆盖最大轮数（必须同时用 --tag 换一个输出名）")
    p.add_argument("--tag", default=None, help="调试用：输出名，默认 <模型>（或 <模型>_lr<lr>）")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--allow-lstm-without-import", action="store_true",
                   help="--model lstm 时不检查是否已导入旧的 22 组（会把完整的 30 组全部重新训练）")
    args = p.parse_args()

    spec = SPECS[args.model]
    default_tag = run_tag(args.model, args.lr)
    tag = args.tag or default_tag
    max_epochs = args.max_epochs or MAX_EPOCHS
    if max_epochs != MAX_EPOCHS and tag == default_tag:
        raise SystemExit("改了 --max-epochs 时必须用 --tag 换一个输出名，以免和正式结果混在一起")

    configs = spec.grid()
    if args.windows:
        bad = [w for w in args.windows if w not in {c[0] for c in configs}]
        if bad:
            raise SystemExit(f"不在网格里的窗口长度：{bad}")
        configs = [c for c in configs if c[0] in args.windows]
    if args.only:
        names = {spec.config_name(w, hp) for w, hp in configs}
        bad = [n for n in args.only if n not in names]
        if bad:
            raise SystemExit(f"不在网格里的配置：{bad}（可选：{sorted(names)}）")
        configs = [c for c in configs if spec.config_name(*c) in args.only]

    paths = output_paths(tag)
    if tag == "lstm" and not args.allow_lstm_without_import:
        df = read_table(paths["csv"], spec)
        n_imported = 0 if df is None else int(df["source"].astype(str).str.startswith("imported").sum())
        if n_imported == 0:
            raise SystemExit("results/tables/sweep_lstm.csv 里还没有导入旧的 22 组 LSTM 结果。请先运行\n"
                             "  python src/import_lstm_sweep.py --verify\n  python src/import_lstm_sweep.py\n"
                             "再用本脚本补跑其余 8 组（确实要全部重跑请加 --allow-lstm-without-import）")

    eval_start = D.EVAL_START_INDEX
    max_w = max(c[0] for c in spec.grid())
    if eval_start < max_w - 1:
        raise SystemExit(f"评估起点 {eval_start} 小于最长窗口 {max_w} 的 {max_w - 1}")

    paths["search"].mkdir(parents=True, exist_ok=True)
    log = Logger(paths["search"] / "sweep_log.txt")
    folders = list(D.TEMP_FOLDERS)
    settings = make_settings(args.model, args.lr, max_epochs, folders, eval_start)
    device = pick_device(args.device)
    setup_torch(device)
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  模型 {args.model}，输出名 '{tag}'，lr {args.lr:g}")
    log(f"torch {torch.__version__}，设备 {device}"
        + (f"（{torch.cuda.get_device_name(device)}）" if device.type == "cuda" else
           f"（CPU 线程 {torch.get_num_threads()}）"))
    log(f"配置 {len(configs)} 组：{', '.join(spec.config_name(*c) for c in configs)}")
    log(f"验证误差和早停的评估起点：序列内第 {eval_start} 行及之后")

    train_seqs, val_seqs, stats = load_train_val(folders, log)
    check_norm_stats(stats, paths["search"] / "norm_stats.json", log)

    df = run_search(spec, configs, train_seqs, val_seqs, stats, settings, paths, device, log)
    if df is not None:
        df = df.sort_values("val_rmse_pct", kind="stable")
        show = ["config", "val_rmse_pct", "val_mae_pct", "best_epoch", "epochs_trained",
                "train_time_s", "n_params", "infer_ms_gpu", "infer_ms_cpu", "source"]
        log(f"\n{rel(paths['csv'])}（{len(df)} 组，按验证 RMSE 排序，单位 %）：")
        log(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        missing = [spec.config_name(*c) for c in spec.grid() if spec.config_name(*c) not in set(df["config"])]
        if missing:
            log(f"完整网格还差 {len(missing)} 组：{', '.join(missing)}")
    log.close()


if __name__ == "__main__":
    main()
