"""LSTM 超参数对比。

- 基础网格（18 组）：W x H x L = {30, 60, 120} x {32, 64, 128} x {1, 2}，输出名 lstm_sweep
- 扩展网格（4 组，--ext）：W x H x L = {180, 240} x {32, 128} x {1}，输出名 lstm_sweep_ext

只用训练集训练、验证集选模型，完全不读测试集（读到测试文件会直接报错）。
可以中断后继续：已经跑完的组合（runs/ 里有结果 JSON 且 models/ 里有权重）直接跳过。

验证误差和早停只在"序列内行号 >= --eval-start"的点上计算（默认 239，即最长窗口 240
能覆盖的点）。最初的 18 组是用 119 跑的；要在 lstm_sweep 目录里续跑那 18 组，
必须加 --eval-start 119，否则会因为设置不一致而报错停止。

用法（在仓库根目录）：
    python src/sweep_lstm.py --ext              # 扩展的 4 组（评估起点 239）
    python src/sweep_lstm.py --ext --smoke      # 只用 25 degC、W180/W240 两组、每组 2 轮
    python src/sweep_lstm.py --eval-start 119   # 最初的 18 组（与当初的设置完全一致）
    python src/sweep_lstm.py --smoke            # 只用 25 degC、2 组、每组 2 轮

合并 18 + 4 组的结果表见 src/combine_lstm_sweep.py。

输出（默认 --name lstm_sweep）：
    results/lstm_sweep/sweep_results.csv     汇总表，按验证 RMSE 从小到大
    results/lstm_sweep/norm_stats.json       标准化参数（只用训练集计算）
    results/lstm_sweep/sweep_log.txt         运行日志
    results/lstm_sweep/runs/<配置>.json       每组的完整结果（也是"已完成"的标记）
    results/lstm_sweep/runs/<配置>_history.csv 每轮的 loss
    results/lstm_sweep/curves/<配置>.png      训练曲线
    models/lstm_sweep/<配置>.pt              验证 loss 最低那一轮的权重
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
# 扩展网格：最佳配置落在 W=120 的边界上，补跑更长的窗口
EXT_GRID_W = [180, 240]
EXT_GRID_H = [32, 128]
EXT_GRID_L = [1]

SEED = 42
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15
EVAL_BATCH_SIZE = 2048  # 只影响评估时的显存占用，不影响结果

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
    """同时输出到屏幕和日志文件。"""

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
        raise RuntimeError("指定了 --device cuda，但 torch.cuda.is_available() 为 False")
    return torch.device(arg)


def write_json(path: Path, obj) -> None:
    """先写临时文件再改名，中断时不会留下半个文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


@torch.no_grad()
def predict(model: torch.nn.Module, ds: D.WindowDataset) -> torch.Tensor:
    model.eval()
    return torch.cat([model(xb) for xb, _ in ds.iter_batches(EVAL_BATCH_SIZE)])


def metrics_pct(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """RMSE、MAE，单位为 SOC 百分点（误差 x 100）。"""
    err = (pred.astype(np.float64) - y.astype(np.float64)) * 100.0
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


# ---------------------------------------------------------------------------
def train_one(w, h, l, train_seqs, val_seqs, stats, settings, dirs, device, log) -> dict:
    name = config_name(w, h, l)
    set_seed(SEED)
    train_ds = D.WindowDataset(train_seqs, stats, w, None, log).to(device)
    val_ds = D.WindowDataset(val_seqs, stats, w, settings["eval_start_index"], log).to(device)
    y_val = val_ds.y[val_ds.ends]
    model = LSTMSOC(hidden_size=h, num_layers=l).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(SEED)  # 每轮打乱训练窗口顺序
    log(f"  训练窗口 {len(train_ds):,}，验证点 {len(val_ds):,}，参数量 {n_params:,}")

    max_epochs = settings["max_epochs"]
    best_loss, best_epoch, best_state, bad = math.inf, 0, None, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        te = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for xb, yb in train_ds.iter_batches(BATCH_SIZE, shuffle=True, generator=gen):
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
            + ("  *" if improved else f"  ({bad}/{PATIENCE})"))
        if bad >= PATIENCE:
            log(f"  [{name}] 验证 loss 连续 {PATIENCE} 轮没有创新低，停止")
            break
    train_time = time.perf_counter() - t0
    epochs_trained = len(history)

    # 用最佳一轮的权重计算验证指标
    model.load_state_dict(best_state)
    pred = predict(model, val_ds).cpu().numpy()
    y = y_val.cpu().numpy()
    rmse, mae = metrics_pct(pred, y)
    per_temp = {}
    for folder in D.TEMP_FOLDERS:
        mask = np.array([val_ds.seqs[k].info.folder == folder for k in val_ds.seq_of])
        if mask.any():
            r, m = metrics_pct(pred[mask], y[mask])
            per_temp[folder] = {"rmse_pct": r, "mae_pct": m, "n_points": int(mask.sum())}

    # 保存权重（附带重建模型和评估所需的全部信息）
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
    # 结果 JSON 最后写：它存在就表示这一组已经完整跑完
    write_json(dirs["runs"] / f"{name}.json", result)
    log(f"  [{name}] 完成：最佳轮 {best_epoch}/{epochs_trained}，验证 RMSE {rmse:.3f}%，"
        f"MAE {mae:.3f}%，用时 {train_time / 60:.1f} min")
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
    p = argparse.ArgumentParser(description="LSTM 超参数对比")
    p.add_argument("--ext", action="store_true",
                   help=f"跑扩展网格 W{EXT_GRID_W} x H{EXT_GRID_H} x L{EXT_GRID_L}，输出名 {EXT_NAME}")
    p.add_argument("--smoke", action="store_true",
                   help="快速检查：只用 25 degC、2 组、每组 2 轮，输出到 *_smoke 目录")
    p.add_argument("--windows", nargs="+", type=int, default=None, help="只跑这些 W")
    p.add_argument("--hidden", nargs="+", type=int, default=None, help="只跑这些 H")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="只跑这些 L")
    p.add_argument("--eval-start", type=int, default=D.EVAL_START_INDEX,
                   help="验证误差和早停的评估起点（序列内行号），默认 "
                        f"{D.EVAL_START_INDEX}；最初的 18 组用的是 {D.EVAL_START_INDEX_ORIGINAL}")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="调试用：覆盖最大轮数（必须同时用 --name 指定另一个输出名）")
    p.add_argument("--name", default=None,
                   help=f"输出子目录名，默认 {DEFAULT_NAME}（--ext 时为 {EXT_NAME}）")
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
            raise SystemExit(f"不在{'扩展' if args.ext else '基础'}网格里的取值：{bad}"
                             + ("" if args.ext else "（W=180、240 要加 --ext）"))
        configs = [(w, h, l) for w in ws for h in hs for l in ls]
        name = args.name or formal_name
    if name in (DEFAULT_NAME, EXT_NAME) and max_epochs != MAX_EPOCHS:
        raise SystemExit("改了 --max-epochs 时必须用 --name 换一个输出目录，以免和正式结果混在一起")
    max_w = max(c[0] for c in configs)
    if args.eval_start < max_w - 1:
        raise SystemExit(f"--eval-start {args.eval_start} 小于最长窗口 {max_w} 的 {max_w - 1}，"
                         "各窗口长度就不是在同一批点上比较了")

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
    log(f"torch {torch.__version__}，设备 {device}"
        + (f"（{torch.cuda.get_device_name(device)}）" if device.type == "cuda" else
           f"（CPU 线程 {torch.get_num_threads()}）"))
    log(f"配置 {len(configs)} 组：{', '.join(config_name(*c) for c in configs)}")
    log(f"验证误差和早停的评估起点：序列内第 {args.eval_start} 行及之后")

    # ---- 划分，并打开测试集保护 ----
    D.block_test_files()
    splits = D.build_splits(folders)
    try:  # 自检：保护确实生效
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
    stats_path = dirs["results"] / "norm_stats.json"
    if stats_path.exists():
        old = D.load_norm_stats(stats_path)
        if old["computed_from"] != stats["computed_from"] or not (
                np.allclose(old["mean"], stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(old["std"], stats["std"], rtol=1e-9, atol=0)):
            raise SystemExit(f"{stats_path} 与本次计算的标准化参数不一致（数据变了？），"
                             "请删除该目录下的旧结果后重跑")
    else:
        D.save_norm_stats(stats, stats_path)
    log("标准化参数（训练集）：" + "，".join(
        f"{f} mean={m:.4f} std={s:.4f}" for f, m, s in zip(D.FEATURES, stats["mean"], stats["std"])))

    # ---- 开始训练前，先确认这个目录里已有的结果都是用同样的设置跑的 ----
    for run_json in sorted(dirs["runs"].glob("W*_H*_L*.json")):
        with open(run_json, "r", encoding="utf-8") as f:
            done_settings = json.load(f)["settings"]
        if done_settings != settings:
            diff = {k: (done_settings.get(k), settings.get(k))
                    for k in set(done_settings) | set(settings)
                    if done_settings.get(k) != settings.get(k)}
            hint = ""
            if set(diff) == {"eval_start_index"}:
                hint = (f"。{name} 里已有的结果是用 --eval-start {done_settings['eval_start_index']} "
                        "跑的，续跑请加上这个参数")
            raise SystemExit(f"{run_json.relative_to(ROOT).as_posix()} 是用不同设置跑的"
                             f"（已有 vs 本次：{diff}）{hint}")

    # ---- 逐组训练，已完成的跳过 ----
    for i, (w, h, l) in enumerate(configs, 1):
        cname = config_name(w, h, l)
        run_json = dirs["runs"] / f"{cname}.json"
        if run_json.exists() and (dirs["models"] / f"{cname}.pt").exists():
            with open(run_json, "r", encoding="utf-8") as f:
                done = json.load(f)
            log(f"[{i}/{len(configs)}] {cname} 已完成（验证 RMSE {done['val_rmse_pct']:.3f}%），跳过")
            continue
        log(f"[{i}/{len(configs)}] {cname} 开始训练")
        train_one(w, h, l, train_seqs, val_seqs, stats, settings, dirs, device, log)
        write_summary(dirs, folders, log)

    df = write_summary(dirs, folders, log)
    if df is not None:
        show = ["rank", "config", "val_rmse_pct", "val_mae_pct", "best_epoch",
                "epochs_trained", "train_time_s", "n_params"]
        log("\n汇总（按验证 RMSE 排序，单位 %）：")
        log(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        log(f"\n已写入 {(dirs['results'] / 'sweep_results.csv').relative_to(ROOT).as_posix()}")
    log.close()


if __name__ == "__main__":
    main()
