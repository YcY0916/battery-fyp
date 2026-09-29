"""把最初的 18 组和扩展的 4 组合并成一张 22 组的结果表（不重新训练）。

最初 18 组（results/lstm_sweep、models/lstm_sweep）的验证误差是在"序列内第 119 行及之后"的点上算的；
扩展的 4 组（results/lstm_sweep_ext、models/lstm_sweep_ext）有 W=240，要统一在第 239 行及之后比较。
本脚本读入全部 22 个已保存的模型，只在验证集上用同一个评估起点（默认 239）重新计算
验证 RMSE / MAE 和各温度的 RMSE，然后合并排序。

- 只读验证集，测试集保护与 sweep_lstm.py 相同（读到测试文件会直接报错）。
- 不修改 results/lstm_sweep/ 里的任何文件，旧的 18 组结果表原样保留。
- 同时在各组自己训练时用的评估起点上重算一次，与当时记录的验证 RMSE 对照，确认模型读取无误。

用法（在仓库根目录）：
    python src/combine_lstm_sweep.py
    python src/combine_lstm_sweep.py --smoke     # 合并两个 smoke 目录，只检查代码能否跑通

输出：
    results/lstm_sweep_ext/sweep_results_22.csv   22 组汇总，按验证 RMSE 从小到大
    results/lstm_sweep_ext/combine_log.txt        运行日志
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
CHECK_TOL_PCT = 0.01  # 重算的验证 RMSE 与训练时记录值之差超过它就提醒


def per_temperature(pred, y, ds):
    out = {}
    folder_of = np.array([s.info.folder for s in ds.seqs])[ds.seq_of]
    for folder in D.TEMP_FOLDERS:
        m = folder_of == folder
        if m.any():
            out[folder] = metrics_pct(pred[m], y[m])[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="合并 18 + 4 组 LSTM 对比结果（不重新训练，只重算验证误差）")
    p.add_argument("--eval-start", type=int, default=D.EVAL_START_INDEX,
                   help=f"统一的验证评估起点（序列内行号），默认 {D.EVAL_START_INDEX}")
    p.add_argument("--smoke", action="store_true",
                   help=f"合并 {SMOKE_NAME} 和 {EXT_SMOKE_NAME}（有几组算几组），只检查代码")
    p.add_argument("--allow-missing", action="store_true",
                   help="有组合还没跑完时也生成表（只包含已完成的组合）")
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
    D.block_test_files()  # 最先打开测试集保护
    device = pick_device(args.device)
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  合并 {base_name} + {ext_name}，"
        f"统一评估起点 {args.eval_start}，设备 {device}")

    # ---- 找出已完成的组合 ----
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
        raise SystemExit("以下组合还没有跑完（或缺少结果 JSON / 权重）：\n  " + "\n  ".join(missing)
                         + "\n先跑完它们，或加 --allow-missing 只合并已完成的组合")
    if not found:
        raise SystemExit("没有找到任何已完成的组合")
    log(f"找到 {len(found)} 组" + (f"，缺少 {len(missing)} 组：{missing}" if missing else ""))

    runs = {}
    for name, cname, run_json, ckpt_path in found:
        with open(run_json, "r", encoding="utf-8") as f:
            r = json.load(f)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        runs[(name, cname)] = (r, ckpt)

    # ---- 一致性检查：除评估起点外设置相同；标准化参数相同；评估起点覆盖最长窗口 ----
    ref_settings = {k: v for k, v in next(iter(runs.values()))[0]["settings"].items()
                    if k != "eval_start_index"}
    ref_stats = next(iter(runs.values()))[1]["norm_stats"]
    for (name, cname), (r, ckpt) in runs.items():
        s = {k: v for k, v in r["settings"].items() if k != "eval_start_index"}
        if s != ref_settings:
            raise SystemExit(f"{name}/{cname} 的训练设置与其他组不同：{s} vs {ref_settings}")
        if list(ckpt["features"]) != D.FEATURES:
            raise SystemExit(f"{name}/{cname} 的输入特征是 {ckpt['features']}")
        st = ckpt["norm_stats"]
        if not (np.allclose(st["mean"], ref_stats["mean"], rtol=1e-9, atol=0)
                and np.allclose(st["std"], ref_stats["std"], rtol=1e-9, atol=0)):
            raise SystemExit(f"{name}/{cname} 的标准化参数与其他组不同")
    for name in sources:
        sp = ROOT / "results" / name / "norm_stats.json"
        if sp.exists():
            st = D.load_norm_stats(sp)
            if not (np.allclose(st["mean"], ref_stats["mean"], rtol=1e-9, atol=0)
                    and np.allclose(st["std"], ref_stats["std"], rtol=1e-9, atol=0)):
                raise SystemExit(f"{sp.relative_to(ROOT)} 与模型里的标准化参数不一致")
    for (name, cname), (r, _) in runs.items():
        if name == ext_name and r["settings"]["eval_start_index"] != args.eval_start:
            raise SystemExit(f"{name}/{cname} 训练时的评估起点是 {r['settings']['eval_start_index']}，"
                             f"与本次的 {args.eval_start} 不同")
    max_w = max(r["window"] for r, _ in runs.values())
    if args.eval_start < max_w - 1:
        raise SystemExit(f"--eval-start {args.eval_start} 小于最长窗口 {max_w} 的 {max_w - 1}")
    log("一致性检查通过：训练设置（评估起点除外）和标准化参数全部相同")

    # ---- 只读验证集 ----
    folders = ref_settings["folders"]
    splits = D.build_splits(folders)
    try:
        D.read_raw(splits["test"][0])
        raise AssertionError("测试集保护没有生效")
    except D.TestSetAccessError:
        log(f"测试集保护已开启：{len(splits['test'])} 个测试文件不会被读取（已自检）")
    log("读取验证集 ...")
    val_seqs = D.load_sequences(splits["val"], log)

    # ---- 逐组重算 ----
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
        flag = "" if abs(diff) <= CHECK_TOL_PCT else "  <-- 与记录值相差较大，请检查"
        log(f"  {name}/{cname}: 起点 {args.eval_start} 的验证 RMSE {rmse:.3f}%  MAE {mae:.3f}%  "
            f"| 起点 {trained_start} 重算 {check_rmse:.3f}% vs 记录 {r['val_rmse_pct']:.3f}%{flag}")

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
    log(f"\n{len(df)} 组汇总（验证点：序列内第 {args.eval_start} 行及之后，共 {df['n_val_points'].iloc[0]:,} 个；单位 %）：")
    log(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\n说明：val_rmse_pct_recorded 是各组训练时记录的验证 RMSE（评估起点见 early_stop_eval_start），"
        "不能和 val_rmse_pct 直接比较。")
    log(f"已写入 {out_path.relative_to(ROOT).as_posix()}（{base_name}/sweep_results.csv 未改动）")
    log.close()


if __name__ == "__main__":
    main()
