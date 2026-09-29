"""smoke test：不需要真实数据，也不需要 GPU，CPU 上几分钟内跑完。

检查内容：
1. 前向传播：四个模型 x 各自完整网格的全部结构超参数 x 窗口 {30, 240}，随机张量输入 (batch, W, 3)，
   输出形状必须是 (batch,)；输出头都是 Linear(·, 1)；位置编码覆盖最长窗口，超出时报错。
2. 训练和搜索流程（用随机生成的小序列，走真实的标准化、WindowDataset、sweep_lstm.fit）：
   每个模型 2 组配置、每组 2 轮 -> CSV 追加、权重、推理时间；再跑一次确认断点续跑会跳过已完成的配置；
   --lr 覆盖时另存一张表。
3. LSTM 行为不变：同一份假数据上，原来的 sweep_lstm.train_one 与统一流程训练出的权重、每轮 loss 逐位相同。
4. 导入旧 LSTM 结果：核对（重跑一组旧配置、对照记录值）-> 导入 -> sweep_models 只补跑缺少的配置。
5. 最终评估流程：选最佳配置 -> 2 个种子重训 -> 按工况评估 -> 汇总表；续跑时不重新训练。
6. 测试集保护：打开后读取测试文件会报错（需要仓库里的 manifest.json，只读 manifest，不读数据）。

所有输出写到 results/ 下一个以 _smoke 结尾的临时目录（已被 .gitignore 忽略），结束后删除。

用法（在仓库根目录）：
    python src/smoke_test_models.py
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
import final_eval as FE  # noqa: E402
import import_lstm_sweep as IM  # noqa: E402
import sweep_lstm as SL  # noqa: E402
import sweep_models as SM  # noqa: E402
from models import POS_MAX_LEN, SPECS, WINDOWS, TransformerSOC  # noqa: E402

ROOT = D.ROOT
SMOKE_EPOCHS = 2
SMOKE_EVAL_START = 19          # 假数据上的评估起点（最长窗口 20 - 1）
SMOKE_WINDOWS = (10, 20)
LAT_KW = {"warmup": 2, "iters": 5}  # 推理计时只检查能跑通，次数取小
FOLDERS = ["n10degC", "25degC"]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f"：{detail}" if detail else ""), flush=True)


class Quiet:
    """收集日志，不打印。"""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        self.lines.append(msg)


# ---------------------------------------------------------------------------
# 假数据
# ---------------------------------------------------------------------------
def fake_sequences(split: str, n: int, rng: np.random.Generator, names: list[str] | None = None) -> list:
    seqs = []
    for i in range(n):
        folder = FOLDERS[i % len(FOLDERS)]
        length = 180 + int(rng.integers(0, 60))
        soc = np.clip(1.0 - np.cumsum(rng.uniform(0, 3e-3, length)), 0, 1)
        current = rng.normal(-1.0, 1.0, length)
        voltage = 3.0 + 1.2 * soc + 0.05 * current + rng.normal(0, 0.01, length)
        temp = (25.0 if folder == "25degC" else -10.0) + rng.normal(0, 0.5, length)
        x = np.stack([voltage, current, temp], axis=1)
        test_name = names[i % len(names)] if names else f"Mixed{i + 1}"
        info = D.SeqInfo(id=f"{folder}/fake_{split}{i}_{test_name}", file=f"{folder}/fake_{split}{i}.csv",
                         folder=folder, temp_c=25 if folder == "25degC" else -10,
                         test_type="drive" if split == "test" else "mixed", test_name=test_name,
                         part=1, n_parts=1, split=split)
        seqs.append(D.Sequence(info, np.arange(length, dtype=np.int64), x, soc))
    return seqs


def smoke_configs(spec) -> list[tuple[int, dict]]:
    """每个模型 2 组：窗口 10 配网格里第一组超参数，窗口 20 配最后一组。"""
    combos = [hp for w, hp in spec.grid([SMOKE_WINDOWS[0]])]
    return [(SMOKE_WINDOWS[0], combos[0]), (SMOKE_WINDOWS[1], combos[-1])]


# ---------------------------------------------------------------------------
# 1. 前向传播
# ---------------------------------------------------------------------------
def test_forward() -> None:
    print("\n1. 前向传播（随机张量）")
    torch.manual_seed(0)
    for name, spec in SPECS.items():
        bad = []
        for w in (min(WINDOWS), max(WINDOWS)):
            for _, hp in spec.grid([w]):
                model = spec.build(hp)
                for mode in ("train", "eval"):
                    getattr(model, mode)()
                    for b in (4, 1):
                        with torch.no_grad():
                            out = model(torch.randn(b, w, 3))
                        if out.shape != (b,) or not torch.isfinite(out).all():
                            bad.append(f"{spec.config_name(w, hp)} {mode} batch={b} -> {tuple(out.shape)}")
                head_ok = isinstance(model.fc, nn.Linear) and model.fc.out_features == 1
                if not head_ok:
                    bad.append(f"{spec.config_name(w, hp)} 输出头不是 Linear(·, 1)")
        n = len(spec.grid([0]))
        check(f"{name}：{n} 组结构 x W{{{min(WINDOWS)}, {max(WINDOWS)}}} x batch{{4, 1}} x train/eval，输出 (batch,)",
              not bad, "; ".join(bad[:3]))

    x = torch.randn(2, 30, 3)
    lstm, gru = SPECS["lstm"].build({"hidden_size": 32, "num_layers": 1}), SPECS["gru"].build(
        {"hidden_size": 32, "num_layers": 1})
    check("GRU 与 LSTM 结构相同（只换循环层）",
          type(gru.gru) is nn.GRU and gru.gru.batch_first and gru.fc.in_features == lstm.fc.in_features
          and gru(x).shape == lstm(x).shape)
    cnn = SPECS["cnn"].build({"filters": 64, "num_layers": 3})
    convs = [m for m in cnn.conv if isinstance(m, nn.Conv1d)]
    check("CNN：每块 Conv1d(k=3, p=1) -> BatchNorm1d -> ReLU，滤波器数相同",
          len(convs) == 3 and all(c.kernel_size == (3,) and c.padding == (1,) and c.out_channels == 64 for c in convs)
          and [type(m) for m in cnn.conv[:3]] == [nn.Conv1d, nn.BatchNorm1d, nn.ReLU])
    tf = SPECS["transformer"].build({"d_model": 64, "num_layers": 2})
    layer = tf.encoder.layers[0]
    check("Transformer：nhead=4、dim_feedforward=4*d_model、dropout=0.1、batch_first",
          layer.self_attn.num_heads == 4 and layer.linear1.out_features == 256
          and layer.dropout.p == 0.1 and layer.self_attn.batch_first and len(tf.encoder.layers) == 2)
    check(f"位置编码 max_len={POS_MAX_LEN} 覆盖最长窗口 {max(WINDOWS)}",
          tf.pos_enc.pe.shape[1] >= max(WINDOWS) and "pe" not in tf.state_dict())
    try:
        TransformerSOC(32, 1, max_len=64)(torch.randn(1, 65, 3))
        check("窗口超过 max_len 时报错", False)
    except ValueError:
        check("窗口超过 max_len 时报错", True)
    torch.manual_seed(1)
    a = SPECS["lstm"].build({"hidden_size": 64, "num_layers": 2})
    torch.manual_seed(1)
    b = SL.LSTMSOC(hidden_size=64, num_layers=2)
    check("注册表里的 lstm 就是原来的 LSTMSOC（同种子初始化逐位相同）",
          type(a) is SL.LSTMSOC and all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))
    counts = {n: len(s.grid()) for n, s in SPECS.items()}
    check("网格组数：lstm 30、gru 30、cnn 30、transformer 20",
          counts == {"lstm": 30, "gru": 30, "cnn": 30, "transformer": 20}, str(counts))


# ---------------------------------------------------------------------------
# 2-5. 假数据上的流程
# ---------------------------------------------------------------------------
def test_pipeline(tmp: Path) -> None:
    rng = np.random.default_rng(0)
    train = fake_sequences("train", 4, rng)
    val = fake_sequences("val", 2, rng)
    test = fake_sequences("test", 8, rng, names=D.DRIVE_CYCLES)
    stats = D.compute_norm_stats(train)
    dev = torch.device("cpu")
    SM.setup_torch(dev)
    log = Quiet()

    # ---- 2. 搜索 ----
    print("\n2. 搜索流程（假数据，每组 2 轮）")
    for name in ("gru", "cnn", "transformer"):
        spec = SPECS[name]
        paths = SM.output_paths(name, root=tmp)
        settings = SM.make_settings(name, SL.LR, SMOKE_EPOCHS, FOLDERS, SMOKE_EVAL_START)
        t0 = time.perf_counter()
        df = SM.run_search(spec, smoke_configs(spec), train, val, stats, settings, paths, dev, log, LAT_KW)
        cols_ok = list(df.columns) == SM.csv_columns(spec)
        files_ok = all((paths["models"] / f"{c}.pt").exists() for c in df["config"]) and \
            all((paths["curves"] / f"{c}.png").exists() for c in df["config"])
        vals_ok = bool((df["val_rmse_pct"] > 0).all() and (df["infer_ms_cpu"] > 0).all()
                       and (df["n_params"] > 0).all() and (df["train_time_s"] > 0).all()
                       and (df["epochs_trained"] == SMOKE_EPOCHS).all())
        check(f"{name}：2 组各训练 2 轮，CSV 两行、列正确、权重和曲线已保存、推理时间已测",
              len(df) == 2 and cols_ok and files_ok and vals_ok,
              f"{time.perf_counter() - t0:.1f}s，验证 RMSE " + "、".join(f"{v:.2f}%" for v in df["val_rmse_pct"]))
        n_before = len(log.lines)
        df2 = SM.run_search(spec, smoke_configs(spec), train, val, stats, settings, paths, dev, log, LAT_KW)
        skipped = sum("已完成（" in m for m in log.lines[n_before:])
        check(f"{name}：断点续跑，已完成的 2 组直接跳过", len(df2) == 2 and skipped == 2)
        ck = torch.load(paths["models"] / f"{df['config'].iloc[0]}.pt", weights_only=True)
        _, _, m = SM.model_from_checkpoint(ck)
        check(f"{name}：权重可以重建模型", ck["model"] == name and isinstance(m, type(spec.build(smoke_configs(spec)[0][1]))))

    tag = SM.run_tag("transformer", 5e-4)
    check("--lr 0.0005 另存为 transformer_lr0.0005", tag == "transformer_lr0.0005"
          and SM.output_paths(tag)["csv"].name == "sweep_transformer_lr0.0005.csv")
    spec = SPECS["transformer"]
    paths = SM.output_paths(tag, root=tmp)
    settings = SM.make_settings("transformer", 5e-4, SMOKE_EPOCHS, FOLDERS, SMOKE_EVAL_START)
    df = SM.run_search(spec, smoke_configs(spec)[:1], train, val, stats, settings, paths, dev, log, LAT_KW)
    check("--lr 覆盖：新表里 lr = 0.0005，原表不受影响",
          len(df) == 1 and np.isclose(df["lr"].iloc[0], 5e-4)
          and len(pd.read_csv(SM.output_paths("transformer", root=tmp)["csv"])) == 2)
    try:
        bad = SM.make_settings("transformer", 5e-4, SMOKE_EPOCHS, FOLDERS, SMOKE_EVAL_START)
        SM.run_search(spec, smoke_configs(spec), train, val, stats, bad,
                      SM.output_paths("transformer", root=tmp), dev, log, LAT_KW)
        check("设置不同（lr 不同）时拒绝续跑到同一张表", False)
    except SystemExit:
        check("设置不同（lr 不同）时拒绝续跑到同一张表", True)

    # ---- 3. LSTM 行为不变 ----
    print("\n3. LSTM 行为不变（原 train_one vs 统一流程）")
    old_sweep = "lstm_fake_sweep"
    dirs = {"results": tmp / "results" / old_sweep, "runs": tmp / "results" / old_sweep / "runs",
            "curves": tmp / "results" / old_sweep / "curves", "models": tmp / "models" / old_sweep}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    old_runs = {}
    # 第一组模拟"最初 18 组"：早停评估起点比统一起点小；第二组与统一起点相同
    for (w, h, l), start in (((10, 32, 1), 9), ((20, 64, 2), SMOKE_EVAL_START)):
        lstm_settings = {"folders": FOLDERS, "features": D.FEATURES, "target": D.TARGET,
                         "resample": "1 s, floor(time_s); V/I/T mean, soc last",
                         "eval_start_index": start, "seed": SL.SEED, "lr": SL.LR,
                         "batch_size": SL.BATCH_SIZE, "max_epochs": SMOKE_EPOCHS, "patience": SL.PATIENCE,
                         "optimizer": "Adam", "loss": "MSE", "dropout": 0.0}
        r_old = SL.train_one(w, h, l, train, val, stats, lstm_settings, dirs, dev, log)
        old_runs[SL.config_name(w, h, l)] = (r_old, start)
        new_settings = SM.make_settings("lstm", SL.LR, SMOKE_EPOCHS, FOLDERS, start)
        r_new = SM.train_config(SPECS["lstm"], w, {"hidden_size": h, "num_layers": l}, train, val, stats,
                                new_settings, dev, log)
        ck_old = torch.load(dirs["models"] / f"{SL.config_name(w, h, l)}.pt", weights_only=True)
        same_w = all(torch.equal(ck_old["state_dict"][k], r_new["best_state"][k]) for k in ck_old["state_dict"])
        h_old = pd.read_csv(dirs["runs"] / f"{SL.config_name(w, h, l)}_history.csv", float_precision="round_trip")
        h_new = pd.DataFrame(r_new["history"])
        same_h = h_old["train_loss"].tolist() == h_new["train_loss"].tolist() and \
            h_old["val_loss"].tolist() == h_new["val_loss"].tolist()
        check(f"{SL.config_name(w, h, l)}：权重逐位相同、每轮 train/val loss 相同、验证 RMSE 相同",
              same_w and same_h and r_old["val_rmse_pct"] == r_new["val_rmse_pct"]
              and r_old["best_epoch"] == r_new["best_epoch"],
              f"验证 RMSE {r_old['val_rmse_pct']:.6f} vs {r_new['val_rmse_pct']:.6f}")

    # ---- 4. 导入旧 LSTM 结果 ----
    print("\n4. 导入旧 LSTM 结果（核对 -> 导入 -> 补跑）")
    rows = []
    for cname, (r, start) in old_runs.items():
        ck = torch.load(dirs["models"] / f"{cname}.pt", weights_only=True)
        _, _, m = SM.model_from_checkpoint(ck)
        ds = D.WindowDataset(val, stats, r["window"], SMOKE_EVAL_START, log)
        rmse, mae, per_t = SL.val_metrics(m, ds)  # 与 combine_lstm_sweep.py 一样在统一起点上重算
        rows.append({"rank": 0, "config": cname, "window": r["window"], "hidden_size": r["hidden_size"],
                     "num_layers": r["num_layers"], "sweep": old_sweep, "val_rmse_pct": rmse, "val_mae_pct": mae,
                     **{f"val_rmse_pct_{f}": per_t.get(f, {}).get("rmse_pct") for f in D.TEMP_FOLDERS},
                     "early_stop_eval_start": start, "val_rmse_pct_recorded": r["val_rmse_pct"],
                     "recheck_diff_pct": 0.0, "best_epoch": r["best_epoch"], "epochs_trained": r["epochs_trained"],
                     "stopped_early": r["stopped_early"], "train_time_s": r["train_time_s"],
                     "n_params": r["n_params"], "n_train_windows": r["n_train_windows"],
                     "n_val_points": len(ds), "device_name": "cpu"})
    table_path = tmp / "results" / old_sweep / "sweep_results_22.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    table = IM.load_table22(table_path)
    verify_dir, verify_models = tmp / "results" / "lstm_verify", tmp / "models" / "lstm_verify"
    lstm_csv = SM.output_paths("lstm", root=tmp)["csv"]
    try:
        IM.import_rows(table, tmp / "models", lstm_csv, verify_dir, dev, log, SMOKE_EVAL_START,
                       latency_kw=LAT_KW, max_epochs=SMOKE_EPOCHS, windows=list(SMOKE_WINDOWS))
        check("没有核对时拒绝导入", False)
    except SystemExit:
        check("没有核对时拒绝导入", True)
    cname = "W10_H32_L1"  # 早停起点 9 != 统一起点 19 的那组
    row = table.loc[table["config"] == cname].iloc[0]
    ck = torch.load(IM.old_ckpt_path(row, tmp / "models"), weights_only=True)
    rec = IM.verify_config(row, ck, train, val, stats, dev, log, verify_dir, verify_models,
                           SMOKE_EVAL_START, IM.DEFAULT_TOL)
    check("核对：新代码重跑旧配置，验证 RMSE、最佳轮、权重与记录完全一致",
          rec["passed"] and rec["val_rmse_diff_pct"] == 0 and rec["max_abs_weight_diff"] == 0
          and rec["best_epoch_rerun"] == rec["best_epoch_recorded"]
          and math.isclose(rec["val_rmse_pct_common_rerun"], rec["val_rmse_pct_common_table"], rel_tol=1e-12),
          f"差 {rec['val_rmse_diff_pct']:+.2e}，权重最大差 {rec['max_abs_weight_diff']:.1e}")
    n = IM.import_rows(table, tmp / "models", lstm_csv, verify_dir, dev, log, SMOKE_EVAL_START,
                       latency_kw=LAT_KW, max_epochs=SMOKE_EPOCHS, windows=list(SMOKE_WINDOWS))
    df = SM.read_table(lstm_csv, SPECS["lstm"])
    check("导入：2 组写进 sweep_lstm.csv，数值与旧表相同，推理时间已测",
          n == 2 and len(df) == 2 and df["source"].str.startswith("imported").all()
          and np.allclose(df.set_index("config").loc[table["config"], "val_rmse_pct"], table["val_rmse_pct"])
          and (df["infer_ms_cpu"] > 0).all()
          and df.set_index("config").loc[cname, "early_stop_eval_start"] == 9)
    n = IM.import_rows(table, tmp / "models", lstm_csv, verify_dir, dev, log, SMOKE_EVAL_START,
                       latency_kw=LAT_KW, max_epochs=SMOKE_EPOCHS, windows=list(SMOKE_WINDOWS))
    check("重复导入不会产生重复的行", n == 0 and len(SM.read_table(lstm_csv, SPECS["lstm"])) == 2)
    spec = SPECS["lstm"]
    configs = [(10, {"hidden_size": 32, "num_layers": 1}), (20, {"hidden_size": 64, "num_layers": 2}),
               (20, {"hidden_size": 32, "num_layers": 2})]
    settings = SM.make_settings("lstm", SL.LR, SMOKE_EPOCHS, FOLDERS, SMOKE_EVAL_START)
    n_before = len(log.lines)
    df = SM.run_search(spec, configs, train, val, stats, settings, SM.output_paths("lstm", root=tmp), dev, log, LAT_KW)
    trained = sum("开始训练" in m for m in log.lines[n_before:])
    check("补跑：已导入的 2 组跳过，只训练缺少的 1 组", len(df) == 3 and trained == 1
          and (df["source"] == "sweep_models").sum() == 1)

    # ---- 5. 最终评估 ----
    print("\n5. 最终评估流程（2 个种子，每次 2 轮）")
    paths = FE.final_paths(root=tmp)
    selections = [FE.select_best(m, FE.candidate_tables(m, paths["tables"]), paths["tables"], True, log)
                  for m in ("lstm", "gru", "cnn", "transformer")]
    tf_sel = selections[-1]
    all_tf = pd.concat([pd.read_csv(p) for p in FE.candidate_tables("transformer", paths["tables"])])
    check("选配置：取验证 RMSE 最低的一组（transformer 同时比较了 lr 0.001 和 0.0005 两张表）",
          math.isclose(tf_sel["search_val_rmse_pct"], all_tf["val_rmse_pct"].min(), rel_tol=1e-5)
          and len(FE.candidate_tables("transformer", paths["tables"])) == 2)
    try:
        FE.select_best("gru", FE.candidate_tables("gru", paths["tables"]), paths["tables"], False, Quiet())
        check("搜索表没跑完完整网格时默认拒绝", False)
    except SystemExit:
        check("搜索表没跑完完整网格时默认拒绝", True)
    seeds = [42, 43]
    t0 = time.perf_counter()
    runs = FE.train_phase(selections, seeds, train, val, stats, FOLDERS, SMOKE_EPOCHS, paths, dev, log)
    check("重训：4 个模型 x 2 个种子，权重已保存", len(runs) == 8
          and all(FE._abs(r["model_file"]).exists() for r in runs), f"{time.perf_counter() - t0:.1f}s")
    gru_sel = selections[1]
    gru_search = pd.read_csv(SM.output_paths("gru", root=tmp)["csv"]).set_index("config")
    r42 = next(r for r in runs if r["model"] == "gru" and r["seed"] == 42)
    r43 = next(r for r in runs if r["model"] == "gru" and r["seed"] == 43)
    check("seed 42 的重训与搜索阶段结果相同；换种子结果不同",
          math.isclose(r42["val_rmse_pct"], gru_search.loc[gru_sel["config"], "val_rmse_pct"], rel_tol=1e-5)
          and r42["val_rmse_pct"] != r43["val_rmse_pct"])
    n_before = len(log.lines)
    runs2 = FE.train_phase(selections, seeds, train, val, stats, FOLDERS, SMOKE_EPOCHS, paths, dev, log)
    check("续跑：已训练的 8 次全部跳过", len(runs2) == 8
          and sum("跳过" in m for m in log.lines[n_before:]) == 8 and not any("开始训练" in m for m in log.lines[n_before:]))
    run_df, file_df = FE.test_phase(runs, test, stats, dev, log)
    lat = FE.measure_selected_latencies(selections, runs, log, **LAT_KW)
    summary, formatted = FE.summarize(selections, run_df, lat)
    files = FE.write_outputs(summary, formatted, run_df, file_df, paths)
    num_cols = [f"test_{m}_pct_{s}_{k}" for m in ("rmse", "mae") for s in FE.SCOPES for k in ("mean", "std")]
    check("测试：每个工况和总体都有 RMSE/MAE，汇总表 4 行（均值、标准差、参数量、推理时间）",
          len(summary) == 4 and (summary["n_seeds"] == 2).all() and np.isfinite(summary[num_cols].to_numpy()).all()
          and (summary["n_params"] > 0).all() and (summary["infer_ms_cpu"] > 0).all()
          and len(file_df) == 8 * len(test) and all(p.exists() for p in files.values())
          and formatted["test_rmse_all (%)"].str.contains("±").all())
    # "all" 是把全部测试点合在一起算的，逐个核对一次
    one = run_df.iloc[0]
    sub = file_df[(file_df["model"] == one["model"]) & (file_df["seed"] == one["seed"])]
    pooled = math.sqrt((sub["rmse_pct"] ** 2 * sub["n_points"]).sum() / sub["n_points"].sum())
    check("总体 RMSE = 全部测试点合在一起计算（不是对文件取平均）",
          math.isclose(pooled, one["test_rmse_pct_all"], rel_tol=1e-9), f"{pooled:.6f} vs {one['test_rmse_pct_all']:.6f}")
    print("\n  假数据上的汇总表（数值没有意义，只检查格式）：")
    print("  " + formatted[["model", "config", "lr", "n_params", "test_rmse_all (%)", "test_mae_all (%)"]]
          .to_string(index=False).replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# 6. 测试集保护
# ---------------------------------------------------------------------------
def test_guard() -> None:
    print("\n6. 测试集保护")
    for f in ("sweep_models.py", "import_lstm_sweep.py"):
        src = (Path(__file__).resolve().parent / f).read_text(encoding="utf-8")
        check(f"{f} 不会关闭测试集保护", "allow_test_reads" not in src and "_TEST_GUARD_ON" not in src)
    if not D.MANIFEST_PATH.exists():
        check("读取测试文件会报错（跳过：没有 manifest.json）", True)
        return
    try:
        D.block_test_files()
        splits = D.build_splits()
        try:
            D.read_raw(splits["test"][0])
            check("打开保护后读取测试文件会报错", False)
        except D.TestSetAccessError:
            check("打开保护后读取测试文件会报错", True, f"{len(splits['test'])} 个测试文件")
    except FileNotFoundError as e:
        check("读取测试文件会报错（跳过：数据文件不完整）", True, str(e)[:80])
    finally:
        D._TEST_GUARD_ON = False


def main() -> None:
    t0 = time.perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}")
    tmp = Path(tempfile.mkdtemp(prefix="models_", suffix="_smoke", dir=ROOT / "results"))
    try:
        test_forward()
        test_pipeline(tmp)
        test_guard()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check("没有异常", False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n共 {len(_results)} 项，失败 {len(failed)} 项，用时 {time.perf_counter() - t0:.0f}s")
    if failed:
        print("失败：" + "；".join(failed))
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
