"""Kollmeyer LG 18650HG2 数据集清洗（v2，2026-09-28）。

把 D 盘上每个原始 CSV 清洗成独立的时间序列（遇到断档再拆段），列为
    time_s, voltage, current, temperature, soc, ah
不切窗口、不标准化、不划分训练/测试集，这些留到各自的模型阶段。

所有人工决定都以常量的形式写在本文件前半部分，不从文件名自动推断。
每条决定的理由见 data/processed/kollmeyer_clean/DATA_REPORT.md。

用法：
    python src/preprocess_kollmeyer.py
或在 notebook 里：
    import preprocess_kollmeyer as pk
    result = pk.run()
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径
RAW_ROOT = (r"D:\kdata\LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC "
            r"Estimator Script\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "kollmeyer_clean")

TEMPS = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
NOMINAL_T = {"n20degC": -20, "n10degC": -10, "0degC": 0, "10degC": 10, "25degC": 25, "40degC": 40}

# ---------------------------------------------------------------- 原始 CSV 格式（208 个文件都已核对）
CSV_SKIPROWS = 30  # 第 1-28 行是说明，第 29 行列名，第 30 行单位，第 31 行起是数据
CSV_COLS = ["TimeStamp", "Step", "Status", "ProgTime", "StepTime", "Cycle", "CycleLevel", "Procedure",
            "Voltage", "Current", "Temperature", "Capacity", "WhAccu", "Cnt", "Extra"]
EXPECTED_COLNAMES = ("Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,"
                     "Voltage,Current,Temperature,Capacity,WhAccu,Cnt,")
EXPECTED_UNITS = ",,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],"
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Time Stamp = 程序开始时的系统时间 + 实际经过时间（暂停也计入）。
# 2018-11-04 02:00 夏令时结束之前启动的程序是 EDT (UTC-4)，之后启动的是 EST (UTC-5)。
DST_END_LOCAL = pd.Timestamp("2018-11-04 02:00:00")

# ---------------------------------------------------------------- 清洗规则
TAIL_PROCEDURE = "NN_Char_Charge"        # HPPC / 0.5C / 2C 结束后的"升温 -> 1C 充电 -> 降温"段
TAIL_TYPES = ("HPPC", "Dis_0p5C", "Dis_2C")
LOG_LEVELS_S = (0.1, 1.0, 10.0, 60.0)    # 测试仪的采样间隔档位
SPLIT_MARGIN_S = 3.0                     # 断档：ProgTime 或 UTC 时间戳的间隔超过"采样间隔 + 3 s"
TEMP_TOLERANCE_C = 3.0                   # 测试段最低温度与文件夹温度相差 <= 3 degC 算一致
LA92_582_RESUME_LOCAL = pd.Timestamp("2018-11-26 00:00:00")  # 582_LA92 只保留暂停 13.2 h 后恢复运行的部分

# ---------------------------------------------------------------- 收录 / 排除（a 放错文件夹，b 测试中止或记录出错，c 正常但特殊）
_562 = "实验舱未开，电池在室温约 23-25 degC；程序在 Charge10 重启、跳过了 Cap_1C，停机容量只有约 0.2 Ah"
EXCLUDE = {
    ("25degC", "551_HWFET"): ("b", "HWFET 曲线没有执行：全程电流为 0，文件里只有 10 分钟静置"),
    ("25degC", "552_PausCycl"): ("b", "工况没有执行：4 次 TABLE 步骤都立即结束，只有静置和一次 0.015 Ah 的补充充电"),
    ("40degC", "562_Mixed4"): ("b", _562 + "，只放电 0.19 Ah"),
    ("40degC", "562_Mixed5"): ("b", _562 + "，只放电 0.19 Ah"),
    ("40degC", "562_Mixed6"): ("b", _562 + "，只放电 0.19 Ah"),
    ("40degC", "562_Mixed7"): ("b", _562 + "，只放电 0.19 Ah"),
    ("40degC", "562_Mixed8"): ("b", _562 + "，只放电 0.19 Ah"),
    ("40degC", "562_PausCycl"): ("b", _562 + "，只运行 192 s"),
    ("10degC", "571_Cap_1C"): ("a", "实际在约 0.7-3.6 degC 下放电（实验舱只在这一步偏冷），不是 10 degC 数据"),
    ("40degC", "557_Cap_1C"): ("b", "放电过程中电池从 38.4 degC 降到 1.3 degC，不在任何恒定温度下"),
}
CHARGE_EXCLUSION = ("-", "例行 1C 恒流恒压充电，按设计在 22-25 degC 进行（与文件夹温度无关），60 s 采样，不属于测试")

# _excluded 里 soc 列的填法：("batch", 批次) 用该批次满容量；("own",) 用文件自己测出的容量；None 留空
EXCLUDED_SOC = {
    ("25degC", "551_HWFET"): ("batch", "551"),
    ("25degC", "552_PausCycl"): ("batch", "552"),
    ("10degC", "571_Cap_1C"): ("own",),
    ("40degC", "557_Cap_1C"): ("own",),
}

# ---------------------------------------------------------------- 满容量（统一规则，用户 2026-09-28 确认）
# 规则 1：自己的 Cap_1C 文件存在且在正确温度下测 -> 用它
OWN_CAP1C = {"610": "n20degC", "611": "n20degC", "596": "n10degC", "602": "n10degC",
             "589": "0degC", "576": "10degC", "551": "25degC", "552": "25degC"}
# 规则 2：自己的 Cap_1C 缺失 -> 停机容量反推（末尾 |Ah| / 0.95），与老化模型预测相差 <= 0.5 % 才采用
BACKCALC = {"556": ("40degC", ["UDDS", "HWFET", "LA92", "US06", "Mixed1", "Mixed2"]),
            "567": ("10degC", ["US06", "Mixed1", "Mixed2"]),
            "582": ("10degC", ["LA92"]),
            "590": ("0degC", ["Mixed4", "Mixed5", "Mixed6", "Mixed7", "Mixed8"]),
            "601": ("n10degC", ["US06", "Mixed1", "Mixed2"]),
            "604": ("n10degC", ["Mixed3", "Mixed6", "Mixed7", "Mixed8"])}
RULE2_TOLERANCE = 0.005
RULE2_ACCEPTED = ["582", "590", "601", "604"]
# 556：40 degC 没有同温度的有效容量，只做跨温度检验（假设 40 degC 比 25 degC 高 0.5-1.0 %），用户接受
CROSS_TEMP_556 = {"reference": "552", "gain_range": (0.005, 0.010)}
# 规则 3（以及未通过规则 2 的 567）：同温度时间最近的有效容量 + 老化修正
AGED_FROM = {"571": ("10degC", "576", "571_Cap_1C"),
             "557": ("40degC", "556", "557_Cap_1C"),
             "567": ("10degC", "576", "567_Cap_1C (missing)")}
RULE3_END_SOC_RANGE = (0.05, 0.12)
# 特性测试批次：借用紧接着的工况批次的容量，按每个文件开始时刻与参照之间的放电量做老化修正
CHAR_REFERENCE = {"607": "610", "593": "596", "585": "589", "575": "576", "549": "551", "555": "556"}
# 老化模型：同温度成对的 1C 容量测试，衰减按放电量线性计
AGING_PAIRS_POOLED = [("551", "552"), ("596", "602")]
AGING_PAIR_N20 = ("610", "611")  # -20 degC 专用速率（只有这一对数据）
# 做过但不在数据集里的测试（放电量是估计值，只用于累计放电量）；时间为当地时间 EST
MISSING_TESTS = [
    ("556_Cap_1C (missing)", "2018-11-05 19:40", 2.66, "556_Charge1 与 556_Charge2 之间 1.1 h 的空档"),
    ("567_Cap_1C (missing)", "2018-11-12 19:27", 2.61, "567 程序 17:21 开始，CyclesA 的 Cap_1C 在 +2.1 h"),
    ("567_UDDS/HWFET/LA92 (missing)", "2018-11-13 04:00", 3.56 + 2.61 + 3.48, "按 576_UDDS、576_HWFET、582_LA92 的放电量估计"),
    ("582_Cap_1C (missing)", "2018-11-25 15:10", 2.54, "582 程序开始 7.9 h 后才出现 LA92"),
    ("590_Cap_1C (missing)", "2018-12-01 10:00", 2.45, "589_Charge8 与 590_Charge10 之间 1 h 的空档"),
    ("601_Cap_1C (missing)", "2018-12-14 22:20", 2.24, "601 程序开始 6.8 h 后才出现 US06"),
    ("604_Cap_1C (missing)", "2018-12-17 21:50", 2.22, "604_Charge17 与 604_Charge9 之间 56 min 的空档"),
]

# ---------------------------------------------------------------- 附注（写进 manifest）
FILE_NOTES = {
    ("10degC", "575_HPPC"): ["575"], ("10degC", "575_C20DisCh"): ["575"],
    ("10degC", "575_Dis_0p5C"): ["575"], ("10degC", "575_Dis_2C"): ["575"],
    ("0degC", "589_Cap_1C"): ["对应的 .mat 文件名误写为 10degC；实测温度 -0.6 至 2.2 degC，是 0 degC 测试"],
}
NOTE_575 = ("表头电池名是 LG ICR18650MG1（软件里选错了电池档案），温度传感器、电阻和容量趋势都表明仍是同一颗 HG2；"
            "测试电流按 2.9 Ah 设定，比其他温度低 3.3 %，HPPC 每个 SOC 台阶是 0.145 Ah")

COLUMN_SPEC = [("time_s", "%.3f"), ("voltage", "%.5f"), ("current", "%.5f"), ("temperature", "%.5f"),
               ("soc", "%.6f"), ("ah", "%.5f")]


# ================================================================ 读取
def classify(name):
    batch, test = name.split("_", 1)
    if test in ("UDDS", "HWFET", "LA92", "US06"):
        kind = "drive"
    elif test.startswith("Mixed"):
        kind = "mixed"
    elif test.startswith("Charge"):
        kind = "charge"
    else:
        kind = test  # Cap_1C, C20DisCh, Dis_0p5C, Dis_2C, HPPC, PausCycl
    return batch, test, kind


def list_raw_files(raw_root=RAW_ROOT):
    return [(folder, f[:-4]) for folder in TEMPS
            for f in sorted(os.listdir(os.path.join(raw_root, folder))) if f.lower().endswith(".csv")]


def _hms_seconds(series):
    """'06:40:55.195'（小时可超过 24）-> 秒。"""
    p = series.astype(str).str.split(":", expand=True).apply(pd.to_numeric, errors="coerce")
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy(dtype=float)


def read_header(path):
    with open(path, "rb") as f:
        lines = [f.readline().decode("latin-1").rstrip("\r\n") for _ in range(CSV_SKIPROWS)]
    meta = {}
    for ln in lines[2:25]:
        if "," in ln:
            k, v = ln.split(",", 1)
            meta.setdefault(k.strip(), v.strip())
    meta["Cell comment"] = lines[25].split(",", 1)[1].strip() if "," in lines[25] else ""
    return meta, lines[28] == EXPECTED_COLNAMES and lines[29] == EXPECTED_UNITS


def read_raw(folder, name, raw_root=RAW_ROOT):
    path = os.path.join(raw_root, folder, name + ".csv")
    meta, header_ok = read_header(path)
    if not header_ok:
        raise ValueError(f"{folder}/{name}: CSV 表头格式与预期不同")
    df = pd.read_csv(path, skiprows=CSV_SKIPROWS, header=None, names=CSV_COLS, low_memory=False, encoding="latin-1")
    ts = df["TimeStamp"].astype(str).str.strip()
    ts = ts.where(ts.str.contains(":"), ts + " 12:00:00 AM")  # 正好午夜的记录只写了日期
    start_local = pd.to_datetime(meta["Start Time"], format=TS_FORMAT)
    utc_offset = -4 if start_local < DST_END_LOCAL else -5
    ts_local = pd.to_datetime(ts, format=TS_FORMAT)
    out = pd.DataFrame({
        "raw_row": np.arange(len(df)),
        "ts_local": ts_local,
        "ts_utc": ts_local - pd.Timedelta(hours=utc_offset),
        "t": _hms_seconds(df["ProgTime"]),
        "step_t": _hms_seconds(df["StepTime"]),
        "step": pd.to_numeric(df["Step"], errors="coerce"),
        "status": df["Status"].astype(str).str.strip(),
        "procedure": df["Procedure"].astype(str).str.strip(),
        "V": pd.to_numeric(df["Voltage"], errors="coerce"),
        "I": pd.to_numeric(df["Current"], errors="coerce"),
        "T": pd.to_numeric(df["Temperature"], errors="coerce"),
        "Ah": pd.to_numeric(df["Capacity"], errors="coerce"),
    })
    return {"folder": folder, "name": name, "path": path, "meta": meta, "utc_offset_h": utc_offset,
            "program_start_local": start_local, "df": out}


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ 第一遍：逐文件汇总（时间线）
def step_segments(df):
    """按 (Step, Status) 变化切成的连续段 [(a, b), ...]。"""
    key = df["step"].astype("float").fillna(-1).astype(int).astype(str) + "|" + df["status"]
    k = key.to_numpy()
    ch = np.r_[0, np.where(k[1:] != k[:-1])[0] + 1, len(k)]
    return list(zip(ch[:-1], ch[1:]))


def logging_interval(df):
    """每一行所在步骤的采样间隔（步骤内间隔中位数，向上取到 0.1/1/10/60 s 档位）。"""
    t = df["t"].to_numpy()
    out = np.full(len(t), LOG_LEVELS_S[0])
    for a, b in step_segments(df):
        med = float(np.nanmedian(np.diff(t[a:b]))) if b - a > 2 else LOG_LEVELS_S[0]
        out[a:b] = next((s for s in LOG_LEVELS_S if s >= 0.9 * med), med)
    return out


def ah_totals(t, I, Ah):
    """按 Ah 计数器累计的放电 / 充电 Ah（跳过计数器清零和时间倒退处）。"""
    ok = ~(np.isnan(t) | np.isnan(I) | np.isnan(Ah))
    t, I, Ah = t[ok], I[ok], Ah[ok]
    dA, dt = np.diff(Ah), np.diff(t)
    pred = 0.5 * (I[1:] + I[:-1]) * dt / 3600
    reset = (np.abs(Ah[1:]) < 5e-4) & (np.abs(Ah[:-1]) > 5e-3) & (np.abs(dA - pred) > 0.5 * np.abs(dA))
    use = ~reset & (dt >= 0)
    return float(-dA[use & (dA < 0)].sum()), float(dA[use & (dA > 0)].sum()), int(reset.sum())


def state_times(df):
    """静置 / 放电 / 充电（按电流正负）以及记录断档各占的秒数（按 ProgTime 计）。"""
    t, I = df["t"].to_numpy(), df["I"].to_numpy()
    thr = np.full(len(t), 3.0)
    for a, b in step_segments(df):
        if b - a > 1:
            thr[a:b] = max(3.0, 3 * float(np.nanmedian(np.diff(t[a:b]))))
    dt = np.diff(t)
    ok = ~np.isnan(dt) & (dt >= 0)
    gap = ok & (dt > np.maximum(thr[:-1], thr[1:]))
    live, i0 = ok & ~gap, I[:-1]
    return (float(dt[live & (np.abs(i0) < 1e-3)].sum()), float(dt[live & (i0 <= -1e-3)].sum()),
            float(dt[live & (i0 >= 1e-3)].sum()), float(dt[gap].sum()))


def summarize(raw):
    df = raw["df"]
    batch, test, kind = classify(raw["name"])
    t, I, Ah, T = (df[c].to_numpy() for c in ("t", "I", "Ah", "T"))
    dis, cha, resets = ah_totals(t, I, Ah)
    rest_s, dis_s, cha_s, gap_s = state_times(df)
    dts = np.diff(t)
    dts = dts[(dts > 0) & ~np.isnan(dts)]
    levels = pd.Series(np.round(dts, 1)).value_counts(normalize=True)
    wall = (df["ts_local"].max() - df["ts_local"].min()).total_seconds()
    recorded = float(np.nanmax(t) - np.nanmin(t))
    s = {
        "folder": raw["folder"], "name": raw["name"], "batch": batch, "test": test, "test_type": kind,
        "program": raw["meta"].get("Program", ""), "battery_name_in_header": raw["meta"].get("Battery Name", ""),
        "utc_offset_h": raw["utc_offset_h"], "rows": len(df),
        "start_local": df["ts_local"].min(), "end_local": df["ts_local"].max(),
        "start_utc": df["ts_utc"].min(), "end_utc": df["ts_utc"].max(),
        "wall_s": wall, "recorded_s": recorded, "hidden_pause_s": wall - recorded if wall - recorded > 60 else 0.0,
        "T_min": float(np.nanmin(T)), "T_median": float(np.nanmedian(T)), "T_max": float(np.nanmax(T)),
        "Ah_discharged": dis, "Ah_charged": cha, "ah_counter_resets": resets,
        "rest_s": rest_s, "discharge_s": dis_s, "charge_s": cha_s, "logging_gap_s": gap_s,
        "dt_median_s": float(np.median(dts)) if len(dts) else float("nan"),
        "logging_levels_s": " ".join(f"{x:g}" for x in sorted(levels[levels >= 0.02].index)),
        "ah_min": float(np.nanmin(Ah)), "ah_last": float(Ah[~np.isnan(Ah)][-1]),
    }
    if kind in TAIL_TYPES or kind == "C20DisCh":
        nn = np.where(df["procedure"].to_numpy() == TAIL_PROCEDURE)[0]
        s["anchor_ah"] = float(Ah[0] - I[0] * df["step_t"].iloc[0] / 3600)  # 第一个放电步骤开始时的 Ah
        if len(nn):
            s.update(tail_first=int(nn[0]), tail_last=int(nn[-1]), ah_test_end=float(Ah[nn[0] - 1]),
                     ah_tail_start=float(Ah[nn[0]]), ah_tail_end=float(Ah[nn[-1]]))
    return s


# ================================================================ 老化模型与满容量
def aged(c_ref, ah_ref, ah_target, rate):
    """线性衰减：之后的容量 = 之前的容量 x (1 - rate x 放电量)。"""
    if ah_target >= ah_ref:
        return c_ref * (1 - rate * (ah_target - ah_ref))
    return c_ref / (1 - rate * (ah_ref - ah_target))


def throughput_events(summ):
    rows = [{"id": s["name"], "folder": s["folder"], "time_utc": s["start_utc"], "Ah_discharged": s["Ah_discharged"]}
            for s in summ.values()]
    for tid, local, ah, basis in MISSING_TESTS:
        rows.append({"id": tid, "folder": "(不在数据集)", "time_utc": pd.Timestamp(local) + pd.Timedelta(hours=5),
                     "Ah_discharged": ah, "basis": basis})
    ev = pd.DataFrame(rows).sort_values("time_utc", kind="stable").reset_index(drop=True)
    ev["cum_before_Ah"] = ev["Ah_discharged"].cumsum() - ev["Ah_discharged"]
    return ev


def compute_capacities(summ):
    ev = throughput_events(summ)
    cum = dict(zip(ev["id"], ev["cum_before_Ah"]))
    by = {s["name"]: s for s in summ.values()}
    own = {b: -by[f"{b}_Cap_1C"]["ah_min"] for b in OWN_CAP1C}
    back = {}
    for b, (folder, tests) in BACKCALC.items():
        ends = [-by[f"{b}_{t}"]["ah_last"] for t in tests]
        if max(ends) - min(ends) > 1e-3:
            raise ValueError(f"{b}: 各文件末尾放电量不一致 {ends}")
        back[b] = float(np.median(ends) / 0.95)

    def rate_of(pair):
        a, b = pair
        return (1 - own[b] / own[a]) / (cum[f"{b}_Cap_1C"] - cum[f"{a}_Cap_1C"])
    pair_rates = {f"{a}->{b}": rate_of((a, b)) for a, b in AGING_PAIRS_POOLED}
    r_pool = (sum(1 - own[b] / own[a] for a, b in AGING_PAIRS_POOLED)
              / sum(cum[f"{b}_Cap_1C"] - cum[f"{a}_Cap_1C"] for a, b in AGING_PAIRS_POOLED))
    r_n20 = rate_of(AGING_PAIR_N20)
    aging = {"rate_pooled_per_Ah": r_pool, "pair_rates_per_Ah": pair_rates, "rate_n20degC_per_Ah": r_n20,
             "n20degC_note": "-20 degC 的速率只来自 610 -> 611 这一对数据"}

    caps, checks = {}, []
    for b, folder in OWN_CAP1C.items():
        caps[b] = {"capacity_Ah": own[b], "rule": "1", "source": f"{b}_Cap_1C（本批次实测）", "folder": folder}
    for b, (folder, _) in BACKCALC.items():
        refs = [(k, abs(cum[f"{k}_Cap_1C"] - cum[f"{b}_Cap_1C (missing)"])) for k, f in OWN_CAP1C.items() if f == folder]
        rec = {"batch": b, "folder": folder, "backcalc_Ah": back[b]}
        if refs:
            k = min(refs, key=lambda x: x[1])[0]
            preds = {name: aged(own[k], cum[f"{k}_Cap_1C"], cum[f"{b}_Cap_1C (missing)"], r)
                     for name, r in [("pooled", r_pool)] + list(pair_rates.items())}
            dev = {name: back[b] / p - 1 for name, p in preds.items()}
            rec.update(reference=k, delta_Ah=cum[f"{b}_Cap_1C (missing)"] - cum[f"{k}_Cap_1C"],
                       predicted_Ah=preds["pooled"], deviation=dev["pooled"],
                       deviation_range=(min(dev.values()), max(dev.values())),
                       passed=abs(dev["pooled"]) <= RULE2_TOLERANCE)
        else:
            ref = CROSS_TEMP_556["reference"]
            c25 = aged(own[ref], cum[f"{ref}_Cap_1C"], cum[f"{b}_Cap_1C (missing)"], r_pool)
            lo, hi = (c25 * (1 + g) for g in CROSS_TEMP_556["gain_range"])
            rec.update(reference=None, cross_temperature_estimate=(lo, hi), cross_temperature_c25=c25,
                       passed=None, cross_check_ok=lo <= back[b] <= hi)
            if not rec["cross_check_ok"]:
                raise ValueError(f"{b}: 跨温度检验不通过 {back[b]:.4f} 不在 {lo:.4f}-{hi:.4f}")
        checks.append(rec)
        if b in RULE2_ACCEPTED:
            if not rec["passed"]:
                raise ValueError(f"{b}: 规则 2 检验不通过（偏差 {rec['deviation'] * 100:+.2f} %），需要重新确认")
            caps[b] = {"capacity_Ah": back[b], "rule": "2", "folder": folder,
                       "source": f"停机容量反推（与老化预测相差 {rec['deviation'] * 100:+.2f} %，参照 {rec['reference']}）"}
        elif b == "556":
            caps[b] = {"capacity_Ah": back[b], "rule": "2（无同温度参照）", "folder": folder,
                       "source": f"停机容量反推；40 degC 无同温度有效容量，只做跨温度检验（{lo:.4f}-{hi:.4f}）"}
    dev567 = next(c["deviation"] for c in checks if c["batch"] == "567")
    for b, (folder, ref, target) in AGED_FROM.items():
        c_ref = caps[ref]["capacity_Ah"]
        ref_key = f"{ref}_Cap_1C" if ref in OWN_CAP1C else f"{ref}_Cap_1C (missing)"
        cap = aged(c_ref, cum[ref_key], cum[target], r_pool)
        why = ("规则 3（本批次 Cap_1C 在错误温度下测得）" if b != "567"
               else f"规则 2 未通过（反推值 {back['567']:.4f} 与预测相差 {dev567 * 100:+.2f} %），改用规则 3 的做法")
        if b != "567":
            ends = [-s["ah_last"] for s in summ.values() if s["batch"] == b and s["test_type"] in ("drive", "mixed")]
            end_soc = 1 - float(np.median(ends)) / cap
            if not RULE3_END_SOC_RANGE[0] <= end_soc <= RULE3_END_SOC_RANGE[1]:
                raise ValueError(f"{b}: 规则 3 修正后末尾 SOC {end_soc * 100:.2f} % 不在 5-12 % 内，需要重新确认")
        caps[b] = {"capacity_Ah": cap, "rule": "3" if b != "567" else "3（规则 2 未通过）", "folder": folder,
                   "source": f"{why}：{ref} 的 {c_ref:.4f} Ah 按老化修正 {cum[target] - cum[ref_key]:+.1f} Ah",
                   "before_correction_Ah": c_ref, "delta_Ah": cum[target] - cum[ref_key], "rate_per_Ah": r_pool}
    # 特性测试文件：逐个文件按开始时刻修正
    file_caps = {}
    for b, ref in CHAR_REFERENCE.items():
        folder = caps[ref]["folder"]
        ref_key = f"{ref}_Cap_1C" if ref in OWN_CAP1C else f"{ref}_Cap_1C (missing)"
        rate = r_n20 if folder == "n20degC" else r_pool
        for kind in ("HPPC", "C20DisCh", "Dis_0p5C", "Dis_2C"):
            name = f"{b}_{kind}"
            c = aged(caps[ref]["capacity_Ah"], cum[ref_key], cum[name], rate)
            file_caps[(folder, name)] = {
                "capacity_Ah": c, "rule": "特性测试（借用 + 老化修正）", "folder": folder,
                "source": f"借用 {ref} 的 {caps[ref]['capacity_Ah']:.4f} Ah，按 {cum[ref_key] - cum[name]:.1f} Ah 老化修正"
                          f"（速率 {rate * 100:.4f} %/Ah{'，只来自 610/611 一对' if folder == 'n20degC' else ''}）",
                "before_correction_Ah": caps[ref]["capacity_Ah"], "delta_Ah": cum[name] - cum[ref_key], "rate_per_Ah": rate}
    return caps, file_caps, checks, aging, ev


# ================================================================ 第二遍：清洗、拆段
def blocks_of(raw, summary):
    """[(block, row_index_array, excluded, reason)]。block: main / tail / attempt1；另返回被丢弃的标记行数。"""
    df, kind = raw["df"], summary["test_type"]
    n, key = len(df), (raw["folder"], raw["name"])
    markers = df["status"].to_numpy() == "STO"
    if raw["name"] == "582_LA92" and raw["folder"] == "10degC":
        resumed = (df["ts_local"] >= LA92_582_RESUME_LOCAL).to_numpy()
        return [("main", np.where(resumed)[0], False, None),
                ("attempt1", np.where(~resumed)[0], True,
                 "11/25 第一次运行约 103 s 后过压（4.2345 V）停机，暂停 13.2 h；记录器把这段数据重复写入并插在后面")], 0
    if kind in TAIL_TYPES and "tail_first" in summary:
        a, z = summary["tail_first"], summary["tail_last"]
        dropped = n - (z + 1)  # NN_Char_Charge 之后的程序结束行（Ah 计数器清零）
        return [("main", np.arange(0, a), False, None),
                ("tail", np.arange(a, z + 1), True, "测试结束后的升温 -> 1C 充电 -> 降温段（Procedure = NN_Char_Charge）")], dropped
    keep = np.where(~markers)[0]
    return [("main", keep, key in EXCLUDE or kind == "charge",
             EXCLUDE[key][1] if key in EXCLUDE else (CHARGE_EXCLUSION[1] if kind == "charge" else None))], int(markers.sum())


def clean_rows(df, idx, sort=False):
    t = df["t"].to_numpy()
    bad = df[["V", "I", "T", "Ah", "t"]].isna().any(axis=1).to_numpy()
    n_nan = int(bad[idx].sum())
    idx = idx[~bad[idx]]
    if sort:
        idx = idx[np.argsort(t[idx], kind="stable")]
    keep = np.r_[True, np.diff(t[idx]) != 0]
    n_dup = int((~keep).sum())
    idx = idx[keep]
    if len(idx) > 1 and not np.all(np.diff(t[idx]) > 0):
        raise ValueError("清洗后 ProgTime 不是严格递增")
    return idx, n_nan, n_dup


def split_rows(df, idx, nominal):
    t = df["t"].to_numpy()[idx]
    tw = (df["ts_utc"].to_numpy()[idx] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    lim = np.maximum(nominal[idx[:-1]], nominal[idx[1:]]) + SPLIT_MARGIN_S
    dtp, dtw = np.diff(t), np.diff(tw)
    cut = np.where((dtp > lim) | (dtw > lim))[0] + 1
    info = [{"split_before_raw_row": int(idx[k]), "dt_progtime_s": round(float(dtp[k - 1]), 3),
             "dt_timestamp_s": float(dtw[k - 1]),
             "reason": ("隐藏暂停（ProgTime 连续，时间戳跳过 %.1f h）" % (dtw[k - 1] / 3600)) if dtp[k - 1] <= lim[k - 1]
             else ("记录断档 %.1f s" % dtp[k - 1])} for k in cut]
    bounds = np.r_[0, cut, len(idx)]
    return list(zip(bounds[:-1], bounds[1:])), info


def dis0p5c_start_soc(summ, folder, batch, cap):
    """选项 B：Dis_0p5C 的起始 SOC 由尾段 Ah 平衡反推，以同温度 Dis_2C 尾段的偏差为基准。"""
    a = summ[(folder, f"{batch}_Dis_0p5C")]
    if "tail_first" not in a:
        return 1.0, None
    b = summ[(folder, f"{batch}_Dis_2C")]
    excess = lambda s, anchor: (s["ah_tail_end"] - s["ah_tail_start"]) - (anchor - s["ah_test_end"])
    deficit = excess(a, a["anchor_ah"]) - excess(b, 0.0)
    return 1 - deficit / cap, deficit


def series_stats(df, idx, nominal_T, excluded_kind=None):
    sub = df.iloc[idx]
    T, st = sub["T"].to_numpy(), sub["status"].to_numpy()
    act = np.isin(st, ["DCH", "TABLE"])
    Ta = T[act] if act.any() else T
    if excluded_kind in ("tail", "charge"):
        check = "n/a（按设计在 22-25 degC 充电）"
    else:
        check = "pass" if abs(float(Ta.min()) - nominal_T) <= TEMP_TOLERANCE_C else "fail"
    dts = np.diff(sub["t"].to_numpy())
    return {
        "rows": len(idx), "duration_s": round(float(sub["t"].iloc[-1] - sub["t"].iloc[0]), 3),
        "start_local": str(sub["ts_local"].iloc[0]), "end_local": str(sub["ts_local"].iloc[-1]),
        "start_utc": str(sub["ts_utc"].iloc[0]) + "Z", "end_utc": str(sub["ts_utc"].iloc[-1]) + "Z",
        "sampling_interval_median_s": round(float(np.median(dts)), 3) if len(dts) else None,
        "sampling_intervals_present_s": sorted({float(x) for x in np.round(dts[dts > 0], 1)} & set(LOG_LEVELS_S))
        if len(dts) else [],
        "temperature": {"T_min_active": round(float(Ta.min()), 3), "T_median_active": round(float(np.median(Ta)), 3),
                        "T_max_active": round(float(Ta.max()), 3), "T_start": round(float(T[0]), 3), "check": check},
        "voltage_range": [round(float(sub["V"].min()), 5), round(float(sub["V"].max()), 5)],
        "current_range": [round(float(sub["I"].min()), 5), round(float(sub["I"].max()), 5)],
    }


def write_csv(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [c for c, _ in COLUMN_SPEC]
    arr = frame[cols].to_numpy(dtype=float)
    with open(path, "w", newline="") as f:
        f.write(",".join(cols) + "\n")
        np.savetxt(f, arr, fmt=[fmt for _, fmt in COLUMN_SPEC], delimiter=",")
    if np.isnan(arr[:, 4]).any():  # 留空的 soc
        with open(path, "r") as f:
            text = f.read()
        with open(path, "w", newline="") as f:
            f.write(text.replace(",nan,", ",,"))


# ================================================================ 主流程
def _clear_outputs(out_dir):
    for top in [out_dir, os.path.join(out_dir, "_excluded")]:
        for folder in TEMPS:
            d = os.path.join(top, folder)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".csv"):
                        os.remove(os.path.join(d, f))


def run(out_dir=OUT_DIR, raw_root=RAW_ROOT, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    files = list_raw_files(raw_root)
    log(f"原始 CSV：{len(files)} 个")

    # ---- 第一遍
    summ = {}
    for folder, name in files:
        raw = read_raw(folder, name, raw_root)
        summ[(folder, name)] = summarize(raw)
        summ[(folder, name)]["md5"] = file_md5(raw["path"])
    caps, file_caps, checks, aging, events = compute_capacities(summ)
    log(f"老化速率：合并 {aging['rate_pooled_per_Ah'] * 100:.4f} %/Ah，-20 degC {aging['rate_n20degC_per_Ah'] * 100:.4f} %/Ah")

    def capacity_of(folder, name, batch):
        if (folder, name) in file_caps:
            return file_caps[(folder, name)]
        return caps.get(batch)

    # ---- 第二遍
    _clear_outputs(out_dir)
    series, raw_records = [], []
    for folder, name in files:
        s = summ[(folder, name)]
        batch, kind = s["batch"], s["test_type"]
        raw = read_raw(folder, name, raw_root)
        df = raw["df"]
        nominal = logging_interval(df)
        blocks, n_marker = blocks_of(raw, s)
        cap = capacity_of(folder, name, batch)
        key = (folder, name)
        # soc 基准
        soc_anchor, ah_anchor, anchor_note = 1.0, 0.0, "测试开始时充满（24 degC 下 1C 充电至 4.2 V / 50 mA），soc = 1"
        if kind == "Dis_0p5C":
            ah_anchor = s["anchor_ah"]
            soc_anchor, deficit = dis0p5c_start_soc(summ, folder, batch, cap["capacity_Ah"])
            anchor_note = ("放电开始时的 Ah 计数值接着 C/20 测试，没有清零；" +
                           ("起始 SOC = %.4f，按尾段 Ah 平衡反推（比充满少 %.4f Ah，以同温度 Dis_2C 尾段为基准）"
                            % (soc_anchor, deficit) if deficit is not None else "25 degC 没有尾段，起始 SOC 记为 1"))
        test_end_soc = None
        out_ids = []
        for bname, idx, excluded, reason in blocks:
            if not len(idx):
                continue
            idx, n_nan, n_dup = clean_rows(df, idx, sort=(bname == "attempt1"))
            bounds, split_info = split_rows(df, idx, nominal)
            ah = df["Ah"].to_numpy()[idx]
            c, cap_rule, cap_source, cap_before = None, None, None, None
            if bname == "tail":
                c, cap_rule, cap_source = cap["capacity_Ah"], cap["rule"], cap["source"]
                cap_before = cap.get("before_correction_Ah")
                soc_block = test_end_soc + (ah - ah[0]) / c
                soc_rule = "接着测试结束时的 SOC 往下算（尾段开始时 Ah 计数器已清零）：soc_end_of_test + (ah - ah[0]) / 满容量"
            elif excluded and (kind == "charge" or (key not in EXCLUDED_SOC and bname != "attempt1")):
                soc_block = np.full(len(idx), np.nan)
                soc_rule = "留空（起始 SOC 或当时的满容量无法确定）"
            elif excluded:
                how = ("batch", "582") if bname == "attempt1" else EXCLUDED_SOC[key]
                if how[0] == "batch":
                    c, cap_rule, cap_source = caps[how[1]]["capacity_Ah"], caps[how[1]]["rule"], caps[how[1]]["source"]
                    soc_rule = f"1 + ah / {c:.4f}（{how[1]} 批次的满容量）"
                else:
                    c, cap_rule = -s["ah_min"], "文件自己测出的容量"
                    cap_source = "本文件自己测出的 1C 容量（在错误温度下），与主数据集不是同一把尺子"
                    soc_rule = f"1 + ah / {c:.4f}（文件自己测出的容量；与主数据集不是同一把尺子）"
                soc_block = 1 + ah / c
            else:
                c, cap_rule, cap_source = cap["capacity_Ah"], cap["rule"], cap["source"]
                cap_before = cap.get("before_correction_Ah")
                soc_block = soc_anchor + (ah - ah_anchor) / c
                soc_rule = f"{soc_anchor:.6f} + (ah - {ah_anchor:.5f}) / {c:.5f}"
                test_end_soc = float(soc_block[-1])
            for k, (a, z) in enumerate(bounds, 1):
                pidx, soc = idx[a:z], soc_block[a:z]
                n_parts = len(bounds)
                stem = {"main": name, "tail": f"{name}_tail", "attempt1": f"{name}_attempt1"}[bname]
                sid = stem if n_parts == 1 else f"{stem}_part{k}"
                rel = os.path.join("_excluded", folder, sid + ".csv") if excluded else os.path.join(folder, sid + ".csv")
                sub = df.iloc[pidx]
                frame = pd.DataFrame({"time_s": sub["t"].to_numpy() - sub["t"].iloc[0], "voltage": sub["V"].to_numpy(),
                                      "current": sub["I"].to_numpy(), "temperature": sub["T"].to_numpy(),
                                      "soc": soc, "ah": sub["Ah"].to_numpy()})
                write_csv(os.path.join(out_dir, rel), frame)
                st = series_stats(df, pidx, NOMINAL_T[folder], excluded_kind="tail" if bname == "tail" else
                                  ("charge" if kind == "charge" else None))
                notes = [NOTE_575 if n_ == "575" else n_ for n_ in FILE_NOTES.get(key, [])]
                if kind == "C20DisCh":
                    notes.append("C/20 充电段按 Ah 累计的 SOC 与放电段不闭合：本文件充电 Ah 是放电 Ah 的 %.1f %%，"
                                 "充电结束时 soc = %.4f；推测是 0.15 A 小电流下约 20 h 的电流零点偏移累积，原样保留"
                                 % (s["Ah_charged"] / s["Ah_discharged"] * 100, soc_block[-1]))
                if kind == "Dis_0p5C" and bname == "main":
                    notes.append(anchor_note)
                if name == "582_LA92" and bname == "main":
                    notes.append("只保留 13.2 h 暂停后恢复运行的数据（2018-11-26 10:11:10 起）；文件名和表头日期是 11/25 的第一次运行")
                if split_info:
                    notes.append("按断档拆段：" + "；".join(f"第 {x['split_before_raw_row']} 行前，{x['reason']}" for x in split_info))
                entry = {
                    "id": f"{'_excluded/' if excluded else ''}{folder}/{sid}", "file": rel.replace(os.sep, "/"),
                    "excluded": bool(excluded), "exclusion_reason": reason if excluded else None,
                    "exclusion_class": (EXCLUDE[key][0] if key in EXCLUDE else "-") if excluded and bname == "main" else None,
                    "source_file": f"{folder}/{name}.csv", "source_md5": s["md5"], "block": bname,
                    "source_rows": [int(pidx[0]), int(pidx[-1])], "part": k, "n_parts": n_parts, "split_points": split_info,
                    "folder": folder, "nominal_temperature_C": NOMINAL_T[folder], "batch": batch,
                    "test_type": kind, "test_name": s["test"], "program": s["program"],
                    "battery_name_in_header": s["battery_name_in_header"], "utc_offset_h": raw["utc_offset_h"],
                    **st,
                    "rows_dropped_in_block": {"nan": n_nan, "duplicate_progtime": n_dup},
                    "program_end_marker_rows_in_file": n_marker,
                    "capacity_Ah": round(c, 5) if c is not None else None, "capacity_rule": cap_rule,
                    "capacity_source": cap_source,
                    "capacity_before_aging_correction_Ah": round(cap_before, 5) if cap_before is not None else None,
                    "soc_rule": soc_rule,
                    "soc_anchor": {"soc": round(soc_anchor, 6), "ah": round(ah_anchor, 5), "note": anchor_note}
                    if bname == "main" and not excluded else None,
                    "soc_start": None if np.isnan(soc[0]) else round(float(soc[0]), 6),
                    "soc_end": None if np.isnan(soc[-1]) else round(float(soc[-1]), 6),
                    "soc_min": None if np.isnan(soc).all() else round(float(np.nanmin(soc)), 6),
                    "soc_max": None if np.isnan(soc).all() else round(float(np.nanmax(soc)), 6),
                    "notes": notes,
                }
                series.append(entry)
                out_ids.append(entry["id"])
        status = ("excluded" if (key in EXCLUDE or kind == "charge") else
                  ("included (tail -> _excluded)" if kind in TAIL_TYPES and "tail_first" in s else
                   ("included (first attempt -> _excluded)" if name == "582_LA92" else "included")))
        raw_records.append({"folder": folder, "name": name, "status": status,
                            "exclusion_class": EXCLUDE[key][0] if key in EXCLUDE else ("-" if kind == "charge" else None),
                            "exclusion_reason": EXCLUDE[key][1] if key in EXCLUDE else (CHARGE_EXCLUSION[1] if kind == "charge" else None),
                            "output_series": out_ids})
    log(f"写出序列：主数据集 {sum(not x['excluded'] for x in series)} 条，_excluded {sum(x['excluded'] for x in series)} 条")

    # ---- 时间线
    tl = []
    rec_by = {(r["folder"], r["name"]): r for r in raw_records}
    for (folder, name), s in summ.items():
        r = rec_by[(folder, name)]
        tl.append({
            "file": f"{name}.csv", "folder": folder, "batch": s["batch"], "test_type": s["test_type"], "test": s["test"],
            "start_local": s["start_local"], "end_local": s["end_local"], "utc_offset_h": s["utc_offset_h"],
            "start_utc": s["start_utc"], "end_utc": s["end_utc"],
            "duration_h": s["wall_s"] / 3600, "recorded_h": s["recorded_s"] / 3600, "hidden_pause_h": s["hidden_pause_s"] / 3600,
            "T_min": s["T_min"], "T_median": s["T_median"], "T_max": s["T_max"],
            "Ah_discharged": s["Ah_discharged"], "Ah_charged": s["Ah_charged"],
            "rest_h": s["rest_s"] / 3600, "discharge_h": s["discharge_s"] / 3600, "charge_h": s["charge_s"] / 3600,
            "logging_gap_h": s["logging_gap_s"] / 3600, "sampling_interval_median_s": s["dt_median_s"],
            "logging_levels_s": s["logging_levels_s"], "rows": s["rows"], "program": s["program"],
            "battery_name_in_header": s["battery_name_in_header"], "status": r["status"],
            "exclusion_class": r["exclusion_class"], "exclusion_reason": r["exclusion_reason"],
            "output_series": ";".join(r["output_series"]), "source_md5": s["md5"]})
    tl = pd.DataFrame(tl).sort_values(["start_utc", "end_utc"]).reset_index(drop=True)
    tl.insert(0, "order", np.arange(1, len(tl) + 1))
    tl.to_csv(os.path.join(out_dir, "file_timeline.csv"), index=False, float_format="%.4f", encoding="utf-8-sig")

    # ---- manifest
    cap_table = []
    for b, cinfo in sorted(caps.items(), key=lambda x: (TEMPS.index(x[1]["folder"]), x[0])):
        cap_table.append({"batch": b, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in cinfo.items()}})
    for (folder, name), cinfo in file_caps.items():
        cap_table.append({"batch": name.split("_")[0], "file": name, **{k: (round(v, 5) if isinstance(v, float) else v)
                                                                          for k, v in cinfo.items()}})
    manifest = {
        "generated_by": "src/preprocess_kollmeyer.py", "raw_root": raw_root,
        "columns": {"time_s": "s，从该序列第一行起算（ProgTime）", "voltage": "V，端电压", "current": "A，放电为负、充电为正",
                    "temperature": "degC，电池表面温度（AD592）", "soc": "0-1 的小数；定义见 soc_rule",
                    "ah": "测试仪原始累计 Ah（Capacity 列），只作标签参考，不能作为模型输入"},
        "soc_definition": "soc = 1 - 放出的 Ah / 满容量；满容量 = 该批次在实际测试温度下的 1C 容量（统一规则），"
                          "与 Kollmeyer 示例用的标称 3.0 Ah 不同",
        "aging_model": {**{k: v for k, v in aging.items()}, "missing_tests": [
            {"id": m[0], "time_local_EST": m[1], "Ah_discharged_estimate": m[2], "basis": m[3]} for m in MISSING_TESTS]},
        "rule2_checks": checks, "capacities": cap_table, "raw_files": raw_records, "series": series,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, default=_json_default)
    return {"manifest": manifest, "timeline": tl, "capacities": caps, "file_capacities": file_caps, "checks": checks,
            "aging": aging, "events": events}


def tables(result):
    """notebook 用的汇总表：满容量、规则 2 检验、收录/排除、拆段、温度核对。"""
    m = result["manifest"]
    S = pd.DataFrame(m["series"])
    cap = pd.DataFrame(m["capacities"])
    cap["folder"] = pd.Categorical(cap["folder"], TEMPS, ordered=True)
    cap = cap.sort_values(["folder", "batch"]).reset_index(drop=True)
    main = S[~S["excluded"]]
    end_soc = (main[main["test_type"].isin(["drive", "mixed"])].groupby("batch")["soc_end"]
               .agg(["min", "max"]).rename(columns={"min": "end_soc_min", "max": "end_soc_max"}))
    cap = cap.merge(end_soc, left_on="batch", right_index=True, how="left")
    cap.loc[cap["file"].notna(), ["end_soc_min", "end_soc_max"]] = np.nan
    checks = pd.DataFrame(m["rule2_checks"])
    splits = pd.DataFrame([{"series_block": s["source_file"] + " / " + s["block"], **p}
                           for s in m["series"] if s["part"] == 1 for p in s["split_points"]])
    raw = pd.DataFrame(m["raw_files"])
    raw["test_type"] = raw["name"].map(lambda n: classify(n)[2])
    inc = raw.pivot_table(index="test_type", columns="status", values="name", aggfunc="count", fill_value=0)
    temp = pd.DataFrame([{"id": s["id"], **s["temperature"]} for s in m["series"]])
    return {"capacities": cap, "rule2_checks": checks, "splits": splits, "inclusion": inc,
            "excluded_files": raw[raw["status"] == "excluded"][["folder", "name", "exclusion_class", "exclusion_reason"]],
            "temperature": temp, "series": S}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    run(out)
