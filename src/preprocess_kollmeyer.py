"""Cleaning of the Kollmeyer LG 18650HG2 data set (v2, 2026-09-28).

Cleans every raw CSV on drive D into independent time series (split again at logging gaps), with the columns
    time_s, voltage, current, temperature, soc, ah
No windowing, no normalisation, no train/test split; those belong to the model stages.

Every manual decision is written as a constant in the first half of this file, never inferred from file names.
The reason for every decision is in data/processed/kollmeyer_clean/DATA_REPORT.md.

Usage:
    python src/preprocess_kollmeyer.py
or in a notebook:
    import preprocess_kollmeyer as pk
    result = pk.run()
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
RAW_ROOT = (r"D:\kdata\LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC "
            r"Estimator Script\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "kollmeyer_clean")

TEMPS = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
NOMINAL_T = {"n20degC": -20, "n10degC": -10, "0degC": 0, "10degC": 10, "25degC": 25, "40degC": 40}

# ---------------------------------------------------------------- raw CSV format (checked for all 208 files)
CSV_SKIPROWS = 30  # rows 1-28 are a description, row 29 the column names, row 30 the units, data from row 31
CSV_COLS = ["TimeStamp", "Step", "Status", "ProgTime", "StepTime", "Cycle", "CycleLevel", "Procedure",
            "Voltage", "Current", "Temperature", "Capacity", "WhAccu", "Cnt", "Extra"]
EXPECTED_COLNAMES = ("Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,"
                     "Voltage,Current,Temperature,Capacity,WhAccu,Cnt,")
EXPECTED_UNITS = ",,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],"
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# Time Stamp = system time at program start + real elapsed time (pauses included).
# Programs started before the end of daylight saving time on 2018-11-04 02:00 are EDT (UTC-4), later ones EST (UTC-5).
DST_END_LOCAL = pd.Timestamp("2018-11-04 02:00:00")

# ---------------------------------------------------------------- cleaning rules
TAIL_PROCEDURE = "NN_Char_Charge"        # the "warm up -> 1C charge -> cool down" part after HPPC / 0.5C / 2C
TAIL_TYPES = ("HPPC", "Dis_0p5C", "Dis_2C")
LOG_LEVELS_S = (0.1, 1.0, 10.0, 60.0)    # logging intervals of the tester
SPLIT_MARGIN_S = 3.0                     # gap: ProgTime or UTC time-stamp step larger than "logging interval + 3 s"
TEMP_TOLERANCE_C = 3.0                   # the test part is consistent if its minimum temperature is within 3 degC of the folder temperature
LA92_582_RESUME_LOCAL = pd.Timestamp("2018-11-26 00:00:00")  # 582_LA92: keep only the part resumed after the 13.2 h pause

# ---------------------------------------------------------------- included / excluded (a wrong folder, b test aborted or logging error, c normal but special)
_562 = "test chamber not switched on, the cell was at room temperature of about 23-25 degC; the program restarted at Charge10 and skipped Cap_1C, the cutoff capacity is only about 0.2 Ah"
EXCLUDE = {
    ("25degC", "551_HWFET"): ("b", "the HWFET profile was not executed: the current is 0 throughout, the file only holds a 10-minute rest"),
    ("25degC", "552_PausCycl"): ("b", "the cycle was not executed: all 4 TABLE steps ended at once; only rests and one 0.015 Ah top-up charge"),
    ("40degC", "562_Mixed4"): ("b", _562 + "; only 0.19 Ah discharged"),
    ("40degC", "562_Mixed5"): ("b", _562 + "; only 0.19 Ah discharged"),
    ("40degC", "562_Mixed6"): ("b", _562 + "; only 0.19 Ah discharged"),
    ("40degC", "562_Mixed7"): ("b", _562 + "; only 0.19 Ah discharged"),
    ("40degC", "562_Mixed8"): ("b", _562 + "; only 0.19 Ah discharged"),
    ("40degC", "562_PausCycl"): ("b", _562 + "; ran for only 192 s"),
    ("10degC", "571_Cap_1C"): ("a", "actually discharged at about 0.7-3.6 degC (the chamber was cold only in this step); not 10 degC data"),
    ("40degC", "557_Cap_1C"): ("b", "during the discharge the cell cooled from 38.4 degC to 1.3 degC; not at any constant temperature"),
}
CHARGE_EXCLUSION = ("-", "routine 1C constant-current constant-voltage charge, done at 22-25 degC by design (independent of the folder temperature), 60 s logging; not a test")

# How the soc column is filled in _excluded: ("batch", batch) uses that batch's full capacity; ("own",) the capacity measured in the file itself; None leaves it empty
EXCLUDED_SOC = {
    ("25degC", "551_HWFET"): ("batch", "551"),
    ("25degC", "552_PausCycl"): ("batch", "552"),
    ("10degC", "571_Cap_1C"): ("own",),
    ("40degC", "557_Cap_1C"): ("own",),
}

# ---------------------------------------------------------------- full capacity (one rule set, confirmed by the user on 2026-09-28)
# Rule 1: the batch's own Cap_1C file exists and was measured at the right temperature -> use it
OWN_CAP1C = {"610": "n20degC", "611": "n20degC", "596": "n10degC", "602": "n10degC",
             "589": "0degC", "576": "10degC", "551": "25degC", "552": "25degC"}
# Rule 2: own Cap_1C missing -> back-calculate from the cutoff capacity (final |Ah| / 0.95); accepted only if within 0.5 % of the aging-model prediction
BACKCALC = {"556": ("40degC", ["UDDS", "HWFET", "LA92", "US06", "Mixed1", "Mixed2"]),
            "567": ("10degC", ["US06", "Mixed1", "Mixed2"]),
            "582": ("10degC", ["LA92"]),
            "590": ("0degC", ["Mixed4", "Mixed5", "Mixed6", "Mixed7", "Mixed8"]),
            "601": ("n10degC", ["US06", "Mixed1", "Mixed2"]),
            "604": ("n10degC", ["Mixed3", "Mixed6", "Mixed7", "Mixed8"])}
RULE2_TOLERANCE = 0.005
RULE2_ACCEPTED = ["582", "590", "601", "604"]
# 556: no valid same-temperature capacity at 40 degC, only a cross-temperature check (assuming 40 degC is 0.5-1.0 % above 25 degC); accepted by the user
CROSS_TEMP_556 = {"reference": "552", "gain_range": (0.005, 0.010)}
# Rule 3 (and 567, which failed rule 2): the nearest valid capacity at the same temperature + aging correction
AGED_FROM = {"571": ("10degC", "576", "571_Cap_1C"),
             "557": ("40degC", "556", "557_Cap_1C"),
             "567": ("10degC", "576", "567_Cap_1C (missing)")}
RULE3_END_SOC_RANGE = (0.05, 0.12)
# Characterisation batches: borrow the capacity of the drive-cycle batch that follows, with an aging correction for the discharge between each file's start and the reference
CHAR_REFERENCE = {"607": "610", "593": "596", "585": "589", "575": "576", "549": "551", "555": "556"}
# Aging model: paired 1C capacity tests at the same temperature; the fade is linear in the discharged Ah
AGING_PAIRS_POOLED = [("551", "552"), ("596", "602")]
AGING_PAIR_N20 = ("610", "611")  # rate for -20 degC only (one pair of data only)
# Tests that were run but are not in the data set (discharge is an estimate, used only for the cumulative discharge); local time EST
MISSING_TESTS = [
    ("556_Cap_1C (missing)", "2018-11-05 19:40", 2.66, "1.1 h gap between 556_Charge1 and 556_Charge2"),
    ("567_Cap_1C (missing)", "2018-11-12 19:27", 2.61, "program 567 started at 17:21; the CyclesA Cap_1C is at +2.1 h"),
    ("567_UDDS/HWFET/LA92 (missing)", "2018-11-13 04:00", 3.56 + 2.61 + 3.48, "estimated from the discharge of 576_UDDS, 576_HWFET, 582_LA92"),
    ("582_Cap_1C (missing)", "2018-11-25 15:10", 2.54, "LA92 appears only 7.9 h after program 582 started"),
    ("590_Cap_1C (missing)", "2018-12-01 10:00", 2.45, "1 h gap between 589_Charge8 and 590_Charge10"),
    ("601_Cap_1C (missing)", "2018-12-14 22:20", 2.24, "US06 appears only 6.8 h after program 601 started"),
    ("604_Cap_1C (missing)", "2018-12-17 21:50", 2.22, "56 min gap between 604_Charge17 and 604_Charge9"),
]

# ---------------------------------------------------------------- notes (written into the manifest)
FILE_NOTES = {
    ("10degC", "575_HPPC"): ["575"], ("10degC", "575_C20DisCh"): ["575"],
    ("10degC", "575_Dis_0p5C"): ["575"], ("10degC", "575_Dis_2C"): ["575"],
    ("0degC", "589_Cap_1C"): ["the matching .mat file name says 10degC by mistake; measured temperature -0.6 to 2.2 degC, a 0 degC test"],
}
NOTE_575 = ("the cell name in the header is LG ICR18650MG1 (wrong cell profile chosen in the software); the temperature sensor, resistance and capacity trend all show it is still the same HG2 cell; "
            "test currents were set for 2.9 Ah, 3.3 % lower than at the other temperatures; each HPPC SOC step is 0.145 Ah")

COLUMN_SPEC = [("time_s", "%.3f"), ("voltage", "%.5f"), ("current", "%.5f"), ("temperature", "%.5f"),
               ("soc", "%.6f"), ("ah", "%.5f")]


# ================================================================ reading
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
    """'06:40:55.195' (hours may exceed 24) -> seconds."""
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
        raise ValueError(f"{folder}/{name}: CSV header format differs from the expected one")
    df = pd.read_csv(path, skiprows=CSV_SKIPROWS, header=None, names=CSV_COLS, low_memory=False, encoding="latin-1")
    ts = df["TimeStamp"].astype(str).str.strip()
    ts = ts.where(ts.str.contains(":"), ts + " 12:00:00 AM")  # records exactly at midnight only carry the date
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


# ================================================================ pass 1: summary per file (timeline)
def step_segments(df):
    """Continuous parts split at changes of (Step, Status): [(a, b), ...]."""
    key = df["step"].astype("float").fillna(-1).astype(int).astype(str) + "|" + df["status"]
    k = key.to_numpy()
    ch = np.r_[0, np.where(k[1:] != k[:-1])[0] + 1, len(k)]
    return list(zip(ch[:-1], ch[1:]))


def logging_interval(df):
    """Logging interval of the step of every row (median interval inside the step, rounded up to the 0.1/1/10/60 s levels)."""
    t = df["t"].to_numpy()
    out = np.full(len(t), LOG_LEVELS_S[0])
    for a, b in step_segments(df):
        med = float(np.nanmedian(np.diff(t[a:b]))) if b - a > 2 else LOG_LEVELS_S[0]
        out[a:b] = next((s for s in LOG_LEVELS_S if s >= 0.9 * med), med)
    return out


def ah_totals(t, I, Ah):
    """Discharged / charged Ah accumulated from the Ah counter (skipping counter resets and time going backwards)."""
    ok = ~(np.isnan(t) | np.isnan(I) | np.isnan(Ah))
    t, I, Ah = t[ok], I[ok], Ah[ok]
    dA, dt = np.diff(Ah), np.diff(t)
    pred = 0.5 * (I[1:] + I[:-1]) * dt / 3600
    reset = (np.abs(Ah[1:]) < 5e-4) & (np.abs(Ah[:-1]) > 5e-3) & (np.abs(dA - pred) > 0.5 * np.abs(dA))
    use = ~reset & (dt >= 0)
    return float(-dA[use & (dA < 0)].sum()), float(dA[use & (dA > 0)].sum()), int(reset.sum())


def state_times(df):
    """Seconds of rest / discharge / charge (by the sign of the current) and of logging gaps (by ProgTime)."""
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
        s["anchor_ah"] = float(Ah[0] - I[0] * df["step_t"].iloc[0] / 3600)  # Ah at the start of the first discharge step
        if len(nn):
            s.update(tail_first=int(nn[0]), tail_last=int(nn[-1]), ah_test_end=float(Ah[nn[0] - 1]),
                     ah_tail_start=float(Ah[nn[0]]), ah_tail_end=float(Ah[nn[-1]]))
    return s


# ================================================================ aging model and full capacity
def aged(c_ref, ah_ref, ah_target, rate):
    """Linear fade: later capacity = earlier capacity x (1 - rate x discharged Ah)."""
    if ah_target >= ah_ref:
        return c_ref * (1 - rate * (ah_target - ah_ref))
    return c_ref / (1 - rate * (ah_ref - ah_target))


def throughput_events(summ):
    rows = [{"id": s["name"], "folder": s["folder"], "time_utc": s["start_utc"], "Ah_discharged": s["Ah_discharged"]}
            for s in summ.values()]
    for tid, local, ah, basis in MISSING_TESTS:
        rows.append({"id": tid, "folder": "(not in the data set)", "time_utc": pd.Timestamp(local) + pd.Timedelta(hours=5),
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
            raise ValueError(f"{b}: the final discharge differs between files {ends}")
        back[b] = float(np.median(ends) / 0.95)

    def rate_of(pair):
        a, b = pair
        return (1 - own[b] / own[a]) / (cum[f"{b}_Cap_1C"] - cum[f"{a}_Cap_1C"])
    pair_rates = {f"{a}->{b}": rate_of((a, b)) for a, b in AGING_PAIRS_POOLED}
    r_pool = (sum(1 - own[b] / own[a] for a, b in AGING_PAIRS_POOLED)
              / sum(cum[f"{b}_Cap_1C"] - cum[f"{a}_Cap_1C"] for a, b in AGING_PAIRS_POOLED))
    r_n20 = rate_of(AGING_PAIR_N20)
    aging = {"rate_pooled_per_Ah": r_pool, "pair_rates_per_Ah": pair_rates, "rate_n20degC_per_Ah": r_n20,
             "n20degC_note": "the -20 degC rate comes only from the pair 610 -> 611"}

    caps, checks = {}, []
    for b, folder in OWN_CAP1C.items():
        caps[b] = {"capacity_Ah": own[b], "rule": "1", "source": f"{b}_Cap_1C (measured in this batch)", "folder": folder}
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
                raise ValueError(f"{b}: cross-temperature check failed: {back[b]:.4f} not in {lo:.4f}-{hi:.4f}")
        checks.append(rec)
        if b in RULE2_ACCEPTED:
            if not rec["passed"]:
                raise ValueError(f"{b}: rule 2 check failed (deviation {rec['deviation'] * 100:+.2f} %), confirm again")
            caps[b] = {"capacity_Ah": back[b], "rule": "2", "folder": folder,
                       "source": f"back-calculated from the cutoff capacity ({rec['deviation'] * 100:+.2f} % from the aging prediction, reference {rec['reference']})"}
        elif b == "556":
            caps[b] = {"capacity_Ah": back[b], "rule": "2 (no same-temperature reference)", "folder": folder,
                       "source": f"back-calculated from the cutoff capacity; no valid same-temperature capacity at 40 degC, cross-temperature check only ({lo:.4f}-{hi:.4f})"}
    dev567 = next(c["deviation"] for c in checks if c["batch"] == "567")
    for b, (folder, ref, target) in AGED_FROM.items():
        c_ref = caps[ref]["capacity_Ah"]
        ref_key = f"{ref}_Cap_1C" if ref in OWN_CAP1C else f"{ref}_Cap_1C (missing)"
        cap = aged(c_ref, cum[ref_key], cum[target], r_pool)
        why = ("rule 3 (this batch's Cap_1C was measured at the wrong temperature)" if b != "567"
               else f"rule 2 failed (back-calculated {back['567']:.4f}, {dev567 * 100:+.2f} % from the prediction), rule 3 used instead")
        if b != "567":
            ends = [-s["ah_last"] for s in summ.values() if s["batch"] == b and s["test_type"] in ("drive", "mixed")]
            end_soc = 1 - float(np.median(ends)) / cap
            if not RULE3_END_SOC_RANGE[0] <= end_soc <= RULE3_END_SOC_RANGE[1]:
                raise ValueError(f"{b}: final SOC after the rule-3 correction {end_soc * 100:.2f} % is not within 5-12 %, confirm again")
        caps[b] = {"capacity_Ah": cap, "rule": "3" if b != "567" else "3 (rule 2 failed)", "folder": folder,
                   "source": f"{why}: {c_ref:.4f} Ah of {ref} with an aging correction of {cum[target] - cum[ref_key]:+.1f} Ah",
                   "before_correction_Ah": c_ref, "delta_Ah": cum[target] - cum[ref_key], "rate_per_Ah": r_pool}
    # characterisation files: corrected file by file according to their start time
    file_caps = {}
    for b, ref in CHAR_REFERENCE.items():
        folder = caps[ref]["folder"]
        ref_key = f"{ref}_Cap_1C" if ref in OWN_CAP1C else f"{ref}_Cap_1C (missing)"
        rate = r_n20 if folder == "n20degC" else r_pool
        for kind in ("HPPC", "C20DisCh", "Dis_0p5C", "Dis_2C"):
            name = f"{b}_{kind}"
            c = aged(caps[ref]["capacity_Ah"], cum[ref_key], cum[name], rate)
            file_caps[(folder, name)] = {
                "capacity_Ah": c, "rule": "characterisation test (borrowed + aging correction)", "folder": folder,
                "source": f"borrowed {caps[ref]['capacity_Ah']:.4f} Ah of {ref}, aging correction for {cum[ref_key] - cum[name]:.1f} Ah"
                          f" (rate {rate * 100:.4f} %/Ah{', from the 610/611 pair only' if folder == 'n20degC' else ''})",
                "before_correction_Ah": caps[ref]["capacity_Ah"], "delta_Ah": cum[name] - cum[ref_key], "rate_per_Ah": rate}
    return caps, file_caps, checks, aging, ev


# ================================================================ pass 2: cleaning and splitting
def blocks_of(raw, summary):
    """[(block, row_index_array, excluded, reason)]. block: main / tail / attempt1; also returns the number of dropped marker rows."""
    df, kind = raw["df"], summary["test_type"]
    n, key = len(df), (raw["folder"], raw["name"])
    markers = df["status"].to_numpy() == "STO"
    if raw["name"] == "582_LA92" and raw["folder"] == "10degC":
        resumed = (df["ts_local"] >= LA92_582_RESUME_LOCAL).to_numpy()
        return [("main", np.where(resumed)[0], False, None),
                ("attempt1", np.where(~resumed)[0], True,
                 "first run on 11/25 stopped on over-voltage (4.2345 V) after about 103 s, then a 13.2 h pause; the logger wrote this data again and inserted it later")], 0
    if kind in TAIL_TYPES and "tail_first" in summary:
        a, z = summary["tail_first"], summary["tail_last"]
        dropped = n - (z + 1)  # program-end rows after NN_Char_Charge (Ah counter reset)
        return [("main", np.arange(0, a), False, None),
                ("tail", np.arange(a, z + 1), True, "warm up -> 1C charge -> cool down part after the test (Procedure = NN_Char_Charge)")], dropped
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
        raise ValueError("ProgTime is not strictly increasing after cleaning")
    return idx, n_nan, n_dup


def split_rows(df, idx, nominal):
    t = df["t"].to_numpy()[idx]
    tw = (df["ts_utc"].to_numpy()[idx] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    lim = np.maximum(nominal[idx[:-1]], nominal[idx[1:]]) + SPLIT_MARGIN_S
    dtp, dtw = np.diff(t), np.diff(tw)
    cut = np.where((dtp > lim) | (dtw > lim))[0] + 1
    info = [{"split_before_raw_row": int(idx[k]), "dt_progtime_s": round(float(dtp[k - 1]), 3),
             "dt_timestamp_s": float(dtw[k - 1]),
             "reason": ("hidden pause (ProgTime continuous, time stamp jumps %.1f h)" % (dtw[k - 1] / 3600)) if dtp[k - 1] <= lim[k - 1]
             else ("logging gap %.1f s" % dtp[k - 1])} for k in cut]
    bounds = np.r_[0, cut, len(idx)]
    return list(zip(bounds[:-1], bounds[1:])), info


def dis0p5c_start_soc(summ, folder, batch, cap):
    """Option B: the start SOC of Dis_0p5C is back-calculated from the Ah balance of the tail part, using the offset of the same-temperature Dis_2C tail as reference."""
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
        check = "n/a (charged at 22-25 degC by design)"
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
    if np.isnan(arr[:, 4]).any():  # empty soc
        with open(path, "r") as f:
            text = f.read()
        with open(path, "w", newline="") as f:
            f.write(text.replace(",nan,", ",,"))


# ================================================================ main flow
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
    log(f"Raw CSV files: {len(files)}")

    # ---- pass 1
    summ = {}
    for folder, name in files:
        raw = read_raw(folder, name, raw_root)
        summ[(folder, name)] = summarize(raw)
        summ[(folder, name)]["md5"] = file_md5(raw["path"])
    caps, file_caps, checks, aging, events = compute_capacities(summ)
    log(f"Aging rate: pooled {aging['rate_pooled_per_Ah'] * 100:.4f} %/Ah, -20 degC {aging['rate_n20degC_per_Ah'] * 100:.4f} %/Ah")

    def capacity_of(folder, name, batch):
        if (folder, name) in file_caps:
            return file_caps[(folder, name)]
        return caps.get(batch)

    # ---- pass 2
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
        # soc reference point
        soc_anchor, ah_anchor, anchor_note = 1.0, 0.0, "fully charged at the start of the test (1C charge to 4.2 V / 50 mA at 24 degC), soc = 1"
        if kind == "Dis_0p5C":
            ah_anchor = s["anchor_ah"]
            soc_anchor, deficit = dis0p5c_start_soc(summ, folder, batch, cap["capacity_Ah"])
            anchor_note = ("the Ah counter at the start of the discharge continues from the C/20 test, not reset; " +
                           ("start SOC = %.4f, back-calculated from the Ah balance of the tail part (%.4f Ah below full, with the same-temperature Dis_2C tail as reference)"
                            % (soc_anchor, deficit) if deficit is not None else "no tail part at 25 degC, start SOC set to 1"))
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
                soc_rule = "continues from the SOC at the end of the test (Ah counter reset at the start of the tail): soc_end_of_test + (ah - ah[0]) / full capacity"
            elif excluded and (kind == "charge" or (key not in EXCLUDED_SOC and bname != "attempt1")):
                soc_block = np.full(len(idx), np.nan)
                soc_rule = "empty (start SOC or full capacity at that time cannot be determined)"
            elif excluded:
                how = ("batch", "582") if bname == "attempt1" else EXCLUDED_SOC[key]
                if how[0] == "batch":
                    c, cap_rule, cap_source = caps[how[1]]["capacity_Ah"], caps[how[1]]["rule"], caps[how[1]]["source"]
                    soc_rule = f"1 + ah / {c:.4f} (full capacity of batch {how[1]})"
                else:
                    c, cap_rule = -s["ah_min"], "capacity measured in the file itself"
                    cap_source = "1C capacity measured in this file itself (at the wrong temperature); not the same yardstick as the main data set"
                    soc_rule = f"1 + ah / {c:.4f} (capacity measured in the file itself; not the same yardstick as the main data set)"
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
                    notes.append("the Ah-counted SOC of the C/20 charge does not close with the discharge: the charged Ah of this file is %.1f %% of the discharged Ah, "
                                 "soc = %.4f at the end of the charge; probably an accumulated current offset over about 20 h at the small current of 0.15 A; kept as is"
                                 % (s["Ah_charged"] / s["Ah_discharged"] * 100, soc_block[-1]))
                if kind == "Dis_0p5C" and bname == "main":
                    notes.append(anchor_note)
                if name == "582_LA92" and bname == "main":
                    notes.append("only the data resumed after the 13.2 h pause are kept (from 2018-11-26 10:11:10); the file name and header date belong to the first run on 11/25")
                if split_info:
                    notes.append("split at logging gaps: " + "; ".join(f"before row {x['split_before_raw_row']}, {x['reason']}" for x in split_info))
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
    log(f"Series written: main data set {sum(not x['excluded'] for x in series)}, _excluded {sum(x['excluded'] for x in series)}")

    # ---- timeline
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
    tl.to_csv(os.path.join(out_dir, "file_timeline.csv"), index=False, float_format="%.4f", encoding="utf-8")

    # ---- manifest
    cap_table = []
    for b, cinfo in sorted(caps.items(), key=lambda x: (TEMPS.index(x[1]["folder"]), x[0])):
        cap_table.append({"batch": b, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in cinfo.items()}})
    for (folder, name), cinfo in file_caps.items():
        cap_table.append({"batch": name.split("_")[0], "file": name, **{k: (round(v, 5) if isinstance(v, float) else v)
                                                                          for k, v in cinfo.items()}})
    manifest = {
        "generated_by": "src/preprocess_kollmeyer.py", "raw_root": raw_root,
        "columns": {"time_s": "s, counted from the first row of the series (ProgTime)", "voltage": "V, terminal voltage", "current": "A, negative for discharge, positive for charge",
                    "temperature": "degC, cell surface temperature (AD592)", "soc": "fraction 0-1; definition in soc_rule",
                    "ah": "raw cumulative Ah of the tester (Capacity column), only a reference for the label, never a model input"},
        "soc_definition": "soc = 1 - discharged Ah / full capacity; full capacity = the batch's 1C capacity at the actual test temperature (one rule set), "
                          "different from the nominal 3.0 Ah used in the Kollmeyer example",
        "aging_model": {**{k: v for k, v in aging.items()}, "missing_tests": [
            {"id": m[0], "time_local_EST": m[1], "Ah_discharged_estimate": m[2], "basis": m[3]} for m in MISSING_TESTS]},
        "rule2_checks": checks, "capacities": cap_table, "raw_files": raw_records, "series": series,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, default=_json_default)
    return {"manifest": manifest, "timeline": tl, "capacities": caps, "file_capacities": file_caps, "checks": checks,
            "aging": aging, "events": events}


def tables(result):
    """Summary tables for the notebook: full capacity, rule 2 checks, included/excluded, splits, temperature checks."""
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
