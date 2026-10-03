"""Analysis helpers of the v2 experiment: load and check run folders, build tables.

Not part of the training code (changing this file does not change the code fingerprint).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import v2_common as V

COLS = ["Overall"] + V.FOLDERS                       # heatmap / table columns
COL_LABEL = {"Overall": "Overall", **{f: V.temp_label(f) for f in V.FOLDERS}}


class CheckError(SystemExit):
    pass


def sample_std(values) -> float:
    v = np.asarray(list(values), dtype=np.float64)
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def scan_runs(paths: V.Paths) -> tuple[dict, list[str], list[str]]:
    """Return ({run: record}, incomplete runs, problems). Hashes are verified."""
    runs, incomplete, problems = {}, [], []
    if not paths.runs.exists():
        return runs, incomplete, problems
    for d in sorted(paths.runs.iterdir()):
        if d.name.startswith((".", "_")):
            continue
        if not d.is_dir() or not V.RUN_RE.match(d.name):
            problems.append(f"unexpected entry in runs/ (a second copy?): {d.name}")
            continue
        st, why = V.run_status(paths, d.name)
        if st != "complete":
            incomplete.append(f"{d.name} ({why})")
            continue
        meta = V.read_json(d / "run_meta.json")
        met = V.read_json(d / "metrics.json")
        if meta["run"] != d.name or met["run"] != d.name:
            problems.append(f"{d.name}: files describe run {meta['run']!r}")
            continue
        if meta["exp"] != paths.exp:
            problems.append(f"{d.name}: trained for experiment {meta['exp']!r}, not {paths.exp!r}")
            continue
        runs[d.name] = {"meta": meta, "metrics": met}
    if paths.models.exists():
        for d in sorted(paths.models.iterdir()):
            if d.name.startswith((".", "_")):
                continue
            if not V.RUN_RE.match(d.name):
                problems.append(f"unexpected entry in models/: {d.name}")
            elif not (paths.runs / d.name).exists():
                problems.append(f"weights without a run folder: models/{paths.exp}/{d.name}")
    return runs, incomplete, problems


def check_consistency(runs: dict, log) -> None:
    """All runs must share code fingerprint, normalisation file, data and hyperparameters."""
    def key_hp(m):
        return json.dumps(m["hyperparameters"], sort_keys=True)
    checks = [("training code fingerprint", lambda m: m["code_fingerprint"]),
              ("normalisation sha256", lambda m: m["norm_stats_sha256"]),
              ("data sha256 summary", lambda m: m["data_sha256_summary"]),
              ("hyperparameters", key_hp)]
    errors = []
    for label, fn in checks:
        groups = defaultdict(list)
        for r, rec in runs.items():
            groups[fn(rec["meta"])].append(r)
        if len(groups) > 1:
            lines = [f"    {str(k)[:60]}: {', '.join(sorted(v))}" for k, v in groups.items()]
            errors.append(f"  runs differ in {label}:\n" + "\n".join(lines))
    for r, rec in runs.items():
        if rec["meta"]["norm_stats_sha256"] != V.EXPECTED_NORM_SHA256:
            errors.append(f"  {r}: normalisation sha256 is not the protocol value")
    if errors:
        raise CheckError("Consistency check FAILED:\n" + "\n".join(errors))
    if runs:
        m0 = next(iter(runs.values()))["meta"]
        commits = sorted({rec["meta"]["git_commit"][:12] for rec in runs.values()})
        log(f"Consistency OK: {len(runs)} runs share code fingerprint {m0['code_fingerprint'][:12]}, "
            f"normalisation {m0['norm_stats_sha256'][:12]}, data {m0['data_sha256_summary'][:12]}, "
            f"identical hyperparameters (max epochs {m0['hyperparameters']['max_epochs']}). "
            f"git commits: {', '.join(commits)}")


def run_row(rec: dict) -> dict:
    m, met = rec["meta"], rec["metrics"]
    row = {"run": m["run"], "config": m["config"], "L": m["num_layers"], "W": m["window"],
           "H": m["hidden_size"], "seed": m["seed"], "stage": m["stage"],
           "val_rmse_Overall": met["overall"]["rmse_pct"], "val_mae_Overall": met["overall"]["mae_pct"]}
    for f in V.FOLDERS:
        row[f"val_rmse_{f}"] = met["per_temp"][f]["rmse_pct"]
        row[f"val_mae_{f}"] = met["per_temp"][f]["mae_pct"]
    row.update({"n_params": m["n_params"], "latency_s": m["window"] / V.FS_HZ,
                "best_epoch": m["best_epoch"], "epochs_trained": m["epochs_trained"],
                "sec_per_epoch": m["sec_per_epoch"], "train_time_s": m["train_time_s"],
                "platform": m["platform"], "hostname": m["hostname"], "gpu_name": m["gpu_name"],
                "torch_version": m["torch_version"], "cuda_version": m["cuda_version"],
                "cudnn_version": m["cudnn_version"], "git_commit": m["git_commit"][:12],
                "start_time": m["start_time"], "end_time": m["end_time"]})
    return row


def recheck_metrics(paths: V.Paths, runs: dict) -> None:
    """Recompute every metrics.json from its npz (rows >= 359); stop on any difference."""
    bad = []
    for r, rec in runs.items():
        m = V.metrics_from_npz(paths.run_dir(r) / "val_predictions.npz", V.EVAL_START)
        if abs(m["overall"]["rmse_pct"] - rec["metrics"]["overall"]["rmse_pct"]) > 1e-9 or \
                m["overall"]["n_points"] != rec["metrics"]["overall"]["n_points"]:
            bad.append(r)
    if bad:
        raise CheckError(f"metrics.json differs from val_predictions.npz for: {', '.join(bad)}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def add_worst(df: pd.DataFrame, prefix: str = "val_rmse_", suffix: str = "") -> pd.DataFrame:
    cols = [f"{prefix}{f}{suffix}" for f in V.FOLDERS]
    vals = df[cols].to_numpy(dtype=float)
    idx = np.nanargmax(vals, axis=1)
    df[f"worst_temp{suffix}"] = [V.temp_label(V.FOLDERS[i]) for i in idx]
    df[f"worst_temp_rmse{suffix}"] = vals[np.arange(len(df)), idx]
    return df


def fmt_table(rows: list[list[str]], aligns: str) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(r, widths, aligns)]
        out.append("  ".join(cells).rstrip())
    return out


def step2_text(rank: pd.DataFrame, platforms: dict, n_expected: int, exp: str) -> str:
    head = ["rank", "config", "overall", "d_rank1"] + [V.temp_label(f) for f in V.FOLDERS] + \
           ["worst", "params", "W/fs(s)", "best/ep", "platform"]
    rows = [head]
    top = float(rank["val_rmse_Overall"].iloc[0])
    for _, r in rank.iterrows():
        rows.append([str(int(r["rank"])), r["config"], f"{r['val_rmse_Overall']:.3f}",
                     f"+{r['val_rmse_Overall'] - top:.3f}"]
                    + [f"{r[f'val_rmse_{f}']:.3f}" for f in V.FOLDERS]
                    + [f"{r['worst_temp']}", f"{int(r['n_params']):,}", f"{r['latency_s']:.0f}",
                       f"{int(r['best_epoch'])}/{int(r['epochs_trained'])}", r["platform"]])
    lines = ["STEP 2 SUMMARY (copy everything between the two lines of '=')", "=" * 100,
             f"experiment {exp}; seed {V.STEP2_SEED} only (one training run per config); "
             f"validation RMSE in SOC %, rows >= {V.EVAL_START}",
             f"{len(rank)}/{n_expected} configs; runs per platform: "
             + ", ".join(f"{k} {v}" for k, v in sorted(platforms.items())),
             "d_rank1 = overall RMSE minus the rank-1 overall RMSE; worst = temperature with the highest RMSE",
             ""]
    lines += fmt_table(rows, "rllr" + "r" * len(V.FOLDERS) + "lrrrl")
    lines += ["=" * 100]
    return "\n".join(lines) + "\n"


def step3_tables(runs: dict, cands: list[str], bench: pd.DataFrame | None):
    """(summary, per_seed) data frames for the step-3 configs; neutral order (L, W, H)."""
    per_seed, summary = [], []
    for c in sorted(cands, key=V.config_sort_key):
        rs = [runs[V.run_name(c, s)] for s in V.ALL_SEEDS if V.run_name(c, s) in runs]
        rows = [run_row(r) for r in rs]
        per_seed += rows
        l, w, h = V.parse_config(c)
        row = {"config": c, "L": l, "W": w, "H": h, "n_seeds": len(rows),
               "seeds": " ".join(str(r["seed"]) for r in rows)}
        for col in COLS:
            v = [r[f"val_rmse_{col}"] for r in rows]
            row[f"rmse_{col}_mean"] = float(np.mean(v)) if v else np.nan
            row[f"rmse_{col}_std"] = sample_std(v)
        v = [r["val_mae_Overall"] for r in rows]
        row["mae_Overall_mean"], row["mae_Overall_std"] = (float(np.mean(v)) if v else np.nan), sample_std(v)
        for r in rows:
            row[f"rmse_Overall_seed{r['seed']}"] = r["val_rmse_Overall"]
        row["n_params"] = V.n_params(c)
        row["latency_s"] = w / V.FS_HZ
        if bench is not None and c in set(bench["config"]):
            b = bench[bench["config"] == c].iloc[0]
            row["cpu_infer_ms_mean"], row["cpu_infer_ms_std"] = b["inference_ms_mean"], b["inference_ms_std"]
        else:
            row["cpu_infer_ms_mean"] = row["cpu_infer_ms_std"] = np.nan
        row["platforms"] = "; ".join(sorted({f"{r['platform']}:{r['gpu_name']}" for r in rows}))
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(per_seed)


def window_saturation(summary: pd.DataFrame) -> pd.DataFrame:
    """Adjacent windows among candidates that share L and H."""
    rows = []
    for (l, h), g in summary.groupby(["L", "H"]):
        g = g.sort_values("W")
        if len(g) < 2:
            continue
        for (_, a), (_, b) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            for col in COLS:
                ma, mb = a[f"rmse_{col}_mean"], b[f"rmse_{col}_mean"]
                sa, sb = a[f"rmse_{col}_std"], b[f"rmse_{col}_std"]
                pooled = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
                imp = ma - mb
                rows.append({"L": l, "H": h, "W_from": int(a["W"]), "W_to": int(b["W"]), "column": COL_LABEL[col],
                             "rmse_mean_from": ma, "rmse_mean_to": mb,
                             "abs_improvement": imp, "rel_improvement_pct": 100 * imp / ma,
                             "seed_std_from": sa, "seed_std_to": sb, "pooled_seed_std": pooled,
                             "improvement_over_pooled_std": imp / pooled if pooled > 0 else np.nan})
    return pd.DataFrame(rows)


def pm(m: float, s: float, nd: int = 3) -> str:
    return f"{m:.{nd}f} ± {s:.{nd}f}"
