"""Step 4: test-set evaluation of ONE configuration chosen by hand (run only after selection).

    python src/evaluate_paper.py --config W240_H32_L1
    python src/evaluate_paper.py --config W240_H32_L1 --unseen   # also -10 / -20 degC (never trained on)
    python src/evaluate_paper.py --config W60_H32_L1 --smoke

The models of all seeds (42-46) of the config are evaluated on the 15 test files (UDDS, HWFET,
LA92, US06 at 0, 10, 25, 40 degC; 25 degC has no HWFET), scoring only rows with in-sequence
index >= 299 like the validation set. Results are reported as mean +- sample std over the seeds.
"Average" is the mean over the test files of each seed's per-file metric, then mean +- std over
seeds; the RMSE pooled over all test points is also given in test_summary.csv.

Capacity marks come from manifest.json (capacity_rule): 'a' = full capacity back-calculated from
the discharge cutoff capacity (rule 2), 'b' = full capacity borrowed from batch 576's Cap_1C with an
aging correction (rule 3; this 10 degC US06 label is known to be biased); no mark = the batch's own
Cap_1C test.

--unseen evaluates the same models on the four standard cycles at -10 and -20 degC, temperatures
never seen in training. These results are stored separately and are never used for any choice.

Outputs in results/<name>/test/<config>/:
    per_file_metrics.csv          one row per (seed, test file)
    test_results.csv              one row per test file: RMSE / MAE mean and std over seeds, + average row
    test_rmse_table.csv           temperature x cycle, "mean ± std" (RMSE, SOC percentage points)
    test_mae_table.csv            same for MAE
    test_summary.csv              average over the test files (and pooled) RMSE / MAE, mean ± std over seeds
    test_table.tex                IEEE-ready LaTeX table with footnotes
    predictions_seed42/*.csv      seed-42 predictions per test file (time_s, soc_true, soc_pred), for figures
    summary.json                  settings and headline numbers
    unseen_temp_results.csv       (--unseen) per file, mean ± std over seeds, + average row
    unseen_per_file_metrics.csv   (--unseen) one row per (seed, file)
    unseen_predictions_seed42/    (--unseen) seed-42 predictions
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
import paper_common as C  # noqa: E402
import sweep_lstm as SL  # noqa: E402
from lstm_model import LSTMSOC  # noqa: E402

QUIET = lambda *_: None  # noqa: E731
UNSEEN_FOLDERS = ["n20degC", "n10degC"]
MARK_TEX = {"a": r"$^\mathrm{a}$", "b": r"$^\mathrm{b}$"}
# Expected marks for the 0-40 degC test set (checked against the manifest)
EXPECTED_MARKS = {"10degC/582_LA92": "a", "40degC/556_UDDS": "a", "40degC/556_HWFET": "a",
                  "40degC/556_LA92": "a", "40degC/556_US06": "a", "10degC/567_US06": "b"}


def capacity_marks(ids: list[str]) -> dict[str, str]:
    series = {e["id"]: e for e in D.load_manifest()["series"]}
    marks = {}
    for i in ids:
        rule = str(series[i].get("capacity_rule") or "")
        marks[i] = "a" if rule.startswith("2") else "b" if rule.startswith("3") else ""
    return marks


def load_models(ctx: C.Context, cname: str):
    paths = ctx.paths
    missing = [s for s in ctx.seeds if not paths.model_file(C.run_name(cname, s)).exists()]
    if missing:
        raise SystemExit(f"Missing weights for {cname} seeds {missing} in {paths.rel(paths.models)}/ "
                         "(finish step 3 and copy the weights to this machine)")
    file_sha = C.file_sha256(paths.norm_stats) if paths.norm_stats.exists() else None
    models, stats, sha = {}, None, None
    for s in ctx.seeds:
        ck = torch.load(paths.model_file(C.run_name(cname, s)), map_location="cpu", weights_only=True)
        if ck["config_name"] != cname or int(ck["seed"]) != s:
            raise SystemExit(f"{C.run_name(cname, s)}.pt holds {ck['config_name']} seed {ck['seed']}")
        if list(ck["features"]) != D.FEATURES:
            raise SystemExit(f"{C.run_name(cname, s)}.pt was trained on {ck['features']}")
        if ck["settings"]["temperatures_C"] != ctx.temps:
            raise SystemExit(f"{C.run_name(cname, s)}.pt was trained on {ck['settings']['temperatures_C']} degC")
        if sha is None:
            stats, sha = ck["norm_stats"], ck["norm_stats_sha256"]
        elif ck["norm_stats_sha256"] != sha:
            raise SystemExit(f"Seeds of {cname} used different normalisation files")
        cfg = ck["config"]
        m = LSTMSOC(cfg["hidden_size"], cfg["num_layers"], cfg["input_size"])
        m.load_state_dict(ck["state_dict"])
        models[s] = m
    if file_sha is not None and file_sha != sha:
        raise SystemExit(f"Weights used normalisation sha256 {sha[:12]}, but {paths.rel(paths.norm_stats)} "
                         f"is {file_sha[:12]}")
    return models, stats, sha


def evaluate(models: dict, stats: dict, window: int, seqs: list, marks: dict, device, pred_dir: Path,
             log) -> tuple[pd.DataFrame, dict]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], {}
    for seed, model in models.items():
        model.to(device)
        for seq in seqs:
            info = seq.info
            ds = D.WindowDataset([seq], stats, window, C.EVAL_START, QUIET).to(device)
            pred = SL.predict(model, ds).cpu().numpy().astype(np.float64)
            loc = ds.local_index
            y = seq.y[loc]
            rmse, mae = SL.metrics_pct(pred, y)
            err = (pred - y) * 100
            errors[(seed, info.id)] = err
            rows.append({"seed": seed, "file": info.id, "temperature": C.temp_label(info.folder),
                         "temp_c": info.temp_c, "cycle": info.test_name, "capacity_mark": marks[info.id],
                         "n_points": len(loc), "rmse_pct": rmse, "mae_pct": mae,
                         "max_abs_err_pct": float(np.max(np.abs(err)))})
            if seed == C.SWEEP_SEED:
                pd.DataFrame({"time_s": seq.sec[loc] - seq.sec[0], "local_index": loc, "soc_true": y,
                              "soc_pred": pred}).to_csv(
                    pred_dir / f"{info.folder}_{info.test_name}.csv", index=False, float_format="%.6f")
        log(f"  seed {seed}: " + "  ".join(
            f"{r['temperature']} {r['cycle']} {r['rmse_pct']:.2f}" for r in rows if r["seed"] == seed))
    return pd.DataFrame(rows), errors


def per_file_table(per: pd.DataFrame, n_files_label: str) -> pd.DataFrame:
    g = per.groupby(["file", "temperature", "temp_c", "cycle", "capacity_mark"], sort=False)
    out = g.agg(rmse_pct_mean=("rmse_pct", "mean"), rmse_pct_std=("rmse_pct", lambda v: C.sample_std(v)),
                mae_pct_mean=("mae_pct", "mean"), mae_pct_std=("mae_pct", lambda v: C.sample_std(v)),
                n_seeds=("seed", "nunique"), n_points=("n_points", "first")).reset_index()
    cyc = {c: i for i, c in enumerate(D.DRIVE_CYCLES)}
    out = out.sort_values(["temp_c", "cycle"], key=lambda s: s.map(cyc) if s.name == "cycle" else s)
    avg = average_over_files(per)
    out = pd.concat([out, pd.DataFrame([{
        "file": n_files_label, "temperature": "all", "temp_c": np.nan, "cycle": "all", "capacity_mark": "",
        "rmse_pct_mean": avg["rmse"][0], "rmse_pct_std": avg["rmse"][1],
        "mae_pct_mean": avg["mae"][0], "mae_pct_std": avg["mae"][1],
        "n_seeds": per["seed"].nunique(), "n_points": per.groupby("seed")["n_points"].sum().iloc[0]}])],
        ignore_index=True)
    return out


def average_over_files(per: pd.DataFrame) -> dict:
    by_seed = per.groupby("seed")[["rmse_pct", "mae_pct"]].mean()
    return {"rmse": (float(by_seed["rmse_pct"].mean()), C.sample_std(by_seed["rmse_pct"])),
            "mae": (float(by_seed["mae_pct"].mean()), C.sample_std(by_seed["mae_pct"]))}


def pm(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def grid_table(res: pd.DataFrame, metric: str, folders: list[str]) -> pd.DataFrame:
    tab = pd.DataFrame(index=[C.temp_label(f) for f in folders], columns=D.DRIVE_CYCLES, dtype=object)
    tab[:] = "--"
    for _, r in res[res["temperature"] != "all"].iterrows():
        tab.loc[r["temperature"], r["cycle"]] = pm(r[f"{metric}_pct_mean"], r[f"{metric}_pct_std"]) + r["capacity_mark"]
    tab.index.name = "temperature"
    return tab


def latex_table(res: pd.DataFrame, folders: list[str], cname: str, n_seeds: int, n_files: int) -> str:
    w, h, l = C.parse_config(cname)
    lines = [
        f"% Generated by src/evaluate_paper.py on {datetime.now():%Y-%m-%d %H:%M} "
        f"({cname}, {n_seeds} seeds, rows >= {C.EVAL_START}). Needs no extra packages.",
        r"\begin{table}[!t]", r"\centering",
        rf"\caption{{Test-set SOC estimation error of the selected LSTM ($W={w}$, $H={h}$, $L={l}$), "
        rf"mean $\pm$ std over {n_seeds} seeds}}",
        r"\label{tab:test_results}", r"\begin{tabular}{llcc}", r"\hline",
        r"Temp. & Cycle & RMSE (\%) & MAE (\%) \\", r"\hline"]
    body = res[res["temperature"] != "all"]
    used_marks, has_missing = set(), False
    for folder in folders:
        label = rf"{C.temp_c(folder)}\,$^\circ$C"
        first = True
        for cyc in D.DRIVE_CYCLES:
            r = body[(body["temperature"] == C.temp_label(folder)) & (body["cycle"] == cyc)]
            t = label if first else ""
            first = False
            if r.empty:
                lines.append(rf"{t} & {cyc} & -- & -- \\")
                has_missing = True
                continue
            r = r.iloc[0]
            mark = r["capacity_mark"]
            used_marks.add(mark)
            lines.append(rf"{t} & {cyc}{MARK_TEX.get(mark, '')} & "
                         rf"${r['rmse_pct_mean']:.2f} \pm {r['rmse_pct_std']:.2f}$ & "
                         rf"${r['mae_pct_mean']:.2f} \pm {r['mae_pct_std']:.2f}$ \\")
        lines.append(r"\hline")
    avg = res[res["temperature"] == "all"].iloc[0]
    lines += [rf"\multicolumn{{2}}{{l}}{{Average ({n_files} files)}} & "
              rf"${avg['rmse_pct_mean']:.2f} \pm {avg['rmse_pct_std']:.2f}$ & "
              rf"${avg['mae_pct_mean']:.2f} \pm {avg['mae_pct_std']:.2f}$ \\", r"\hline", r"\end{tabular}"]
    notes = []
    if has_missing:
        notes.append(r"--: no data (25\,$^\circ$C HWFET raw record corrupted).")
    if "a" in used_marks:
        notes.append(r"$^\mathrm{a}$Full capacity back-calculated from the discharge cutoff capacity.")
    if "b" in used_marks:
        notes.append(r"$^\mathrm{b}$Full capacity from batch 576 Cap\_1C with aging correction; "
                     r"label known to be biased.")
    if used_marks - {""}:
        notes.append(r"Others: full capacity from the batch's own Cap\_1C test.")
    if notes:
        lines += [r"\vspace{2pt}", r"\parbox{\columnwidth}{\raggedright\footnotesize",
                  "\\\\\n".join(notes) + "}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Paper step 4: test-set evaluation of one chosen config")
    C.add_common_args(p)
    p.add_argument("--config", required=True, help="the config you chose, e.g. W240_H32_L1")
    p.add_argument("--unseen", action="store_true",
                   help="also evaluate -10 and -20 degC (never trained on); stored separately")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    ctx = C.resolve(args)
    paths = ctx.paths
    cname = args.config
    w, h, l = C.parse_config(cname)
    if cname not in ctx.config_names:
        raise SystemExit(f"{cname} is not in the grid {ctx.config_names}")
    device = SL.pick_device(args.device)
    out = paths.test / cname
    log = C.Tee(out / "evaluate_log.txt")
    log("=" * 78)
    log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  TEST-SET evaluation of {cname} ({paths.name}), seeds {ctx.seeds}, "
        f"rows >= {C.EVAL_START}, device {C.device_label(device)}")
    if paths.selected.exists():
        rec = C.read_json(paths.selected)["recommended_config"]
        if rec != cname:
            log(f"Note: select_config.py recommended {rec}; evaluating {cname} as requested")

    models, stats, sha = load_models(ctx, cname)
    splits = D.build_splits(ctx.folders)
    log(f"Reading the {len(splits['test'])} test files: " + ", ".join(s.id for s in splits["test"]))
    test_seqs = D.load_sequences(splits["test"], log)
    marks = capacity_marks([s.id for s in splits["test"]])
    for i, m in EXPECTED_MARKS.items():
        if i in marks and marks[i] != m:
            log(f"WARNING: manifest gives {i} capacity mark {marks[i]!r}, expected {m!r}")
    for i, m in marks.items():
        if m and i not in EXPECTED_MARKS:
            log(f"WARNING: unexpected capacity mark {m!r} for {i}")

    per, errors = evaluate(models, stats, w, test_seqs, marks, device, out / "predictions_seed42", log)
    per.to_csv(out / "per_file_metrics.csv", index=False, float_format="%.4f")
    n_files = len(test_seqs)
    res = per_file_table(per, f"Average ({n_files} files)")
    res.to_csv(out / "test_results.csv", index=False, float_format="%.4f")
    for metric in ("rmse", "mae"):
        grid_table(res, metric, ctx.folders).to_csv(out / f"test_{metric}_table.csv", encoding="utf-8")

    avg = average_over_files(per)
    pooled = {}
    for metric in ("rmse", "mae"):
        vals = []
        for s in ctx.seeds:
            e = np.concatenate([errors[(s, q.info.id)] for q in test_seqs])
            vals.append(float(np.sqrt(np.mean(e ** 2))) if metric == "rmse" else float(np.mean(np.abs(e))))
        pooled[metric] = (float(np.mean(vals)), C.sample_std(vals))
    summary = pd.DataFrame([
        {"metric": "RMSE", "basis": f"mean over {n_files} files", "mean_pct": avg["rmse"][0], "std_pct": avg["rmse"][1]},
        {"metric": "MAE", "basis": f"mean over {n_files} files", "mean_pct": avg["mae"][0], "std_pct": avg["mae"][1]},
        {"metric": "RMSE", "basis": "pooled over all test points", "mean_pct": pooled["rmse"][0], "std_pct": pooled["rmse"][1]},
        {"metric": "MAE", "basis": "pooled over all test points", "mean_pct": pooled["mae"][0], "std_pct": pooled["mae"][1]}])
    summary["n_seeds"] = len(ctx.seeds)
    summary.to_csv(out / "test_summary.csv", index=False, float_format="%.4f")
    (out / "test_table.tex").write_text(latex_table(res, ctx.folders, cname, len(ctx.seeds), n_files),
                                        encoding="utf-8", newline="\n")

    log(f"\nTest RMSE (SOC percentage points, mean ± std over {len(ctx.seeds)} seeds):")
    log(grid_table(res, "rmse", ctx.folders).to_string())
    log(f"\nTest MAE:\n{grid_table(res, 'mae', ctx.folders).to_string()}")
    log(f"\nAverage over {n_files} files: RMSE {pm(*avg['rmse'])}, MAE {pm(*avg['mae'])}; "
        f"pooled RMSE {pm(*pooled['rmse'])}")
    log("Marks: a = capacity back-calculated from cutoff capacity; b = batch-576 capacity with aging "
        "correction (label known to be biased)")

    record = {"config": cname, "seeds": ctx.seeds, "eval_start_index": C.EVAL_START,
              "n_test_files": n_files, "test_files": [s.id for s in splits["test"]],
              "average_over_files": {"rmse_pct": avg["rmse"], "mae_pct": avg["mae"]},
              "pooled": {"rmse_pct": pooled["rmse"], "mae_pct": pooled["mae"]},
              "norm_stats_sha256": sha, "device": str(device), "git_commit": C.git_info()[0],
              "evaluated_at": datetime.now().isoformat(timespec="seconds")}

    if args.unseen:
        log(f"\n--unseen: {UNSEEN_FOLDERS} (never seen in training; not used for any choice)")
        su = D.build_splits(UNSEEN_FOLDERS)
        useqs = D.load_sequences(su["test"], log)
        umarks = capacity_marks([s.id for s in su["test"]])
        uper, _ = evaluate(models, stats, w, useqs, umarks, device, out / "unseen_predictions_seed42", log)
        uper.to_csv(out / "unseen_per_file_metrics.csv", index=False, float_format="%.4f")
        ures = per_file_table(uper, f"Average ({len(useqs)} files)")
        ures.to_csv(out / "unseen_temp_results.csv", index=False, float_format="%.4f")
        log(grid_table(ures, "rmse", UNSEEN_FOLDERS).to_string())
        ua = average_over_files(uper)
        log(f"Unseen average over {len(useqs)} files: RMSE {pm(*ua['rmse'])}, MAE {pm(*ua['mae'])}")
        record["unseen_average_over_files"] = {"rmse_pct": ua["rmse"], "mae_pct": ua["mae"]}

    C.write_json(out / "summary.json", record)
    log(f"\nResults written to {paths.rel(out)}/")
    log.close()


if __name__ == "__main__":
    main()
