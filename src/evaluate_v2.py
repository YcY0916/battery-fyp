"""Test-set evaluation of the ONE final config chosen by the author (run once, on the laptop).

    python src/evaluate_v2.py --exp paper_0to40_v2 --config L1_W300_H64 --unseen

The weights of seeds 42-46 of the config are evaluated on the 15 test files (UDDS, HWFET, LA92,
US06 at 0, 10, 25, 40 °C; 25 °C has no HWFET), scoring rows >= 359 like the validation set.
--unseen also evaluates the four cycles at −20 and −10 °C (never used in training; stored
separately, never used for any choice). Results: mean ± sample std (ddof = 1) over the seeds.
"Average" = mean over the test files of each seed's per-file metric, then mean ± std over seeds.

For paper_0to40_v2 the script refuses to run a second time (test/TEST_DONE.json) and only accepts
a config listed in shards/step3_candidates.txt whose five runs are complete.
Before reading, the test files are checked against protocol/v2/data_sha256.json.

Capacity marks (from manifest.json capacity_rule): a = full capacity back-calculated from the
discharge cutoff capacity (rule 2); b = full capacity borrowed from batch 576 Cap_1C with an aging
correction (rule 3; this 10 °C US06 label is known to be biased); no mark = the batch's own Cap_1C.

Outputs in results/<exp>/test/<config>/:
    per_file_metrics.csv, test_results.csv, test_rmse_table.csv, test_mae_table.csv,
    test_summary.csv, test_table.tex, summary.json, predictions_seed42/<folder>_<cycle>.csv,
    unseen_per_file_metrics.csv, unseen_temp_results.csv, unseen_predictions_seed42/ (--unseen)
and results/<exp>/test/TEST_DONE.json (written last).

--dry-run-on-val (temporary experiments only, for the smoke test): the validation files stand in
for the test files, so the whole code path runs without opening any test file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
import sweep_lstm as SL  # noqa: E402
import v2_analysis as A  # noqa: E402
import v2_common as V  # noqa: E402
from lstm_model import LSTMSOC  # noqa: E402

QUIET = lambda *_: None  # noqa: E731
MARK_TEX = {"a": r"$^\mathrm{a}$", "b": r"$^\mathrm{b}$"}
EXPECTED_MARKS = {"10degC/582_LA92": "a", "40degC/556_UDDS": "a", "40degC/556_HWFET": "a",
                  "40degC/556_LA92": "a", "40degC/556_US06": "a", "10degC/567_US06": "b"}


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def capacity_marks(ids: list[str]) -> dict[str, str]:
    series = {e["id"]: e for e in D.load_manifest()["series"]}
    out = {}
    for i in ids:
        rule = str(series[i].get("capacity_rule") or "")
        out[i] = "a" if rule.startswith("2") else "b" if rule.startswith("3") else ""
    return out


def load_models(paths: V.Paths, config: str, seeds: list[int], log):
    models, stats = {}, None
    for s in seeds:
        run = V.run_name(config, s)
        st, why = V.run_status(paths, run)
        if st != "complete":
            raise SystemExit(f"{run} is not complete ({st} {why}); collect all step-3 runs first")
        ck = torch.load(paths.model_file(run), map_location="cpu", weights_only=True)
        if ck["config_name"] != config or int(ck["seed"]) != s:
            raise SystemExit(f"{run}/model.pt holds {ck['config_name']} seed {ck['seed']}")
        if ck["norm_stats_sha256"] != V.EXPECTED_NORM_SHA256:
            raise SystemExit(f"{run} used another normalisation file")
        stats = ck["norm_stats"]
        c = ck["config"]
        m = LSTMSOC(c["hidden_size"], c["num_layers"], c["input_size"])
        m.load_state_dict(ck["state_dict"])
        models[s] = m
    log(f"Loaded {len(models)} models of {config} (seeds {', '.join(map(str, models))})")
    return models, stats


def evaluate(models, stats, window, seqs, marks, device, pred_dir: Path, log) -> tuple[pd.DataFrame, dict]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], {}
    for seed, model in models.items():
        model.to(device)
        for seq in seqs:
            info = seq.info
            ds = D.WindowDataset([seq], stats, window, V.EVAL_START, QUIET).to(device)
            pred = SL.predict(model, ds).cpu().numpy().astype(np.float64)
            loc = ds.local_index
            y = seq.y[loc]
            rmse, mae = SL.metrics_pct(pred, y)
            err = (pred - y) * 100
            errors[(seed, info.id)] = err
            rows.append({"seed": seed, "file": info.id, "temperature": V.temp_label(info.folder),
                         "temp_c": info.temp_c, "cycle": info.test_name, "capacity_mark": marks[info.id],
                         "n_points": len(loc), "rmse_pct": rmse, "mae_pct": mae,
                         "max_abs_err_pct": float(np.max(np.abs(err)))})
            if seed == V.STEP2_SEED:
                df = pd.DataFrame({"time_s": seq.sec[loc] - seq.sec[0], "local_index": loc,
                                   "soc_true": y, "soc_pred": pred})
                V.write_text_atomic(pred_dir / f"{info.folder}_{info.test_name}.csv",
                                    df.to_csv(index=False, float_format="%.6f", lineterminator="\n"))
        log(f"  seed {seed}: " + "  ".join(f"{r['temperature']} {r['cycle']} {r['rmse_pct']:.2f}"
                                          for r in rows if r["seed"] == seed))
    return pd.DataFrame(rows), errors


def per_file_table(per: pd.DataFrame) -> pd.DataFrame:
    g = per.groupby(["file", "temperature", "temp_c", "cycle", "capacity_mark"], sort=False)
    out = g.agg(rmse_pct_mean=("rmse_pct", "mean"), rmse_pct_std=("rmse_pct", A.sample_std),
                mae_pct_mean=("mae_pct", "mean"), mae_pct_std=("mae_pct", A.sample_std),
                n_seeds=("seed", "nunique"), n_points=("n_points", "first")).reset_index()
    order = {c: i for i, c in enumerate(D.DRIVE_CYCLES)}
    out["_o"] = out["cycle"].map(lambda c: order.get(c, 99))
    out = out.sort_values(["temp_c", "_o"]).drop(columns="_o")
    out["n_files"] = 1
    avg = average_over_files(per)
    out = pd.concat([out, pd.DataFrame([{
        "file": f"Average ({per['file'].nunique()} files)", "temperature": "all", "temp_c": np.nan,
        "cycle": "all", "capacity_mark": "", "rmse_pct_mean": avg["rmse"][0], "rmse_pct_std": avg["rmse"][1],
        "mae_pct_mean": avg["mae"][0], "mae_pct_std": avg["mae"][1], "n_seeds": per["seed"].nunique(),
        "n_points": int(per.groupby("seed")["n_points"].sum().iloc[0]), "n_files": per["file"].nunique()}])],
        ignore_index=True)
    return out


def average_over_files(per: pd.DataFrame) -> dict:
    by_seed = per.groupby("seed")[["rmse_pct", "mae_pct"]].mean()
    return {"rmse": (float(by_seed["rmse_pct"].mean()), A.sample_std(by_seed["rmse_pct"])),
            "mae": (float(by_seed["mae_pct"].mean()), A.sample_std(by_seed["mae_pct"]))}


def grid_table(res: pd.DataFrame, metric: str, folders: list[str]) -> pd.DataFrame:
    cycles = list(dict.fromkeys(res.loc[res["cycle"] != "all", "cycle"]))
    tab = pd.DataFrame(index=[V.temp_label(f) for f in folders], columns=cycles, dtype=object)
    tab[:] = "--"
    for _, r in res[res["cycle"] != "all"].iterrows():
        mark = f" ({r['capacity_mark']})" if r["capacity_mark"] else ""
        tab.loc[r["temperature"], r["cycle"]] = A.pm(r[f"{metric}_pct_mean"], r[f"{metric}_pct_std"], 2) + mark
    tab.index.name = "temperature"
    return tab


def latex_table(res: pd.DataFrame, folders: list[str], config: str) -> str:
    l, w, h = V.parse_config(config)
    n_seeds = int(res["n_seeds"].max())
    lines = [f"% Generated by src/evaluate_v2.py ({config}, {n_seeds} seeds, rows >= {V.EVAL_START}).",
             r"\begin{table}[!t]", r"\centering",
             rf"\caption{{Test-set SOC estimation error of the final LSTM ($L={l}$, $W={w}$, $H={h}$), "
             rf"mean $\pm$ std over {n_seeds} seeds}}",
             r"\label{tab:test_results}", r"\begin{tabular}{llcc}", r"\hline",
             r"Temp. & Cycle & RMSE (\%) & MAE (\%) \\", r"\hline"]
    body = res[res["cycle"] != "all"]
    used, missing = set(), False
    for f in folders:
        first = True
        for cyc in D.DRIVE_CYCLES:
            r = body[(body["temperature"] == V.temp_label(f)) & (body["cycle"] == cyc)]
            t = rf"{int(f.replace('n', '-').replace('degC', ''))}\,$^\circ$C" if first else ""
            first = False
            if r.empty:
                lines.append(rf"{t} & {cyc} & -- & -- \\")
                missing = True
                continue
            r = r.iloc[0]
            used.add(r["capacity_mark"])
            lines.append(rf"{t} & {cyc}{MARK_TEX.get(r['capacity_mark'], '')} & "
                         rf"${r['rmse_pct_mean']:.2f} \pm {r['rmse_pct_std']:.2f}$ & "
                         rf"${r['mae_pct_mean']:.2f} \pm {r['mae_pct_std']:.2f}$ \\")
        lines.append(r"\hline")
    a = res[res["cycle"] == "all"].iloc[0]
    lines += [rf"\multicolumn{{2}}{{l}}{{Average ({int(a['n_files'])} files)}} & "
              rf"${a['rmse_pct_mean']:.2f} \pm {a['rmse_pct_std']:.2f}$ & "
              rf"${a['mae_pct_mean']:.2f} \pm {a['mae_pct_std']:.2f}$ \\", r"\hline", r"\end{tabular}"]
    notes = []
    if missing:
        notes.append(r"--: no data (25\,$^\circ$C HWFET raw record corrupted).")
    if "a" in used:
        notes.append(r"$^\mathrm{a}$Full capacity back-calculated from the discharge cutoff capacity.")
    if "b" in used:
        notes.append(r"$^\mathrm{b}$Full capacity from batch 576 Cap\_1C with aging correction; label known to be biased.")
    if used - {""}:
        notes.append(r"Others: full capacity from the batch's own Cap\_1C test.")
    if notes:
        lines += [r"\vspace{2pt}", r"\parbox{\columnwidth}{\raggedright\footnotesize", "\\\\\n".join(notes) + "}"]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    V.write_text_atomic(path, df.to_csv(index=index, float_format="%.4f", lineterminator="\n"))


def run_set(name, models, stats, w, seqs, out: Path, folders, log, prefix=""):
    marks = capacity_marks([s.info.id for s in seqs]) if seqs[0].info.split == "test" else \
        {s.info.id: "" for s in seqs}
    if seqs[0].info.split == "test" and not prefix:
        for i, m in EXPECTED_MARKS.items():
            if i in marks and marks[i] != m:
                log(f"WARNING: manifest gives {i} capacity mark {marks[i]!r}, expected {m!r}")
    per, errors = evaluate(models, stats, w, seqs, marks, torch.device("cpu"),
                           out / f"{prefix}predictions_seed42", log)
    save_csv(per, out / f"{prefix}per_file_metrics.csv")
    res = per_file_table(per)
    save_csv(res, out / (f"{prefix}test_results.csv" if not prefix else "unseen_temp_results.csv"))
    for metric in ("rmse", "mae"):
        save_csv(grid_table(res, metric, folders), out / f"{prefix}test_{metric}_table.csv", index=True)
    log(f"\n{name} RMSE (SOC %, mean ± std over {len(models)} seeds):\n{grid_table(res, 'rmse', folders).to_string()}")
    return per, res, errors


def main() -> None:
    p = argparse.ArgumentParser(description="v2 test-set evaluation of the final config (once)")
    p.add_argument("--exp", default=V.FORMAL_EXP)
    p.add_argument("--config", required=True)
    p.add_argument("--unseen", action="store_true", help="also −20 and −10 °C (never trained on)")
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--data-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--dry-run-on-val", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seeds", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    formal = args.exp == V.FORMAL_EXP
    if formal and (args.dry_run_on_val or args.seeds):
        raise SystemExit("--dry-run-on-val and --seeds are only for temporary experiments")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else V.ALL_SEEDS
    D.set_data_root(args.data_root)
    paths = V.Paths(args.out_root, args.exp)
    config = args.config
    l, w, h = V.parse_config(config)
    done_marker = paths.test / "TEST_DONE.json"
    if formal:
        if done_marker.exists():
            raise SystemExit(f"The test set was already evaluated ({paths.rel(done_marker)}: "
                             f"{V.read_json(done_marker)['config']}). The protocol allows one evaluation.")
        if config not in V.read_step3_candidates():
            raise SystemExit(f"{config} is not one of the step-3 configs")
    out = paths.test / config
    log = Log(out / "evaluate_log.txt")
    log("=" * 78)
    log(f"{V.now_iso()}  {'DRY RUN ON VALIDATION FILES' if args.dry_run_on_val else 'TEST-SET evaluation'} "
        f"of {config} ({args.exp}), seeds {seeds}, rows >= {V.EVAL_START}")
    models, stats = load_models(paths, config, seeds, log)
    code = V.check_code_version(formal, log)

    if args.dry_run_on_val:
        D.block_test_files()
        seqs = D.load_sequences(D.build_splits(V.FOLDERS)["val"], log)
        useqs = seqs if args.unseen else None
        log("Dry run: validation files stand in for the test files; no test file is opened")
    else:
        roles = ("test", "unseen_test") if args.unseen else ("test",)
        V.check_data_files(log, roles=roles, exist_only=("manifest", "train", "val") +
                           (() if args.unseen else ("unseen_test",)))
        seqs = D.load_sequences(D.build_splits(V.FOLDERS)["test"], log)
        useqs = D.load_sequences(D.build_splits(V.UNSEEN_FOLDERS)["test"], log) if args.unseen else None

    per, res, errors = run_set("Test", models, stats, w, seqs, out, V.FOLDERS, log)
    avg = average_over_files(per)
    pooled = {}
    for metric in ("rmse", "mae"):
        vals = []
        for s in models:
            e = np.concatenate([errors[(s, q.info.id)] for q in seqs])
            vals.append(float(np.sqrt(np.mean(e ** 2))) if metric == "rmse" else float(np.mean(np.abs(e))))
        pooled[metric] = (float(np.mean(vals)), A.sample_std(vals))
    n = len(seqs)
    summary = pd.DataFrame([
        {"metric": "RMSE", "basis": f"mean over {n} files", "mean_pct": avg["rmse"][0], "std_pct": avg["rmse"][1]},
        {"metric": "MAE", "basis": f"mean over {n} files", "mean_pct": avg["mae"][0], "std_pct": avg["mae"][1]},
        {"metric": "RMSE", "basis": "pooled over all test points", "mean_pct": pooled["rmse"][0], "std_pct": pooled["rmse"][1]},
        {"metric": "MAE", "basis": "pooled over all test points", "mean_pct": pooled["mae"][0], "std_pct": pooled["mae"][1]}])
    summary["n_seeds"] = len(models)
    save_csv(summary, out / "test_summary.csv")
    V.write_text_atomic(out / "test_table.tex", latex_table(res, V.FOLDERS, config))
    log(f"\nAverage over {n} files: RMSE {A.pm(*avg['rmse'], 2)}, MAE {A.pm(*avg['mae'], 2)}; "
        f"pooled RMSE {A.pm(*pooled['rmse'], 2)}")
    log("Marks: a = capacity back-calculated from the cutoff capacity; b = batch-576 capacity with aging "
        "correction (label known to be biased)")
    record = {"config": config, "exp": args.exp, "seeds": list(models), "eval_start_index": V.EVAL_START,
              "dry_run_on_validation_files": bool(args.dry_run_on_val),
              "n_test_files": n, "test_files": [s.info.id for s in seqs],
              "average_over_files": {"rmse_pct": avg["rmse"], "mae_pct": avg["mae"]},
              "pooled": {"rmse_pct": pooled["rmse"], "mae_pct": pooled["mae"]},
              "norm_stats_sha256": V.EXPECTED_NORM_SHA256, "git_commit": code["git_commit"],
              "evaluated_at": V.now_iso(), "device": "cpu"}
    if useqs is not None:
        log("\n--unseen: −20 and −10 °C (never seen in training; not used for any choice)")
        uper, ures, _ = run_set("Unseen", models, stats, w, useqs, out,
                                V.FOLDERS if args.dry_run_on_val else V.UNSEEN_FOLDERS, log, prefix="unseen_")
        ua = average_over_files(uper)
        log(f"Unseen average over {len(useqs)} files: RMSE {A.pm(*ua['rmse'], 2)}, MAE {A.pm(*ua['mae'], 2)}")
        record["unseen_average_over_files"] = {"rmse_pct": ua["rmse"], "mae_pct": ua["mae"]}
    V.write_json(out / "summary.json", record)
    V.write_json(done_marker, {"config": config, "evaluated_at": record["evaluated_at"],
                               "unseen": bool(args.unseen), "dry_run": bool(args.dry_run_on_val)})
    log(f"\nResults written to {paths.rel(out)}/")


if __name__ == "__main__":
    main()
