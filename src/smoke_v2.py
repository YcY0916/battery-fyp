"""End-to-end smoke test of the v2 pipeline (temporary experiment v2_smoke; never paper_0to40_v2).

    python src/smoke_v2.py --fresh --platform laptop

All 4 temperatures, every run limited to 2 epochs. Runs and checks, in order:
    1. train L1_W360_H32 and L2_W180_H32 (seed 42) with run_shard.py; check their files;
    2. rerun with --skip-existing: nothing is trained; without it: the script refuses;
    3. interruption: a run stopped by --deadline-h is discarded; a half-written run folder is
       retrained by --skip-existing;
    4. guards: the formal experiment rejects wrong seeds, --max-epochs and temporary stages;
       results/paper_0to40 cannot be written;
    5. seed 43 of L1_W60_H32 and L1_W360_H32 (to exercise the step-3 tables);
    6. collect_runs.py from a zip (copy, re-collect, conflict detection);
    7. merge_v2.py (step2 and step3, --allow-missing), plot_v2.py step2 (+ --highlight) and step3;
    8. benchmark_v2.py on two configs;
    9. evaluate_v2.py --dry-run-on-val (validation files stand in; no test file is opened) and
       plot_v2.py final;
   10. check_english.py on the tracked files and the smoke outputs.
Output: results/v2_smoke/ and models/v2_smoke/ (ignored by git). Prints SMOKE TEST PASSED at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import v2_common as V  # noqa: E402

SRC = Path(__file__).resolve().parent
EXP = "v2_smoke"
EPOCHS = "2"
CHILD_LOG = V.REPO_ROOT / "results" / EXP / "logs" / "smoke_children.log"
RESULTS: list[tuple[str, bool, str]] = []


def run(args: list[str], expect: int | set = 0, label: str = "") -> subprocess.CompletedProcess:
    """Run a child script; its output is decoded as UTF-8 (the children force UTF-8 output) and
    appended in full to results/v2_smoke/logs/smoke_children.log."""
    cmd = [sys.executable, *args]
    env = dict(os.environ, PYTHONUTF8="1")
    t0 = time.time()
    print(f"\n$ {' '.join(args)}", flush=True)
    p = subprocess.run(cmd, cwd=V.REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = (p.stdout or "") + (p.stderr or "")
    CHILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CHILD_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 78}\n$ {' '.join(args)}\nexit {p.returncode}\n{out}")
    tail = "\n".join(out.strip().splitlines()[-12:])
    print(tail)
    ok = p.returncode in (expect if isinstance(expect, set) else {expect})
    RESULTS.append((label or " ".join(args[:3]), ok, f"exit {p.returncode}, {time.time() - t0:.0f} s"))
    print(f"--> {'PASS' if ok else 'FAIL'}: {label} (exit {p.returncode}, expected {expect})", flush=True)
    if not ok:
        print(out)
    if "\ufffd" in out:  # a byte that is not valid UTF-8 was replaced while decoding
        RESULTS.append((f"output of '{label}' is valid UTF-8", False, ""))
        print(f"--> FAIL: output of '{label}' is not valid UTF-8 (see {CHILD_LOG.name})", flush=True)
    p.out = out
    return p


def check(cond: bool, label: str) -> None:
    RESULTS.append((label, bool(cond), ""))
    print(f"--> {'PASS' if cond else 'FAIL'}: {label}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="v2 end-to-end smoke test")
    p.add_argument("--fresh", action="store_true", help="delete the previous smoke output first")
    p.add_argument("--platform", default="laptop", choices=["laptop", "colab", "kaggle", "other"])
    p.add_argument("--cpu", action="store_true", help="allow CPU training (only if there is no GPU)")
    args = p.parse_args()
    paths = V.Paths(V.REPO_ROOT, EXP)
    if args.fresh:
        for d in (paths.results, paths.models):
            if d.exists():
                shutil.rmtree(d)
    base = ["src/run_shard.py", "--exp", EXP, "--stage", "smoke", "--platform", args.platform,
            "--max-epochs", EPOCHS] + (["--allow-cpu"] if args.cpu else [])
    t_all = time.time()
    formal_existed = (V.REPO_ROOT / "results" / V.FORMAL_EXP).exists()

    # 0. console encoding: a child process prints the allowed symbols through a pipe
    sym = "\u00b0 \u00b1 \u2212 \u00d7 \u00b5"
    q = run(["-c", "import sys; sys.path.insert(0, 'src'); import utf8_stdio; utf8_stdio.setup(); "
             f"print({sym!r})"], 0, "child prints ° ± − × µ through a pipe")
    check(q.out.strip() == sym, "the symbols arrive as UTF-8 and decode unchanged")

    # 1. two configs, seed 42
    run(base + ["--configs", "L1_W360_H32,L2_W180_H32", "--seeds", "42"], 0, "train L1_W360_H32 and L2_W180_H32")
    for r in ("L1_W360_H32_seed42", "L2_W180_H32_seed42"):
        check(V.run_status(paths, r)[0] == "complete", f"{r} complete with verified hashes")
        meta = V.read_json(paths.run_dir(r) / "run_meta.json")
        need = ["git_commit", "platform", "gpu_name", "torch_version", "cuda_version", "cudnn_version",
                "python_version", "norm_stats_sha256", "data_sha256_summary", "hyperparameters", "seed",
                "start_time", "end_time", "epochs_trained", "best_epoch", "sec_per_epoch", "train_time_s",
                "code_fingerprint"]
        check(all(k in meta for k in need), f"{r} run_meta.json has all required fields")
        met = V.read_json(paths.run_dir(r) / "metrics.json")
        check(met["eval_start_index"] == 359 and set(met["per_temp"]) == set(V.FOLDERS),
              f"{r} metrics.json: rows >= 359, overall + 4 temperatures")
        check(meta["norm_stats_sha256"] == V.EXPECTED_NORM_SHA256, f"{r} normalisation sha256 = v1 file")
    print("  L1_W360_H32 val RMSE {:.3f}%, {:.1f} s/epoch; L2_W180_H32 val RMSE {:.3f}%, {:.1f} s/epoch".format(
        *[v for r in ("L1_W360_H32_seed42", "L2_W180_H32_seed42") for v in
          (V.read_json(paths.run_dir(r) / "run_meta.json")["val_rmse_pct"],
           V.read_json(paths.run_dir(r) / "run_meta.json")["sec_per_epoch"])]))

    # 2. resume behaviour
    q = run(base + ["--configs", "L1_W360_H32,L2_W180_H32", "--seeds", "42", "--skip-existing"], 0,
            "--skip-existing trains nothing")
    check("Nothing to train." in q.out, "--skip-existing printed 'Nothing to train.'")
    run(base + ["--configs", "L1_W360_H32", "--seeds", "42"], 1, "without --skip-existing: refuses")

    # 3. interruption
    run(base + ["--configs", "L1_W60_H32", "--seeds", "42", "--test-abort-epoch", "1"], 3,
        "interruption during a run (exit 3)")
    check(V.run_status(paths, "L1_W60_H32_seed42")[0] == "absent", "unfinished run folder was discarded")
    d = paths.run_dir("L1_W60_H32_seed42")
    d.mkdir(parents=True)
    (d / "run_meta.json").write_text("{half written", encoding="utf-8")
    check(V.run_status(paths, "L1_W60_H32_seed42")[0] == "incomplete", "half-written folder seen as incomplete")
    run(base + ["--configs", "L1_W60_H32", "--seeds", "42", "--skip-existing"], 0, "incomplete run is retrained")
    check(V.run_status(paths, "L1_W60_H32_seed42")[0] == "complete", "L1_W60_H32_seed42 complete after retry")

    # 4. guards
    run(["src/run_shard.py", "--exp", V.FORMAL_EXP, "--stage", "step2", "--configs", "L1_W60_H32",
         "--seeds", "43", "--platform", args.platform, "--dry-run"], 1, "formal step2 rejects seed 43")
    run(["src/run_shard.py", "--exp", V.FORMAL_EXP, "--stage", "step2", "--configs", "L1_W60_H32",
         "--seeds", "42", "--platform", args.platform, "--max-epochs", "2"], 1, "formal rejects --max-epochs")
    run(["src/run_shard.py", "--exp", V.FORMAL_EXP, "--stage", "smoke", "--configs", "L1_W60_H32",
         "--seeds", "42", "--platform", args.platform], 1, "formal rejects the smoke stage")
    run(["src/run_shard.py", "--exp", V.FORMAL_EXP, "--stage", "step2", "--configs", "L1_W70_H32",
         "--seeds", "42", "--platform", args.platform], 1, "formal rejects a config outside the grid")
    run(["src/run_shard.py", "--exp", "paper_0to40", "--stage", "smoke", "--configs", "L1_W60_H32",
         "--seeds", "42", "--platform", args.platform], 1, "v1 name paper_0to40 refused")
    try:
        V.Paths(V.REPO_ROOT, "paper_0to40")
        check(False, "Paths refuses results/paper_0to40")
    except SystemExit:
        check(True, "Paths refuses results/paper_0to40")
    check((V.REPO_ROOT / "results" / V.FORMAL_EXP).exists() == formal_existed,
          f"results/{V.FORMAL_EXP} was not created or changed by the guard tests")
    for f in V.step2_configs():
        V.parse_config(f)
    check(len(V.step2_configs()) == 30 and len(set(V.step2_configs())) == 30, "the step-2 grid has 30 configs")
    shards = sorted((V.SHARDS_DIR).glob("step2_*.txt"))
    listed = []
    for s in shards:
        sys.path.insert(0, str(SRC))
        from run_shard import parse_shard_file
        stage, runs = parse_shard_file(s)
        listed += [V.run_name(c, x) for c, x in runs]
        check(stage == "step2", f"{s.name}: stage step2")
    check(sorted(listed) == sorted(V.run_name(c, 42) for c in V.step2_configs()),
          f"step-2 shard files cover the 30 runs exactly once ({len(listed)} lines in {len(shards)} files)")

    # 5. seed 43 of two configs (for the step-3 tables)
    run(base + ["--configs", "L1_W60_H32,L1_W360_H32", "--seeds", "43"], 0, "train seed 43 of two configs")

    # 6. collect from a zip
    with tempfile.TemporaryDirectory() as tmp:
        remote = Path(tmp) / "remote" / "battery-fyp-v2"
        rp = V.Paths(remote, EXP)
        r = "L1_W60_H32_seed43"
        shutil.copytree(paths.run_dir(r), rp.run_dir(r))
        shutil.copytree(paths.model_dir(r), rp.model_dir(r))
        shutil.rmtree(paths.run_dir(r))
        shutil.rmtree(paths.model_dir(r))
        z = Path(tmp) / "remote.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for f in (Path(tmp) / "remote").rglob("*"):
                zf.write(f, f.relative_to(Path(tmp) / "remote"))
        run(["src/collect_runs.py", "--exp", EXP, "--source", str(z)], 0, "collect copies a run from a zip")
        check(V.run_status(paths, r)[0] == "complete", "collected run complete with verified hashes")
        q = run(["src/collect_runs.py", "--exp", EXP, "--source", str(z)], 0, "collect again: already present")
        check("Already present (identical): 1" in q.out, "second collect reports the run as already present")
        log = rp.run_dir(r) / "run.log"
        log.write_text(log.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
        marker = rp.run_dir(r) / V.COMPLETE_MARKER
        rec = json.loads(marker.read_text(encoding="utf-8"))
        rec["files"]["run.log"] = V.file_sha256(log)
        marker.write_text(json.dumps(rec), encoding="utf-8")
        run(["src/collect_runs.py", "--exp", EXP, "--source", str(Path(tmp) / "remote")], 1,
            "collect detects a run trained twice (conflict)")

    # 7. merge and step-2 / step-3 figures
    cand = paths.results / "step3_candidates_smoke.txt"
    V.write_text_atomic(cand, "L1_W60_H32\nL1_W360_H32\n")
    run(["src/merge_v2.py", "--exp", EXP, "--stage", "step2"], 1, "merge without --allow-missing stops (27 missing)")
    run(["src/merge_v2.py", "--exp", EXP, "--stage", "step2", "--allow-missing"], 0, "merge step2 (partial)")
    check((paths.tables / "step2_ranking.csv").exists() and (paths.tables / "step2_summary.txt").exists(),
          "step2_ranking.csv and step2_summary.txt written")
    run(["src/plot_v2.py", "--exp", EXP, "step2"], 0, "plot step2 heatmaps")
    run(["src/plot_v2.py", "--exp", EXP, "step2", "--highlight", "L1_W360_H32,L2_W180_H32"], 0,
        "plot step2 heatmaps with --highlight")
    run(["src/merge_v2.py", "--exp", EXP, "--stage", "step3", "--allow-missing", "--candidates-file", str(cand)],
        0, "merge step3 (partial)")
    run(["src/plot_v2.py", "--exp", EXP, "step3"], 0, "plot step3 heatmap and dot plot")

    # 8. CPU inference timing
    run(["src/benchmark_v2.py", "--exp", EXP, "--configs", "L1_W60_H32,L1_W360_H32", "--yes"], 0, "CPU benchmark")
    run(["src/merge_v2.py", "--exp", EXP, "--stage", "step3", "--allow-missing", "--candidates-file", str(cand)],
        0, "merge step3 with inference times")
    check((paths.tables / "window_saturation.csv").exists(), "window_saturation.csv written (W60 vs W360, L1 H32)")

    # 9. evaluation code path on the validation files, final figures
    run(["src/evaluate_v2.py", "--exp", EXP, "--config", "L1_W360_H32", "--seeds", "42,43", "--unseen",
         "--dry-run-on-val"], 0, "evaluate (dry run on validation files, no test file opened)")
    run(["src/plot_v2.py", "--exp", EXP, "final", "--config", "L1_W360_H32", "--trace-cycle", "Mixed8"], 0,
        "plot final figures")
    figs = sorted(f.name for f in paths.figs.glob("*.png"))
    check(len(figs) == 10 and all((paths.figs / f.replace(".png", ".pdf")).exists() for f in figs),
          f"10 figures as PNG + PDF: {', '.join(figs)}")

    # 10. English check
    run(["src/check_english.py", "--exp", EXP], 0, "check_english on tracked files and smoke outputs")

    n_ok = sum(ok for _, ok, _ in RESULTS)
    print("\n" + "=" * 78)
    for label, ok, info in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {label}{'  (' + info + ')' if info else ''}")
    print("=" * 78)
    print(f"{n_ok}/{len(RESULTS)} checks passed in {(time.time() - t_all) / 60:.1f} min")
    if n_ok != len(RESULTS):
        print("SMOKE TEST FAILED")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
