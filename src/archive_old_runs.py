"""Move the outputs of the old 6-temperature experiment into archive/2026-09_6temp/.

Results and model files live on the user's machine (results/ and models/ are not
committed), so this script only reorganises local files:

- default is a dry run: it prints what would be moved and changes nothing;
- --apply performs the move, keeping the original directory structure
  (results/lstm_sweep/... -> archive/2026-09_6temp/results/lstm_sweep/...);
- files are only moved, never deleted or overwritten (an existing destination file
  is reported as a conflict and left in place); directories that become empty
  after the move are removed;
- nothing under data/ is ever touched, and files tracked by git are left in place
  (e.g. results/lstm_sweep/README.md);
- after --apply, archive/2026-09_6temp/ARCHIVE_README.md describes the archived
  experiment and ARCHIVE_MANIFEST.csv lists every moved file.

Usage (from the repository root):
    python src/archive_old_runs.py              # dry run
    python src/archive_old_runs.py --apply      # really move
    python src/archive_old_runs.py --smoke      # self-test on a fake tree (smoke folder)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SUBDIR = Path("archive") / "2026-09_6temp"
SMOKE_DIR = Path("results") / "paper_0to40_smoke" / "archive_test"

# (glob pattern relative to --out_root, where the pattern comes from)
TARGETS: list[tuple[str, str]] = [
    # master: src/sweep_lstm.py, src/archive/combine_lstm_sweep.py, src/archive/evaluate_lstm.py
    ("results/lstm_sweep*", "master (sweep_lstm / combine_lstm_sweep / evaluate_lstm)"),
    ("models/lstm_sweep*", "master (sweep_lstm)"),
    ("results/timing_*", "master (timing runs, see .gitignore)"),
    ("models/timing_*", "master (timing runs, see .gitignore)"),
    # listed by the user; not produced by any code in this repository
    ("results/paper_figs", "user list"),
    ("results/figures/multitemp", "user list"),
    ("models/pretrained_soc_*.pt", "user list"),
    # feature/gru-cnn-transformer branch (also the old 6-temperature split)
    ("results/tables", "feature/gru-cnn-transformer"),
    ("results/search", "feature/gru-cnn-transformer"),
    ("results/final_eval", "feature/gru-cnn-transformer"),
    ("results/lstm_verify", "feature/gru-cnn-transformer"),
    ("results/models_*_smoke", "feature/gru-cnn-transformer (smoke_test_models)"),
    ("models/final", "feature/gru-cnn-transformer"),
    ("models/lstm_verify", "feature/gru-cnn-transformer"),
    ("models/lstm", "feature/gru-cnn-transformer"),
    ("models/lstm_lr*", "feature/gru-cnn-transformer"),
    ("models/gru", "feature/gru-cnn-transformer"),
    ("models/gru_lr*", "feature/gru-cnn-transformer"),
    ("models/cnn", "feature/gru-cnn-transformer"),
    ("models/cnn_lr*", "feature/gru-cnn-transformer"),
    ("models/transformer", "feature/gru-cnn-transformer"),
    ("models/transformer_lr*", "feature/gru-cnn-transformer"),
]

README_TEMPLATE = """# Archive: 6-temperature LSTM experiment (2026-09)

This folder was created by `src/archive_old_runs.py --apply` (last run: {now}). It holds the outputs of the old experiment from before the paper was restricted to 0-40 °C.
The files were only moved here from their original locations; their content was not changed and no file was deleted. Nothing under `data/` was touched.

## Which experiment these results belong to

- Temperatures: 6 (−20, −10, 0, 10, 25, 40 °C).
- Split (old split, by whole files):
  - validation: the highest-numbered Mixed of each temperature, 6 files (611_Mixed8, 604_Mixed8, 590_Mixed8, 571_Mixed8, 552_Mixed8, 557_Mixed3);
  - training: the other Mixed files, 37 sequences (including the _part1/_part2 of 602_Mixed4 and 571_Mixed6);
  - test: every UDDS, HWFET, LA92 and US06, 23 files (25 °C has no HWFET).
- Hyperparameter comparison: 22 configs.
  - first 18: W in {{30, 60, 120}} × H in {{32, 64, 128}} × L in {{1, 2}}, early-stopping evaluation from row 119;
  - extension, 4 configs: W in {{180, 240}} × H in {{32, 128}} × L = 1, evaluation from row 239;
  - the merged table (`results/lstm_sweep_ext/sweep_results_22.csv`) recomputes every validation error from row 239.
- Best config: **W240_H32_L1**.
- Code: git tag `archive-6temp-2026-09` (master commit be7b363).
- Outputs of the feature/gru-cnn-transformer branch (GRU / 1D-CNN / Transformer comparison, same old 6-temperature split), if present, are archived here as well.

## Moves

{table}

{kept}The complete per-file list is in `ARCHIVE_MANIFEST.csv` (appended by every --apply).
"""


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def git_tracked(root: Path, rel: Path) -> set[Path]:
    """Files under root/rel that are tracked by git (empty set if git is unavailable)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", rel.as_posix()],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {(root / p).resolve() for p in out.decode("utf-8").split("\0") if p}


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect(root: Path, archive_root: Path):
    """Return (plans, missing). Each plan describes one matched target."""
    forbidden = [root / "data", archive_root.parent]
    plans, missing, seen = [], [], set()
    for pattern, origin in TARGETS:
        matches = sorted(root.glob(pattern))
        if not matches:
            missing.append((pattern, origin))
            continue
        for src in matches:
            key = src.resolve()
            if key in seen:
                continue
            seen.add(key)
            if not is_inside(src, root):
                raise SystemExit(f"Refusing to move {src}: it is outside {root}")
            for bad in forbidden:
                if is_inside(src, bad) or is_inside(bad, src):
                    raise SystemExit(f"Refusing to move {src}: it overlaps {bad}")
            rel = src.relative_to(root)
            files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
            tracked = git_tracked(root, rel)
            move = [f for f in files if f.resolve() not in tracked]
            keep = [f for f in files if f.resolve() in tracked]
            conflicts = [f for f in move if (archive_root / f.relative_to(root)).exists()]
            plans.append({"src": src, "rel": rel, "origin": origin, "is_dir": src.is_dir(),
                          "move": [f for f in move if f not in conflicts], "keep": keep,
                          "conflicts": conflicts,
                          "bytes": sum(f.stat().st_size for f in move if f not in conflicts)})
    return plans, missing


def print_plan(root: Path, archive_root: Path, plans, missing, apply: bool) -> None:
    print(f"Root:    {root}")
    print(f"Archive: {archive_root}")
    print("Mode:    " + ("APPLY (files will be moved)" if apply else "DRY RUN (nothing is changed)"))
    print()
    if not plans:
        print("Nothing to move.")
    for p in plans:
        slash = "/" if p["is_dir"] else ""
        dest = (archive_root / p["rel"]).relative_to(root).as_posix()
        print(f"MOVE  {p['rel'].as_posix()}{slash}  ->  {dest}{slash}   "
              f"({len(p['move'])} files, {human_size(p['bytes'])})   [{p['origin']}]")
        for f in p["keep"]:
            print(f"      keep in place (tracked by git): {f.relative_to(root).as_posix()}")
        for f in p["conflicts"]:
            print(f"      CONFLICT, destination exists, left in place: {f.relative_to(root).as_posix()}")
    if missing:
        print("\nNot found (skipped):")
        for pattern, origin in missing:
            print(f"      {pattern}   [{origin}]")
    total = sum(len(p["move"]) for p in plans)
    print(f"\nTotal: {total} files, {human_size(sum(p['bytes'] for p in plans))}")


def remove_empty_dirs(top: Path) -> None:
    """Remove directories under (and including) top that are empty after the move."""
    if not top.is_dir():
        return
    for d in sorted((p for p in top.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        top.rmdir()
    except OSError:
        pass


def apply_plan(root: Path, archive_root: Path, plans) -> list[dict]:
    moved = []
    now = datetime.now().isoformat(timespec="seconds")
    for p in plans:
        for f in p["move"]:
            dest = archive_root / f.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():  # appeared after planning: never overwrite
                print(f"CONFLICT, left in place: {f.relative_to(root).as_posix()}")
                continue
            size = f.stat().st_size
            shutil.move(str(f), str(dest))
            moved.append({"moved_at": now, "target": p["rel"].as_posix(),
                          "source": f.relative_to(root).as_posix(),
                          "destination": dest.relative_to(root).as_posix(), "bytes": size,
                          "origin": p["origin"]})
        if p["is_dir"]:
            remove_empty_dirs(p["src"])
    return moved


def write_records(root: Path, archive_root: Path, plans, moved: list[dict]) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest = archive_root / "ARCHIVE_MANIFEST.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["moved_at", "target", "source", "destination", "bytes",
                                          "origin"])
        if new:
            w.writeheader()
        w.writerows(moved)

    # README lists every target ever archived (read back from the manifest)
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: dict[str, list] = {}
    for r in rows:
        g = groups.setdefault(r["target"], [0, 0, r["origin"], r["moved_at"]])
        g[0] += 1
        g[1] += int(r["bytes"])
    lines = ["| Original path | Source | Files | Size | Moved at |", "|---|---|---|---|---|"]
    for top, (n, size, origin, when) in sorted(groups.items()):
        lines.append(f"| `{top}` | {origin} | {n} | {human_size(size)} | {when} |")
    kept = [f.relative_to(root).as_posix() for p in plans for f in p["keep"]]
    kept_txt = ("These files are tracked by git and were left in place: " + ", ".join(f"`{k}`" for k in kept) + ".\n\n"
                if kept else "")
    (archive_root / "ARCHIVE_README.md").write_text(
        README_TEMPLATE.format(now=datetime.now().strftime("%Y-%m-%d %H:%M"),
                               table="\n".join(lines), kept=kept_txt),
        encoding="utf-8", newline="\n")


def run(root: Path, apply: bool) -> int:
    archive_root = root / ARCHIVE_SUBDIR
    plans, missing = collect(root, archive_root)
    print_plan(root, archive_root, plans, missing, apply)
    if not apply:
        print("\nDry run only. Re-run with --apply to move these files.")
        return 0
    moved = apply_plan(root, archive_root, plans)
    if moved or plans:
        write_records(root, archive_root, plans, moved)
        print(f"\nMoved {len(moved)} files. Wrote {(archive_root / 'ARCHIVE_README.md').relative_to(root)} "
              f"and ARCHIVE_MANIFEST.csv")
    return len(moved)


def smoke() -> None:
    """Build a fake tree in the smoke folder, archive it, and check the result."""
    fake = REPO_ROOT / SMOKE_DIR
    if fake.exists():
        shutil.rmtree(fake)
    files = ["results/lstm_sweep/runs/W30_H32_L1.json", "results/lstm_sweep/curves/W30_H32_L1.png",
             "results/lstm_sweep_ext/sweep_results_22.csv", "results/lstm_sweep_smoke/x.json",
             "results/timing_w240/log.txt", "results/paper_figs/fig1.png",
             "results/figures/multitemp/a.png", "results/tables/sweep_gru.csv",
             "results/search/gru/sweep_log.txt", "results/final_eval/final_log.txt",
             "results/lstm_verify/v.json", "models/lstm_sweep/W30_H32_L1.pt",
             "models/lstm_sweep_ext/W240_H32_L1.pt", "models/pretrained_soc_25degC.pt",
             "models/gru/W60_H64_L1.pt", "models/transformer_lr0.0005/a.pt", "models/final/b.pt"]
    untouched = ["data/processed/kollmeyer_clean/25degC/552_Mixed8.csv",
                 "results/paper_0to40/runs/W60_H32_L1_seed42.json", "models/paper_0to40/x.pt",
                 "models/lstm_other/keep.pt"]
    for rel in files + untouched:
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(rel, encoding="utf-8")

    print("---- smoke: dry run ----")
    assert run(fake, apply=False) == 0
    assert all((fake / rel).exists() for rel in files), "dry run must not move anything"
    print("\n---- smoke: apply ----")
    n = run(fake, apply=True)
    assert n == len(files), f"expected {len(files)} moved files, got {n}"
    for rel in files:
        assert not (fake / rel).exists() and (fake / ARCHIVE_SUBDIR / rel).read_text(encoding="utf-8") == rel
    for rel in untouched:
        assert (fake / rel).read_text(encoding="utf-8") == rel, f"{rel} must not be touched"
    assert (fake / ARCHIVE_SUBDIR / "ARCHIVE_README.md").exists()
    print("\n---- smoke: apply again (nothing left to move) ----")
    assert run(fake, apply=True) == 0
    print(f"\nSMOKE OK: {len(files)} files archived, data/ and new outputs untouched ({fake})")


def main() -> None:
    p = argparse.ArgumentParser(description="Archive the old 6-temperature results and models "
                                            "(dry run unless --apply)")
    p.add_argument("--apply", action="store_true", help="really move the files")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT,
                   help="folder that contains results/ and models/ (default: repository root)")
    p.add_argument("--smoke", action="store_true", help="self-test on a fake tree in the smoke folder")
    args = p.parse_args()
    if args.smoke:
        smoke()
        return
    run(args.out_root.resolve(), args.apply)


if __name__ == "__main__":
    main()
