"""Copy finished runs from Colab / Kaggle outputs into this checkout's results/<exp> and models/<exp>.

    python src/collect_runs.py --exp paper_0to40_v2 --source "C:/Users/chiay/Downloads/v2_kaggle_step2.zip"
    python src/collect_runs.py --exp paper_0to40_v2 --source "C:/Users/chiay/Downloads/battery-fyp-v2"

A source is a .zip file or a folder; it may contain the results/<exp>/runs/... and
models/<exp>/... trees at any depth (e.g. a Google Drive download adds one top folder).
Only complete runs (COMPLETE.json present and every hash correct) are copied.

For every run of the source:
    absent here                 -> copied (to a temporary name first, then renamed)
    complete here, same hashes  -> skipped (already collected)
    complete here, other hashes -> CONFLICT: the same run was trained twice; nothing is
                                   overwritten, the script exits with an error
    incomplete here             -> CONFLICT: a local attempt exists; nothing is overwritten
    incomplete in the source    -> reported, not copied (train it again)
Shard logs (results/<exp>/logs/*.log) are copied too. Every action is appended to
results/<exp>/collect_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_common as V  # noqa: E402


def find_roots(base: Path, exp: str) -> list[Path]:
    """Folders R such that R/results/<exp>/runs exists."""
    roots = []
    for runs in base.rglob("runs"):
        if runs.is_dir() and runs.parent.name == exp and runs.parent.parent.name == "results":
            roots.append(runs.parent.parent.parent)
    return sorted(set(roots))


def copy_tree_atomic(src: Path, dst: Path) -> None:
    V.check_not_protected(dst)
    tmp = dst.with_name(f".incoming_{dst.name}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect finished runs from Colab / Kaggle outputs")
    p.add_argument("--exp", required=True)
    p.add_argument("--source", required=True, nargs="+", type=Path, help=".zip file(s) or folder(s)")
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--dry-run", action="store_true", help="only report, copy nothing")
    args = p.parse_args()
    local = V.Paths(args.out_root, args.exp)

    copied, same, conflicts, src_incomplete = [], [], [], []
    log_rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for src in args.source:
            if not src.exists():
                raise SystemExit(f"Source not found: {src}")
            base = src
            if src.is_file():
                if src.suffix.lower() != ".zip":
                    raise SystemExit(f"{src}: expected a .zip file or a folder")
                base = Path(tmpdir) / f"src{len(list(Path(tmpdir).iterdir()))}"
                with zipfile.ZipFile(src) as z:
                    z.extractall(base)
            roots = find_roots(base, args.exp)
            if not roots:
                raise SystemExit(f"{src}: no results/{args.exp}/runs folder inside")
            for root in roots:
                remote = V.Paths(root, args.exp)
                print(f"Source {src} -> {remote.results}")
                for d in sorted(remote.runs.iterdir()):
                    if d.name.startswith((".", "_")) or not d.is_dir():
                        continue
                    if not V.RUN_RE.match(d.name):
                        conflicts.append(f"{d.name}: unexpected folder name in {src}")
                        continue
                    run = d.name
                    st, why = V.run_status(remote, run)
                    if st != "complete":
                        src_incomplete.append(f"{run} ({why}) in {src.name}")
                        continue
                    meta = V.read_json(d / "run_meta.json")
                    lst, lwhy = V.run_status(local, run)
                    action = ""
                    if lst == "absent":
                        if not args.dry_run:
                            local.runs.mkdir(parents=True, exist_ok=True)
                            local.models.mkdir(parents=True, exist_ok=True)
                            copy_tree_atomic(remote.model_dir(run), local.model_dir(run))
                            copy_tree_atomic(remote.run_dir(run), local.run_dir(run))
                            if V.run_status(local, run)[0] != "complete":
                                raise SystemExit(f"{run}: copy failed verification")
                        copied.append(run)
                        action = "copied"
                    elif lst == "complete":
                        a = V.read_json(local.run_dir(run) / V.COMPLETE_MARKER)["files"]
                        b = V.read_json(d / V.COMPLETE_MARKER)["files"]
                        if a == b:
                            same.append(run)
                            action = "already_present"
                        else:
                            lm = V.read_json(local.run_dir(run) / "run_meta.json")
                            conflicts.append(f"{run}: trained twice (here: {lm['platform']} {lm['gpu_name']}, "
                                             f"{src.name}: {meta['platform']} {meta['gpu_name']})")
                            action = "conflict"
                    else:
                        conflicts.append(f"{run}: an incomplete local attempt exists ({lwhy}); "
                                         f"{src.name} has a complete one")
                        action = "conflict"
                    log_rows.append({"time": V.now_iso(), "source": str(src), "run": run,
                                     "platform": meta["platform"], "gpu_name": meta["gpu_name"],
                                     "hostname": meta["hostname"], "action": action})
                if remote.logs.exists() and not args.dry_run:
                    for f in remote.logs.glob("*.log"):
                        dst = local.logs / f.name
                        if not dst.exists():
                            local.logs.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, dst)

    if log_rows and not args.dry_run:
        path = local.results / "collect_log.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0]), lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(log_rows)
    print(f"\nCopied {len(copied)} runs{' (dry run, nothing written)' if args.dry_run else ''}: "
          + (", ".join(copied) or "-"))
    print(f"Already present (identical): {len(same)}" + (f": {', '.join(same)}" if same else ""))
    if src_incomplete:
        print(f"Incomplete in the source, NOT copied ({len(src_incomplete)}): " + ", ".join(src_incomplete))
    if conflicts:
        print(f"CONFLICTS ({len(conflicts)}), nothing overwritten:\n  " + "\n  ".join(conflicts))
        print("Stop and ask before doing anything else with these runs.")
        raise SystemExit(1)
    print("No conflicts.")


if __name__ == "__main__":
    main()
