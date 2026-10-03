"""Print the seconds per epoch measured by the timing probe (and, optionally, other experiments).

    python src/probe_summary.py --exp timing_v2
    python src/probe_summary.py --exp timing_v2 --source "C:/Users/chiay/Downloads/v2_kaggle_probe.zip"

Reads every complete run of the experiment (in this checkout and in the given .zip files or
folders, which are only read) and prints one line per run: platform, GPU, note, config,
epochs, the median time of epochs 2 onwards (epoch 1 includes CUDA start-up) and the mean.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_common as V  # noqa: E402
from collect_runs import find_roots  # noqa: E402


def rows_for(paths: V.Paths, origin: str) -> list[dict]:
    out = []
    if not paths.runs.exists():
        return out
    for d in sorted(paths.runs.iterdir()):
        if not d.is_dir() or not V.RUN_RE.match(d.name) or V.run_status(paths, d.name)[0] != "complete":
            continue
        m = V.read_json(d / "run_meta.json")
        h = pd.read_csv(d / "train_log.csv")
        later = h[h["epoch"] >= 2]["epoch_time_s"]
        out.append({"platform": m["platform"], "gpu": m["gpu_name"], "visible": m["cuda_visible_devices"],
                    "note": m["note"], "run": d.name, "epochs": len(h),
                    "s_epoch1": float(h["epoch_time_s"].iloc[0]),
                    "s_epoch_median_2plus": float(later.median()) if len(later) else float("nan"),
                    "torch": m["torch_version"], "origin": origin})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Seconds per epoch of the timing probe")
    p.add_argument("--exp", default="timing_v2")
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--source", nargs="*", type=Path, default=[], help=".zip files or folders (read only)")
    args = p.parse_args()
    rows = rows_for(V.Paths(args.out_root, args.exp), "this checkout")
    with tempfile.TemporaryDirectory() as tmp:
        for k, src in enumerate(args.source):
            base = src
            if src.is_file():
                base = Path(tmp) / f"s{k}"
                with zipfile.ZipFile(src) as z:
                    z.extractall(base)
            for root in find_roots(base, args.exp):
                rows += rows_for(V.Paths(root, args.exp), src.name)
    if not rows:
        raise SystemExit(f"No complete runs of {args.exp} found")
    df = pd.DataFrame(rows).sort_values(["platform", "gpu", "visible", "note", "run"])
    print("TIMING PROBE (copy everything between the two lines of '=')")
    print("=" * 100)
    print(df.drop(columns=["origin"]).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("=" * 100)


if __name__ == "__main__":
    main()
