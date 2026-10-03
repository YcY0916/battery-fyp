"""Print the training times of the v1 runs (read only; used only to estimate the v2 run time).

    python src/print_v1_timing.py

Reads results/paper_0to40/sweep_results.csv and multiseed_results.csv (never writes anything) and
prints sec_per_epoch, epochs_trained and train_time_s per config, plus the device they ran on.
No v1 result is reused in v2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "paper_0to40"


def show(name: str, cols: list[str]) -> None:
    path = ROOT / name
    if not path.exists():
        print(f"{path}: not found")
        return
    df = pd.read_csv(path)
    have = [c for c in cols if c in df.columns]
    print(f"\n{name} ({len(df)} rows; columns shown: {', '.join(have)})")
    print(df[have].to_string(index=False, float_format=lambda v: f"{v:.1f}"))


pd.set_option("display.width", 200)
print("V1 TIMING (copy everything between the two lines of '=')")
print("=" * 100)
show("sweep_results.csv", ["config", "sec_per_epoch", "epochs_trained", "best_epoch", "train_time_s",
                           "train_host", "train_device"])
show("multiseed_results.csv", ["config", "n_seeds", "sec_per_epoch", "epochs_trained", "epochs_trained_mean",
                               "best_epoch_mean", "train_time_s", "train_time_s_mean", "train_time_s_std",
                               "train_host", "train_device"])
print("=" * 100)
