"""Data handling for the LSTM models: reading, filtering, splitting, resampling, normalisation, window Dataset.

Data definitions: data/processed/kollmeyer_clean/DATA_REPORT.md and manifest.json.

- Inputs: voltage, current, temperature; output: soc at the last time step of the window.
  The ah column is never read (D8: ah is the source of the SOC label and must not be an input).
- Only drive cycles (manifest test_type == "drive") and Mixed (test_type == "mixed") are used,
  selected by the manifest test type, never guessed from file names.
- Split by whole files:
  validation = the highest-numbered Mixed file of each temperature; training = the other Mixed
  files (including _part1/_part2); test = all UDDS / HWFET / LA92 / US06.
- Every sequence (every part) is resampled to 1 s by floor(time_s):
  voltage, current and temperature are averaged within the second, soc is the last row of the second.
- The normalisation mean/std come from the training set only (after resampling).
- Windows are sliced from the original arrays by index when sampled, never expanded in advance.

Running this file prints the file lists and window counts of the three sets (it reads the test
files, only to check the split):
    python src/lstm_data.py
    python src/lstm_data.py --paper     # 0-40 degC paper setting (windows 60-300, start 299)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "kollmeyer_clean"
MANIFEST_PATH = DATA_DIR / "manifest.json"

FEATURES = ["voltage", "current", "temperature"]
TARGET = "soc"
FORBIDDEN_INPUTS = {"ah"}
assert not FORBIDDEN_INPUTS & set(FEATURES), "ah must not be an input"

# The 6 temperature folders, from cold to hot
TEMP_FOLDERS = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
# Nominal temperature (degC) -> folder name, used by the --temps command-line options
TEMP_C_TO_FOLDER = {-20: "n20degC", -10: "n10degC", 0: "0degC", 10: "10degC", 25: "25degC", 40: "40degC"}
DRIVE_CYCLES = ["UDDS", "HWFET", "LA92", "US06"]

# Checked one by one against the task description: the highest-numbered Mixed file of each temperature
EXPECTED_VAL_IDS = {
    "n20degC": "n20degC/611_Mixed8",
    "n10degC": "n10degC/604_Mixed8",
    "0degC": "0degC/590_Mixed8",
    "10degC": "10degC/571_Mixed8",
    "25degC": "25degC/552_Mixed8",
    "40degC": "40degC/557_Mixed3",
}
# Drive cycles missing from the dataset (D29: 551_HWFET is excluded)
MISSING_DRIVE = {("25degC", "HWFET")}

# The validation error (and the validation loss used for early stopping) only uses points with
# in-sequence index >= the evaluation start, the first point the longest window W_max can cover
# (W_max - 1), so that every config is scored on the same points.
# The original 18 configs (W <= 120) used 119; after adding W = 180, 240 it became 239 (the default
# here; sweep_lstm.py accepts --eval-start). The 0-40 degC paper pipeline uses 299 (paper_common.EVAL_START).
EVAL_START_INDEX = 239
EVAL_START_INDEX_ORIGINAL = 119  # evaluation start used for early stopping by the original 18 configs


def set_data_root(data_root: Path | str) -> None:
    """Read data from <data_root>/data/processed/kollmeyer_clean (default: the repository copy).

    Must be called before block_test_files() / build_splits().
    """
    global DATA_DIR, MANIFEST_PATH
    DATA_DIR = Path(data_root).resolve() / "data" / "processed" / "kollmeyer_clean"
    MANIFEST_PATH = DATA_DIR / "manifest.json"
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest.json not found: {MANIFEST_PATH} (check --data_root)")


def folders_for_temps(temps: list[int]) -> list[str]:
    """Nominal temperatures in degC (e.g. [0, 10, 25, 40]) -> folder names, from cold to hot."""
    unknown = sorted(set(temps) - set(TEMP_C_TO_FOLDER))
    if unknown:
        raise ValueError(f"Unknown temperatures {unknown}; available: {sorted(TEMP_C_TO_FOLDER)}")
    wanted = {TEMP_C_TO_FOLDER[t] for t in temps}
    return [f for f in TEMP_FOLDERS if f in wanted]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


class TestSetAccessError(RuntimeError):
    """A test file was read during the hyperparameter comparison."""


# ---------------------------------------------------------------------------
# Test-set guard
# ---------------------------------------------------------------------------
_TEST_GUARD_ON = False
_TEST_PATHS: set[Path] = set()


def block_test_files() -> None:
    """Switch on the guard: from now on, reading any test file (drive cycle) raises an error.

    The list of test files comes straight from the manifest (every temperature, every sequence with
    test_type == "drive", including those in _excluded), independent of the caller's split.
    """
    global _TEST_GUARD_ON
    manifest = load_manifest()
    _TEST_PATHS.clear()
    for e in manifest["series"]:
        if e["test_type"] == "drive":
            _TEST_PATHS.add(_series_path(e["file"]).resolve())
    _TEST_GUARD_ON = True


def _check_not_test(path: Path) -> None:
    if _TEST_GUARD_ON and path.resolve() in _TEST_PATHS:
        raise TestSetAccessError(
            f"Reading test files is not allowed during the comparison: {_rel(path)}. "
            "The test set may only be read by src/evaluate_paper.py after the config has been chosen."
        )


# ---------------------------------------------------------------------------
# Manifest and split
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeqInfo:
    id: str             # e.g. "10degC/571_Mixed6_part2"
    file: str           # path relative to DATA_DIR (/-separated, as in the manifest)
    folder: str         # temperature folder, e.g. "n20degC"
    temp_c: int         # nominal temperature
    test_type: str      # "mixed" or "drive"
    test_name: str      # "Mixed6", "UDDS", ...
    part: int
    n_parts: int
    split: str          # "train" / "val" / "test"

    @property
    def path(self) -> Path:
        return _series_path(self.file)


def _series_path(file: str) -> Path:
    # manifest paths are /-separated; excluded sequences already carry the "_excluded/" prefix
    return DATA_DIR / Path(*file.split("/"))


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _mixed_number(test_name: str) -> int:
    m = re.fullmatch(r"Mixed(\d+)", test_name)
    if m is None:
        raise ValueError(f"Unrecognised Mixed name: {test_name}")
    return int(m.group(1))


def build_splits(folders: list[str] | None = None) -> dict[str, list[SeqInfo]]:
    """Filter by the manifest test type and split. folders=None means all 6 temperatures.

    Only reads the manifest, never a data file. Raises an error on any mismatch with the task description.
    """
    folders = list(TEMP_FOLDERS) if folders is None else list(folders)
    unknown = set(folders) - set(TEMP_FOLDERS)
    if unknown:
        raise ValueError(f"Unknown temperature folders: {sorted(unknown)}")

    manifest = load_manifest()
    entries = [
        e for e in manifest["series"]
        if not e["excluded"] and e["folder"] in folders
        and e["test_type"] in ("mixed", "drive")
    ]

    # The highest-numbered Mixed of each temperature is the validation file
    val_ids: dict[str, str] = {}
    for folder in folders:
        mixed = [e for e in entries if e["folder"] == folder and e["test_type"] == "mixed"]
        if not mixed:
            raise ValueError(f"{folder} has no Mixed file")
        top = max(_mixed_number(e["test_name"]) for e in mixed)
        chosen = [e for e in mixed if _mixed_number(e["test_name"]) == top]
        if len(chosen) != 1:
            raise ValueError(f"{folder}: the highest-numbered Mixed{top} has {len(chosen)} parts; decide by hand")
        if chosen[0]["id"] != EXPECTED_VAL_IDS[folder]:
            raise ValueError(
                f"{folder}: the validation file is {chosen[0]['id']}, but the task description says "
                f"{EXPECTED_VAL_IDS[folder]}"
            )
        val_ids[folder] = chosen[0]["id"]

    splits: dict[str, list[SeqInfo]] = {"train": [], "val": [], "test": []}
    for e in entries:
        if e["test_type"] == "drive":
            if e["test_name"] not in DRIVE_CYCLES:
                raise ValueError(f"Unknown drive cycle: {e['id']} ({e['test_name']})")
            split = "test"
        elif e["id"] == val_ids[e["folder"]]:
            split = "val"
        else:
            split = "train"
        info = SeqInfo(
            id=e["id"], file=e["file"], folder=e["folder"],
            temp_c=int(e["nominal_temperature_C"]), test_type=e["test_type"],
            test_name=e["test_name"], part=int(e["part"]), n_parts=int(e["n_parts"]),
            split=split,
        )
        if not info.path.exists():
            raise FileNotFoundError(f"File listed in the manifest does not exist: {info.path}")
        splits[split].append(info)

    # The test set must hold each of the 4 cycles once per temperature (25 degC has no HWFET)
    for folder in folders:
        got = sorted(s.test_name for s in splits["test"] if s.folder == folder)
        want = sorted(c for c in DRIVE_CYCLES if (folder, c) not in MISSING_DRIVE)
        if got != want:
            raise ValueError(f"{folder}: test files {got} differ from the expected {want}")

    # The three sets must not overlap (by sequence and by original Mixed / drive-cycle file)
    ids = {k: {s.id for s in v} for k, v in splits.items()}
    raw = {k: {(s.folder, s.id.split("/")[1].split("_part")[0]) for s in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b] or raw[a] & raw[b]:
            raise ValueError(f"{a} and {b} overlap: {sorted(ids[a] & ids[b]) or sorted(raw[a] & raw[b])}")

    order = {f: i for i, f in enumerate(TEMP_FOLDERS)}
    for k in splits:
        splits[k].sort(key=lambda s: (order[s.folder], s.id))
    return splits


# ---------------------------------------------------------------------------
# Reading and resampling
# ---------------------------------------------------------------------------
@dataclass
class Sequence:
    info: SeqInfo
    sec: np.ndarray  # (n,) int64, floor(time_s)
    x: np.ndarray    # (n, 3) float64, unnormalised voltage/current/temperature
    y: np.ndarray    # (n,) float64, soc (0-1)

    def __len__(self) -> int:
        return len(self.y)


def read_raw(info: SeqInfo) -> pd.DataFrame:
    """Read the raw CSV of one sequence (without the ah column). All data reading goes through here."""
    path = info.path
    _check_not_test(path)
    if info.split == "test" and _TEST_GUARD_ON:
        raise TestSetAccessError(f"Reading test files is not allowed during the comparison: {info.id}")
    cols = ["time_s", *FEATURES, TARGET]
    df = pd.read_csv(path, usecols=cols)
    assert not FORBIDDEN_INPUTS & set(df.columns)
    if not np.all(np.diff(df["time_s"].to_numpy()) > 0):
        raise ValueError(f"{info.id}: time_s is not strictly increasing")
    return df


def resample_1s(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group by floor(time_s): mean voltage / current / temperature, soc of the last row of each second."""
    sec = np.floor(df["time_s"].to_numpy()).astype(np.int64)
    g = df.groupby(sec, sort=True)
    x = g[FEATURES].mean().to_numpy(dtype=np.float64)
    y = g[TARGET].last().to_numpy(dtype=np.float64)
    return g.size().index.to_numpy(dtype=np.int64), x, y


def load_sequences(infos: list[SeqInfo], log=print) -> list[Sequence]:
    seqs = []
    for info in infos:
        sec, x, y = resample_1s(read_raw(info))
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError(f"{info.id}: NaN after resampling")
        seqs.append(Sequence(info, sec, x, y))
    log(f"  read {len(seqs)} sequences, {sum(len(s) for s in seqs):,} rows after resampling")
    return seqs


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def compute_norm_stats(train_seqs: list[Sequence]) -> dict:
    if any(s.info.split != "train" for s in train_seqs):
        raise ValueError("Normalisation stats may only be computed from the training set")
    x = np.concatenate([s.x for s in train_seqs], axis=0)
    return {
        "features": FEATURES,
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "n_rows": int(len(x)),
        "computed_from": [s.info.id for s in train_seqs],
        "note": "Computed from the training set only (all rows after 1 s resampling); validation and test sets use these values unchanged",
    }


def save_norm_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_norm_stats(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    if list(stats["features"]) != FEATURES:
        raise ValueError(f"Normalisation features {stats['features']} differ from {FEATURES}")
    return (x - np.asarray(stats["mean"])) / np.asarray(stats["std"])


# ---------------------------------------------------------------------------
# Window Dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Sliding windows (stride 1), sliced by index when sampled; windows are never expanded in advance.

    Windows are cut within each sequence, never across sequences. Sample i is the window of length W
    whose last row is ends[i]; its label is the soc of row ends[i].

    min_end_index: keep only window ends with in-sequence index >= min_end_index.
        Training set: W-1 (every point with a complete window);
        validation / test sets: the evaluation start (EVAL_START_INDEX = 239 in the old sweep, 299 in
        the paper pipeline), so that every window length is scored on the same points.
    """

    def __init__(self, seqs: list[Sequence], stats: dict, window: int,
                 min_end_index: int | None = None, log=print):
        self.window = int(window)
        first_end = self.window - 1 if min_end_index is None else max(self.window - 1, int(min_end_index))
        xs, ys, ends, seq_of, local_idx = [], [], [], [], []
        self.seqs: list[Sequence] = []
        self.skipped: list[tuple[str, int]] = []
        offset = 0
        for s in seqs:
            n = len(s)
            if n < self.window:
                self.skipped.append((s.info.id, n))
                continue
            k = len(self.seqs)
            self.seqs.append(s)
            xs.append(normalize(s.x, stats).astype(np.float32))
            ys.append(s.y.astype(np.float32))
            loc = np.arange(first_end, n, dtype=np.int64)
            ends.append(offset + loc)
            local_idx.append(loc)
            seq_of.append(np.full(len(loc), k, dtype=np.int64))
            offset += n
        if self.skipped:
            log(f"  window length {self.window}: sequences shorter than the window were skipped: "
                + ", ".join(f"{sid} ({n} rows)" for sid, n in self.skipped))
        if not self.seqs:
            raise ValueError("No sequence is longer than the window")
        # Original (row-wise) arrays; windows are sliced from them when sampled
        self.x = torch.from_numpy(np.concatenate(xs))
        self.y = torch.from_numpy(np.concatenate(ys))
        self.ends = torch.from_numpy(np.concatenate(ends))
        self.seq_of = np.concatenate(seq_of)        # index of the (kept) sequence of each sample
        self.local_index = np.concatenate(local_idx)  # in-sequence row index of each sample
        self._rel = torch.arange(-self.window + 1, 1, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        e = int(self.ends[i])
        return self.x[e - self.window + 1:e + 1], self.y[e]

    def get_batch(self, idx: torch.Tensor):
        """Fetch a batch of windows (vectorised slicing); only this batch is materialised."""
        idx = idx.to(self.ends.device)
        e = self.ends[idx]
        rows = e[:, None] + self._rel[None, :]
        return self.x[rows], self.y[e]

    def to(self, device) -> "WindowDataset":
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.ends = self.ends.to(device)
        self._rel = self._rel.to(device)
        return self

    def iter_batches(self, batch_size: int, shuffle: bool = False,
                     generator: torch.Generator | None = None):
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            yield self.get_batch(order[start:start + batch_size])


# ---------------------------------------------------------------------------
# Run directly: print the split
# ---------------------------------------------------------------------------
def _print_split_summary(folders, windows, eval_start):
    splits = build_splits(folders)
    seqs = {k: load_sequences(v, log=lambda *_: None) for k, v in splits.items()}
    stats = compute_norm_stats(seqs["train"])
    for split in ("train", "val", "test"):
        print(f"\n===== {split}: {len(splits[split])} sequences =====")
        header = f"{'sequence':32s}{'temp':>6s}{'rows(1s)':>10s}" + "".join(
            f"{'W=' + str(w):>10s}" for w in windows)
        print(header)
        for s in seqs[split]:
            if split == "train":
                counts = [max(0, len(s) - w + 1) for w in windows]
            else:
                counts = [max(0, len(s) - max(w - 1, eval_start)) if len(s) >= w else 0
                          for w in windows]
            print(f"{s.info.id:32s}{s.info.temp_c:>6d}{len(s):>10,d}"
                  + "".join(f"{c:>10,d}" for c in counts))
        totals = []
        for w in windows:
            min_end = None if split == "train" else eval_start
            ds = WindowDataset(seqs[split], stats, w, min_end, log=lambda *_: None)
            totals.append(len(ds))
            if ds.skipped:
                print(f"  W={w} skipped (shorter than the window): {ds.skipped}")
        print(f"{'total':32s}{'':>6s}{sum(len(s) for s in seqs[split]):>10,d}"
              + "".join(f"{t:>10,d}" for t in totals))
        print(f"  windows per temperature ({split}):")
        for folder in folders or TEMP_FOLDERS:
            part = [s for s in seqs[split] if s.info.folder == folder]
            if not part:
                continue
            counts = []
            for w in windows:
                min_end = None if split == "train" else eval_start
                ds = WindowDataset(part, stats, w, min_end, log=lambda *_: None)
                counts.append(len(ds))
            print(f"  {folder:30s}{len(part):>6d}{sum(len(s) for s in part):>10,d}"
                  + "".join(f"{c:>10,d}" for c in counts))
    print("\nNote: training windows end at row W-1 of each sequence onwards; validation / test windows end at row "
          f"{eval_start} onwards (the same points for every W <= {eval_start + 1}).")
    ids = [{s.id for s in v} for v in splits.values()]
    print("No overlap between the three sets (by sequence):", not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]))
    raw = [{(s.folder, s.id.split("/")[1].split("_part")[0]) for s in v} for v in splits.values()]
    print("No overlap between the original files (parts follow their file):",
          not (raw[0] & raw[1] or raw[0] & raw[2] or raw[1] & raw[2]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print the file lists and window counts of the train / validation / test sets")
    p.add_argument("--folders", nargs="+", default=None, help="only these temperature folders")
    p.add_argument("--temps", nargs="+", type=int, default=None,
                   help="nominal temperatures in degC (e.g. 0 10 25 40); overrides --folders")
    p.add_argument("--paper", action="store_true",
                   help="paper setting: temps 0 10 25 40, windows 60-300, evaluation start 299")
    p.add_argument("--smoke", action="store_true", help="only 25 degC (quick check)")
    p.add_argument("--data_root", type=Path, default=None,
                   help="folder that contains data/processed/kollmeyer_clean (default: repository root)")
    p.add_argument("--windows", nargs="+", type=int, default=[30, 60, 120, 180, 240])
    p.add_argument("--eval-start", type=int, default=EVAL_START_INDEX,
                   help=f"evaluation start (row index) of the validation / test sets, default {EVAL_START_INDEX}")
    a = p.parse_args()
    if a.data_root is not None:
        set_data_root(a.data_root)
    if a.paper:
        import paper_common as PC  # local import: paper_common imports this module
        a.temps = a.temps or list(PC.DEFAULT_TEMPS)
        a.windows, a.eval_start = sorted({c[0] for c in PC.SWEEP_CONFIGS}), PC.EVAL_START
    if a.smoke:
        a.temps = [25]
    if a.temps is not None:
        a.folders = folders_for_temps(a.temps)
    _print_split_summary(a.folders, a.windows, a.eval_start)
