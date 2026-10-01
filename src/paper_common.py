"""Shared settings and helpers for the 0-40 degC paper pipeline (GECOST 2027).

Every paper script (sweep_paper.py, merge_results.py, select_config.py,
benchmark_inference.py, evaluate_paper.py, plot_paper_figs.py) takes the same
--temps / --data_root / --out_root / --name / --smoke options and resolves its
input and output locations through PaperPaths, so the same layout is used on the
Windows laptop and on Colab (where --out_root points to Google Drive).

Data reading, splitting, resampling, normalisation and windowing come from
lstm_data.py; the model from lstm_model.py; the training loop from sweep_lstm.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
from sweep_lstm import config_name, metrics_pct, predict  # noqa: E402,F401

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------
DEFAULT_TEMPS = [0, 10, 25, 40]
DEFAULT_NAME = "paper_0to40"
# All validation / test metrics (and the early-stopping validation loss) use only rows with
# in-sequence index >= 299, i.e. the first point the longest window (W = 300) can cover.
EVAL_START = 299

GRID_W_L1 = [60, 120, 180, 240, 300]
GRID_H = [32, 64, 128]
L2_WINDOW = 240
SWEEP_CONFIGS = [(w, h, 1) for w in GRID_W_L1 for h in GRID_H] + [(L2_WINDOW, h, 2) for h in GRID_H]
SWEEP_SEED = 42
MULTISEED_SEEDS = [42, 43, 44, 45, 46]
TOP_K = 4  # multi-seed candidates = top TOP_K by validation RMSE + all L=1 H at the best W

# --smoke: 25 degC only, 2 configs, 2 seeds, 2 epochs. The two configs land in different
# shards for --shard k/2 (checked in sweep_paper.py), so the shard logic is exercised too.
SMOKE_TEMPS = [25]
SMOKE_CONFIGS = [(60, 32, 1), (300, 32, 1)]
SMOKE_SEEDS = [42, 43]
SMOKE_EPOCHS = 2

CONFIG_RE = re.compile(r"^W(\d+)_H(\d+)_L(\d+)$")
RUN_RE = re.compile(r"^(W\d+_H\d+_L\d+)_seed(\d+)$")


def parse_config(name: str) -> tuple[int, int, int]:
    m = CONFIG_RE.match(name)
    if m is None:
        raise ValueError(f"Bad config name {name!r}; expected e.g. W240_H32_L1")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_name(config: str, seed: int) -> str:
    return f"{config}_seed{seed}"


def temp_label(folder: str) -> str:
    """'n10degC' -> '-10 degC' (for tables and logs)."""
    return folder.replace("n", "-").replace("degC", " degC")


def temp_c(folder: str) -> int:
    return int(folder.replace("n", "-").replace("degC", ""))


# ---------------------------------------------------------------------------
# Paths and common command-line options
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PaperPaths:
    out_root: Path
    name: str

    @property
    def results(self) -> Path:
        return self.out_root / "results" / self.name

    @property
    def models(self) -> Path:
        return self.out_root / "models" / self.name

    runs = property(lambda self: self.results / "runs")
    val_preds = property(lambda self: self.results / "val_preds")
    sweep_curves = property(lambda self: self.results / "sweep" / "curves")
    multiseed_curves = property(lambda self: self.results / "multiseed" / "curves")
    logs = property(lambda self: self.results / "logs")
    timing = property(lambda self: self.results / "timing")
    norm_stats = property(lambda self: self.results / "norm_stats.json")
    candidates = property(lambda self: self.results / "candidates.json")
    sweep_csv = property(lambda self: self.results / "sweep_results.csv")
    multiseed_csv = property(lambda self: self.results / "multiseed_results.csv")
    all_runs_csv = property(lambda self: self.results / "all_runs.csv")
    selected = property(lambda self: self.results / "selected_config.json")
    bench_csv = property(lambda self: self.results / "inference_benchmark.csv")
    test = property(lambda self: self.results / "test")
    figs = property(lambda self: self.results / "figs")
    notes = property(lambda self: self.results / "figure_notes.md")

    def model_file(self, run: str) -> Path:
        return self.models / f"{run}.pt"

    def run_json(self, run: str) -> Path:
        return self.runs / f"{run}.json"

    def rel(self, path: Path) -> str:
        """Path relative to --out_root (for logs and JSON), with / separators."""
        try:
            return path.resolve().relative_to(self.out_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


@dataclass(frozen=True)
class Context:
    temps: list[int]
    folders: list[str]
    paths: PaperPaths
    smoke: bool

    @property
    def configs(self) -> list[tuple[int, int, int]]:
        return list(SMOKE_CONFIGS) if self.smoke else list(SWEEP_CONFIGS)

    @property
    def config_names(self) -> list[str]:
        return [config_name(*c) for c in self.configs]

    @property
    def seeds(self) -> list[int]:
        return list(SMOKE_SEEDS) if self.smoke else list(MULTISEED_SEEDS)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temps", nargs="+", type=int, default=None,
                   help=f"nominal temperatures in degC (default: {' '.join(map(str, DEFAULT_TEMPS))}); "
                        "e.g. -20 -10 0 10 25 40 reproduces the old 6-temperature split")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT,
                   help="folder that contains data/processed/kollmeyer_clean (default: repository root)")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT,
                   help="folder where results/ and models/ are written (default: repository root; "
                        "on Colab point it to Google Drive)")
    p.add_argument("--name", default=None,
                   help=f"experiment folder name (default: {DEFAULT_NAME} for 0 10 25 40, "
                        "derived from --temps otherwise; '_smoke' is appended with --smoke)")
    p.add_argument("--smoke", action="store_true",
                   help="quick end-to-end check: 25 degC only, 2 configs, 2 seeds, 2 epochs, "
                        "separate *_smoke output folder")


def default_name(temps: list[int]) -> str:
    if sorted(temps) == DEFAULT_TEMPS:
        return DEFAULT_NAME
    return "paper_" + "_".join(f"n{-t}" if t < 0 else str(t) for t in sorted(temps))


def resolve(args: argparse.Namespace) -> Context:
    temps = sorted(args.temps) if args.temps else list(DEFAULT_TEMPS)
    name = args.name or default_name(temps)
    if args.smoke:
        temps = list(SMOKE_TEMPS)
        if not name.endswith("_smoke"):
            name += "_smoke"
    D.set_data_root(args.data_root)
    folders = D.folders_for_temps(temps)
    return Context(temps=temps, folders=folders,
                   paths=PaperPaths(out_root=Path(args.out_root).resolve(), name=name),
                   smoke=bool(args.smoke))


# ---------------------------------------------------------------------------
# Machine assignment (--shard k/n)
# ---------------------------------------------------------------------------
def shard_index(config: str, n: int) -> int:
    """Stable shard (1..n) of a config name: sha256, never Python's randomised hash()."""
    digest = hashlib.sha256(config.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n + 1


def parse_shard(text: str) -> tuple[set[int], int]:
    """'1/2' -> ({1}, 2); '1,3/4' -> ({1, 3}, 4)."""
    m = re.fullmatch(r"(\d+(?:,\d+)*)/(\d+)", text.strip())
    if m is None:
        raise ValueError(f"Bad --shard {text!r}; expected k/n, e.g. 1/2 or 1,3/4")
    n = int(m.group(2))
    ks = {int(k) for k in m.group(1).split(",")}
    if n < 1 or not all(1 <= k <= n for k in ks):
        raise ValueError(f"Bad --shard {text!r}: every k must be in 1..{n}")
    return ks, n


# ---------------------------------------------------------------------------
# Run environment
# ---------------------------------------------------------------------------
def git_info() -> tuple[str, bool | None]:
    """(commit hash, has uncommitted changes to tracked files) of the code being run."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def cpu_model() -> str:
    try:
        if sys.platform.startswith("win"):
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return platform.processor() or "unknown"


def device_label(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def environment_info(device: torch.device, norm_sha256: str) -> dict:
    commit, dirty = git_info()
    return {"hostname": socket.gethostname(), "device": str(device), "gpu": device_label(device),
            "cpu_model": cpu_model(), "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "platform": platform.platform(), "git_commit": commit, "git_dirty": dirty,
            "norm_stats_sha256": norm_sha256}


# ---------------------------------------------------------------------------
# Normalisation statistics: computed once from the training set, then always read back
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    """Atomic write with '\\n' line endings, so the bytes are identical on Windows and Linux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_or_create_norm_stats(path: Path, train_seqs: list, temps: list[int], log,
                              save: bool = True) -> tuple[dict, str]:
    """Return (stats, sha256 of the file).

    If the file exists it is read as is (never recomputed); it must have been computed from
    exactly the current training sequences. Otherwise it is computed from train_seqs and saved.
    """
    train_ids = [s.info.id for s in train_seqs]
    if path.exists():
        stats = D.load_norm_stats(path)
        if list(stats["features"]) != D.FEATURES:
            raise SystemExit(f"{path}: features {stats['features']} != {D.FEATURES}")
        if list(stats["computed_from"]) != train_ids:
            raise SystemExit(f"{path} was computed from a different training set:\n"
                             f"  file:    {stats['computed_from']}\n  current: {train_ids}")
        sha = file_sha256(path)
        log(f"Normalisation stats read from {path} (sha256 {sha[:12]}), not recomputed")
        return stats, sha
    stats = D.compute_norm_stats(train_seqs)
    stats["note"] = ("Computed once from the training set only (all rows after 1 s resampling); "
                     "validation and test sets use these values unchanged.")
    stats["temperatures_C"] = list(temps)
    if not save:
        log("Normalisation stats computed in memory (file not written)")
        return stats, "not-saved"
    write_json(path, stats)
    sha = file_sha256(path)
    log(f"Normalisation stats computed from {len(train_ids)} training sequences and saved to "
        f"{path} (sha256 {sha[:12]}). Copy this exact file to every other machine.")
    return stats, sha


# ---------------------------------------------------------------------------
# Validation predictions (npz): all points from W-1, so metrics can be recomputed later
# with a later evaluation start without retraining
# ---------------------------------------------------------------------------
def save_val_predictions(path: Path, model: torch.nn.Module, val_seqs: list, stats: dict,
                         window: int, device: torch.device) -> None:
    ds = D.WindowDataset(val_seqs, stats, window, None, log=lambda *_: None).to(device)
    pred = predict(model, ds).cpu().numpy().astype(np.float32)
    y = ds.y[ds.ends].cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, pred=pred, y_true=y,
                        local_index=ds.local_index.astype(np.int32),
                        seq_index=ds.seq_of.astype(np.int16),
                        seq_ids=np.array([s.info.id for s in ds.seqs]),
                        seq_folders=np.array([s.info.folder for s in ds.seqs]),
                        window=np.int32(window))
    os.replace(tmp, path)


def metrics_from_npz(path: Path, eval_start: int) -> dict:
    """Validation RMSE / MAE (SOC percentage points) on rows >= eval_start, overall and per folder."""
    z = np.load(path)
    keep = z["local_index"] >= eval_start
    pred, y = z["pred"][keep], z["y_true"][keep]
    folders = z["seq_folders"][z["seq_index"][keep]]
    rmse, mae = metrics_pct(pred, y)
    out = {"val_rmse_pct": rmse, "val_mae_pct": mae, "n_val_points": int(keep.sum()), "per_temp": {}}
    for folder in D.TEMP_FOLDERS:
        m = folders == folder
        if m.any():
            r, a = metrics_pct(pred[m], y[m])
            out["per_temp"][folder] = {"rmse_pct": r, "mae_pct": a, "n_points": int(m.sum())}
    return out


def sample_std(values) -> float:
    """Sample standard deviation (ddof = 1); NaN for fewer than two values."""
    v = np.asarray(list(values), dtype=np.float64)
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


class Tee:
    """Print and append to a log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
