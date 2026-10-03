"""Core settings and helpers of the v2 experiment (paper_0to40_v2), shared by every v2 script.

The rules are written down in docs/protocol_v2.md (git tag v2-protocol). This module holds the
parts that affect training: the grid, the evaluation start, the hashes that must match, the
pre-flight checks and the layout of one run folder. Analysis-only helpers live in v2_analysis.py,
so that changing a table or a figure never changes the training code fingerprint.

Training itself (data, model, loop) is unchanged from the v1 code:
    lstm_data.py   reading, split, 1 Hz resampling, normalisation, window Dataset
    lstm_model.py  LSTM -> last time step -> Linear(H, 1)
    sweep_lstm.py  fit() (Adam, MSE, early stopping), predict(), metrics_pct(), set_seed()
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_data as D  # noqa: E402
import sweep_lstm as SL  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "protocol" / "v2"
PROTOCOL_NORM_STATS = PROTOCOL_DIR / "norm_stats.json"
PROTOCOL_DATA_SHA = PROTOCOL_DIR / "data_sha256.json"
SHARDS_DIR = REPO_ROOT / "shards"
STEP3_CANDIDATES_FILE = SHARDS_DIR / "step3_candidates.txt"

# ---------------------------------------------------------------------------
# Experiment definition (docs/protocol_v2.md, sections 4.1-4.4)
# ---------------------------------------------------------------------------
FORMAL_EXP = "paper_0to40_v2"
PROTOCOL_TAG = "v2-protocol"
TEMPS_C = [0, 10, 25, 40]
FOLDERS = D.folders_for_temps(TEMPS_C)          # ['0degC', '10degC', '25degC', '40degC']
UNSEEN_TEMPS_C = [-20, -10]
UNSEEN_FOLDERS = D.folders_for_temps(UNSEEN_TEMPS_C)
FS_HZ = 1.0                                      # every sequence is resampled to 1 Hz
EVAL_START = 359                                 # longest window (360) - 1
STEP2_SEED = 42
STEP3_SEEDS = [43, 44, 45, 46]
ALL_SEEDS = [STEP2_SEED] + STEP3_SEEDS
GRID_H = [32, 64, 128]
GRID_W = {1: [60, 120, 180, 240, 300, 360], 2: [180, 240, 300, 360]}
N_STEP3_CONFIGS = 6

# Old v1 folders: read only, no v2 script may write into them
PROTECTED_DIRS = [("results", "paper_0to40"), ("models", "paper_0to40")]
# The old normalisation file (v1, results/paper_0to40/norm_stats.json); the v2 file must be identical
EXPECTED_NORM_SHA256 = "00c97f004a5f80a4144dda3e4fe5bf8400e36462c36cef5dba67a12666156a5b"
# Largest relative difference allowed between the recomputed and the protocol normalisation stats
# (see docs/protocol_v2.md, changelog entry 2 of 2026-10-03)
NORM_REL_TOL = 1e-9
NORM_NOTE = ("Computed once from the training set only (all rows after 1 s resampling); "
             "validation and test sets use these values unchanged.")

# Files whose content decides the training result. Every formal run checks that they do not
# differ from the tag v2-protocol, and stores their combined hash (code fingerprint).
TRAINING_FILES = ["src/lstm_data.py", "src/lstm_model.py", "src/sweep_lstm.py",
                  "src/v2_common.py", "src/run_shard.py",
                  "protocol/v2/norm_stats.json", "protocol/v2/data_sha256.json"]

STAGES = ["step2", "step3", "probe", "smoke"]
FORMAL_STAGES = ["step2", "step3"]

CONFIG_RE = re.compile(r"^L(\d+)_W(\d+)_H(\d+)$")
RUN_RE = re.compile(r"^(L\d+_W\d+_H\d+)_seed(\d+)$")


def config_name(l: int, w: int, h: int) -> str:
    return f"L{l}_W{w}_H{h}"


def parse_config(name: str) -> tuple[int, int, int]:
    """'L1_W300_H64' -> (L, W, H) = (1, 300, 64)."""
    m = CONFIG_RE.match(name)
    if m is None:
        raise ValueError(f"Bad config name {name!r}; expected e.g. L1_W300_H64")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_name(config: str, seed: int) -> str:
    return f"{config}_seed{seed}"


def parse_run(name: str) -> tuple[str, int]:
    m = RUN_RE.match(name)
    if m is None:
        raise ValueError(f"Bad run name {name!r}; expected e.g. L1_W300_H64_seed42")
    return m.group(1), int(m.group(2))


def step2_configs() -> list[str]:
    """The 30 step-2 configs: L1 first, then L2; inside each, by W then H."""
    return [config_name(l, w, h) for l in sorted(GRID_W) for w in GRID_W[l] for h in GRID_H]


def config_sort_key(name: str) -> tuple[int, int, int]:
    return parse_config(name)


def n_params(config: str) -> int:
    from lstm_model import LSTMSOC, count_parameters
    l, _, h = parse_config(config)
    return count_parameters(LSTMSOC(hidden_size=h, num_layers=l))


def latency_s(config: str) -> float:
    """Initial delay before the first estimate: W / f_s."""
    return parse_config(config)[1] / FS_HZ


def temp_label(folder: str) -> str:
    """'0degC' -> '0 °C', 'n10degC' -> '−10 °C'."""
    t = int(folder.replace("n", "-").replace("degC", ""))
    return f"{t} °C" if t >= 0 else f"−{-t} °C"


def read_step3_candidates(path: Path = STEP3_CANDIDATES_FILE) -> list[str]:
    """The 6 configs chosen by the author for step 3 (one per line, '#' comments)."""
    if not path.exists():
        raise SystemExit(f"{path} not found. The 6 step-3 configs are chosen by the author after "
                         "step 2 and committed in this file.")
    names = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n]
    for n in names:
        parse_config(n)
        if n not in step2_configs():
            raise SystemExit(f"{path}: {n} is not one of the 30 step-2 configs")
    if len(names) != len(set(names)):
        raise SystemExit(f"{path}: duplicate configs")
    return names


# ---------------------------------------------------------------------------
# Hyperparameters, read from the existing training code (never typed twice)
# ---------------------------------------------------------------------------
def hyperparameters(max_epochs: int | None = None) -> dict:
    return {
        "temperatures_C": TEMPS_C, "folders": FOLDERS,
        "features": list(D.FEATURES), "target": D.TARGET,
        "resample": "1 s by floor(time_s); voltage/current/temperature mean, soc last row",
        "sampling_rate_hz": FS_HZ,
        "split": "whole files; val = highest-numbered Mixed per temperature; train = other Mixed; "
                 "test = UDDS/HWFET/LA92/US06",
        "label": "soc of the last row of the window",
        "train_window_ends": "every row from W-1",
        "eval_start_index": EVAL_START,
        "early_stopping_metric": f"validation MSE on rows >= {EVAL_START}",
        "optimizer": "Adam", "lr": SL.LR, "batch_size": SL.BATCH_SIZE, "loss": "MSE",
        "max_epochs": SL.MAX_EPOCHS if max_epochs is None else int(max_epochs),
        "patience": SL.PATIENCE, "dropout": 0.0, "eval_batch_size": SL.EVAL_BATCH_SIZE,
        "restore_best_weights": True, "shuffle": "torch.randperm with torch.Generator().manual_seed(seed) every epoch",
        "seeding": "random, numpy, torch, torch.cuda seeded with the run seed before the model is built",
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "weight_init": "PyTorch default",
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
class Paths:
    def __init__(self, out_root: Path, exp: str):
        self.out_root = Path(out_root).resolve()
        self.exp = exp
        self.results = self.out_root / "results" / exp
        self.models = self.out_root / "models" / exp
        self.runs = self.results / "runs"
        self.logs = self.results / "logs"
        self.norm_stats = self.results / "norm_stats.json"
        self.tables = self.results / "tables"
        self.figs = self.results / "figs"
        self.notes = self.results / "figure_notes.md"
        self.bench_csv = self.results / "inference_benchmark.csv"
        self.test = self.results / "test"
        check_not_protected(self.results)
        check_not_protected(self.models)

    def run_dir(self, run: str) -> Path:
        return self.runs / run

    def model_dir(self, run: str) -> Path:
        return self.models / run

    def model_file(self, run: str) -> Path:
        return self.models / run / "model.pt"

    def rel(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.out_root).as_posix()
        except ValueError:
            return Path(path).as_posix()


def check_not_protected(path: Path) -> None:
    """Stop if path is (inside) results/paper_0to40 or models/paper_0to40 (v1, read only)."""
    parts = Path(path).resolve().parts
    for prot in PROTECTED_DIRS:
        if any(parts[i:i + len(prot)] == prot for i in range(len(parts) - len(prot) + 1)):
            raise SystemExit(f"Refusing to write to {Path(path).resolve()}: the v1 folders results/paper_0to40 and "
                             "models/paper_0to40 are read only.")


# ---------------------------------------------------------------------------
# Hashes and atomic writes
# ---------------------------------------------------------------------------
def lf_bytes(path: Path) -> bytes:
    """File bytes with CRLF turned into LF, so a Windows checkout hashes like a Linux one."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(lf_bytes(path)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(mapping: dict[str, str]) -> str:
    """One sha256 over sorted 'key sha' lines."""
    text = "".join(f"{k} {v}\n" for k, v in sorted(mapping.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tmp_name(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json(path: Path, obj) -> None:
    """Atomic JSON write with LF line endings (identical bytes on Windows and Linux)."""
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Data files expected by the protocol
# ---------------------------------------------------------------------------
def data_rel(info: D.SeqInfo) -> str:
    return info.file


def protocol_file_list() -> dict[str, str]:
    """Every data file of the v2 experiment -> its role (train / val / test / unseen_test / manifest)."""
    out = {"manifest.json": "manifest"}
    sp = D.build_splits(FOLDERS)
    for split in ("train", "val", "test"):
        for s in sp[split]:
            out[s.file] = split
    for s in D.build_splits(UNSEEN_FOLDERS)["test"]:
        out[s.file] = "unseen_test"
    return out


def compute_data_hashes() -> dict:
    """Used once by make_protocol_files.py to write protocol/v2/data_sha256.json."""
    roles = protocol_file_list()
    files = {}
    for rel, role in sorted(roles.items()):
        p = D.DATA_DIR / Path(*rel.split("/"))
        files[rel] = {"role": role, "sha256_lf": lf_sha256(p), "bytes_lf": len(lf_bytes(p))}
    return {"hash_rule": "sha256 of the file bytes after replacing CRLF by LF",
            "data_dir": "data/processed/kollmeyer_clean", "files": files}


def check_data_files(log, roles: tuple[str, ...] = ("manifest", "train", "val"),
                     exist_only: tuple[str, ...] = ("test", "unseen_test")) -> dict:
    """Hash the files of the given roles against protocol/v2/data_sha256.json; check that the
    files of the exist_only roles are present (they are not opened). Returns digests."""
    proto = read_json(PROTOCOL_DATA_SHA)
    bad, got = [], {}
    for rel, rec in sorted(proto["files"].items()):
        p = D.DATA_DIR / Path(*rel.split("/"))
        if not p.exists():
            bad.append(f"missing: {rel}")
            continue
        if rec["role"] in roles:
            sha = lf_sha256(p)
            got[rel] = sha
            if sha != rec["sha256_lf"]:
                bad.append(f"sha256 differs: {rel} ({sha[:12]} vs protocol {rec['sha256_lf'][:12]})")
        elif rec["role"] not in exist_only:
            bad.append(f"unknown role {rec['role']} for {rel}")
    current = protocol_file_list()
    if set(current) != set(proto["files"]):
        extra = sorted(set(current) - set(proto["files"]))
        lost = sorted(set(proto["files"]) - set(current))
        bad.append(f"the split differs from the protocol: new {extra}, gone {lost}")
    if bad:
        raise SystemExit("Data check FAILED:\n  " + "\n  ".join(bad))
    n_exist = sum(1 for r in proto["files"].values() if r["role"] in exist_only)
    log(f"Data check OK: {len(got)} files hashed ({', '.join(roles)}), {n_exist} files present "
        f"({', '.join(exist_only)}; not opened)")
    return {"data_sha256_summary": digest_of(got),
            "data_files_checked": len(got),
            "data_protocol_sha256": lf_sha256(PROTOCOL_DATA_SHA)}


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
def norm_stats_bytes(stats: dict) -> bytes:
    """The exact bytes the v1 code wrote (paper_common.load_or_create_norm_stats)."""
    return (json.dumps(stats, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build_norm_stats(train_seqs: list) -> dict:
    stats = D.compute_norm_stats(train_seqs)
    stats["note"] = NORM_NOTE
    stats["temperatures_C"] = list(TEMPS_C)
    return stats


def compare_norm_stats(reference: dict, fresh: dict) -> float:
    """Largest relative difference between two normalisation stats (over every mean and std).

    Raises ValueError if they were not computed from the same training files, features or row count."""
    for key in ("features", "computed_from", "n_rows"):
        if fresh[key] != reference[key]:
            raise ValueError(f"'{key}' differs: recomputed {fresh[key]!r} vs protocol {reference[key]!r}")
    rel = 0.0
    for key in ("mean", "std"):
        ref, new = np.asarray(reference[key], dtype=np.float64), np.asarray(fresh[key], dtype=np.float64)
        rel = max(rel, float(np.max(np.abs(new - ref) / np.abs(ref))))
    return rel


def check_norm_stats(train_seqs: list, paths: Paths, log, perturb: float = 0.0) -> tuple[dict, str, float]:
    """Normalisation used for training = protocol/v2/norm_stats.json, read as is on every platform.

    1. The file's sha256 must be EXPECTED_NORM_SHA256 (= the v1 file); otherwise stop.
    2. The stats are recomputed from the training set and compared number by number; the largest
       relative difference must be <= NORM_REL_TOL; otherwise stop. The recomputed values are never
       used for training (different numpy / Python versions can differ in the last bits).
    3. The protocol file is copied byte for byte to results/<exp>/norm_stats.json.
    perturb (tests only): multiply the recomputed mean and std by (1 + perturb) before comparing.
    Returns (stats from the protocol file, its sha256, largest relative difference)."""
    proto_bytes = lf_bytes(PROTOCOL_NORM_STATS)
    proto_sha = hashlib.sha256(proto_bytes).hexdigest()
    if proto_sha != EXPECTED_NORM_SHA256:
        raise SystemExit(f"Normalisation check FAILED: {PROTOCOL_NORM_STATS.name} sha256 {proto_sha} "
                         f"!= expected {EXPECTED_NORM_SHA256}. Stop and ask.")
    stats = json.loads(proto_bytes.decode("utf-8"))
    fresh = build_norm_stats(train_seqs)
    if perturb:
        fresh["mean"] = [v * (1 + perturb) for v in fresh["mean"]]
        fresh["std"] = [v * (1 + perturb) for v in fresh["std"]]
        log(f"TEST: recomputed stats multiplied by (1 + {perturb:g}) before the comparison")
    fresh_sha = hashlib.sha256(norm_stats_bytes(fresh)).hexdigest()
    try:
        rel = compare_norm_stats(stats, fresh)
    except ValueError as e:
        raise SystemExit(f"Normalisation check FAILED: {e}. Stop and ask.")
    if rel > NORM_REL_TOL:
        raise SystemExit(f"Normalisation check FAILED: recomputed stats differ from {PROTOCOL_NORM_STATS.name} "
                         f"by up to {rel:.3e} (relative), more than the tolerance {NORM_REL_TOL:.0e}. Stop and ask.")
    if paths.norm_stats.exists():
        have = file_sha256(paths.norm_stats)
        if have != proto_sha:
            raise SystemExit(f"{paths.norm_stats} exists with sha256 {have[:12]} != {proto_sha[:12]}")
    else:
        write_bytes_atomic(paths.norm_stats, proto_bytes)
    same = "byte-identical" if fresh_sha == proto_sha else f"recomputed sha256 {fresh_sha[:12]}"
    log(f"Normalisation check OK: training uses {PROTOCOL_NORM_STATS.relative_to(REPO_ROOT).as_posix()} "
        f"(sha256 {proto_sha}, = v1 file); recomputed from {len(train_seqs)} training sequences: "
        f"max relative difference {rel:.3e} <= {NORM_REL_TOL:.0e} ({same}); copy at {paths.rel(paths.norm_stats)}")
    return stats, proto_sha, rel


# ---------------------------------------------------------------------------
# Code version
# ---------------------------------------------------------------------------
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def code_fingerprint() -> dict:
    files = {f: lf_sha256(REPO_ROOT / f) for f in TRAINING_FILES}
    return {"code_fingerprint": digest_of(files), "training_files_sha256": files}


def check_code_version(formal: bool, log) -> dict:
    """HEAD must equal the tag v2-protocol, or the training files must not differ from it
    (committed or not). Formal runs stop on failure; other experiments only warn."""
    info = {"git_commit": "unknown", "git_tag": PROTOCOL_TAG, "git_tag_commit": None,
            "git_head_is_tag": False, "training_files_match_tag": False}
    problems = []
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        problems.append("git is not available or this is not a git checkout")
    else:
        info["git_commit"] = head.stdout.strip()
        tag = _git("rev-parse", "-q", "--verify", f"refs/tags/{PROTOCOL_TAG}^{{commit}}")
        if tag.returncode != 0:
            problems.append(f"tag {PROTOCOL_TAG} not found (run: git fetch --tags)")
        else:
            info["git_tag_commit"] = tag.stdout.strip()
            info["git_head_is_tag"] = info["git_tag_commit"] == info["git_commit"]
            diff = _git("diff", "--quiet", PROTOCOL_TAG, "--", *TRAINING_FILES)
            if diff.returncode == 0:
                info["training_files_match_tag"] = True
            else:
                names = _git("diff", "--name-only", PROTOCOL_TAG, "--", *TRAINING_FILES).stdout.split()
                problems.append(f"training files differ from {PROTOCOL_TAG}: {', '.join(names)}")
    status = _git("status", "--porcelain", "--untracked-files=no")
    info["git_dirty_files"] = [ln[3:] for ln in status.stdout.splitlines()] if status.returncode == 0 else None
    if problems:
        msg = "Code version check: " + "; ".join(problems)
        if formal:
            raise SystemExit(msg + ". Formal runs need the protocol code. Stop and ask.")
        log("WARNING (not a formal experiment, continuing): " + msg)
    else:
        log(f"Code version check OK: HEAD {info['git_commit'][:12]}, tag {PROTOCOL_TAG} "
            f"{info['git_tag_commit'][:12]}, training files identical to the tag")
    return info


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def environment(device: torch.device, platform_name: str) -> dict:
    cuda = device.type == "cuda"
    return {
        "platform": platform_name, "hostname": socket.gethostname(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "python_version": platform.python_version(), "os": platform.platform(),
        "numpy_version": np.__version__, "cpu_threads": torch.get_num_threads(),
    }


def check_device(allow_cpu: bool, log) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log(f"GPU check OK: {torch.cuda.get_device_name(dev)} (torch {torch.__version__}, CUDA "
            f"{torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})")
        return dev
    if allow_cpu:
        log("WARNING: no GPU; running on the CPU because --allow-cpu was given (not a formal experiment)")
        return torch.device("cpu")
    raise SystemExit("GPU check FAILED: torch.cuda.is_available() is False. On Kaggle choose "
                     "'GPU T4 x2'; on Colab choose 'T4 GPU'. Stop and ask if this persists.")


# ---------------------------------------------------------------------------
# One run folder: results/<exp>/runs/<run>/ and models/<exp>/<run>/model.pt
# ---------------------------------------------------------------------------
RUN_FILES = ["run_meta.json", "metrics.json", "train_log.csv", "val_predictions.npz",
             "run.log", "curve.png"]
COMPLETE_MARKER = "COMPLETE.json"


def run_status(paths: Paths, run: str, verify_hashes: bool = True) -> tuple[str, str]:
    """('complete' | 'incomplete' | 'absent', reason)."""
    d = paths.run_dir(run)
    if not d.exists() and not paths.model_dir(run).exists():
        return "absent", ""
    marker = d / COMPLETE_MARKER
    if not marker.exists():
        return "incomplete", "no completion marker"
    try:
        rec = read_json(marker)
    except (OSError, ValueError) as e:
        return "incomplete", f"unreadable marker ({e})"
    for name, sha in rec["files"].items():
        p = paths.model_file(run) if name == "model.pt" else d / name
        if not p.exists():
            return "incomplete", f"missing {name}"
        if verify_hashes and file_sha256(p) != sha:
            return "incomplete", f"hash mismatch {name}"
    return "complete", ""


def clear_run(paths: Paths, run: str) -> None:
    for d in (paths.run_dir(run), paths.model_dir(run)):
        check_not_protected(d)
        if d.exists():
            shutil.rmtree(d)


def write_complete_marker(paths: Paths, run: str) -> None:
    d = paths.run_dir(run)
    files = {name: file_sha256(d / name) for name in RUN_FILES}
    files["model.pt"] = file_sha256(paths.model_file(run))
    write_json(d / COMPLETE_MARKER, {"run": run, "files": files, "completed_at": now_iso()})


def save_val_predictions(path: Path, model: torch.nn.Module, val_seqs: list, stats: dict,
                         window: int, device: torch.device) -> None:
    """Validation predictions for every point from W-1 (metrics can be recomputed later)."""
    ds = D.WindowDataset(val_seqs, stats, window, None, log=lambda *_: None).to(device)
    pred = SL.predict(model, ds).cpu().numpy().astype(np.float32)
    y = ds.y[ds.ends].cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, pred=pred, y_true=y,
                        local_index=ds.local_index.astype(np.int32),
                        seq_index=ds.seq_of.astype(np.int16),
                        seq_ids=np.array([s.info.id for s in ds.seqs]),
                        seq_folders=np.array([s.info.folder for s in ds.seqs]),
                        window=np.int32(window), eval_start=np.int32(EVAL_START))
    os.replace(tmp, path)


def metrics_from_npz(path: Path, eval_start: int = EVAL_START) -> dict:
    """Validation RMSE / MAE (SOC percentage points) on rows >= eval_start, overall and per folder."""
    z = np.load(path)
    keep = z["local_index"] >= eval_start
    pred, y = z["pred"][keep], z["y_true"][keep]
    folders = z["seq_folders"][z["seq_index"][keep]]
    rmse, mae = SL.metrics_pct(pred, y)
    out = {"eval_start_index": eval_start, "overall": {"rmse_pct": rmse, "mae_pct": mae,
                                                       "n_points": int(keep.sum())}, "per_temp": {}}
    for folder in D.TEMP_FOLDERS:
        m = folders == folder
        if m.any():
            r, a = SL.metrics_pct(pred[m], y[m])
            out["per_temp"][folder] = {"rmse_pct": r, "mae_pct": a, "n_points": int(m.sum())}
    return out
