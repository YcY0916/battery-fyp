"""LSTM 阶段的数据处理：读取、筛选、划分、重采样、标准化、窗口 Dataset。

数据定义见 data/processed/kollmeyer_clean/DATA_REPORT.md 和 manifest.json。

- 输入：voltage, current, temperature；输出：窗口最后一个时间点的 soc。
  ah 列在读取时就不读进来（D8：ah 是 SOC 标签的来源，不能作为输入）。
- 只用驾驶工况（manifest test_type == "drive"）和 Mixed（test_type == "mixed"），
  按 manifest 的测试类型筛选，不按文件名猜。
- 按整个文件划分：
  验证集 = 每个温度编号最大的 Mixed 文件；训练集 = 其余 Mixed（含 _part1/_part2）；
  测试集 = 全部 UDDS / HWFET / LA92 / US06。
- 每条序列（每个 part）各自按 floor(time_s) 重采样到 1 s：
  电压、电流、温度取组内平均，soc 取这一秒最后一行。
- 标准化的 mean/std 只用训练集（重采样后）计算。
- 窗口在取样时按索引即时从原始数组切出，不预先展开。

直接运行本文件会打印三个集合的文件清单和窗口数（会读取测试文件，只用于核对划分）：
    python src/lstm_data.py
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
assert not FORBIDDEN_INPUTS & set(FEATURES), "ah 不能作为输入"

# 6 个温度文件夹，按温度从低到高
TEMP_FOLDERS = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
# Nominal temperature (degC) -> folder name, used by the --temps command-line options
TEMP_C_TO_FOLDER = {-20: "n20degC", -10: "n10degC", 0: "0degC", 10: "10degC", 25: "25degC", 40: "40degC"}
DRIVE_CYCLES = ["UDDS", "HWFET", "LA92", "US06"]

# 与任务说明逐一核对用：每个温度编号最大的 Mixed 文件
EXPECTED_VAL_IDS = {
    "n20degC": "n20degC/611_Mixed8",
    "n10degC": "n10degC/604_Mixed8",
    "0degC": "0degC/590_Mixed8",
    "10degC": "10degC/571_Mixed8",
    "25degC": "25degC/552_Mixed8",
    "40degC": "40degC/557_Mixed3",
}
# 数据集里没有的驾驶工况（D29：551_HWFET 被排除）
MISSING_DRIVE = {("25degC", "HWFET")}

# 验证误差（和早停用的验证 loss）只在"序列内行号 >= 评估起点"的点上计算，
# 评估起点取最长窗口 W_max 能覆盖的第一个点，即 W_max - 1，保证各配置比较的是同一批点。
# 最初 18 组（W <= 120）用的是 119；加入 W = 180、240 之后统一改为 239，这里是默认值，
# sweep_lstm.py / evaluate_lstm.py 都可以用 --eval-start 另外指定。
EVAL_START_INDEX = 239
EVAL_START_INDEX_ORIGINAL = 119  # 最初 18 组训练时（早停）用的评估起点


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
    """超参数对比阶段读到了测试文件。"""


# ---------------------------------------------------------------------------
# 测试集保护
# ---------------------------------------------------------------------------
_TEST_GUARD_ON = False
_TEST_PATHS: set[Path] = set()


def block_test_files() -> None:
    """打开保护：之后任何读取测试文件（驾驶工况）的操作都会直接报错。

    测试文件清单直接从 manifest 取（所有温度、所有 test_type == "drive" 的序列，
    包括 _excluded 里的），不依赖调用方传入的划分结果。
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
            f"对比阶段禁止读取测试文件：{_rel(path)}。"
            "测试集只能由 src/evaluate_lstm.py 在选定配置之后读取。"
        )


# ---------------------------------------------------------------------------
# manifest 与划分
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeqInfo:
    id: str             # 例如 "10degC/571_Mixed6_part2"
    file: str           # 相对 DATA_DIR 的路径（manifest 里用 / 分隔）
    folder: str         # 温度文件夹，例如 "n20degC"
    temp_c: int         # 名义温度
    test_type: str      # "mixed" 或 "drive"
    test_name: str      # "Mixed6"、"UDDS" ……
    part: int
    n_parts: int
    split: str          # "train" / "val" / "test"

    @property
    def path(self) -> Path:
        return _series_path(self.file)


def _series_path(file: str) -> Path:
    # manifest 里的 file 用 / 分隔，排除的序列本身就带 "_excluded/" 前缀
    return DATA_DIR / Path(*file.split("/"))


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _mixed_number(test_name: str) -> int:
    m = re.fullmatch(r"Mixed(\d+)", test_name)
    if m is None:
        raise ValueError(f"无法识别的 Mixed 名称：{test_name}")
    return int(m.group(1))


def build_splits(folders: list[str] | None = None) -> dict[str, list[SeqInfo]]:
    """按 manifest 的测试类型筛选并划分。folders=None 表示全部 6 个温度。

    只读 manifest，不读任何数据文件。发现与任务说明不一致时直接报错。
    """
    folders = list(TEMP_FOLDERS) if folders is None else list(folders)
    unknown = set(folders) - set(TEMP_FOLDERS)
    if unknown:
        raise ValueError(f"未知的温度文件夹：{sorted(unknown)}")

    manifest = load_manifest()
    entries = [
        e for e in manifest["series"]
        if not e["excluded"] and e["folder"] in folders
        and e["test_type"] in ("mixed", "drive")
    ]

    # 每个温度编号最大的 Mixed 做验证集
    val_ids: dict[str, str] = {}
    for folder in folders:
        mixed = [e for e in entries if e["folder"] == folder and e["test_type"] == "mixed"]
        if not mixed:
            raise ValueError(f"{folder} 没有 Mixed 文件")
        top = max(_mixed_number(e["test_name"]) for e in mixed)
        chosen = [e for e in mixed if _mixed_number(e["test_name"]) == top]
        if len(chosen) != 1:
            raise ValueError(f"{folder} 编号最大的 Mixed{top} 有 {len(chosen)} 段，需要人工决定")
        if chosen[0]["id"] != EXPECTED_VAL_IDS[folder]:
            raise ValueError(
                f"{folder} 的验证文件是 {chosen[0]['id']}，与任务说明的 "
                f"{EXPECTED_VAL_IDS[folder]} 不一致"
            )
        val_ids[folder] = chosen[0]["id"]

    splits: dict[str, list[SeqInfo]] = {"train": [], "val": [], "test": []}
    for e in entries:
        if e["test_type"] == "drive":
            if e["test_name"] not in DRIVE_CYCLES:
                raise ValueError(f"未知的驾驶工况：{e['id']} ({e['test_name']})")
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
            raise FileNotFoundError(f"manifest 中的文件不存在：{info.path}")
        splits[split].append(info)

    # 测试集应为每个温度 4 个工况各一个（25 degC 没有 HWFET）
    for folder in folders:
        got = sorted(s.test_name for s in splits["test"] if s.folder == folder)
        want = sorted(c for c in DRIVE_CYCLES if (folder, c) not in MISSING_DRIVE)
        if got != want:
            raise ValueError(f"{folder} 的测试文件 {got} 与预期 {want} 不一致")

    # 三个集合之间不能有重叠（按序列，也按原始 Mixed/工况文件）
    ids = {k: {s.id for s in v} for k, v in splits.items()}
    raw = {k: {(s.folder, s.id.split("/")[1].split("_part")[0]) for s in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b] or raw[a] & raw[b]:
            raise ValueError(f"{a} 与 {b} 有重叠：{sorted(ids[a] & ids[b]) or sorted(raw[a] & raw[b])}")

    order = {f: i for i, f in enumerate(TEMP_FOLDERS)}
    for k in splits:
        splits[k].sort(key=lambda s: (order[s.folder], s.id))
    return splits


# ---------------------------------------------------------------------------
# 读取与重采样
# ---------------------------------------------------------------------------
@dataclass
class Sequence:
    info: SeqInfo
    sec: np.ndarray  # (n,) int64，floor(time_s)
    x: np.ndarray    # (n, 3) float64，未标准化的 voltage/current/temperature
    y: np.ndarray    # (n,) float64，soc（0-1）

    def __len__(self) -> int:
        return len(self.y)


def read_raw(info: SeqInfo) -> pd.DataFrame:
    """读取一条序列的原始 CSV（不读 ah 列）。所有数据读取都经过这里。"""
    path = info.path
    _check_not_test(path)
    if info.split == "test" and _TEST_GUARD_ON:
        raise TestSetAccessError(f"对比阶段禁止读取测试文件：{info.id}")
    cols = ["time_s", *FEATURES, TARGET]
    df = pd.read_csv(path, usecols=cols)
    assert not FORBIDDEN_INPUTS & set(df.columns)
    if not np.all(np.diff(df["time_s"].to_numpy()) > 0):
        raise ValueError(f"{info.id} 的 time_s 不是严格递增")
    return df


def resample_1s(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 floor(time_s) 分组：电压/电流/温度取平均，soc 取该秒最后一行。"""
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
            raise ValueError(f"{info.id} 重采样后有 NaN")
        seqs.append(Sequence(info, sec, x, y))
    log(f"  读取 {len(seqs)} 条序列，重采样后共 {sum(len(s) for s in seqs):,} 行")
    return seqs


# ---------------------------------------------------------------------------
# 标准化
# ---------------------------------------------------------------------------
def compute_norm_stats(train_seqs: list[Sequence]) -> dict:
    if any(s.info.split != "train" for s in train_seqs):
        raise ValueError("标准化参数只能用训练集计算")
    x = np.concatenate([s.x for s in train_seqs], axis=0)
    return {
        "features": FEATURES,
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "n_rows": int(len(x)),
        "computed_from": [s.info.id for s in train_seqs],
        "note": "只用训练集（1 s 重采样之后的全部行）计算，验证集和测试集原样套用",
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
        raise ValueError(f"标准化参数的特征 {stats['features']} 与 {FEATURES} 不一致")
    return (x - np.asarray(stats["mean"])) / np.asarray(stats["std"])


# ---------------------------------------------------------------------------
# 窗口 Dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """滑动窗口（步长 1），取样时按索引即时切片，不预先展开窗口。

    每条序列各自切窗口，不跨序列。样本 i 是以 ends[i] 为最后一行、长度 W 的窗口，
    标签是 ends[i] 这一行的 soc。

    min_end_index：只保留"序列内行号 >= min_end_index"的窗口终点。
        训练集用 W-1（所有能切出完整窗口的点）；
        验证集/测试集用评估起点（默认 EVAL_START_INDEX = 239），保证各窗口长度评估同一批点。
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
            log(f"  窗口长度 {self.window}：以下序列比窗口短，已跳过："
                + "，".join(f"{sid}（{n} 行）" for sid, n in self.skipped))
        if not self.seqs:
            raise ValueError("没有任何序列长于窗口")
        # 原始（逐行）数组，窗口在取样时从这里切
        self.x = torch.from_numpy(np.concatenate(xs))
        self.y = torch.from_numpy(np.concatenate(ys))
        self.ends = torch.from_numpy(np.concatenate(ends))
        self.seq_of = np.concatenate(seq_of)        # 每个样本属于第几条（保留的）序列
        self.local_index = np.concatenate(local_idx)  # 每个样本在序列内的行号
        self._rel = torch.arange(-self.window + 1, 1, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        e = int(self.ends[i])
        return self.x[e - self.window + 1:e + 1], self.y[e]

    def get_batch(self, idx: torch.Tensor):
        """一次取一批窗口（向量化切片），只生成这一批的窗口。"""
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
# 直接运行：打印划分
# ---------------------------------------------------------------------------
def _print_split_summary(folders, windows, eval_start):
    splits = build_splits(folders)
    seqs = {k: load_sequences(v, log=lambda *_: None) for k, v in splits.items()}
    stats = compute_norm_stats(seqs["train"])
    for split in ("train", "val", "test"):
        print(f"\n===== {split}：{len(splits[split])} 条序列 =====")
        header = f"{'序列':32s}{'温度':>6s}{'行数(1s)':>10s}" + "".join(
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
                print(f"  W={w} 跳过（比窗口短）：{ds.skipped}")
        print(f"{'合计':32s}{'':>6s}{sum(len(s) for s in seqs[split]):>10,d}"
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
    print("\n说明：训练集窗口终点为每条序列的第 W-1 行起；验证/测试集窗口终点为第 "
          f"{eval_start} 行起（W <= {eval_start + 1} 时各 W 相同）。")
    ids = [{s.id for s in v} for v in splits.values()]
    print("三个集合两两无重叠：", not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]))
    raw = [{(s.folder, s.id.split("/")[1].split("_part")[0]) for s in v} for v in splits.values()]
    print("No overlap between the original files (parts follow their file):",
          not (raw[0] & raw[1] or raw[0] & raw[2] or raw[1] & raw[2]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="打印训练/验证/测试集的文件清单和窗口数")
    p.add_argument("--folders", nargs="+", default=None, help="只看这些温度文件夹")
    p.add_argument("--temps", nargs="+", type=int, default=None,
                   help="nominal temperatures in degC (e.g. 0 10 25 40); overrides --folders")
    p.add_argument("--paper", action="store_true",
                   help="paper setting: temps 0 10 25 40, windows 60-300, evaluation start 299")
    p.add_argument("--smoke", action="store_true", help="only 25 degC (quick check)")
    p.add_argument("--data_root", type=Path, default=None,
                   help="folder that contains data/processed/kollmeyer_clean (default: repository root)")
    p.add_argument("--windows", nargs="+", type=int, default=[30, 60, 120, 180, 240])
    p.add_argument("--eval-start", type=int, default=EVAL_START_INDEX,
                   help=f"验证/测试集的评估起点（行号），默认 {EVAL_START_INDEX}")
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
