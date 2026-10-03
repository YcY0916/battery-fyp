"""Check that tracked files and experiment outputs contain only English text.

    python src/check_english.py                         # tracked files + results/paper_0to40_v2 (if present)
    python src/check_english.py --exp v2_smoke          # tracked files + results/v2_smoke
    python src/check_english.py --exp paper_0to40_v2 --exp timing_v2

Allowed characters: printable ASCII, tab, line feeds, and only these non-ASCII symbols:
° (U+00B0), ± (U+00B1), − (U+2212), × (U+00D7), µ (U+00B5).
Scanned:
    - every file tracked by git (binary files such as .png / .pdf / .pt / .npz are skipped);
    - every .csv / .json / .md / .txt / .tex / .log file under results/<exp>/ for each --exp;
    - the names of all those files and of every file and folder under results/<exp>/ and models/<exp>/.
Every offending file is listed with its line numbers; the exit status is 1 if anything was found.
One tracked file is skipped on purpose and listed as SKIPPED: results/paper_0to40/README.md, because
the v1 folder results/paper_0to40 is read only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED = set("°±−×µ")
BINARY_EXT = {".png", ".pdf", ".pt", ".npz", ".npy", ".zip", ".jpg", ".jpeg", ".gif", ".pkl", ".h5", ".xlsx"}
OUTPUT_EXT = {".csv", ".json", ".md", ".txt", ".tex", ".log"}
SKIPPED = {"results/paper_0to40/README.md": "inside the read-only v1 folder results/paper_0to40"}
MAX_LINES_SHOWN = 40


def bad_char(ch: str) -> bool:
    o = ord(ch)
    if o < 128:
        return not (ch.isprintable() or ch in "\t\n\r")
    return ch not in ALLOWED


def scan_text(path: Path) -> list[tuple[int, str]]:
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return []  # binary
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return [(0, f"not valid UTF-8 ({e})")]
    hits = []
    for n, line in enumerate(text.split("\n"), 1):
        bad = sorted({ch for ch in line if bad_char(ch)})
        if bad:
            hits.append((n, " ".join(f"U+{ord(c):04X}" for c in bad)))
    return hits


def bad_name(rel: str) -> bool:
    return any(bad_char(ch) or ord(ch) > 127 for ch in rel)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"], capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def main() -> None:
    p = argparse.ArgumentParser(description="Check that files contain only English text (and the allowed symbols)")
    p.add_argument("--exp", action="append", default=None,
                   help="experiment output folder(s) to scan (default: paper_0to40_v2 if it exists)")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT)
    p.add_argument("--no-tracked", action="store_true", help="skip the tracked files")
    args = p.parse_args()

    problems: dict[str, list[tuple[int, str]]] = {}
    n_files = 0
    skipped = []
    if not args.no_tracked:
        for rel in tracked_files():
            if rel in SKIPPED:
                skipped.append(rel)
                continue
            if bad_name(rel):
                problems.setdefault(rel, []).append((0, "non-ASCII character in the path"))
            path = REPO_ROOT / rel
            if path.suffix.lower() in BINARY_EXT or not path.is_file():
                continue
            n_files += 1
            hits = scan_text(path)
            if hits:
                problems.setdefault(rel, []).extend(hits)
    exps = args.exp
    if exps is None:
        exps = ["paper_0to40_v2"] if (args.out_root / "results" / "paper_0to40_v2").exists() else []
    for exp in exps:
        for top in ("results", "models"):
            base = args.out_root / top / exp
            if not base.exists():
                if top == "results":
                    print(f"Note: {base} does not exist")
                continue
            for path in sorted(base.rglob("*")):
                rel = path.relative_to(args.out_root).as_posix()
                if bad_name(rel):
                    problems.setdefault(rel, []).append((0, "non-ASCII character in the path"))
                if path.is_file() and path.suffix.lower() in OUTPUT_EXT:
                    n_files += 1
                    hits = scan_text(path)
                    if hits:
                        problems.setdefault(rel, []).extend(hits)

    for rel in skipped:
        print(f"SKIPPED {rel} ({SKIPPED[rel]})")
    if problems:
        total = sum(len(v) for v in problems.values())
        print(f"FAILED: {total} lines with characters that are not allowed, in {len(problems)} files:")
        for rel, hits in sorted(problems.items()):
            shown = ", ".join(str(n) for n, _ in hits[:MAX_LINES_SHOWN])
            more = f" (+{len(hits) - MAX_LINES_SHOWN} more)" if len(hits) > MAX_LINES_SHOWN else ""
            print(f"  {rel}: lines {shown}{more}")
            for n, what in hits[:3]:
                print(f"      line {n}: {what}")
        sys.exit(1)
    print(f"OK: {n_files} files scanned ({'tracked files' if not args.no_tracked else 'no tracked files'}"
          f"{', outputs of ' + ', '.join(exps) if exps else ''}); only English text and the allowed "
          "symbols ° ± − × µ found.")


if __name__ == "__main__":
    main()
