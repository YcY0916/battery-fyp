"""Write (or check) the machine-readable protocol files of v2 in protocol/v2/.

    protocol/v2/norm_stats.json    normalisation stats from the training set (must have the v1 sha256)
    protocol/v2/data_sha256.json   sha256 (CRLF -> LF) of every data file of the v2 experiment

Run once before the tag v2-protocol is created; afterwards only --check is used:
    python src/make_protocol_files.py            # write both files
    python src/make_protocol_files.py --check    # recompute and compare, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import lstm_data as D  # noqa: E402
import v2_common as V  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Write or check protocol/v2/norm_stats.json and data_sha256.json")
    p.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    p.add_argument("--data-root", type=Path, default=V.REPO_ROOT)
    args = p.parse_args()
    D.set_data_root(args.data_root)

    D.block_test_files()
    splits = D.build_splits(V.FOLDERS)
    train = D.load_sequences(splits["train"])
    stats = V.build_norm_stats(train)
    norm_bytes = V.norm_stats_bytes(stats)
    sha = hashlib.sha256(norm_bytes).hexdigest()
    print(f"Normalisation stats from {len(train)} training sequences: sha256 {sha}")
    if sha != V.EXPECTED_NORM_SHA256:
        raise SystemExit(f"FAILED: expected the v1 sha256 {V.EXPECTED_NORM_SHA256}. Stop and ask.")
    print("  = v1 file results/paper_0to40/norm_stats.json (OK)")
    hashes = V.compute_data_hashes()
    data_text = json.dumps(hashes, indent=2, ensure_ascii=False) + "\n"
    roles = {}
    for rec in hashes["files"].values():
        roles[rec["role"]] = roles.get(rec["role"], 0) + 1
    print("Data files: " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))

    if args.check:
        ok = True
        for path, want in ((V.PROTOCOL_NORM_STATS, norm_bytes), (V.PROTOCOL_DATA_SHA, data_text.encode("utf-8"))):
            have = V.lf_bytes(path) if path.exists() else b""
            same = have == want
            ok &= same
            print(f"{path.relative_to(V.REPO_ROOT).as_posix()}: {'identical' if same else 'DIFFERENT'}")
        raise SystemExit(0 if ok else 1)
    V.write_bytes_atomic(V.PROTOCOL_NORM_STATS, norm_bytes)
    V.write_text_atomic(V.PROTOCOL_DATA_SHA, data_text)
    print(f"Wrote {V.PROTOCOL_NORM_STATS} and {V.PROTOCOL_DATA_SHA}")


if __name__ == "__main__":
    main()
