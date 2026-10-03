"""Write the step-3 files after the author has chosen the step-3 configs (any number N, e.g. 6 to 8).

    python src/make_step3_files.py --configs L1_W300_H64,L1_W240_H32,L1_W180_H64,L2_W240_H32,L2_W300_H32,L1_W360_H32
    python src/make_step3_files.py --configs ... --dry-run     # only print the plan

Writes
    shards/step3_candidates.txt                  the N chosen configs (read by run_shard.py, merge_v2.py, ...)
    shards/step3_laptop.txt, step3_kaggle_gpu0.txt, step3_kaggle_gpu1.txt, step3_colab.txt
                                                 seeds 43-46; all four seeds of a config on one platform

Rules: a config starts on the platform where its seed-42 run was trained (read from
results/paper_0to40_v2/runs/<config>_seed42/run_meta.json, else from shards/step2_*.txt). Heavy
configs (L2, or W >= 240) never go to Colab. Then whole configs are moved, one at a time, from the
platform that would finish last to another allowed platform while this shortens the slowest
platform by at least 0.3 h. The time model uses the measured timing probe: seconds per epoch = unit x W x L x g(H),
g = 1.0 / 1.2 / 1.5 for H = 32 / 64 / 128, 80 epochs per run (a planning estimate only).
These files do not change any training setting (they are not training files; the tag v2-protocol
is not affected).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import v2_common as V  # noqa: E402
from run_shard import parse_shard_file  # noqa: E402

PLATFORMS = ["laptop", "kaggle_gpu0", "kaggle_gpu1", "colab"]
# seconds per epoch of L2_W360_H128 measured by the timing probe, divided by W x L x g(H) = 1080
UNIT = {"laptop": 26.9 / 1080, "kaggle_gpu0": 38.4 / 1080, "kaggle_gpu1": 38.4 / 1080, "colab": 42.0 / 1080}
G_H = {32: 1.0, 64: 1.2, 128: 1.5}
EPOCHS = 80
MIN_GAIN_H = 0.3  # move a config away from its seed-42 platform only if the slowest platform gains >= 0.3 h


def heavy(config: str) -> bool:
    l, w, _ = V.parse_config(config)
    return l == 2 or w >= 240


def hours(config: str, platform: str) -> float:
    l, w, h = V.parse_config(config)
    return UNIT[platform] * w * l * G_H[h] * EPOCHS * len(V.STEP3_SEEDS) / 3600


def step2_platform(config: str, paths: V.Paths, shard_map: dict) -> str:
    meta = paths.run_dir(V.run_name(config, V.STEP2_SEED)) / "run_meta.json"
    if meta.exists():
        m = V.read_json(meta)
        if m["platform"] == "kaggle":
            return f"kaggle_gpu{m.get('cuda_visible_devices') or '0'}"
        return m["platform"] if m["platform"] in PLATFORMS else "laptop"
    return shard_map.get(config, "laptop")


def main() -> None:
    p = argparse.ArgumentParser(description="Write shards/step3_candidates.txt and the step-3 shard files")
    p.add_argument("--configs", required=True, help="the chosen step-3 configs (N >= 1), comma-separated")
    p.add_argument("--exp", default=V.FORMAL_EXP)
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite existing step-3 files with other content")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    grid = V.step2_configs()
    bad = [c for c in configs if c not in grid]
    if bad or not configs or len(set(configs)) != len(configs):
        raise SystemExit(f"Need one or more different configs from the step-2 grid; got {configs}"
                         + (f" (not in the grid: {bad})" if bad else ""))
    configs.sort(key=V.config_sort_key)
    paths = V.Paths(args.out_root, args.exp)
    shard_map = {}
    for plat in PLATFORMS:
        f = V.SHARDS_DIR / f"step2_{plat}.txt"
        if f.exists():
            for c, _ in parse_shard_file(f)[1]:
                shard_map[c] = plat

    assign = {}
    for c in configs:
        plat = step2_platform(c, paths, shard_map)
        assign[c] = "laptop" if plat == "colab" and heavy(c) else plat

    def loads(a):
        return {pl: sum(hours(c, pl) for c, x in a.items() if x == pl) for pl in PLATFORMS}

    while True:
        cur = loads(assign)
        span = max(cur.values())
        best = None
        for c, pl in assign.items():
            if cur[pl] < span - 1e-9:
                continue
            for other in PLATFORMS:
                if other == pl or (other == "colab" and heavy(c)):
                    continue
                trial = dict(assign, **{c: other})
                new_span = max(loads(trial).values())
                if new_span < span - MIN_GAIN_H and (best is None or new_span < best[0]):
                    best = (new_span, c, other)
        if best is None:
            break
        assign[best[1]] = best[2]

    print(f"Step-3 plan ({len(configs)} configs x seeds {V.STEP3_SEEDS}; estimate at {EPOCHS} epochs per run):")
    for c in configs:
        origin = step2_platform(c, paths, shard_map)
        moved = "" if assign[c] == origin else f"   (seed 42 ran on {origin})"
        print(f"  {c:14s} -> {assign[c]:12s} {hours(c, assign[c]):5.2f} h{moved}")
    for pl, h in loads(assign).items():
        print(f"  {pl:12s} total about {h:4.1f} h")

    files = {V.STEP3_CANDIDATES_FILE: f"# The {len(configs)} step-3 configs chosen by the author (written by src/make_step3_files.py)\n"
                                      + "".join(f"{c}\n" for c in configs)}
    for pl in PLATFORMS:
        mine = [c for c in configs if assign[c] == pl]
        body = "".join(f"{c} {','.join(map(str, V.STEP3_SEEDS))}\n" for c in mine)
        files[V.SHARDS_DIR / f"step3_{pl}.txt"] = (f"# Step 3 on {pl} (written by src/make_step3_files.py)"
                                                   f"{'; no runs here' if not mine else ''}\nstage: step3\n" + body)
    for path, text in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != text and not args.force:
            raise SystemExit(f"{path.relative_to(V.REPO_ROOT).as_posix()} already exists with other content; "
                             "add --force to overwrite (stop and ask if unsure)")
    if args.dry_run:
        print("Dry run: nothing written")
        return
    for path, text in files.items():
        V.write_text_atomic(path, text)
        print(f"Wrote {path.relative_to(V.REPO_ROOT).as_posix()}")
    covered = []
    for pl in PLATFORMS:
        covered += parse_shard_file(V.SHARDS_DIR / f"step3_{pl}.txt")[1]
    want = [(c, s) for c in configs for s in V.STEP3_SEEDS]
    if sorted(covered) != sorted(want):
        raise SystemExit(f"Internal check failed: the step-3 shard files do not cover the {len(want)} runs exactly once")
    print(f"Check OK: the 4 step-3 shard files cover the {len(want)} runs exactly once.")


if __name__ == "__main__":
    main()
