"""Figures of the v2 experiment (PNG 300 dpi + PDF) and results/<exp>/figure_notes.md.

    python src/plot_v2.py --exp paper_0to40_v2 step2
    python src/plot_v2.py --exp paper_0to40_v2 step2 --highlight L1_W300_H64,L1_W240_H32
    python src/plot_v2.py --exp paper_0to40_v2 step3
    python src/plot_v2.py --exp paper_0to40_v2 final --config L1_W300_H64

step2  (needs tables/step2_ranking.csv)
    fig_step2_heatmap_colnorm   colour normalised inside each column (shows the rank inside the column)
    fig_step2_heatmap_global    one colour scale for the whole figure
    (with --highlight the files get the suffix _highlight and the given rows are outlined)
step3  (needs tables/step3_summary.csv, step3_per_seed.csv)
    fig_step3_heatmap           colour = mean over 5 seeds, cell text "mean ± std"
    fig_step3_dots              per column: mean (point), ± std (bar), every seed (small dots)
final  (needs the step-3 tables and test/<config>/ from evaluate_v2.py)
    fig_window_vs_rmse          step 2, seed 42: W vs overall validation RMSE, L1 and L2 lines
    fig_params_vs_rmse          step 3: parameters vs validation RMSE, mean ± std over 5 seeds
    fig_test_per_temp           final config: test RMSE per temperature and cycle, mean ± std
    fig_us06_soc                final config, seed 42: US06 reference vs estimated SOC
Every figure writes a section of figure_notes.md (figs/notes/<figure>.md); figure_notes.md is
rebuilt from all sections in a fixed order. All text is English.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utf8_stdio  # noqa: E402
utf8_stdio.setup()  # UTF-8 console output on Windows too
import v2_analysis as A  # noqa: E402
import v2_common as V  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SINGLE_W, DOUBLE_W = 3.5, 7.16  # IEEE column widths (inches)
# Categorical slots (validated: adjacent CVD delta E >= 9, normal-vision >= 27); H also has its own line style
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
H_STYLE = {32: (BLUE, "-", "o"), 64: (ORANGE, "--", "s"), 128: (AQUA, "-.", "D")}
CYCLE_STYLE = {"UDDS": (BLUE, "o"), "HWFET": (ORANGE, "s"), "LA92": (AQUA, "D"), "US06": (YELLOW, "^")}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ_BLUE = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CPU_NOTE = ("CPU inference time was measured on the laptop CPU (Intel i7-12700H, single thread, batch 1); "
            "it shows the relative trend only and is not the absolute time on the MCU.")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150, "axes.unicode_minus": True,
})

NOTE_ORDER = ["fig_step2_heatmap_colnorm", "fig_step2_heatmap_global", "fig_step2_heatmap_colnorm_highlight",
              "fig_step2_heatmap_global_highlight", "step2_ranking", "fig_step3_heatmap", "fig_step3_dots",
              "step3_tables", "window_saturation", "fig_window_vs_rmse", "fig_params_vs_rmse",
              "fig_test_per_temp", "fig_us06_soc"]


def save(fig, paths: V.Paths, stem: str) -> list[str]:
    paths.figs.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = paths.figs / f"{stem}.{ext}"
        tmp = paths.figs / f".{stem}.tmp.{ext}"
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.03)
        tmp.replace(path)
        out.append(f"figs/{path.name}")
    plt.close(fig)
    return out


def write_note(paths: V.Paths, key: str, text: str) -> None:
    V.write_text_atomic(paths.figs / "notes" / f"{key}.md", text.rstrip() + "\n")
    parts = [f"# Figure and table notes ({paths.exp})", "",
             f"Rebuilt by src/plot_v2.py on {datetime.now():%Y-%m-%d %H:%M}. Errors are in SOC percentage "
             f"points (%). Validation and test metrics use only rows with in-sequence index >= {V.EVAL_START} "
             "of every file, so every window length is scored on the same points. Standard deviations are "
             "sample standard deviations (ddof = 1). No statistical significance test was done.", ""]
    notes_dir = paths.figs / "notes"
    keys = [k for k in NOTE_ORDER if (notes_dir / f"{k}.md").exists()]
    keys += sorted(p.stem for p in notes_dir.glob("*.md") if p.stem not in NOTE_ORDER)
    for k in keys:
        parts += [(notes_dir / f"{k}.md").read_text(encoding="utf-8").rstrip(), ""]
    V.write_text_atomic(paths.notes, "\n".join(parts) + "\n")


def text_color(frac: float) -> str:
    return "white" if frac > 0.55 else INK


# ---------------------------------------------------------------------------
# Step 2 heatmaps
# ---------------------------------------------------------------------------
def heatmap(ax, mat: np.ndarray, labels_txt: list[list[str]], colnorm: bool):
    n_r, n_c = mat.shape
    colors = np.zeros_like(mat)
    if colnorm:
        for j in range(n_c):
            col = mat[:, j]
            lo, hi = np.nanmin(col), np.nanmax(col)
            colors[:, j] = 0.0 if hi == lo else (col - lo) / (hi - lo)
        norm = Normalize(0, 1)
    else:
        lo, hi = np.nanmin(mat), np.nanmax(mat)
        colors = mat
        norm = Normalize(lo, hi)
    im = ax.imshow(colors, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(n_r):
        for j in range(n_c):
            frac = norm(colors[i, j])
            ax.text(j, i, labels_txt[i][j], ha="center", va="center", fontsize=6.3, color=text_color(frac))
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2-pt surface gaps between cells
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    return im


def plot_step2(paths: V.Paths, highlight: list[str]) -> None:
    rank = pd.read_csv(paths.tables / "step2_ranking.csv", encoding="utf-8")
    rank["_k"] = rank["config"].map(V.config_sort_key)
    df = rank.sort_values("_k").reset_index(drop=True)
    unknown = [h for h in highlight if h not in set(df["config"])]
    if unknown:
        raise SystemExit(f"--highlight configs not in the step-2 table: {unknown}")
    cols = ["val_rmse_Overall"] + [f"val_rmse_{f}" for f in V.FOLDERS]
    mat = df[cols].to_numpy(dtype=float)
    txt = [[f"{v:.3f}" for v in row] for row in mat]
    ylabels = [f"{c}  ({int(p):,})" for c, p in zip(df["config"], df["n_params"])]
    suffix = "_highlight" if highlight else ""
    n_l1 = int((df["L"] == 1).sum())
    for colnorm, stem in ((True, "fig_step2_heatmap_colnorm"), (False, "fig_step2_heatmap_global")):
        fig, ax = plt.subplots(figsize=(SINGLE_W + 1.3, 0.75 + 0.2 * len(df)))
        im = heatmap(ax, mat, txt, colnorm)
        ax.set_xticks(range(len(cols)), [A.COL_LABEL[c] for c in A.COLS])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(df)), ylabels, fontsize=6.8)
        ax.axhline(n_l1 - 0.5, color=INK, lw=1.2)
        for i, c in enumerate(df["config"]):
            if c in highlight:
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False, ec=ORANGE, lw=1.8, zorder=5))
                ax.get_yticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color(INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2)
        if colnorm:
            cb.set_ticks([0, 1], labels=["lowest\nin column", "highest\nin column"])
            cb.set_label("Colour: position inside each column")
        else:
            cb.set_label("Validation RMSE (%)")
        ax.set_xlabel(f"Validation RMSE (%), seed {V.STEP2_SEED}, one training run per config; "
                      f"rows >= {V.EVAL_START}\nRow label: config (trainable parameters)", fontsize=6.8)
        files = save(fig, paths, stem + suffix)
        print(f"Wrote {', '.join(files)}")
        what = ("normalised separately inside every column, so the colour shows the rank of a "
                "config inside that column; the numbers are the absolute RMSE values"
                if colnorm else "one colour scale for the whole figure (absolute RMSE)")
        note = [f"## {stem + suffix}", "", f"Files: `{files[0]}`, `{files[1]}`.", "",
                f"Step-2 validation RMSE (%) of all {len(df)} configs, **seed {V.STEP2_SEED}, one training run per "
                f"config**. Rows: L1 configs, then L2 configs (separated by a line), each sorted by W, then H; "
                "the number in brackets is the count of trainable parameters. Columns: Overall (all validation "
                f"points) and each validation temperature. Cell text: RMSE with three decimals. Colour: {what}.",
                "", "The differences between temperatures are much larger than the differences between configs, "
                "which is why both colourings are given."]
        if highlight:
            note += ["", f"Outlined rows (the {len(highlight)} configs chosen by the author for step 3): " + ", ".join(highlight) + "."]
        write_note(paths, stem + suffix, "\n".join(note))
    best = rank.sort_values("val_rmse_Overall").iloc[0]
    write_note(paths, "step2_ranking", "\n".join([
        "## step2_ranking.csv / step2_summary.txt", "",
        f"Files: `tables/step2_ranking.csv`, `tables/step2_summary.txt`. All {len(rank)} step-2 configs sorted by "
        f"overall validation RMSE (seed {V.STEP2_SEED}, single run each), with per-temperature RMSE, the worst "
        "temperature, trainable parameters and the initial delay W / f_s (f_s = 1 Hz).",
        "", f"- Lowest overall RMSE in this single-seed table: {best['config']} ({best['val_rmse_Overall']:.3f}%). "
        "With one seed per config, small gaps between ranks may be run-to-run variation."]))


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def load_step3(paths: V.Paths):
    s = pd.read_csv(paths.tables / "step3_summary.csv", encoding="utf-8")
    s["_k"] = s["config"].map(V.config_sort_key)
    s = s.sort_values("_k").reset_index(drop=True)
    per = pd.read_csv(paths.tables / "step3_per_seed.csv", encoding="utf-8")
    return s, per


def plot_step3(paths: V.Paths) -> None:
    s, per = load_step3(paths)
    mat = s[[f"rmse_{c}_mean" for c in A.COLS]].to_numpy(dtype=float)
    txt = [[A.pm(s.loc[i, f"rmse_{c}_mean"], s.loc[i, f"rmse_{c}_std"]) for c in A.COLS] for i in range(len(s))]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 0.9 + 0.3 * len(s)))
    im = heatmap(ax, mat, txt, colnorm=False)
    ax.set_xticks(range(len(A.COLS)), [A.COL_LABEL[c] for c in A.COLS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(s)), [f"{c}  ({int(p):,})" for c, p in zip(s["config"], s["n_params"])])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Mean validation RMSE (%)")
    cb.ax.tick_params(labelsize=6.5, length=2)
    ax.set_xlabel(f"Validation RMSE (%), mean ± sample std over seeds {', '.join(map(str, V.ALL_SEEDS))}; "
                  f"rows >= {V.EVAL_START}", fontsize=7)
    files = save(fig, paths, "fig_step3_heatmap")
    print(f"Wrote {', '.join(files)}")
    write_note(paths, "fig_step3_heatmap", "\n".join([
        "## fig_step3_heatmap", "", f"Files: `{files[0]}`, `{files[1]}`.", "",
        f"The {len(s)} step-3 configs (rows, in fixed L, W, H order; trainable parameters in brackets) x "
        "Overall and the four validation temperatures. Colour = mean validation RMSE over the five seeds "
        f"({', '.join(map(str, V.ALL_SEEDS))}) on one scale; cell text = mean ± sample std (ddof = 1)."]))

    fig, axes = plt.subplots(1, len(A.COLS), figsize=(DOUBLE_W, 0.8 + 0.28 * len(s)), sharey=True)
    ypos = np.arange(len(s))[::-1]
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, A.COLS):
        for y, (_, r) in zip(ypos, s.iterrows()):
            seeds = per[per["config"] == r["config"]][f"val_rmse_{c}"].to_numpy()
            ax.scatter(seeds, y + rng.uniform(-0.18, 0.18, len(seeds)), s=7, color=MUTED, lw=0, zorder=2)
            ax.errorbar(r[f"rmse_{c}_mean"], y, xerr=r[f"rmse_{c}_std"], fmt="o", color=BLUE, ms=4.5,
                        mec="white", mew=0.6, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.set_title(A.COL_LABEL[c])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks(ypos, list(s["config"]))
    axes[0].set_ylim(-0.6, len(s) - 0.4)
    fig.supxlabel(f"Validation RMSE (%), rows >= {V.EVAL_START}", fontsize=8, y=0.02)
    handles = [Line2D([], [], color=BLUE, marker="o", ls="-", ms=4.5, label="Mean ± std (5 seeds)"),
               Line2D([], [], color=MUTED, marker="o", ls="none", ms=3, label="Single seed")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=0.6)
    files = save(fig, paths, "fig_step3_dots")
    print(f"Wrote {', '.join(files)}")
    write_note(paths, "fig_step3_dots", "\n".join([
        "## fig_step3_dots", "", f"Files: `{files[0]}`, `{files[1]}`.", "",
        f"One panel per column (Overall, then each validation temperature; x axes are independent). For each "
        "config: blue point = mean over the five seeds, bar = ± sample std, grey dots = the five single-seed "
        "values (small vertical jitter only for visibility)."]))

    tab = ["## step3_summary.csv / step3_per_seed.csv", "",
           "Files: `tables/step3_summary.csv`, `tables/step3_per_seed.csv`, `tables/step3_summary.txt`. Per config: "
           "overall and per-temperature validation RMSE mean and sample std over the five seeds, every seed's "
           "overall RMSE, trainable parameters, CPU inference time (`cpu_infer_ms_mean`, `cpu_infer_ms_std`) and "
           "initial delay W / f_s in seconds (f_s = 1 Hz).", "", CPU_NOTE, ""]
    for _, r in s.iterrows():
        cpu = "n/a" if pd.isna(r["cpu_infer_ms_mean"]) else f"{r['cpu_infer_ms_mean']:.3f} ms"
        tab.append(f"- {r['config']}: overall {A.pm(r['rmse_Overall_mean'], r['rmse_Overall_std'])}%, "
                   f"{int(r['n_params']):,} parameters, delay {r['latency_s']:.0f} s, CPU {cpu}.")
    write_note(paths, "step3_tables", "\n".join(tab))

    sat_path = paths.tables / "window_saturation.csv"
    if sat_path.exists():
        sat = pd.read_csv(sat_path, encoding="utf-8")
        L = ["## window_saturation.csv", "", "File: `tables/window_saturation.csv`. Step-3 configs that share L and H "
             "but differ in W, compared between adjacent windows: absolute improvement (mean RMSE of the shorter "
             "window minus that of the longer window), relative improvement, and the pooled seed-to-seed sample "
             "std sqrt((s1^2 + s2^2) / 2). No significance test was done; a ratio below 1 means that the "
             "improvement was small relative to the observed seed-to-seed variation.", ""]
        for _, r in sat[sat["column"] == "Overall"].iterrows():
            L.append(f"- L{r['L']} H{r['H']}, W {r['W_from']} -> {r['W_to']}: improvement "
                     f"{r['abs_improvement']:+.3f} points ({r['rel_improvement_pct']:+.1f} %), pooled seed std "
                     f"{r['pooled_seed_std']:.3f}, ratio {r['improvement_over_pooled_std']:.2f}.")
        write_note(paths, "window_saturation", "\n".join(L))
    else:
        write_note(paths, "window_saturation", "## window_saturation.csv\n\nNot produced: no two step-3 configs "
                   "share L and H with different W.")


# ---------------------------------------------------------------------------
# Final figures
# ---------------------------------------------------------------------------
def plot_window(paths: V.Paths) -> None:
    rank = pd.read_csv(paths.tables / "step2_ranking.csv", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
    for h in V.GRID_H:
        color, ls, _ = H_STYLE[h]
        for l, marker, mfc, lstyle in ((1, "o", color, "-"), (2, "^", "white", "--")):
            d = rank[(rank["H"] == h) & (rank["L"] == l)].sort_values("W")
            if d.empty:
                continue
            ax.plot(d["W"], d["val_rmse_Overall"], color=color, ls=lstyle, lw=1.3, marker=marker, ms=4.5,
                    mfc=mfc, mec=color if l == 2 else "white", mew=1.0 if l == 2 else 0.6, zorder=3)
    ax.set_xticks(sorted(rank["W"].unique()))
    ax.set_xlabel("Window length $W$ (s)")
    ax.set_ylabel("Validation RMSE (%)")
    handles = [Line2D([], [], color=H_STYLE[h][0], lw=1.6, label=f"$H$ = {h}") for h in V.GRID_H]
    handles += [Line2D([], [], color=INK2, ls="-", marker="o", ms=4.5, label="$L$ = 1"),
                Line2D([], [], color=INK2, ls="--", marker="^", ms=4.5, mfc="white", label="$L$ = 2")]
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand",
              borderaxespad=0, handlelength=2.2, columnspacing=0.8)
    files = save(fig, paths, "fig_window_vs_rmse")
    print(f"Wrote {', '.join(files)}")
    write_note(paths, "fig_window_vs_rmse", "\n".join([
        "## fig_window_vs_rmse", "", f"Files: `{files[0]}`, `{files[1]}` (single column).", "",
        f"Overall validation RMSE versus window length W for all {len(rank)} step-2 configs. Colour = hidden "
        "size H; L = 1: solid lines with filled circles; L = 2: dashed lines with open triangles. "
        f"**Every point is a single training run (seed {V.STEP2_SEED}).**"]))


def plot_params(paths: V.Paths, final: str) -> None:
    s, per = load_step3(paths)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.5))
    for _, r in s.iterrows():
        seeds = per[per["config"] == r["config"]]["val_rmse_Overall"]
        ax.scatter(np.full(len(seeds), r["n_params"]), seeds, s=7, color=MUTED, lw=0, zorder=2)
        is_final = r["config"] == final
        ax.errorbar(r["n_params"], r["rmse_Overall_mean"], yerr=r["rmse_Overall_std"],
                    fmt="D" if is_final else "o", color=ORANGE if is_final else BLUE, ms=5, mec="white",
                    mew=0.6, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.annotate(r["config"].replace("_", " "), (r["n_params"], r["rmse_Overall_mean"]), xytext=(5, 2),
                    textcoords="offset points", fontsize=6.3, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(s["n_params"].min() / 1.6, s["n_params"].max() * 2.6)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Validation RMSE (%)")
    handles = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=5, label="Mean ± std (5 seeds)"),
               Line2D([], [], color=ORANGE, marker="D", ls="none", ms=5, label="Final config"),
               Line2D([], [], color=MUTED, marker="o", ls="none", ms=3, label="Single seed")]
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0,
              columnspacing=1.0, handlelength=1.2)
    files = save(fig, paths, "fig_params_vs_rmse")
    print(f"Wrote {', '.join(files)}")
    write_note(paths, "fig_params_vs_rmse", "\n".join([
        "## fig_params_vs_rmse", "", f"Files: `{files[0]}`, `{files[1]}` (single column).", "",
        f"Trainable parameters (log scale) versus overall validation RMSE of the {len(s)} step-3 configs: "
        "point = mean over seeds 42-46, bar = ± sample std, grey dots = single seeds. The orange diamond is "
        f"the final config chosen by the author: {final}."]))


def plot_test(paths: V.Paths, final: str) -> None:
    tdir = paths.test / final
    res = pd.read_csv(tdir / "test_results.csv", encoding="utf-8")
    res = res[res["cycle"] != "all"]
    groups = [("seen", res)]
    if (tdir / "unseen_temp_results.csv").exists():
        u = pd.read_csv(tdir / "unseen_temp_results.csv", encoding="utf-8")
        groups.append(("unseen", u[u["cycle"] != "all"]))
    temps = []
    for kind, g in groups:
        for t in sorted(g["temp_c"].unique()):
            temps.append((kind, int(t)))
    temps.sort(key=lambda x: x[1])
    xpos = {t: i for i, t in enumerate(temps)}
    fig, ax = plt.subplots(figsize=(DOUBLE_W * 0.62, 2.5))
    present = list(dict.fromkeys(pd.concat([g for _, g in groups])["cycle"]))
    cycles = [c for c in CYCLE_STYLE if c in present] + [c for c in present if c not in CYCLE_STYLE]
    style = {c: CYCLE_STYLE.get(c, (INK2, "v")) for c in cycles}
    off = {c: (k - (len(cycles) - 1) / 2) * 0.16 for k, c in enumerate(cycles)}
    for kind, g in groups:
        for _, r in g.iterrows():
            color, marker = style[r["cycle"]]
            x = xpos[(kind, int(r["temp_c"]))] + off[r["cycle"]]
            ax.errorbar(x, r["rmse_pct_mean"], yerr=r["rmse_pct_std"], fmt=marker, color=color, ms=4.5,
                        mec="white", mew=0.5, elinewidth=0.9, capsize=2, zorder=3,
                        mfc=color if kind == "seen" else "white")
    unseen_x = [xpos[t] for t in temps if t[0] == "unseen"]
    if unseen_x:
        ax.axvspan(min(unseen_x) - 0.5, max(unseen_x) + 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.text(np.mean(unseen_x), 1.0, "not seen in training", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(temps)), [f"{t} °C" if t >= 0 else f"\u2212{-t} °C" for _, t in temps])
    ax.set_xlabel("Test temperature")
    ax.set_ylabel("Test RMSE (%)")
    ax.legend(handles=[Line2D([], [], color=style[c][0], marker=style[c][1], ls="none", ms=4.5, label=c)
                       for c in cycles], ncol=len(cycles), loc="lower center", bbox_to_anchor=(0.5, 1.06))
    files = save(fig, paths, "fig_test_per_temp")
    print(f"Wrote {', '.join(files)}")
    avg = pd.read_csv(tdir / "test_results.csv", encoding="utf-8")
    avg = avg[avg["cycle"] == "all"].iloc[0]
    L = ["## fig_test_per_temp", "", f"Files: `{files[0]}`, `{files[1]}`.", "",
         f"Test RMSE of the final config {final} per temperature (x) and drive cycle (colour and marker): "
         f"mean ± sample std over seeds 42-46, rows >= {V.EVAL_START}. 25 °C has no HWFET file."
         + (" Open markers on the grey band: −20 °C and −10 °C, temperatures never used in training "
            "(supplementary test only)." if unseen_x else ""), "",
         f"- Average over the {int(avg['n_files'])} 0-40 °C test files: RMSE "
         f"{A.pm(avg['rmse_pct_mean'], avg['rmse_pct_std'], 2)}%, MAE {A.pm(avg['mae_pct_mean'], avg['mae_pct_std'], 2)}%.",
         "- Capacity marks in the test tables: a = full capacity back-calculated from the discharge cutoff "
         "capacity; b = full capacity borrowed from batch 576 Cap_1C with aging correction (label known to be biased)."]
    write_note(paths, "fig_test_per_temp", "\n".join(L))


def plot_us06(paths: V.Paths, final: str, cycle: str) -> None:
    pdir = paths.test / final / "predictions_seed42"
    avail = [f for f in V.FOLDERS if (pdir / f"{f}_{cycle}.csv").exists()]
    if not avail:
        print(f"fig_us06_soc skipped: no {cycle} predictions in {pdir}")
        return
    fig, axes = plt.subplots(2, len(avail), figsize=(DOUBLE_W, 3.1), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    err_lim, stats = 0.0, {}
    for j, f in enumerate(avail):
        d = pd.read_csv(pdir / f"{f}_{cycle}.csv", encoding="utf-8")
        t = d["time_s"] / 60.0
        err = (d["soc_pred"] - d["soc_true"]) * 100
        err_lim = max(err_lim, float(np.abs(err).max()))
        a1, a2 = axes[0, j], axes[1, j]
        a1.plot(t, d["soc_true"] * 100, color=BLUE, lw=1.3, label="Reference SOC")
        a1.plot(t, d["soc_pred"] * 100, color=ORANGE, lw=0.9, ls="--", label="Estimated SOC")
        a2.plot(t, err, color=INK2, lw=0.7)
        a2.axhline(0, color=INK2, lw=0.5, alpha=0.6)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        stats[f] = (rmse, float(np.mean(np.abs(err))), float(np.abs(err).max()))
        a1.set_title(f"{cycle}, {V.temp_label(f)}")
        a1.text(0.98, 0.95, f"RMSE {rmse:.2f}% (seed 42)", transform=a1.transAxes, ha="right", va="top",
                fontsize=6.5, color=INK)
        a2.set_xlabel("Time (min)")
        if j == 0:
            a1.set_ylabel("SOC (%)")
            a2.set_ylabel("Error (%)")
    for a2 in axes[1]:
        a2.set_ylim(-1.1 * err_lim, 1.1 * err_lim)
    axes[0, 0].legend(loc="lower left")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    files = save(fig, paths, "fig_us06_soc")
    print(f"Wrote {', '.join(files)}")
    L = ["## fig_us06_soc", "", f"Files: `{files[0]}`, `{files[1]}` (double column).", "",
         f"Final config {final}, {cycle}, one column per temperature. Top: reference SOC (solid) and estimated "
         "SOC (dashed); bottom: error (estimated minus reference, SOC %). Time starts at the first scored row "
         f"(index {V.EVAL_START}). **The RMSE printed in each panel comes from a single seed (42)**; the "
         "mean ± std over five seeds is in the test tables.", ""]
    for f, (r, m, mx) in stats.items():
        L.append(f"- {V.temp_label(f)}: seed-42 RMSE {r:.2f}%, MAE {m:.2f}%, max |error| {mx:.2f}%.")
    if "10degC" in stats:
        L.append("- The 10 °C US06 label uses capacity mark b (borrowed capacity with aging correction; known bias).")
    write_note(paths, "fig_us06_soc", "\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description="v2 figures and figure_notes.md")
    p.add_argument("--exp", default=V.FORMAL_EXP)
    p.add_argument("--out-root", type=Path, default=V.REPO_ROOT)
    sub = p.add_subparsers(dest="what", required=True)
    s2 = sub.add_parser("step2")
    s2.add_argument("--highlight", default="", help="comma-separated configs to outline")
    sub.add_parser("step3")
    fi = sub.add_parser("final")
    fi.add_argument("--config", required=True, help="the final config chosen by the author")
    fi.add_argument("--trace-cycle", default="US06", help=argparse.SUPPRESS)
    args = p.parse_args()
    paths = V.Paths(args.out_root, args.exp)
    if args.what == "step2":
        plot_step2(paths, [c.strip() for c in args.highlight.split(",") if c.strip()])
    elif args.what == "step3":
        plot_step3(paths)
    else:
        V.parse_config(args.config)
        plot_window(paths)
        plot_params(paths, args.config)
        plot_test(paths, args.config)
        plot_us06(paths, args.config, args.trace_cycle)
    print(f"Updated {paths.rel(paths.notes)}")


if __name__ == "__main__":
    main()
