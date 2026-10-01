"""Step 5: paper figures (IEEE format) and figure_notes.md.

Reads the CSV / JSON / prediction files written by the earlier steps and writes, to
results/<name>/figs/, each figure as PNG (300 dpi) and PDF (fonts embedded as TrueType):
    fig1_window_vs_rmse   (single column) validation RMSE vs W, seed 42; one line per H (L = 1),
                          the three W = 240, L = 2 configs as separate markers
    fig2_params_vs_rmse   (single column) trainable parameters (log x) vs validation RMSE,
                          mean +- std over seeds, multi-seed candidates only, threshold line,
                          final config marked
    fig3_temp_heatmap     (single column) validation RMSE per temperature for every L = 1 config (seed 42)
    fig4_us06_traces      (double column) final config, seed 42, US06 at 0 / 25 / 40 degC:
                          reference vs predicted SOC (top) and error (bottom), time in minutes
and results/<name>/figure_notes.md (in Chinese) with what each figure shows and its key numbers.
A figure whose inputs do not exist yet is skipped with a message. Training time is never plotted.

The "final config" (figures 2 and 4) is --config if given; otherwise the only config evaluated
with evaluate_paper.py; otherwise the recommendation in selected_config.json.

Usage (from the repository root):
    python src/plot_paper_figs.py
    python src/plot_paper_figs.py --config W240_H32_L1
    python src/plot_paper_figs.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SINGLE_W, DOUBLE_W = 3.5, 7.16  # IEEE column widths (inches)
# Colour-blind-safe categorical slots (validated all-pairs, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
H_STYLE = {32: (BLUE, "-"), 64: (ORANGE, "--"), 128: (AQUA, "-.")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

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
    "savefig.dpi": 300, "figure.dpi": 150,
})


def save(fig, figs_dir: Path, stem: str) -> list[str]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = figs_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        out.append(path.name)
    plt.close(fig)
    return out


def degc(folder: str) -> str:
    return f"{C.temp_c(folder)} °C"


# ---------------------------------------------------------------------------
def fig1(sweep: pd.DataFrame, figs: Path) -> dict:
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.35))
    l1 = sweep[sweep["num_layers"] == 1]
    l2 = sweep[sweep["num_layers"] == 2]
    hs = sorted(sweep["hidden_size"].unique())
    for h in hs:
        color, ls = H_STYLE.get(h, (INK2, "-"))
        d = l1[l1["hidden_size"] == h].sort_values("window")
        if not d.empty:
            ax.plot(d["window"], d["val_rmse_pct"], color=color, ls=ls, lw=1.4, marker="o", ms=4.5,
                    mec="white", mew=0.6, zorder=3)
        d2 = l2[l2["hidden_size"] == h]
        if not d2.empty:
            ax.plot(d2["window"], d2["val_rmse_pct"], ls="none", marker="^", ms=6, mfc="white",
                    mec=color, mew=1.2, zorder=4)
    windows = sorted(sweep["window"].unique())
    ax.set_xticks(windows)
    ax.set_xlabel("Window length $W$ (s)")
    ax.set_ylabel("Validation RMSE (%)")
    handles = [Line2D([], [], color=H_STYLE.get(h, (INK2, "-"))[0], ls=H_STYLE.get(h, (INK2, "-"))[1],
                      lw=1.4, label=f"$H$ = {h}") for h in hs]
    handles.append(Line2D([], [], color=INK2, ls="none", marker="o", ms=4.5, label="$L$ = 1"))
    if not l2.empty:
        handles.append(Line2D([], [], color=INK2, ls="none", marker="^", ms=6, mfc="white", mew=1.2,
                              label=f"$L$ = 2 ($W$ = {int(l2['window'].iloc[0])})"))
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2),
              mode="expand", borderaxespad=0, handlelength=2.2, columnspacing=0.8)
    files = save(fig, figs, "fig1_window_vs_rmse")
    best = sweep.sort_values("val_rmse_pct").iloc[0]
    per_h = {int(h): l1[l1["hidden_size"] == h].sort_values("val_rmse_pct").iloc[0]
             for h in hs if not l1[l1["hidden_size"] == h].empty}
    return {"files": files, "best": best, "per_h": per_h, "l2": l2, "l1": l1,
            "max_w": max(windows)}


def place_labels(ax, fig, pts: list[tuple[float, float, str]]) -> None:
    """Direct labels next to points, trying a few offsets so that labels do not overlap."""
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
               (-5, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top")]
    for x, y, text in sorted(pts):
        for dx, dy, ha, va in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=6.5, color=INK, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
            bb = t.get_window_extent(renderer).expanded(1.05, 1.1)
            if not any(bb.overlaps(o) for o in placed) and ax.bbox.contains(*bb.p0) and ax.bbox.contains(*bb.p1):
                placed.append(bb)
                break
            t.remove()
        else:
            ax.annotate(text, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK)


def fig2(multi: pd.DataFrame, final_cfg: str | None, figs: Path) -> dict:
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.45))
    best = multi.sort_values("val_rmse_pct_mean").iloc[0]
    thr = float(best["val_rmse_pct_mean"] + best["val_rmse_pct_std"])
    ax.axhline(thr, color=INK2, ls=(0, (4, 2)), lw=0.9, zorder=1)
    is_final = multi["config"] == final_cfg
    for sel, color, marker, ms in ((~is_final, BLUE, "o", 5), (is_final, ORANGE, "D", 5)):
        d = multi[sel]
        if d.empty:
            continue
        ax.errorbar(d["n_params"], d["val_rmse_pct_mean"], yerr=d["val_rmse_pct_std"], fmt=marker,
                    color=color, ms=ms, mec="white", mew=0.6, elinewidth=0.9, capsize=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Validation RMSE (%)")
    lo, hi = multi["n_params"].min(), multi["n_params"].max()
    ax.set_xlim(lo / 1.6, hi * 1.9)
    fig.canvas.draw()
    place_labels(ax, fig, [(float(r["n_params"]), float(r["val_rmse_pct_mean"]), r["config"].replace("_", " "))
                           for _, r in multi.iterrows()])
    handles = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=5, label="Candidate (mean ± std)"),
               Line2D([], [], color=INK2, ls=(0, (4, 2)), lw=0.9, label="Threshold")]
    if is_final.any():
        handles.insert(1, Line2D([], [], color=ORANGE, marker="D", ls="none", ms=5, label="Selected"))
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              borderaxespad=0, handlelength=1.8, columnspacing=1.4)
    files = save(fig, figs, "fig2_params_vs_rmse")
    return {"files": files, "best": best, "threshold": thr, "final": final_cfg}


def fig3(sweep: pd.DataFrame, folders: list[str], figs: Path) -> dict:
    l1 = sweep[sweep["num_layers"] == 1].sort_values(["window", "hidden_size"])
    cols = [f for f in folders if f"val_rmse_pct_{f}" in l1.columns]
    mat = l1[[f"val_rmse_pct_{f}" for f in cols]].to_numpy(dtype=float)
    rows = [f"W{int(w)} H{int(h)}" for w, h in zip(l1["window"], l1["hidden_size"])]
    fig, ax = plt.subplots(figsize=(SINGLE_W, 0.6 + 0.2 * len(rows)))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            frac = 0.0 if vmax == vmin else (mat[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if frac > 0.55 else INK)
    ax.set_xticks(range(len(cols)), [degc(f) for f in cols])
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Validation temperature")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Validation RMSE (%)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    files = save(fig, figs, "fig3_temp_heatmap")
    return {"files": files, "matrix": pd.DataFrame(mat, index=rows, columns=cols)}


def fig4(test_dir: Path, folders_wanted: list[str], figs: Path) -> dict:
    pred_dir = test_dir / "predictions_seed42"
    avail = [f for f in folders_wanted if (pred_dir / f"{f}_US06.csv").exists()]
    if not avail:
        return {}
    fig, axes = plt.subplots(2, len(avail), figsize=(DOUBLE_W, 3.1), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    stats = {}
    err_lim = 0.0
    for j, f in enumerate(avail):
        d = pd.read_csv(pred_dir / f"{f}_US06.csv")
        t = d["time_s"] / 60.0
        err = (d["soc_pred"] - d["soc_true"]) * 100
        err_lim = max(err_lim, float(np.abs(err).max()))
        a1, a2 = axes[0, j], axes[1, j]
        a1.plot(t, d["soc_true"] * 100, color=BLUE, lw=1.3, label="Reference SOC")
        a1.plot(t, d["soc_pred"] * 100, color=ORANGE, lw=0.9, ls="--", label="Predicted SOC")
        a2.plot(t, err, color=INK2, lw=0.7)
        a2.axhline(0, color=INK2, lw=0.5, alpha=0.6)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        stats[f] = {"rmse": rmse, "mae": mae, "max_abs": float(np.abs(err).max()),
                    "minutes": float(t.iloc[-1] - t.iloc[0])}
        a1.set_title(f"US06, {degc(f)}")
        a1.text(0.98, 0.95, f"RMSE {rmse:.2f}%", transform=a1.transAxes, ha="right", va="top",
                fontsize=7, color=INK)
        a2.set_xlabel("Time (min)")
        if j == 0:
            a1.set_ylabel("SOC (%)")
            a2.set_ylabel("Error (%)")
    for a2 in axes[1]:
        a2.set_ylim(-1.1 * err_lim, 1.1 * err_lim)
    axes[0, 0].legend(loc="lower left")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    files = save(fig, figs, "fig4_us06_traces")
    return {"files": files, "stats": stats}


# ---------------------------------------------------------------------------
def resolve_final(paths: C.PaperPaths, arg: str | None) -> tuple[str | None, str]:
    if arg:
        return arg, "--config"
    evaluated = sorted(p.name for p in paths.test.glob("W*_H*_L*") if (p / "test_results.csv").exists())
    if len(evaluated) == 1:
        return evaluated[0], "the only config evaluated with evaluate_paper.py"
    if paths.selected.exists():
        return C.read_json(paths.selected)["recommended_config"], "selected_config.json"
    return None, "none (run select_config.py / evaluate_paper.py, or pass --config)"


def notes_md(ctx: C.Context, r1, r2, r3, r4, final_cfg, final_src, test_res) -> str:
    L = [f"# 论文图表说明（{ctx.paths.name}）", "",
         f"由 `src/plot_paper_figs.py` 于 {datetime.now():%Y-%m-%d %H:%M} 生成。所有误差单位是 SOC 百分点（%），"
         f"验证和测试指标都只计算每条序列第 {C.EVAL_START} 行（300 s）之后的点。"
         "标准差都是样本标准差（ddof = 1）。图里没有任何训练时间。", ""]
    if ctx.smoke:
        L += ["> 注意：这是 smoke test 的输出（只用 25 °C、2 组配置、2 个 seed、每组 2 轮），数字没有实际意义。", ""]
    if r1:
        b = r1["best"]
        L += ["## 图 1：窗口长度 W 与验证 RMSE（seed 42）", "",
              f"文件：`figs/{r1['files'][0]}`、`figs/{r1['files'][1]}`（单栏）。", "",
              "横轴是窗口长度 W，纵轴是验证集 RMSE。L = 1 时每个隐藏单元数 H 画一条线（颜色和线型区分 H，圆点），"
              "W = 240、L = 2 的配置用空心三角形单独标出（颜色同样表示 H）。", "",
              f"- 验证 RMSE 最低的配置是 **{b['config']}**：RMSE {b['val_rmse_pct']:.3f}%，MAE {b['val_mae_pct']:.3f}%。"]
        for h, r in r1["per_h"].items():
            L.append(f"- H = {h}（L = 1）时最佳窗口为 W = {int(r['window'])}，RMSE {r['val_rmse_pct']:.3f}%。")
        for _, r in r1["l2"].iterrows():
            same = r1["l1"][(r1["l1"]["window"] == r["window"]) & (r1["l1"]["hidden_size"] == r["hidden_size"])]
            cmp = f"（同 W、同 H 的 L = 1 为 {same['val_rmse_pct'].iloc[0]:.3f}%）" if not same.empty else ""
            L.append(f"- {r['config']}：RMSE {r['val_rmse_pct']:.3f}%{cmp}。")
        if int(b["window"]) == r1["max_w"]:
            L.append(f"- **提醒：最佳配置的 W = {r1['max_w']} 在网格边缘，W 可能还没饱和。**")
        L.append("")
    if r2:
        b = r2["best"]
        L += ["## 图 2：可训练参数量与验证 RMSE（多 seed 候选）", "",
              f"文件：`figs/{r2['files'][0]}`、`figs/{r2['files'][1]}`（单栏）。", "",
              f"横轴是可训练参数量（对数刻度），纵轴是 {len(ctx.seeds)} 个 seed 的验证 RMSE 均值，误差棒为 ± 标准差；"
              "每个点旁标出配置名。水平虚线是选择门槛（最佳均值 + 其标准差），菱形是最终选中的配置。", "",
              f"- 均值最低：**{b['config']}**，{b['val_rmse_pct_mean']:.3f} ± {b['val_rmse_pct_std']:.3f}%，"
              f"参数量 {int(b['n_params']):,}。",
              f"- 门槛 = {b['val_rmse_pct_mean']:.3f} + {b['val_rmse_pct_std']:.3f} = **{r2['threshold']:.3f}%**。"]
        for _, r in r2["table"].iterrows():
            tag = "，不超过门槛" if r["val_rmse_pct_mean"] <= r2["threshold"] + 1e-12 else "，超过门槛"
            L.append(f"- {r['config']}：{r['val_rmse_pct_mean']:.3f} ± {r['val_rmse_pct_std']:.3f}%，"
                     f"参数量 {int(r['n_params']):,}{tag}。")
        L.append(f"- 图中标出的最终配置：**{final_cfg}**（来源：{final_src}）。")
        L.append("")
    if r3:
        m = r3["matrix"]
        L += ["## 图 3：各温度的验证 RMSE 热力图（L = 1，seed 42）", "",
              f"文件：`figs/{r3['files'][0]}`、`figs/{r3['files'][1]}`（单栏）。", "",
              "每行是一个 L = 1 的配置（按 W、H 排序），每列是一个验证温度（每个温度就是一个验证文件），"
              "颜色越深误差越大，格子里写的是 RMSE 数值。", ""]
        for col in m.columns:
            s = m[col]
            L.append(f"- {degc(col)}：RMSE 范围 {s.min():.2f}–{s.max():.2f}%，最低为 {s.idxmin()}，"
                     f"各配置平均 {s.mean():.2f}%。")
        if len(m.columns) > 1:
            hardest = m.mean().idxmax()
            L.append(f"- 平均来看最难的温度是 {degc(hardest)}（各配置平均 {m.mean().max():.2f}%）。")
        L.append("")
    if r4:
        L += ["## 图 4：最终配置在 US06 上的估计结果（seed 42）", "",
              f"文件：`figs/{r4['files'][0]}`、`figs/{r4['files'][1]}`（双栏）。", "",
              f"最终配置 **{final_cfg}**（seed 42）在 US06 工况下的结果，每列一个温度。上排是参考 SOC（实线）和预测 SOC"
              f"（虚线），下排是误差（预测 − 参考，%）。横轴单位为分钟，从第 {C.EVAL_START} 行（第 5 分钟）开始。", ""]
        for f, s in r4["stats"].items():
            line = (f"- {degc(f)}：seed 42 的 RMSE {s['rmse']:.2f}%，MAE {s['mae']:.2f}%，最大绝对误差 {s['max_abs']:.2f}%，"
                    f"时长约 {s['minutes']:.0f} 分钟")
            if test_res is not None:
                row = test_res[(test_res["temperature"] == C.temp_label(f)) & (test_res["cycle"] == "US06")]
                if not row.empty:
                    r = row.iloc[0]
                    line += f"；{len(ctx.seeds)} 个 seed 的 RMSE {r['rmse_pct_mean']:.2f} ± {r['rmse_pct_std']:.2f}%"
            L.append(line + "。")
        L += ["- 40 °C 的四个工况满容量由放电终止容量反推（表格脚注 †）。", ""]
    if test_res is not None:
        a = test_res[test_res["temperature"] == "all"]
        if not a.empty:
            a = a.iloc[0]
            L += ["## 测试集总览（供正文引用）", "",
                  f"- {a['file']}：RMSE {a['rmse_pct_mean']:.2f} ± {a['rmse_pct_std']:.2f}%，"
                  f"MAE {a['mae_pct_mean']:.2f} ± {a['mae_pct_std']:.2f}%（{len(ctx.seeds)} 个 seed）。"
                  "表格见 `test/<配置>/test_table.tex`。", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Paper step 5: figures (PNG 300 dpi + PDF) and figure_notes.md")
    C.add_common_args(p)
    p.add_argument("--config", default=None, help="final config for figures 2 and 4")
    args = p.parse_args()
    ctx = C.resolve(args)
    paths = ctx.paths
    final_cfg, final_src = resolve_final(paths, args.config)
    print(f"Final config: {final_cfg} ({final_src})")
    r1 = r2 = r3 = r4 = None
    test_res = None
    if paths.sweep_csv.exists():
        sweep = pd.read_csv(paths.sweep_csv)
        r1 = fig1(sweep, paths.figs)
        r3 = fig3(sweep, ctx.folders, paths.figs)
        print(f"Figure 1: {r1['files']}\nFigure 3: {r3['files']}")
    else:
        print(f"Figures 1 and 3 skipped: {paths.rel(paths.sweep_csv)} not found (run merge_results.py)")
    if paths.multiseed_csv.exists():
        multi = pd.read_csv(paths.multiseed_csv)
        r2 = fig2(multi, final_cfg, paths.figs)
        r2["table"] = multi.sort_values("val_rmse_pct_mean")
        print(f"Figure 2: {r2['files']}")
    else:
        print(f"Figure 2 skipped: {paths.rel(paths.multiseed_csv)} not found (finish step 3, run merge_results.py)")
    if final_cfg and (paths.test / final_cfg / "test_results.csv").exists():
        test_res = pd.read_csv(paths.test / final_cfg / "test_results.csv")
        r4 = fig4(paths.test / final_cfg, [f for f in ("0degC", "25degC", "40degC") if f in ctx.folders], paths.figs)
        print(f"Figure 4: {r4.get('files', 'no US06 predictions found')}")
    else:
        print(f"Figure 4 skipped: no test results for {final_cfg} (run evaluate_paper.py --config ...)")
    paths.notes.write_text(notes_md(ctx, r1, r2, r3, r4 or None, final_cfg, final_src, test_res),
                           encoding="utf-8", newline="\n")
    print(f"Wrote {paths.rel(paths.notes)} and figures in {paths.rel(paths.figs)}/")


if __name__ == "__main__":
    main()
