"""Report figures: PNG, Agg backend, dpi 120, each figure closed once saved."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 120
PAGE = 50
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SCATTER_CELLS = 20_000
TISSUE_PAGE = 12
TISSUE_CELLS = 50_000
REASON_COLORS = {"pass": "#bbbbbb", "sample_excluded": "#555555", "doublet": "#8e44ad", "high_mito": "#c0392b",
                 "low_genes": "#2980b9", "high_genes": "#e67e22", "low_counts": "#16a085", "no_nucleus": "#d35400"}


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _width(n: int) -> float:
    return min(max(6.0, 0.28 * n + 2), 24.0)


def quantile_boxes(values: pd.Series, groups: pd.Series) -> dict[str, dict]:
    """Box statistics per group from quantiles of all cells: whiskers at 5% and 95%."""
    q = values.groupby(groups, observed=True).quantile(list(QUANTILES)).unstack()
    return {str(g): {"whislo": r[0.05], "q1": r[0.25], "med": r[0.5], "q3": r[0.75], "whishi": r[0.95],
                     "label": str(g)} for g, r in q.iterrows()}


def boxes(stats: dict[str, dict], order: list[str], ylabel: str, title: str, path: Path,
          lines: dict[str, float] | None = None, own: dict[str, dict[str, float]] | None = None) -> Path:
    """Box plots in `order`; `lines` are drawn across, `own` per box where a group's value differs."""
    order = [g for g in order if g in stats]
    fig, ax = plt.subplots(figsize=(_width(len(order)), 4))
    ax.bxp([stats[g] for g in order], showfliers=False, patch_artist=True,
           boxprops={"facecolor": "#d6e4f0"}, medianprops={"color": "black"})
    for name, value in (lines or {}).items():
        ax.axhline(value, color="#c0392b", linestyle="--", linewidth=1, label=f"{name} {value:g}")
    for i, g in enumerate(order, start=1):
        for name, value in (own or {}).get(g, {}).items():
            if lines and lines.get(name) != value:
                ax.hlines(value, i - 0.4, i + 0.4, color="#c0392b", linewidth=2)
    if lines:
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    return save(fig, path)


def pages(order: list[str], size: int = PAGE) -> list[list[str]]:
    return [order[i:i + size] for i in range(0, len(order), size)] or [[]]


def bars(table: pd.DataFrame, ylabel: str, title: str, path: Path, stacked: bool = False,
         colors: list[str] | None = None, line: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(_width(len(table)), 4))
    table.plot.bar(ax=ax, stacked=stacked, color=colors, width=0.8)
    if line is not None:
        ax.axhline(line, color="#c0392b", linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    return save(fig, path)


def heatmap(matrix: pd.DataFrame, title: str, path: Path, cmap: str = "viridis", center: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(_width(matrix.shape[1]), max(2.5, 0.35 * matrix.shape[0] + 1.5)))
    values = matrix.to_numpy(dtype=float)
    kwargs = {}
    if center is not None and np.isfinite(values).any():
        span = max(np.nanmax(np.abs(values - center)), 1e-9)
        kwargs = {"vmin": center - span, "vmax": center + span}
    image = ax.imshow(values, aspect="auto", cmap=cmap, **kwargs)
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), [str(r) for r in matrix.index], fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    return save(fig, path)


def reason_colors(rules: list[str]) -> list:
    spare = iter(plt.get_cmap("tab10").colors)
    return [REASON_COLORS.get(r) or next(spare) for r in rules]


def scatter(x: np.ndarray, y: np.ndarray, x_axis: tuple[str, dict[str, float]], y_axis: tuple[str, dict[str, float]],
            title: str, path: Path, seed: int) -> Path:
    """Cells of one sample, metric against metric, with each axis's thresholds drawn."""
    if len(x) > SCATTER_CELLS:
        keep = np.random.default_rng(seed).choice(len(x), SCATTER_CELLS, replace=False)
        x, y = x[keep], y[keep]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=2, alpha=0.4, color="#34495e", rasterized=True)
    for value in x_axis[1].values():
        ax.axvline(value, color="#c0392b", linestyle="--", linewidth=1)
    for value in y_axis[1].values():
        ax.axhline(value, color="#c0392b", linestyle="--", linewidth=1)
    ax.set_xlabel(x_axis[0])
    ax.set_ylabel(y_axis[0])
    ax.set_title(title)
    return save(fig, path)


def tissue(coords: np.ndarray, passed: np.ndarray, sample: pd.Series, order: list[str], path: Path,
           seed: int = 0) -> Path:
    """Each sample's cells at their spatial coordinates: passing in grey, failing in red."""
    fig, axes = _grid(len(order))
    rng = np.random.default_rng(seed)
    for ax, sid in zip(axes, order):
        cells = np.flatnonzero((sample == sid).to_numpy())
        if len(cells) > TISSUE_CELLS:
            cells = np.sort(rng.choice(cells, TISSUE_CELLS, replace=False))
        ok = passed[cells]
        ax.scatter(*coords[cells][ok].T, s=0.5, color="#bbbbbb", rasterized=True)
        ax.scatter(*coords[cells][~ok].T, s=0.5, color="#c0392b", rasterized=True)
        ax.set_title(f"{sid}: {100 * (~ok).mean():.0f}% fail", fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    return save(fig, path)


def _palette(values: pd.Series) -> dict[str, tuple]:
    levels = [str(v) for v in pd.Series(values).astype("category").cat.categories]
    cmap = plt.get_cmap("tab20" if len(levels) > 10 else "tab10")
    return {lvl: cmap(i % cmap.N) for i, lvl in enumerate(levels)}


def umap(emb: pd.DataFrame, column: str, path: Path, colors: dict[str, str] | None = None) -> Path:
    values = emb[column].astype(str).where(emb[column].notna(), "NA")
    colors = colors or _palette(values)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for level, color in colors.items():
        sel = (values == level).to_numpy()
        if sel.any():
            ax.scatter(emb["umap_1"][sel], emb["umap_2"][sel], s=3, color=color, label=level, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"UMAP by {column}")
    ax.legend(fontsize=6, markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1 + len(colors) // 25)
    return save(fig, path)


def _grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def umap_split(emb: pd.DataFrame, column: str, path: Path) -> Path:
    levels = list(emb[column].astype("category").cat.categories)
    fig, axes = _grid(len(levels))
    for ax, level in zip(axes, levels):
        sel = (emb[column] == level).to_numpy()
        ax.scatter(emb["umap_1"], emb["umap_2"], s=1, color="#dddddd", rasterized=True)
        ax.scatter(emb["umap_1"][sel], emb["umap_2"][sel], s=1, color="#2c3e50", rasterized=True)
        ax.set_title(f"{column}={level} ({sel.sum()})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    return save(fig, path)


def density(emb: pd.DataFrame, column: str, path: Path, subtract_mean: bool, bins: int = 60) -> Path:
    """2-D cell density per group on the UMAP; with `subtract_mean`, minus the mean over groups."""
    levels = list(emb[column].astype("category").cat.categories)
    edges = [np.linspace(emb[a].min(), emb[a].max(), bins + 1) for a in ("umap_1", "umap_2")]
    maps = []
    for level in levels:
        sel = (emb[column] == level).to_numpy()
        h, _, _ = np.histogram2d(emb["umap_1"][sel], emb["umap_2"][sel], bins=edges)
        maps.append(h / max(h.sum(), 1))
    maps = np.array(maps)
    if subtract_mean:
        maps = maps - maps.mean(axis=0)
    span = np.abs(maps).max() or 1.0
    fig, axes = _grid(len(levels))
    for ax, level, m in zip(axes, levels, maps):
        ax.imshow(m.T, origin="lower", cmap="RdBu_r" if subtract_mean else "magma",
                  vmin=-span if subtract_mean else 0, vmax=span)
        ax.set_title(f"{column}={level}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    return save(fig, path)


def composition(emb: pd.DataFrame, column: str, path: Path) -> Path:
    """Fraction of each leiden cluster's cells per value of `column`."""
    table = pd.crosstab(emb["leiden"], emb[column].astype(str), normalize="index")
    table = table.loc[sorted(table.index, key=lambda c: int(c) if str(c).isdigit() else str(c))]
    colors = [_palette(table.columns)[str(c)] for c in table.columns]
    return bars(table, "fraction of cluster", f"leiden composition by {column}", path, stacked=True, colors=colors)
