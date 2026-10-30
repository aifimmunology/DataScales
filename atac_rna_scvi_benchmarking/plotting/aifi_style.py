"""AIFI UMAP style (AIFI_UMAP_STYLE.md): colour finest level, L2-under-L1 legend tree, L1 labels on plot."""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .aifi_palette import L1_COLORS, L2_COLORS, L3_COLORS
from .common import frame, in_frame, point_size, shuffled

UNLABELED = "#DDDDDD"
FALLBACK = plt.get_cmap("tab20")


def _xpos(name, palette):
    return list(palette).index(name) if name in palette else 10_000


def color_map(cats):
    """{category: hex}; AIFI hex where the name is in the L3 palette, tab20 fallback otherwise."""
    out, extra = {}, 0
    for c in cats:
        if c in L3_COLORS:
            out[c] = L3_COLORS[c]
        else:
            out[c] = matplotlib.colors.to_hex(FALLBACK(extra % 20))
            extra += 1
    return out


def legend_tree(l1, l2, l3, cmap):
    """[(L1 header, [(L2 name, swatch hex)])] from the data's own L1/L2 co-occurrence.

    Swatch = AIFI L2 hex when the name matches, else the most abundant child's colour. Headers: AIFI
    order for known names, then by cell count; L2 children: AIFI order, then first L3 position.
    """
    df = pd.DataFrame({"l1": l1, "l2": l2, "l3": l3}).dropna(subset=["l1", "l2"])
    count = df["l1"].value_counts()
    tree = {}
    for l2n, g in df.groupby("l2", observed=True):
        top = g["l3"].dropna().mode()
        swatch = L2_COLORS.get(l2n) or (cmap.get(top.iloc[0], "#999999") if len(top) else "#999999")
        pos = g["l3"].map(lambda c: _xpos(c, L3_COLORS)).min()
        tree.setdefault(g["l1"].mode().iloc[0], []).append((l2n, swatch, pos))
    heads = sorted(tree, key=lambda h: (_xpos(h, L1_COLORS), -count[h]))
    return [(h, [(n, s) for n, s, _ in sorted(tree[h], key=lambda x: (_xpos(x[0], L2_COLORS), x[2]))]) for h in heads]


def density_peaks(coords, labels, bins=80, push=0.35):
    """{label: (x, y)} at each group's 80x80 density peak, pushed 35% outward from the global centroid."""
    m = in_frame(coords)[0]
    coords, labels = coords[m], np.asarray(labels, dtype=object)[m]
    centre = coords.mean(0)
    pts = {}
    for g in pd.unique(labels[pd.notna(labels)]):
        p = coords[labels == g]
        h, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=[bins, bins])
        i, j = np.unravel_index(np.argmax(h), h.shape)
        peak = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        pts[g] = tuple(peak + (peak - centre) * push)
    return pts


def scatter_l3(ax, xy, l3, cmap, s, seed=0):
    """Rasterized scatter in shuffled order; unlabeled cells light grey underneath."""
    cats = pd.Series(l3)
    na = cats.isna().to_numpy()
    if na.any():
        ax.scatter(*xy[na].T, s=s, c=UNLABELED, linewidths=0, rasterized=True)
    keep = np.flatnonzero(~na)
    keep = keep[shuffled(len(keep), seed)]
    ax.scatter(xy[keep, 0], xy[keep, 1], s=s, c=cats.map(cmap).to_numpy()[keep], linewidths=0, rasterized=True)
    frame(ax, xy)
    ax.set_axis_off()


def add_legend(ax, tree, title):
    handles = []
    for head, children in tree:
        handles.append(Line2D([0], [0], marker="none", linestyle="none", label=r"$\bf{%s}$" % re.sub(r"([ +\-/])", r"\\\1", head)))
        handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=s, markeredgecolor="none",
                           markersize=6, label=f"  {n}") for n, s in children]
    many = len(handles) > 20
    if many:
        heads = [i for i, h in enumerate(handles) if h.get_marker() == "none"]
        cut = min(heads, key=lambda i: abs(i - len(handles) / 2))
        left, right = handles[:cut], handles[cut:]
        blank = lambda: Line2D([0], [0], marker="none", linestyle="none", label=" ")
        left += [blank() for _ in range(len(right) - len(left))]
        right += [blank() for _ in range(len(left) - len(right))]
        handles = left + right
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7,
              handletextpad=0.5, labelspacing=0.5 if many else 0.8, borderaxespad=0.0, ncol=2 if many else 1,
              columnspacing=1.2, title=title, title_fontsize=7, alignment="left")


def add_labels(ax, xy, l1, fs=None):
    n = pd.Series(l1).nunique()
    fs = fs or max(5.0, 11.0 - 0.05 * n)
    texts = []
    for g, (px, py) in density_peaks(xy, np.asarray(l1, dtype=object)).items():
        texts.append(ax.text(px, py, g, fontsize=fs, fontweight="semibold", ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6)))
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5))


def level_names(cfg):
    """(colour, legend, label) column names: finest, middle, coarsest of plot_labels."""
    pl = cfg["plot_labels"]
    return pl[2], pl[1], pl[0]


def render(xy, l1, l2, l3, out_png, legend_title, title="", sidecar=None, dpi=300):
    """One 5x5 in AIFI figure: colour L3, L2-under-L1 legend tree, L1 labels."""
    cmap = color_map(pd.Series(l3).dropna().unique())
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.94])
    scatter_l3(ax, xy, l3, cmap, point_size(len(xy)))
    add_labels(ax, xy, l1)
    add_legend(ax, legend_tree(l1, l2, l3, cmap), legend_title)
    if title:
        ax.set_title(title, fontsize=8)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    if sidecar:
        out_png.with_suffix(".params.json").write_text(json.dumps(sidecar, indent=1, default=str))
