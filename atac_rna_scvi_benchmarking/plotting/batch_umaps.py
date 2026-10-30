"""Batch / covariate UMAPs: neutral categorical palette, sequential map for continuous columns."""
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .common import Obs, frame, load_umap, point_size, shuffled
from .grids import draw_grid

MAX_LEGEND_LEVELS = 30
GREY = "#CCCCCC"


def neutral_palette(n, seed=0):
    """n muted, mutually distinct colours (golden-ratio hues, fixed s/v), shuffled once."""
    cols = [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.5, 0.78) for i in range(n)]
    return [cols[i] for i in np.random.default_rng(seed).permutation(n)] if n > 12 else cols


def continuous_values(col, v):
    """(values, colourbar label); nFrags is log10-scaled."""
    v = np.asarray(v, dtype=float)
    return (np.log10(np.clip(v, 1, None)), f"log10 {col}") if col == "nFrags" else (v, col)


def _scatter(ax, xy, col, vals, continuous, state, seed=0):
    order = shuffled(len(xy), seed)
    s = point_size(len(xy))
    if continuous:
        v, label = continuous_values(col, vals)
        lo, hi = np.nanpercentile(v, [1, 99])
        sc = ax.scatter(xy[order, 0], xy[order, 1], c=v[order], s=s, cmap="viridis", vmin=lo, vmax=hi,
                        linewidths=0, rasterized=True)
        state["sc"], state["label"] = sc, label
    else:
        codes, uniq = pd.factorize(pd.Series(vals).astype(str).where(pd.notna(vals)))
        pal = np.array(neutral_palette(len(uniq)) + [matplotlib.colors.to_rgb(GREY)])
        ax.scatter(xy[order, 0], xy[order, 1], c=pal[np.where(codes < 0, len(uniq), codes)][order], s=s,
                   linewidths=0, rasterized=True)
        state.update(levels=len(uniq), uniq=list(uniq), pal=pal)
    frame(ax, xy)
    ax.set_axis_off()


def single(ax_fig, xy, col, vals, continuous, title):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=200)
    state = {}
    _scatter(ax, xy, col, vals, continuous, state)
    if continuous:
        fig.colorbar(state["sc"], ax=ax, shrink=0.6, label=state["label"])
        ax.set_title(title, fontsize=8)
    else:
        n = state["levels"]
        ax.set_title(f"{title} - {col}, {n} levels", fontsize=8)
        if n <= MAX_LEGEND_LEVELS:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=state["pal"][i],
                                      markersize=5, label=str(u)) for i, u in enumerate(state["uniq"])],
                      loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def batch_columns(ds):
    cfg = config.dataset(ds)
    cont = set(cfg.get("continuous_cols", []))
    return [(c, c in cont) for c in cfg["batch_umap_cols"]]


def batch_plots(ds, runs, obs, out_root):
    """umaps/batch/<ds>/<column>/<strategy>_<subset>.png plus grid.png per column; returns written paths."""
    written = []
    for col, cont in batch_columns(ds):
        d = out_root / "batch" / ds / col
        d.mkdir(parents=True, exist_ok=True)
        for (tag, strat), (rid, rep) in sorted(runs.items()):
            rows, xy = load_umap(rid)
            fig = single(None, xy, col, obs.get(col, rows), cont, f"{ds} {strat} f{tag} r{rep}")
            out = d / f"{strat}_{tag}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(out)

        def panel(ax, rid, rows, xy, col=col, cont=cont):
            _scatter(ax, xy, col, obs.get(col, rows), cont, {})

        draw_grid(runs, panel, d / "grid.png", fig_w=2.6, dpi=130)
        written.append(d / "grid.png")
    return written
