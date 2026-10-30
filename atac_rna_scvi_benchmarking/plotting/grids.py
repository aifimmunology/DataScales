"""Grid layout (rows = subset, cols = strategy) and the AIFI per-run / grid renderers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config
from . import aifi_style as A
from .common import SUBSET_LABEL, Obs, columns, load_umap, point_size


def draw_grid(runs, panel_fn, out_png, fig_w=3.0, dpi=150, legend_fn=None):
    """Lay out panel_fn(ax, run_id, rows, xy) over fractions x strategies; missing cells stay blank."""
    fracs, strats = columns(runs)
    fig, axes = plt.subplots(len(fracs), len(strats), figsize=(fig_w * len(strats), fig_w * len(fracs)),
                             dpi=dpi, squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, f in enumerate(fracs):
        for j, s in enumerate(strats):
            if (f, s) in runs:
                rid, rep = runs[(f, s)]
                rows, xy = load_umap(rid)
                panel_fn(axes[i, j], rid, rows, xy)
            if i == 0:
                axes[i, j].set_title(s, fontsize=11)
        axes[i, 0].text(-0.02, 0.5, SUBSET_LABEL.get(f, f), transform=axes[i, 0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=11)
    if legend_fn:
        legend_fn(axes[len(fracs) // 2, -1])
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def aifi_grid(ds, runs, obs, out_png):
    c3, c2, c1 = A.level_names(obs.cfg)
    state = {}

    def panel(ax, rid, rows, xy):
        l3 = obs.get(c3, rows)
        cmap = A.color_map(set(l3[~_isna(l3)]))
        A.scatter_l3(ax, xy, l3, cmap, point_size(len(xy)) * 0.6)
        A.add_labels(ax, xy, obs.get(c1, rows), fs=6)
        state.setdefault("tree", A.legend_tree(obs.get(c1, rows), obs.get(c2, rows), l3, cmap))

    draw_grid(runs, panel, out_png, legend_fn=lambda ax: A.add_legend(ax, state["tree"], c2))


def _isna(a):
    import pandas as pd
    return pd.isna(a)


def aifi_single(ds, runs, obs, out_root):
    """Per-run AIFI PNGs -> umaps/aifi/<ds>/<strategy>/f<frac>.png; returns written paths."""
    c3, c2, c1 = A.level_names(obs.cfg)
    tail = config.params()["tail"]
    written = []
    for (tag, strat), (rid, rep) in sorted(runs.items()):
        rows, xy = load_umap(rid)
        out = out_root / "aifi" / ds / strat / f"{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        A.render(xy, obs.get(c1, rows), obs.get(c2, rows), obs.get(c3, rows), out, c2,
                 title=f"{ds} · {strat} · {SUBSET_LABEL.get(tag, tag)} (rep {rep}, n={len(rows):,})",
                 sidecar=dict(run_id=rid, rep=rep, n_cells=len(rows), umap="umap.npy", tail=tail,
                              colour=c3, legend=c2, labels=c1, palette="AIFI_UMAP_STYLE.md Appendix B"))
        written.append(out)
    return written
