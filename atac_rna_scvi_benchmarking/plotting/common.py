"""Run discovery, saved-UMAP loading and obs access shared by the plotting modules."""
import numpy as np
import pandas as pd

from .. import config, data

STRATEGY_ORDER = ["latent", "pca", "retrain"]
SUBSET_LABEL = {"all": "All cells", "tnk": "T + NK cells", "b": "B cells"}


def runs_for(ds):
    """{(subset, strategy): (run_id, rep)} for ok runs, rep 2 preferred, rep 1 fallback."""
    r = pd.read_parquet(config.REPO / "results" / "runs.parquet")
    r = r[(r.ds == ds) & (r.status == "ok")].sort_values("rep")
    return {(x.subset, x.strategy): (x.run_id, int(x.rep)) for x in r.itertuples()}


def columns(runs):
    """(subsets in config order, strategies in canonical order) present in runs."""
    fracs = [s for s in config.SUBSETS if any(t == s for t, _ in runs)]
    strats = sorted({s for _, s in runs}, key=lambda s: STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 99)
    return fracs, strats


def load_umap(run_id):
    """(store rows, (n, 2) umap) of a finished run."""
    rid = config.RunId.parse(run_id)
    d = rid.out_dir()
    rows = np.load(d / "rows.npy")
    xy = np.load(d / "umap.npy")[:, :2]
    return rows, xy


class Obs:
    """Lazy per-dataset obs columns (label columns under plot names), indexable by store rows."""

    def __init__(self, ds):
        self.ds, self.cfg, self.root = ds, config.dataset(ds), data.open_root(ds)
        self._cache = {}

    def col(self, name):
        if name not in self._cache:
            store = dict(zip(self.cfg["plot_labels"], self.cfg["labels"])).get(name, name)
            self._cache[name] = data.read_obs_cols(self.root, [store], ds=self.ds)[name]
        return self._cache[name]

    def get(self, name, rows):
        return self.col(name).to_numpy()[rows]


def shuffled(n, seed=0):
    return np.random.default_rng(seed).permutation(n)


def in_frame(xy, q=0.05, pad=0.08):
    """Mask of points inside the [q, 100-q] percentile box (padded); far UMAP outliers fall outside."""
    lo, hi = np.percentile(xy, [q, 100 - q], axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    return ((xy >= lo) & (xy <= hi)).all(1), lo, hi


def frame(ax, xy):
    """Set axis limits to the robust box so a few stray points can't shrink the main embedding."""
    _, lo, hi = in_frame(xy)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def point_size(n):
    return float(min(3.0, max(0.5, 120000 / n)))
