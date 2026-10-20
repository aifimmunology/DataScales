"""App-style cell-type subsets (run 2): every cell, T + NK cells, B cells, selected by the dataset's L1 label."""
import numpy as np
import pandas as pd

from . import config, data


def select(labels, pattern):
    """Sorted int64 rows whose label matches `pattern` (regex); "" = every row. Unlabelled rows only in ""."""
    if not pattern:
        return np.arange(len(labels), dtype=np.int64)
    s = pd.Series(np.asarray(labels, dtype=object)).astype(str)
    return np.flatnonzero(s.str.contains(pattern, regex=True).to_numpy()).astype(np.int64)


def build(ds, out_dir, p=None):
    """Write <subset>.npy under out_dir; returns a manifest of cells per (subset, L1 label)."""
    p = p or config.params()
    cfg = config.dataset(ds)
    level = cfg["plot_labels"][p["subsets"]["label_level"]]
    l1 = data.label_obs(ds, data.open_root(ds))[level].astype(object).to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in config.SUBSETS:
        idx = select(l1, p["subsets"][name])
        np.save(out_dir / f"{name}.npy", idx)
        vc = pd.Series(l1[idx]).astype(str).value_counts()
        rows += [dict(ds=ds, subset=name, label=k, n=int(v)) for k, v in vc.items()]
    return pd.DataFrame(rows)
