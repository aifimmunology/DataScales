"""UMAP plots from saved umap.npy + obs. Usage: 30_plots.py [ds ...] [--only aifi|batch]"""
import sys
import time

from scvi_pca import config
from scvi_pca.plotting import batch_umaps, grids
from scvi_pca.plotting.common import Obs, runs_for


def do_ds(ds, only):
    runs = runs_for(ds)
    if not runs:
        print(f"{ds}: no ok runs"); return
    obs, root = Obs(ds), config.REPO / config.params()["paths"]["umaps_root"]
    if only in (None, "aifi"):
        t = time.perf_counter()
        w = grids.aifi_single(ds, runs, obs, root)
        grids.aifi_grid(ds, runs, obs, root / "aifi" / ds / "grid_subset_x_strategy.png")
        print(f"{ds} aifi: {len(w)} singles + grid in {time.perf_counter() - t:.1f}s", flush=True)
    if only in (None, "batch"):
        t = time.perf_counter()
        w = batch_umaps.batch_plots(ds, runs, obs, root)
        print(f"{ds} batch: {len(w)} files in {time.perf_counter() - t:.1f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    only = args.pop(args.index("--only") + 1) if "--only" in args else None
    if "--only" in args:
        args.remove("--only")
    for ds in args or [d for d, v in config.datasets().items() if isinstance(v, dict)]:
        do_ds(ds, only)
