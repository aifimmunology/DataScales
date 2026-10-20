"""Run-2 metrics: 20_metrics.py <ds> ... -> results/metrics/<ds>.parquet (long: ds, subset, strategy, level, metric, value).

Per (ds, subset): evaluation cells (metrics.eval_positions), every ok run's embedding (rep 2 preferred) plus the
uncorrected baseline -> kNN purity here (GPU) and the scib-metrics Benchmarker in the metrics env (GPU lock held).
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from scvi_pca import config, data, metrics as M

WORKER = config.SCRIPTS / "21_scib_worker.py"
KNN_K = 90    # scib Benchmarker neighbour count (max of 15/50/90)


def rows_for(ds, subset, runs):
    cfg = config.dataset(ds)
    root = data.open_root(ds)
    rows = np.load(config.subset_path(ds, subset))
    coarse, fine = M.labels(ds, root, rows)
    pos = M.eval_positions(ds, subset, root, rows, coarse, fine)
    batch = data.read_obs_cols(root, [cfg["batch_col"]])[cfg["batch_col"]].astype(str).to_numpy()[rows][pos]
    coarse, fine = coarse[pos], fine[pos]
    strategies = sorted(s for (sub, s) in runs if sub == subset)
    embs = {s: M.run_embedding(runs[(subset, s)], pos) for s in strategies}
    embs["unintegrated"] = M.unintegrated(ds, subset, root, data.batch_obs(ds, root), rows, pos, runs)
    out, graphs = [], {}
    for s, e in embs.items():
        graphs[f"idx_{s}"], graphs[f"dist_{s}"] = M.knn(e, KNN_K)
        for k, v in M.purity(e, coarse, fine, batch, idx=graphs[f"idx_{s}"]).items():
            out.append(dict(ds=ds, subset=subset, strategy=s, level="", metric=k, value=v, n_eval=len(pos)))
    job = M.cache_dir() / f"scib_{ds}_{subset}.npz"
    np.savez(job, batch=batch, label_coarse=coarse, label_fine=fine, unintegrated=embs["unintegrated"],
             **{f"emb_{s}": embs[s] for s in strategies}, **graphs)
    subprocess.run(["flock", M.GPU_LOCK, "pixi", "run", "-e", "metrics", "python", str(WORKER), str(job)],
                   check=True, cwd=config.SCRIPTS)
    res = json.loads(job.with_suffix(".json").read_text())
    for level, r in res.items():
        tab = pd.DataFrame(r["table"]).set_index("Embedding")
        mtype = tab.loc["Metric Type"] if "Metric Type" in tab.index else pd.Series(dtype=str)
        for emb, rec in tab.drop(index="Metric Type", errors="ignore").iterrows():
            s = emb[4:] if emb.startswith("emb_") else emb
            for metric, v in rec.items():
                out.append(dict(ds=ds, subset=subset, strategy=s, level=level, metric=f"scib:{metric}",
                                value=float(v), n_eval=r["n_eval"], metric_type=str(mtype.get(metric, ""))))
    print(f"[ok] {ds} {subset}: {strategies} on {len(pos):,} cells", flush=True)
    return out


def main(ds_list):
    """One file per dataset (results/metrics/<ds>.parquet) so concurrent runs never clobber each other."""
    out_dir = config.REPO / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in ds_list:
        runs = M.ok_runs(ds)
        new = []
        for subset in config.SUBSETS:
            if any(sub == subset for sub, _ in runs):
                new += rows_for(ds, subset, runs)
        pd.DataFrame(new).to_parquet(out_dir / f"{ds}.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
