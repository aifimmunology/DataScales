"""Execute one run id end to end: load -> strategy t_pre -> shared tail -> artifacts, run.json, runs.parquet."""
import contextlib
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config, data, paths, provenance, results, strategies

paths.add()


@dataclass
class Ctx:
    rid: config.RunId
    params: dict
    rows: np.ndarray
    root: Any = None
    obs: Any = None
    eager: bool = True
    data: Any = None
    n_pcs: Any = None
    rec: Any = None
    t_read_in_pre: float = 0.0
    arrays: dict = field(default_factory=dict)    # extra outputs saved after timing (e.g. unintegrated PCA)
    extra: dict = field(default_factory=dict)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def close(self):
        pass

    def isatty(self):
        return False


def _gpu_setup(strategy):
    from . import measure
    measure.init_nvml()
    if not strategy.USES_RMM:
        import scvi  # noqa: F401  (import + CUDA context outside timed stages, as in a warm service)
        import torch
        torch.zeros(1, device="cuda")
    if strategy.USES_RMM:
        import rerun_umap_on_selection as svc
        svc.warm_up()
        svc._set_zarr_config(svc.ZARR_CONCURRENCY, svc.ZARR_MAX_WORKERS)


def _seed():
    random.seed(0)
    np.random.seed(0)


def _pipeline(ctx, strategy, stack, out_dir):
    from . import measure, tail
    if hasattr(strategy, "cluster"):
        stack.enter_context(strategy.cluster(ctx))
    elif strategy.needs_cluster(ctx):
        import rerun_umap_on_selection as svc
        stack.enter_context(svc.cuda_cluster())
    ctx.rec = rec = measure.Recorder()
    with rec.stage("t_load"):
        strategy.load(ctx)
    with rec.stage("t_pre"):
        emb, rep = strategy.run(ctx)
    stack.close()
    umap, labels = tail.run_tail(rec, emb, rep, ctx.n_pcs, ctx.params)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_host = emb.get() if hasattr(emb, "get") else np.asarray(emb)
    np.save(out_dir / "embedding.npy", emb_host.astype(np.float32))
    np.save(out_dir / "umap.npy", umap)
    np.save(out_dir / "rows.npy", np.asarray(ctx.rows, np.int64))
    for name, arr in ctx.arrays.items():
        np.save(out_dir / f"{name}.npy", np.asarray(arr.get() if hasattr(arr, "get") else arr, np.float32))
    if labels is not None:
        pd.DataFrame({"row": ctx.rows, "leiden": labels}).to_parquet(out_dir / "leiden.parquet", index=False)
    return rec, rep, None if labels is None else int(len(np.unique(labels)))


def _timings(rec, t_read_in_pre=0.0):
    """t_read = getting X/latent into memory wherever it happened (t_load, or the lazy path's persist
    inside t_pre); t_total excludes it on both paths; t_e2e = t_read + t_total."""
    t = {k: rec.stages[k]["wall_s"] for k in ("t_load", "t_pre", "t_neighbors", "t_umap", "t_leiden") if k in rec.stages}
    if "t_load" in t:
        t["t_read"] = t["t_load"] + t_read_in_pre
    if all(k in t for k in ("t_pre", "t_neighbors", "t_umap")):
        t["t_pre_ex_read"] = t["t_pre"] - t_read_in_pre
        t["t_total"] = t["t_pre_ex_read"] + t["t_neighbors"] + t["t_umap"] + t.get("t_leiden", 0.0)
        t["t_e2e"] = t["t_read"] + t["t_total"]
    return t


def execute(run_id, override_items=(), explore=None):
    """Returns 'ok' or 'error'. Never raises for pipeline failures: the error is recorded."""
    rid = config.RunId.parse(run_id)
    overrides = {**strategies.variant_overrides(rid.strategy), **config.parse_overrides(override_items)}
    p = config.params(overrides)
    repo_dir, out_dir = rid.repo_dir(explore), rid.out_dir(explore, p)
    repo_dir.mkdir(parents=True, exist_ok=True)
    log = open(repo_dir / "log.txt", "w")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(real_out, log), _Tee(real_err, log)
    info = {"run_id": run_id, "ds": rid.ds, "strategy": rid.strategy, "subset": rid.subset, "rep": rid.rep,
            "overrides": overrides, "explore": explore, "config": p, "dataset": config.dataset(rid.ds),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": provenance.versions(), "git": provenance.git_state(), "hardware": provenance.hardware(),
            "status": "ok"}
    t_wall, ctx = time.perf_counter(), None
    try:
        strategy = strategies.get(rid.strategy)
        _seed()
        rows = np.load(config.subset_path(rid.ds, rid.frac_tag, p))
        info["n_cells"] = int(len(rows))
        ctx = Ctx(rid=rid, params=p, rows=rows)
        _gpu_setup(strategy)
        ctx.root = data.open_root(rid.ds)
        ctx.obs = data.batch_obs(rid.ds, ctx.root)
        ctx.eager = strategies.base(rid.strategy) != "pca"
        info["eager"] = ctx.eager
        if strategies.base(rid.strategy) in ("pca", "pca_lsi"):
            info["harmony_alpha"] = config.harmony_alpha(p)
        with contextlib.ExitStack() as stack:
            rec, rep, n_clusters = _pipeline(ctx, strategy, stack, out_dir)
        info.update(rep_name=rep, n_clusters=n_clusters, n_pcs=ctx.n_pcs, **_timings(rec, ctx.t_read_in_pre),
                    stages=rec.stages, **rec.run_peaks())
        info.update(ctx.extra)
        samples = repo_dir / "stage_samples.parquet" if explore else config.REPO / "results" / "stage_samples" / f"{run_id}.parquet"
        samples.parent.mkdir(parents=True, exist_ok=True)
        rec.samples_frame().to_parquet(samples, index=False)
    except BaseException as e:
        info.update(status="error", error=repr(e), traceback=traceback.format_exc())
        traceback.print_exc()
        if ctx is not None:
            info.update(ctx.extra)
        if ctx is not None and ctx.rec is not None:
            info.update(stages=ctx.rec.stages, **_timings(ctx.rec, ctx.t_read_in_pre), **ctx.rec.run_peaks())
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        info["wall_total_s"] = time.perf_counter() - t_wall
        for k in ("t_load", "t_read", "t_pre", "t_pre_ex_read", "t_neighbors", "t_umap", "t_leiden", "t_total", "t_e2e", "n_clusters", "rep_name"):
            info.setdefault(k, None)
        info.setdefault("n_cells", None)
        (repo_dir / "run.json").write_text(json.dumps(info, indent=2, default=str))
        if not explore:
            results.upsert(info)
        sys.stdout, sys.stderr = real_out, real_err
        log.close()
    print(f"{run_id}: {info['status']}  t_pre={info['t_pre']}  t_total={info['t_total']}  clusters={info['n_clusters']}")
    return info["status"]
