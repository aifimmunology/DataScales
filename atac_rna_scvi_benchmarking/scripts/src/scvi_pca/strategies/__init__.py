"""Strategy registry. A strategy module exposes:

  needs_cluster(ctx) -> bool   start the per-run dask-cuda cluster before t_load
  load(ctx)                    t_load stage: put inputs on ctx.data
  run(ctx) -> (embedding, rep_name)   t_pre stage: embedding is a cupy/np (n, d) array;
                                      may set ctx.n_pcs (neighbors n_pcs, default None = all)
  USES_RMM = True              module flag; False skips RMM managed-memory init (torch training)

`retrain` / `finetune` hooks live in scvi_train.py / finetune.py with the same interface.
"""
import importlib

MODULES = {"latent": "latent", "pca": "pca", "retrain": "scvi_train"}   # run 2 (run 1 code: results/archive)


VARIANTS = {}


def base(name):
    return VARIANTS[name][0] if name in VARIANTS else name


def variant_overrides(name):
    """Param overrides a sweep variant applies on top of params.toml (explicit --override wins)."""
    return dict(VARIANTS[name][1]) if name in VARIANTS else {}


def get(name):
    b = base(name)
    if b not in MODULES:
        raise KeyError(f"unknown strategy {name!r}; known: {list(MODULES) + list(VARIANTS)}")
    return importlib.import_module(f"{__name__}.{MODULES[b]}")
