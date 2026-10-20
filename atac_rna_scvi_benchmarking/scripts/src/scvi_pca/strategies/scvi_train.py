"""retrain: fit scVI (RNA) / PeakVI (ATAC) from scratch on the subset, return the latent.

Data seam: `fit_latent(adata, modality, p, seed, tm, datamodule, n_cells, latent_loader)` is the only
function that trains. `adata` is a registry-only AnnData; the datamodule and latent loader serve
GPU-resident minibatches. RNA: scvi_streamed.prepare (streamed exact seurat_v3, HVG CSR, GPU); ATAC:
atac_gpu_data.prepare (binarised host CSR, GPU). Any other loader plugs in the same way.
"""
import time

import numpy as np

from .. import config, data

USES_RMM = False
REP = {"rna": "X_scVI", "atac": "X_peakVI"}
PEAKVI_PLAN = {"lr": 1e-4, "weight_decay": 1e-3, "eps": 1e-8, "n_epochs_kl_warmup": 50,
               "n_steps_kl_warmup": None, "optimizer": "AdamW"}
LAST = {}


def needs_cluster(ctx):
    return False


def load(ctx):
    """RNA is streamed from the store inside t_pre (t_load = nothing); ATAC reads the subset CSR eagerly."""
    ctx.data = None if config.dataset(ctx.rid.ds)["modality"] == "rna" else data.load_X(ctx.root, ctx.obs, ctx.rows, eager=True)


class Timers(dict):
    """Named GPU-synced sub-timers: `with tm("t_train"): ...` accumulates seconds."""

    def __call__(self, name):
        from .. import measure
        import contextlib

        @contextlib.contextmanager
        def cm():
            measure.sync()
            t0 = time.perf_counter()
            yield
            measure.sync()
            self[name] = self.get(name, 0.0) + time.perf_counter() - t0
        return cm()


def max_epochs(n_cells, e):
    """clamp(round(cell_budget / n), min_epochs, max_epochs); the upper bound wins so overrides can cap."""
    return int(min(e["max_epochs"], max(e["min_epochs"], round(e["cell_budget"] / n_cells))))


class EpochLog:
    """Lightning callback: per-epoch wall seconds and an ETA line after epoch 1."""

    def __new__(cls, budget):
        import lightning.pytorch as pl

        class _CB(pl.Callback):
            def __init__(self):
                self.t0, self.epoch_s = None, []

            def on_train_epoch_start(self, trainer, module):
                self.t0 = time.perf_counter()

            def on_train_epoch_end(self, trainer, module):
                self.epoch_s.append(time.perf_counter() - self.t0)
                if len(self.epoch_s) == 1:
                    print(f"epoch 1 took {self.epoch_s[0]:.1f}s; ETA for {budget} epochs "
                          f"{self.epoch_s[0] * (budget - 1) / 60:.1f} min", flush=True)
        return _CB()


def _build(adata, modality, p):
    import scvi
    if modality == "rna":
        scvi.model.SCVI.setup_anndata(adata, batch_key=p["scvi"]["batch_key"])
        s = p["scvi"]
        return scvi.model.SCVI(adata, n_latent=s["n_latent"], n_layers=s["n_layers"], gene_likelihood=s["gene_likelihood"])
    scvi.model.PEAKVI.setup_anndata(adata, batch_key=p["peakvi"]["batch_key"])
    return scvi.model.PEAKVI(adata, n_latent=p["peakvi"]["n_latent"])


def _train(model, modality, p, budget, datamodule, cb):
    from scvi.model.base._training_mixin import UnsupervisedTrainingMixin
    e = p["epochs"]
    kw = dict(max_epochs=budget, accelerator=p["train"]["accelerator"], early_stopping=e["early_stopping"], early_stopping_patience=e["patience"],
              early_stopping_monitor="elbo_validation", check_val_every_n_epoch=1, datamodule=datamodule, callbacks=[cb])
    plan = dict(PEAKVI_PLAN) if modality == "atac" else {}
    frac = p["train"].get("kl_warmup_frac") or 0
    if frac:
        plan["n_epochs_kl_warmup"] = max(1, int(np.ceil(frac * budget)))
    if plan:
        kw["plan_kwargs"] = plan
    UnsupervisedTrainingMixin.train(model, **kw)
    return plan.get("n_epochs_kl_warmup", 400)


def _summary(model, budget, cb):
    h = model.history
    tr, va = h["elbo_train"]["elbo_train"].to_numpy(), h["elbo_validation"]["elbo_validation"].to_numpy()
    n = model.trainer.current_epoch
    hist = [{"epoch": i + 1, "elbo_train": float(tr[i]), "elbo_validation": float(va[i]), "wall_s": cb.epoch_s[i]}
            for i in range(min(len(tr), len(va), len(cb.epoch_s)))]
    return {"epochs_run": int(n), "total_steps": int(model.trainer.global_step), "max_epochs_budget": int(budget),
            "early_stopped": bool(n < budget), "final_elbo_train": float(tr[-1]), "final_elbo_validation": float(va[-1]),
            "history": hist}


def fit_latent(adata, modality, p, seed, tm, datamodule=None, n_cells=None, latent_loader=None):
    """Train on `adata` (prepared counts / binarised peaks) and return (latent float32, summary dict).

    The trained model is left in `LAST["model"]`, moved to CPU with the GPU cache released.
    """
    import scvi
    import torch
    scvi.settings.seed = seed
    n_cells = n_cells or adata.n_obs
    budget = max_epochs(n_cells, p["epochs"])
    cb = EpochLog(budget)
    with tm("t_setup"):
        model = _build(adata, modality, p)
    with tm("t_train"):
        warmup = _train(model, modality, p, budget, datamodule, cb)
    with tm("t_latent"):
        kw = {"dataloader": latent_loader} if latent_loader is not None else {"batch_size": p["train"]["latent_batch_size"]}
        lat = model.get_latent_representation(**kw).astype(np.float32)
    summary = _summary(model, budget, cb)
    summary.update(kl_warmup_epochs=int(warmup), kl_weight_final=min(1.0, (summary["epochs_run"] - 1) / warmup))
    model.trainer = None
    model.module.cpu()
    LAST["model"] = model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return lat, summary


def _release():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def run(ctx):
    from .. import atac_gpu_data, scvi_streamed
    p, modality = ctx.params, config.dataset(ctx.rid.ds)["modality"]
    seed = p["train"]["seed_by_rep"][ctx.rid.rep - 1]
    bs = p["train"]["latent_batch_size"]
    tm = Timers()
    if modality == "rna":
        s = scvi_streamed.prepare(ctx.root, ctx.obs, ctx.rows, p, seed, tm, bs)
        ctx.t_read_in_pre = s.t_read_blocked
        ctx.extra["io"] = s.io
    else:
        adata, ctx.data = ctx.data, None
        s = atac_gpu_data.prepare(adata, p, seed, tm, bs)
        del adata
    lat, summary = fit_latent(s.registry_adata, modality, p, seed, tm, datamodule=s.datamodule,
                              n_cells=s.n_cells, latent_loader=s.latent_loader)
    del s
    _release()
    ctx.extra.update(summary, seed=seed, sub_timers=dict(tm), n_features_trained=int(LAST["model"].summary_stats.n_vars))
    return lat, REP[modality]
