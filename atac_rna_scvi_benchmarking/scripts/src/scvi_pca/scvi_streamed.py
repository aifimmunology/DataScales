"""Out-of-core scVI input for `retrain` (rna13m): streamed exact seurat_v3 -> HVG CSR -> GPU datamodule.

Never materialises the full X; host peak is the HVG CSR (~4 B/nnz) plus one read block.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import csr_stream, hvg_matrix, hvg_v3, scvi_gpu_data


@dataclass
class Streamed:
    registry_adata: Any      # pass as `adata` to fit_latent (registry only, never data)
    datamodule: Any          # pass as `datamodule`
    latent_loader: Any       # pass to get_latent_representation(dataloader=...)
    n_cells: int
    hvg: Any                 # seurat_v3 DataFrame (all genes)
    io: dict = field(default_factory=dict)

    @property
    def t_read_blocked(self):
        """Seconds the consumer sat waiting on X reads (non-overlapped I/O) across HVG + extract."""
        return sum(v.get("wait_s", 0.0) for v in self.io.values())


def var_names(root):
    import anndata as ad
    return ad.io.read_elem(root["var"][root["var"].attrs["_index"]])


def prepare(root, obs, rows, p, seed, tm, latent_batch_size=8192):
    """root: zarr store root; obs: data.batch_obs (categorical batch over all rows); rows: sorted subset."""
    bkey, n_top = p["scvi"]["batch_key"], p["hvg"]["scvi_n_top_genes"]
    rows = np.asarray(rows, np.int64)
    csr_stream.set_zarr_io()
    names = np.asarray(var_names(root))
    batch = obs[bkey]
    io = {"hvg": {}, "extract": {}}
    with tm("t_hvg"):
        df = hvg_v3.seurat_v3(root["X"], names, batch.to_numpy(), rows, n_top_genes=n_top, stats=io["hvg"])
    cols = hvg_v3.hvg_columns(df)
    with tm("t_extract"):
        csr = hvg_matrix.extract(root["X"], rows, cols, int(df["nnz"].to_numpy()[cols].sum()), stats=io["extract"])
    with tm("t_to_gpu"):
        g = scvi_gpu_data.GpuCSR(csr, batch.cat.codes.to_numpy()[rows])
    del csr
    cats = batch.cat.categories
    dm = scvi_gpu_data.make_datamodule(g, len(cats), p["epochs"]["train_size"], p["scvi"]["batch_size"], seed)
    return Streamed(scvi_gpu_data.registry_adata(names[cols], cats, bkey), dm,
                    scvi_gpu_data.inference_loader(g, latent_batch_size), len(rows), df, io)
