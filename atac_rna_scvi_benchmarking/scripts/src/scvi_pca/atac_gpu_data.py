"""ATAC (PeakVI) input on the GPU: binarised CSR resident in VRAM, same datamodule/loaders as scvi_gpu_data."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import scvi_gpu_data


class BinaryGpuCSR(scvi_gpu_data.GpuCSR):
    """Binary peak matrix: int64 indptr + int32 indices only; every stored entry is 1."""

    def __init__(self, csr, batch_codes, device="cuda"):
        import torch
        self.device = torch.device(device)
        self.n_rows, self.n_cols = csr.shape
        self.indptr = torch.from_numpy(csr.indptr.astype(np.int64)).to(self.device)
        self.indices = torch.from_numpy(csr.indices.astype(np.int32)).to(self.device)
        self.data = None
        self.batch = torch.from_numpy(np.asarray(batch_codes, np.int64)).to(self.device)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.indptr, self.indices, self.batch))

    def dense(self, idx):
        import torch
        start, stop = self.indptr[idx], self.indptr[idx + 1]
        lens = stop - start
        total = int(lens.sum())
        rowpos = torch.repeat_interleave(torch.arange(len(idx), device=self.device), lens, output_size=total)
        excl = torch.cumsum(lens, 0) - lens
        src = torch.arange(total, device=self.device) + torch.repeat_interleave(start - excl, lens, output_size=total)
        out = torch.zeros((len(idx), self.n_cols), dtype=torch.float32, device=self.device)
        out[rowpos, self.indices[src].long()] = 1.0
        return out


@dataclass
class AtacInput:
    registry_adata: Any
    datamodule: Any
    latent_loader: Any
    n_cells: int


def prepare(adata, p, seed, tm, latent_batch_size=8192):
    """adata: host AnnData (csr X, obs with the categorical batch column); consumed (X binarised in place)."""
    bkey = p["peakvi"]["batch_key"]
    X = adata.X
    with tm("t_binarise"):
        X.data[:] = 1
        X.eliminate_zeros()
    batch = adata.obs[bkey]
    with tm("t_to_gpu"):
        g = BinaryGpuCSR(X, batch.cat.codes.to_numpy())
    cats, names = batch.cat.categories, adata.var_names.to_numpy()
    n = adata.n_obs
    dm = scvi_gpu_data.make_datamodule(g, len(cats), p["epochs"]["train_size"], p["peakvi"]["batch_size"], seed)
    return AtacInput(scvi_gpu_data.registry_adata(names, cats, bkey), dm,
                     scvi_gpu_data.inference_loader(g, latent_batch_size), n)
