"""GPU-resident CSR counts feeding scvi-tools: LightningDataModule (train/val split as DataSplitter)
and an inference loader; each minibatch is densified to float32 on the device."""
import math

import numpy as np


class GpuCSR:
    """HvgCSR moved to the GPU: int64 indptr, int16 indices, counts as int16 bit pattern of uint16."""

    def __init__(self, csr, batch_codes, device="cuda"):
        import torch
        self.device = torch.device(device)
        self.n_rows, self.n_cols = csr.n_rows, csr.n_cols
        self.indptr = torch.from_numpy(csr.indptr).to(self.device)
        self.indices = torch.from_numpy(csr.indices).to(self.device)
        self.data = torch.from_numpy(csr.data.view(np.int16)).to(self.device)
        self.batch = torch.from_numpy(np.asarray(batch_codes, np.int64)).to(self.device)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.indptr, self.indices, self.data, self.batch))

    def dense(self, idx):
        """float32 (len(idx), n_cols) counts for row ids idx (torch long on device)."""
        import torch
        start, stop = self.indptr[idx], self.indptr[idx + 1]
        lens = stop - start
        total = int(lens.sum())
        rowpos = torch.repeat_interleave(torch.arange(len(idx), device=self.device), lens, output_size=total)
        excl = torch.cumsum(lens, 0) - lens
        src = torch.arange(total, device=self.device) + torch.repeat_interleave(start - excl, lens, output_size=total)
        out = torch.zeros((len(idx), self.n_cols), dtype=torch.float32, device=self.device)
        out[rowpos, self.indices[src].long()] = (self.data[src].int() & 0xFFFF).float()
        return out

    def minibatch(self, idx):
        import torch
        return {"X": self.dense(idx), "batch": self.batch[idx][:, None],
                "labels": torch.zeros((len(idx), 1), dtype=torch.long, device=self.device)}


class _Loader:
    """Iterable of scvi tensor dicts over a fixed index set (optionally reshuffled per epoch)."""

    def __init__(self, gcsr, idx, batch_size, shuffle=False, seed=0):
        import torch
        self.gcsr, self.batch_size, self.shuffle = gcsr, batch_size, shuffle
        self.idx = torch.as_tensor(np.asarray(idx, np.int64), device=gcsr.device)
        self.gen = torch.Generator(device=gcsr.device)
        self.gen.manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __iter__(self):
        import torch
        order = self.idx[torch.randperm(len(self.idx), generator=self.gen, device=self.idx.device)] \
            if self.shuffle else self.idx
        for i in range(0, len(order), self.batch_size):
            yield self.gcsr.minibatch(order[i:i + self.batch_size])


def make_datamodule(gcsr, n_batch, train_size, batch_size, seed):
    """LightningDataModule with scvi DataSplitter's split (RandomState(seed) permutation, val first)."""
    import lightning.pytorch as pl
    from scvi.dataloaders._data_splitting import validate_data_split

    class GpuCSRDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.n_obs, self.n_vars, self.n_batch = gcsr.n_rows, gcsr.n_cols, n_batch
            n_train, n_val = validate_data_split(gcsr.n_rows, train_size, None, batch_size, False,
                                                 train_size_is_none=False)
            perm = np.random.RandomState(seed=seed).permutation(gcsr.n_rows)
            self.val_idx, self.train_idx = perm[:n_val], perm[n_val:n_val + n_train]
            self.test_idx = perm[n_val + n_train:]
            self.n_train, self.n_val = n_train, n_val

        def train_dataloader(self):
            return _Loader(gcsr, self.train_idx, batch_size, shuffle=True, seed=seed)

        def val_dataloader(self):
            return _Loader(gcsr, self.val_idx, batch_size) if self.n_val else []

    return GpuCSRDataModule()


def inference_loader(gcsr, batch_size=1024):
    """All rows in order, for get_latent_representation(dataloader=...)."""
    return _Loader(gcsr, np.arange(gcsr.n_rows), batch_size)


def registry_adata(var_names, batch_categories, batch_key="batch_id"):
    """Minimal AnnData (one row per batch category) so SCVI.setup_anndata/SCVI(adata) build the same
    registry and module as the in-memory path; never used for data. X is dense: SCVI.load rejects an
    all-zero sparse X as "minified"."""
    import anndata as ad
    import pandas as pd
    cats = pd.Index(batch_categories)
    obs = pd.DataFrame({batch_key: pd.Categorical(cats, categories=cats)}, index=cats.astype(str))
    X = np.zeros((len(cats), len(var_names)), dtype=np.float32)
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(var_names)))
