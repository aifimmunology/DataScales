"""HVG-only counts CSR (uint16 data, int16 indices) built by streaming the source, plus a zarr cache."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

from . import csr_stream


@dataclass
class HvgCSR:
    indptr: np.ndarray    # int64 (n + 1,)
    indices: np.ndarray   # int16
    data: np.ndarray      # uint16 counts
    n_cols: int

    @property
    def n_rows(self):
        return len(self.indptr) - 1

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.indices.nbytes + self.data.nbytes

    def to_scipy(self):
        """scipy CSR (scipy upcasts indices to int32)."""
        import scipy.sparse as sp
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=(self.n_rows, self.n_cols))


@numba.njit(cache=True)
def _filter_block(indptr, indices, data, sel, colmap, out_ip, out_ix, out_d, row0, pos):
    for k in range(len(sel)):
        r = sel[k]
        for j in range(indptr[r], indptr[r + 1]):
            c = colmap[indices[j]]
            if c >= 0:
                out_ix[pos] = c
                out_d[pos] = data[j]
                pos += 1
        out_ip[row0 + k + 1] = pos
    return pos


def extract(X, rows, cols, nnz, block_nnz=csr_stream.BLOCK_NNZ, stats=None, exact=True):
    """X[rows][:, cols] as HvgCSR; nnz is the exact output nnz (sum of hvg_v3's per-gene `nnz` over cols),
    or with exact=False an upper bound (output arrays are trimmed)."""
    n_genes = int(X.attrs["shape"][1])
    n_rows = int(X["indptr"].shape[0] - 1) if rows is None else len(rows)
    assert len(cols) < 2 ** 15
    colmap = np.full(n_genes, -1, np.int16)
    colmap[np.asarray(cols)] = np.arange(len(cols), dtype=np.int16)
    out = HvgCSR(np.zeros(n_rows + 1, np.int64), np.empty(nnz, np.int16), np.empty(nnz, np.uint16), len(cols))
    pos = 0
    for blk in csr_stream.blocks(X, rows, block_nnz, stats=stats):
        pos = _filter_block(blk.indptr, blk.indices, blk.data, blk.sel, colmap,
                            out.indptr, out.indices, out.data, blk.out0, pos)
    if exact and pos != nnz:
        raise ValueError(f"extracted nnz {pos} != expected {nnz}")
    if pos != nnz:
        out = HvgCSR(out.indptr, out.indices[:pos].copy(), out.data[:pos].copy(), len(cols))
    return out


def cache_path(out_root, ds, frac_tag, n_top):
    return Path(out_root) / "cache" / f"{ds}_hvg{n_top}_f{frac_tag}.zarr"


def write_cache(path, csr, var_names, batch, rows, chunk=50_000_000):
    """Anndata-encoded zarr: X (csr, uint16/int16), obs (batch_id, src_row), var index."""
    import anndata as ad
    import pandas as pd
    import zarr
    from zarr.codecs import ZstdCodec
    t0 = time.perf_counter()
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    Xg = root.create_group("X")
    Xg.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                     "shape": [csr.n_rows, csr.n_cols]})
    for name, arr in (("data", csr.data), ("indices", csr.indices), ("indptr", csr.indptr)):
        z = Xg.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=(min(chunk, max(len(arr), 1)),),
                            compressors=ZstdCodec(level=3))
        z[:] = arr
    obs = pd.DataFrame({"batch_id": pd.Categorical(batch), "src_row": np.asarray(rows, np.int64)},
                       index=pd.Index(np.arange(csr.n_rows).astype(str)))
    ad.io.write_elem(root, "obs", obs)
    ad.io.write_elem(root, "var", pd.DataFrame(index=pd.Index(var_names)))
    root.attrs["scvi_pca"] = json.dumps({"nnz": int(len(csr.data)), "n_cols": csr.n_cols})
    return time.perf_counter() - t0


def read_cache(path):
    """(HvgCSR, obs DataFrame, var_names) from write_cache output."""
    import anndata as ad
    import zarr
    root = zarr.open_group(str(path), mode="r")
    Xg = root["X"]
    csr = HvgCSR(Xg["indptr"][:], Xg["indices"][:], Xg["data"][:], int(Xg.attrs["shape"][1]))
    obs = ad.io.read_elem(root["obs"])
    var = ad.io.read_elem(root["var"])
    return csr, obs, var.index
