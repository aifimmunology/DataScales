from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..config import AppConfig
from ..engine import _stage
from ..errors import ConversionError
from ..layout import _x_compressors
from ..storage import open_store_rw

_BAND_BYTES = 256 * 1024 * 1024


def add_expr_layer(
    store: str,
    cfg: AppConfig,
    *,
    fmt: str = "csc",
    layer: str = "gexp",
    chunk_elems: int = 1_000_000,
    target_sum: float = 1e4,
) -> list[str]:
    """Add a log-normalized expression layer (layers/<layer>) derived from CSR X."""
    import numpy as np

    if fmt not in ("csc", "dense", "csr"):
        raise ConversionError(f"add-expr format must be csc, dense, or csr; got '{fmt}'.")

    store_path = Path(store)
    root, finalize = open_store_rw(
        store_path, cfg, commit_message=f"zarrsmith add-expr {fmt} → layers/{layer}"
    )
    x = root["X"]
    if x.attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(
            f"add-expr requires CSR X; got encoding {x.attrs.get('encoding-type')!r}."
        )

    n_obs, n_vars = (int(v) for v in x.attrs["shape"])
    data_arr, idx_arr = x["data"], x["indices"]
    indptr = np.asarray(x["indptr"][:], dtype=np.int64)
    nnz = int(indptr[-1])
    row_nnz = np.diff(indptr)

    layers = root.require_group("layers")
    if "encoding-type" not in dict(layers.attrs):
        layers.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    if layer in layers:
        if not cfg.io.overwrite:
            raise ConversionError(f"layers/{layer} already exists; pass --overwrite to replace it.")
        del layers[layer]

    bytes_per_row = max(1, nnz // max(1, n_obs)) * 12
    row_step = max(1_000, min(200_000, _BAND_BYTES // bytes_per_row))

    factors = np.zeros(n_obs, dtype=np.float64)
    with _stage(f"Computing scale factors (n_obs={n_obs}, nnz={nnz})"):
        for b0 in range(0, n_obs, row_step):
            b1 = min(b0 + row_step, n_obs)
            seg = np.asarray(data_arr[int(indptr[b0]):int(indptr[b1])], dtype=np.float64)
            if seg.size == 0:
                continue
            offsets = np.minimum((indptr[b0:b1] - indptr[b0]), seg.size - 1)
            sums = np.add.reduceat(seg, offsets)
            sums[row_nnz[b0:b1] == 0] = 0.0
            nz = sums > 0
            factors[b0:b1][nz] = target_sum / sums[nz]

    def _band(b0: int, b1: int):
        s0, s1 = int(indptr[b0]), int(indptr[b1])
        vals = np.log1p(
            np.asarray(data_arr[s0:s1], dtype=np.float64)
            * np.repeat(factors[b0:b1], row_nnz[b0:b1])
        ).astype(np.float32)
        return s0, s1, vals

    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    if fmt == "csr":
        g = _sparse_layer(layers, layer, "csr_matrix", (n_obs, n_vars), nnz,
                          idx_arr.dtype, indptr_dtype, chunk_elems)
        g["indptr"][:] = indptr.astype(indptr_dtype)
        with _stage(f"Writing layers/{layer} (csr, nnz={nnz})"):
            for b0 in range(0, n_obs, row_step):
                b1 = min(b0 + row_step, n_obs)
                s0, s1, vals = _band(b0, b1)
                g["data"][s0:s1] = vals
                g["indices"][s0:s1] = idx_arr[s0:s1]
        finalize()
        return []

    # csc/dense: pass 1 counts nnz per column; pass 2 buckets entries into column
    # bands (disk-backed, so RAM stays one band); each band then writes its slice.
    col_nnz = np.zeros(n_vars, dtype=np.int64)
    flat_step = max(chunk_elems, _BAND_BYTES // 8)
    with _stage("Counting nnz per gene"):
        for s0 in range(0, nnz, flat_step):
            s1 = min(s0 + flat_step, nnz)
            col_nnz += np.bincount(idx_arr[s0:s1], minlength=n_vars)
    csc_indptr = np.concatenate([[0], np.cumsum(col_nnz)]).astype(np.int64)

    if fmt == "dense":
        k = max(1, chunk_elems // n_obs)
        band_cols = max(k, (_BAND_BYTES // (4 * n_obs)) // k * k)
        edges = list(range(0, n_vars, band_cols)) + [n_vars]
    else:
        max_band_nnz = _BAND_BYTES // 12
        edges = [0]
        while edges[-1] < n_vars:
            nxt = int(np.searchsorted(csc_indptr, csc_indptr[edges[-1]] + max_band_nnz))
            edges.append(min(max(nxt, edges[-1] + 1), n_vars))
    n_bands = len(edges) - 1
    band_nnz = [int(csc_indptr[edges[i + 1]] - csc_indptr[edges[i]]) for i in range(n_bands)]

    tmp_root = Path(tempfile.mkdtemp(prefix="zarrsmith_expr_", dir=str(store_path.parent)))
    try:
        buckets = []
        for i, m in enumerate(band_nnz):
            m = max(1, m)
            buckets.append({
                "rows": np.memmap(tmp_root / f"r{i}", dtype=np.int64, mode="w+", shape=(m,)),
                "cols": np.memmap(tmp_root / f"c{i}", dtype=np.int64, mode="w+", shape=(m,)),
                "vals": np.memmap(tmp_root / f"v{i}", dtype=np.float32, mode="w+", shape=(m,)),
            })
        edges_arr = np.asarray(edges[1:], dtype=np.int64)

        cursors = [0] * n_bands
        with _stage(f"Bucketing {nnz} entries into {n_bands} gene bands"):
            for b0 in range(0, n_obs, row_step):
                b1 = min(b0 + row_step, n_obs)
                s0, s1, vals = _band(b0, b1)
                cols = np.asarray(idx_arr[s0:s1], dtype=np.int64)
                rows = np.repeat(np.arange(b0, b1, dtype=np.int64), row_nnz[b0:b1])
                band_ids = np.searchsorted(edges_arr, cols, side="right")
                order = np.argsort(band_ids, kind="stable")
                bounds = np.searchsorted(band_ids[order], np.arange(n_bands + 1))
                for bi in range(n_bands):
                    lo, hi = int(bounds[bi]), int(bounds[bi + 1])
                    if lo == hi:
                        continue
                    sel = order[lo:hi]
                    c = cursors[bi]
                    buckets[bi]["rows"][c:c + hi - lo] = rows[sel]
                    buckets[bi]["cols"][c:c + hi - lo] = cols[sel]
                    buckets[bi]["vals"][c:c + hi - lo] = vals[sel]
                    cursors[bi] = c + hi - lo

        if fmt == "csc":
            indices_dtype = np.int64 if n_obs > np.iinfo(np.int32).max else np.int32
            g = _sparse_layer(layers, layer, "csc_matrix", (n_obs, n_vars), nnz,
                              indices_dtype, indptr_dtype, chunk_elems)
            g["indptr"][:] = csc_indptr.astype(indptr_dtype)
            with _stage(f"Writing layers/{layer} (csc, nnz={nnz})"):
                for bi in range(n_bands):
                    m = band_nnz[bi]
                    if m == 0:
                        continue
                    # entries were appended in ascending row order, so a stable
                    # sort by column yields canonical CSC
                    order = np.argsort(np.asarray(buckets[bi]["cols"][:m]), kind="stable")
                    o0, o1 = int(csc_indptr[edges[bi]]), int(csc_indptr[edges[bi + 1]])
                    g["data"][o0:o1] = np.asarray(buckets[bi]["vals"][:m])[order]
                    g["indices"][o0:o1] = np.asarray(buckets[bi]["rows"][:m])[order].astype(indices_dtype)
        else:
            arr = layers.require_array(
                layer, shape=(n_obs, n_vars), dtype=np.float32,
                chunks=(n_obs, k), compressors=_x_compressors(), overwrite=True,
            )
            arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
            with _stage(f"Writing layers/{layer} (dense, {n_bands} column bands)"):
                for bi in range(n_bands):
                    c0, c1 = edges[bi], edges[bi + 1]
                    block = np.zeros((n_obs, c1 - c0), dtype=np.float32)
                    m = band_nnz[bi]
                    if m:
                        block[
                            np.asarray(buckets[bi]["rows"][:m]),
                            np.asarray(buckets[bi]["cols"][:m]) - c0,
                        ] = np.asarray(buckets[bi]["vals"][:m])
                    arr[:, c0:c1] = block
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    finalize()
    return []


def _sparse_layer(layers, name, enc, shape, nnz, indices_dtype, indptr_dtype, chunk_elems):
    import numpy as np

    g = layers.require_group(name)
    g.attrs.update({"encoding-type": enc, "encoding-version": "0.1.0", "shape": list(shape)})
    n_major = shape[0] if enc == "csr_matrix" else shape[1]
    flat = min(chunk_elems, max(1, nnz))
    g.require_array("data", shape=(nnz,), dtype=np.float32, chunks=(flat,),
                    compressors=_x_compressors(), overwrite=True)
    g.require_array("indices", shape=(nnz,), dtype=indices_dtype, chunks=(flat,),
                    compressors=_x_compressors(), overwrite=True)
    g.require_array("indptr", shape=(n_major + 1,), dtype=indptr_dtype,
                    chunks=(n_major + 1,), overwrite=True)
    for a in ("data", "indices", "indptr"):
        g[a].attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    return g
