from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from annizarr import _layout
from annizarr._config import load_config
from annizarr._layout import x_compressors
from annizarr._ops._result import OpResult
from annizarr._runtime import configure_runtime, stage
from annizarr._storage import is_remote, open_store_rw
from annizarr._writers._encoding import make_sparse_group, set_array_attrs
from annizarr._writers._sparse import write_transposed_sparse
from annizarr._zarr import get_array, get_group, shape_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from annizarr._config import AppConfig
    from annizarr.typing import PathLike, XStorage

logger = logging.getLogger(__name__)

_TARGET_SUM_ATTR = "annizarr_target_sum"
_TARGET_SUM_ATTR_LEGACY = "zarrsmith_target_sum"  # written by pre-merge zarrsmith; read-compat only


def target_sum_attr(attrs: Any) -> float | None:
    value = attrs.get(_TARGET_SUM_ATTR, attrs.get(_TARGET_SUM_ATTR_LEGACY))
    return None if value is None else float(value)


def add_expr(
    store: PathLike,
    *,
    fmt: XStorage = "csc",
    layer: str = "gexp",
    chunk_elems: int = 1_000_000,
    target_sum: float = 1e4,
    overwrite: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Add a log-normalized expression layer (``layers/<layer>``) derived from CSR X.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store to update, in place.
    fmt
        Storage format for the new layer.
    layer
        Layer name.
    chunk_elems
        Chunk size (elements) for the layer.
    target_sum
        Library-size normalization target (per-row sum after normalization).
    overwrite
        Replace an existing ``layers/<layer>`` instead of erroring.
    cfg
        Resolved configuration; ``None`` loads :func:`~annizarr.config.load_config` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op, ``fmt``, and the layer.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``store`` has no CSR X, ``layers/<layer>`` already exists and ``overwrite`` is
        not set, or ``fmt`` is not one of ``"csr"``, ``"csc"``, ``"dense"``.
    """
    if cfg is None:
        cfg = load_config()
    if fmt not in ("csc", "dense", "csr"):
        raise ConversionError(f"add-expr format must be csc, dense, or csr; got '{fmt}'.")

    configure_runtime(cfg.chunks.cpus)
    commit_message = message or f"annizarr add-expr {fmt} → layers/{layer}"
    root, finalize = open_store_rw(store, cfg, commit_message=commit_message, branch=branch)
    if "X" not in root:
        raise ConversionError(f"no X in {store} — not an AnnData zarr store?")
    x = get_group(root, "X")
    if x.attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(f"add-expr requires CSR X; got encoding {x.attrs.get('encoding-type')!r}.")

    n_obs, n_vars = shape_attr(x)
    if max(n_obs, n_vars) > 2**31 - 1:
        raise ConversionError("add-expr supports up to 2^31-1 cells/genes.")
    data_arr, idx_arr = get_array(x, "data"), get_array(x, "indices")
    indptr = np.asarray(get_array(x, "indptr")[:], dtype=np.int64)
    nnz = int(indptr[-1])
    row_nnz = np.diff(indptr)

    layers = root.require_group("layers")
    if "encoding-type" not in dict(layers.attrs):
        layers.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    if layer in layers:
        if not overwrite:
            raise ConversionError(f"layers/{layer} already exists; pass overwrite=True to replace it.")
        del layers[layer]

    bytes_per_row = max(1, nnz // max(1, n_obs)) * 12
    row_step = max(1_000, min(200_000, _layout.BATCH_BYTES // bytes_per_row))

    def _band(b0: int, b1: int) -> tuple[int, int, NDArray[np.float32]]:
        return lognorm_band(data_arr, indptr, row_nnz, target_sum, b0, b1)

    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    if fmt == "csr":
        g = _sparse_layer(
            layers, layer, "csr_matrix", (n_obs, n_vars), nnz, idx_arr.dtype, indptr_dtype, chunk_elems, target_sum
        )
        g["indptr"][:] = indptr.astype(indptr_dtype)
        with stage(f"Writing layers/{layer} (csr, nnz={nnz})"):
            for b0 in range(0, n_obs, row_step):
                b1 = min(b0 + row_step, n_obs)
                s0, s1, vals = _band(b0, b1)
                g["data"][s0:s1] = vals
                g["indices"][s0:s1] = idx_arr[s0:s1]
        snapshot_id = finalize()
        return OpResult(path=str(store), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)

    if fmt == "csc":
        # Factors are computed up front, in the same row_step bands lognorm_band uses,
        # so they match bit-for-bit what the fused per-band computation used to produce;
        # the transpose (bucket-by-column, disk-backed, RAM bounded to one band) then
        # lives once in _writers._sparse, shared with write_matrix's backed CSR->CSC.
        factors = _lognorm_factors(data_arr, indptr, target_sum, row_step, n_obs)
        layer_cfg = replace(cfg, chunks=replace(cfg.chunks, sparse_flat_chunk=chunk_elems))
        with stage(f"Writing layers/{layer} (csc, nnz={nnz})"):
            g = write_transposed_sparse(layers, layer, x, layer_cfg, row_scale=factors, target="csc")
        g.attrs[_TARGET_SUM_ATTR] = float(target_sum)
        snapshot_id = finalize()
        return OpResult(path=str(store), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)

    # fmt == "dense": pass 1 counts nnz per column; pass 2 buckets entries into column
    # bands (disk-backed, so RAM stays one band); each band then scatters into its slice
    # of the dense array. Self-contained (not shared with the csc path above) since the
    # final materialisation — and so the band sizing — differs from a sparse write.
    col_nnz = np.zeros(n_vars, dtype=np.int64)
    flat_step = max(chunk_elems, _layout.BATCH_BYTES // 8)
    with stage("Counting nnz per gene"):
        for s0 in range(0, nnz, flat_step):
            s1 = min(s0 + flat_step, nnz)
            col_nnz += np.bincount(np.asarray(idx_arr[s0:s1]), minlength=n_vars)
    csc_indptr = np.concatenate([[0], np.cumsum(col_nnz)]).astype(np.int64)

    k = max(1, chunk_elems // n_obs)
    band_cols = max(k, (_layout.BATCH_BYTES // (4 * n_obs)) // k * k)
    edges = [*range(0, n_vars, band_cols), n_vars]
    n_bands = len(edges) - 1
    band_nnz = [int(csc_indptr[edges[i + 1]] - csc_indptr[edges[i]]) for i in range(n_bands)]

    # bucket temp files sit next to a local store (same filesystem); system tmp for a remote one
    tmp_dir = None if is_remote(store) else str(Path(store).parent)
    tmp_root = Path(tempfile.mkdtemp(prefix="annizarr_expr_", dir=tmp_dir))
    try:
        buckets: list[dict[str, NDArray[Any]]] = []
        for i, m in enumerate(band_nnz):
            m = max(1, m)
            buckets.append(
                {
                    "rows": np.memmap(tmp_root / f"r{i}", dtype=np.int32, mode="w+", shape=(m,)),
                    "cols": np.memmap(tmp_root / f"c{i}", dtype=np.int32, mode="w+", shape=(m,)),
                    "vals": np.memmap(tmp_root / f"v{i}", dtype=np.float32, mode="w+", shape=(m,)),
                }
            )
        edges_arr = np.asarray(edges[1:], dtype=np.int64)

        cursors = [0] * n_bands
        with stage(f"Bucketing {nnz} entries into {n_bands} gene bands"):
            for b0 in range(0, n_obs, row_step):
                b1 = min(b0 + row_step, n_obs)
                s0, s1, vals = _band(b0, b1)
                cols = np.asarray(idx_arr[s0:s1], dtype=np.int32)
                rows = np.repeat(np.arange(b0, b1, dtype=np.int32), row_nnz[b0:b1])
                band_ids = np.searchsorted(edges_arr, cols, side="right")
                order = np.argsort(band_ids, kind="stable")
                bounds = np.searchsorted(band_ids[order], np.arange(n_bands + 1))
                for bi in range(n_bands):
                    lo, hi = int(bounds[bi]), int(bounds[bi + 1])
                    if lo == hi:
                        continue
                    sel = order[lo:hi]
                    c = cursors[bi]
                    buckets[bi]["rows"][c : c + hi - lo] = rows[sel]
                    buckets[bi]["cols"][c : c + hi - lo] = cols[sel]
                    buckets[bi]["vals"][c : c + hi - lo] = vals[sel]
                    cursors[bi] = c + hi - lo

        arr = layers.require_array(
            layer,
            shape=(n_obs, n_vars),
            dtype=np.float32,
            chunks=(n_obs, k),
            compressors=x_compressors(),
            overwrite=True,
        )
        arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0", _TARGET_SUM_ATTR: float(target_sum)})
        with stage(f"Writing layers/{layer} (dense, {n_bands} column bands)"):
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

    snapshot_id = finalize()
    return OpResult(path=str(store), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)


def lognorm_band(
    data_arr: Any, indptr: NDArray[np.int64], row_nnz: NDArray[np.int64], target_sum: float, b0: int, b1: int
) -> tuple[int, int, NDArray[np.float32]]:
    """Lognorm one row band of CSR data. Factors are row-local, so they fuse into the
    transform: one pass over the band's data. Also used by append --extend-layers."""
    s0, s1 = int(indptr[b0]), int(indptr[b1])
    seg = np.asarray(data_arr[s0:s1], dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(seg)))
    sums = cs[indptr[b0 + 1 : b1 + 1] - indptr[b0]] - cs[indptr[b0:b1] - indptr[b0]]
    factors = np.zeros(b1 - b0)
    nz = sums > 0
    factors[nz] = target_sum / sums[nz]
    vals = np.log1p(seg * np.repeat(factors, row_nnz[b0:b1])).astype(np.float32)
    return s0, s1, vals


def _lognorm_factors(
    data_arr: Any, indptr: NDArray[np.int64], target_sum: float, row_step: int, n_obs: int
) -> NDArray[np.float64]:
    """Per-row ``target_sum / row_sum`` factors, in the same ``row_step`` bands
    :func:`lognorm_band` uses (so the sums match bit-for-bit); feeds
    :func:`~annizarr._writers._sparse.write_transposed_sparse`'s ``row_scale``."""
    factors = np.zeros(n_obs, dtype=np.float64)
    for b0 in range(0, n_obs, row_step):
        b1 = min(b0 + row_step, n_obs)
        s0, s1 = int(indptr[b0]), int(indptr[b1])
        seg = np.asarray(data_arr[s0:s1], dtype=np.float64)
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        sums = cs[indptr[b0 + 1 : b1 + 1] - indptr[b0]] - cs[indptr[b0:b1] - indptr[b0]]
        band_factors = np.zeros(b1 - b0)
        nz = sums > 0
        band_factors[nz] = target_sum / sums[nz]
        factors[b0:b1] = band_factors
    return factors


def introspect_gexp(node: Any) -> tuple[XStorage, int, float | None]:
    # recovers (fmt, chunk_elems, target_sum) from an existing gexp layer
    import zarr

    target_sum = target_sum_attr(node.attrs)
    if isinstance(node, zarr.Array):
        return "dense", node.chunks[0] * node.chunks[1], target_sum
    enc = node.attrs.get("encoding-type")
    fmt_map: dict[str, XStorage] = {"csr_matrix": "csr", "csc_matrix": "csc"}
    fmt = fmt_map.get(enc)
    if fmt is None:
        raise ConversionError(f"cannot re-derive layers/gexp: unsupported encoding {enc!r}.")
    return fmt, int(node["data"].chunks[0]), target_sum


def _sparse_layer(
    layers: Any,
    name: str,
    enc: str,
    shape: tuple[int, int],
    nnz: int,
    indices_dtype: Any,
    indptr_dtype: Any,
    chunk_elems: int,
    target_sum: float,
) -> Any:
    g = make_sparse_group(layers, name, csr=(enc == "csr_matrix"), shape=shape)
    g.attrs[_TARGET_SUM_ATTR] = float(target_sum)
    n_major = shape[0] if enc == "csr_matrix" else shape[1]
    flat = min(chunk_elems, max(1, nnz))
    g.require_array("data", shape=(nnz,), dtype=np.float32, chunks=(flat,), compressors=x_compressors(), overwrite=True)
    g.require_array(
        "indices", shape=(nnz,), dtype=indices_dtype, chunks=(flat,), compressors=x_compressors(), overwrite=True
    )
    g.require_array("indptr", shape=(n_major + 1,), dtype=indptr_dtype, chunks=(n_major + 1,), overwrite=True)
    for a in ("data", "indices", "indptr"):
        set_array_attrs(get_array(g, a))
    return g
