from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import zarr

from ..config import AppConfig
from ..engine import _stage
from ..errors import ConversionError
from ..storage import open_input_group, open_store_rw
from .expr import add_expr_layer

_SEG_BYTES = 256 * 1024 * 1024


def append_cells(
    store: str,
    cells: str,
    cfg: AppConfig,
    *,
    drop_obsp: bool = False,
    refresh_expr: bool = False,
) -> list[str]:
    """Append the cells of another zarr store onto this one, in place."""
    import numpy as np
    import pandas as pd
    from anndata._io.specs import write_elem
    from anndata.io import read_elem

    src = open_input_group(cells)
    store_path = Path(store)
    root, finalize = open_store_rw(
        store_path, cfg, commit_message=f"zarrsmith append {Path(cells).name} → {store_path.name}"
    )
    warnings: list[str] = []

    for g, label in ((root, "store"), (src, "cells")):
        if g["X"].attrs.get("encoding-type") != "csr_matrix":
            raise ConversionError(
                f"append requires CSR X in {label}; got {g['X'].attrs.get('encoding-type')!r}."
            )
    if "raw" in root and len(list(root["raw"])) > 0:
        raise ConversionError("append does not extend raw (it is obs-aligned); drop raw first.")

    layer_keys = list(root["layers"]) if "layers" in root else []
    gexp_params = None
    if layer_keys:
        if not refresh_expr or layer_keys != ["gexp"]:
            raise ConversionError(
                f"store has layers {layer_keys} that new cells would leave stale; pass "
                "--refresh-expr to re-derive layers/gexp (other layers are unsupported)."
            )
        gexp_params = _introspect_gexp(root["layers"]["gexp"])

    obsp_keys = list(root["obsp"]) if "obsp" in root else []
    if obsp_keys:
        if not drop_obsp:
            raise ConversionError(
                f"store has obsp graphs {obsp_keys} invalidated by new cells; pass --drop-obsp."
            )
        for k in obsp_keys:
            del root["obsp"][k]
        warnings.append(f"dropped obsp graphs {obsp_keys} (invalidated by appended cells).")

    var_t, var_s = read_elem(root["var"]), read_elem(src["var"])
    if len(var_t) != len(var_s) or not (var_t.index == var_s.index).all():
        raise ConversionError("var mismatch: names + order must be identical between stores.")

    obs_t, obs_s = read_elem(root["obs"]), read_elem(src["obs"])
    if list(obs_t.columns) != list(obs_s.columns):
        raise ConversionError(
            f"obs schema mismatch: store {list(obs_t.columns)} vs cells {list(obs_s.columns)}."
        )
    for c in obs_t.columns:
        t_cat = isinstance(obs_t[c].dtype, pd.CategoricalDtype)
        s_cat = isinstance(obs_s[c].dtype, pd.CategoricalDtype)
        if t_cat != s_cat or (
            t_cat and set(obs_t[c].cat.categories) != set(obs_s[c].cat.categories)
        ):
            raise ConversionError(
                f"obs column '{c}' categorical mismatch; reconcile categories before append."
            )

    x_t, x_s = root["X"], src["X"]
    if x_t["data"].dtype != x_s["data"].dtype:
        raise ConversionError(
            f"X dtype mismatch: {x_t['data'].dtype} vs {x_s['data'].dtype}."
        )
    n_t, n_vars = (int(v) for v in x_t.attrs["shape"])
    n_s = int(x_s.attrs["shape"][0])
    n_new = n_t + n_s

    obsm_keys = list(root["obsm"]) if "obsm" in root else []
    src_obsm = src["obsm"] if "obsm" in src else None
    for k in obsm_keys:
        if src_obsm is None or k not in src_obsm:
            raise ConversionError(f"cells store lacks obsm entry '{k}' present in store.")
        a, b = root["obsm"][k], src_obsm[k]
        if not isinstance(a, zarr.Array) or not isinstance(b, zarr.Array):
            raise ConversionError(f"append only supports plain-array obsm entries (obsm/{k}).")
        if a.shape[1:] != b.shape[1:] or a.dtype != b.dtype:
            raise ConversionError(f"obsm/{k} shape/dtype mismatch between stores.")

    indptr_t = np.asarray(x_t["indptr"][:], dtype=np.int64)
    indptr_s = np.asarray(x_s["indptr"][:], dtype=np.int64)
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    nnz_new = nnz_t + nnz_s

    with _stage(f"Appending X ({n_s} cells, nnz={nnz_s})"):
        for name in ("data", "indices"):
            dst_a, src_a = x_t[name], x_s[name]
            dst_a.resize((nnz_new,))
            seg = max(1, _SEG_BYTES // max(1, dst_a.dtype.itemsize))
            for s0 in range(0, nnz_s, seg):
                s1 = min(s0 + seg, nnz_s)
                dst_a[nnz_t + s0:nnz_t + s1] = src_a[s0:s1]
        indptr_dtype = np.int64 if nnz_new > np.iinfo(np.int32).max else x_t["indptr"].dtype
        del x_t["indptr"]
        ip = x_t.require_array(
            "indptr", shape=(n_new + 1,), dtype=indptr_dtype, chunks=(n_new + 1,), overwrite=True
        )
        ip.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
        ip[:] = np.concatenate([indptr_t, indptr_s[1:] + nnz_t]).astype(indptr_dtype)
        x_t.attrs["shape"] = [n_new, n_vars]

    with _stage(f"Appending obs + obsm ({n_s} cells)"):
        obs_new = pd.concat([obs_t, obs_s], axis=0)
        if obs_new.index.duplicated().any():
            warnings.append("obs names contain duplicates after append.")
        if "obs" in root:
            del root["obs"]
        write_elem(root, "obs", obs_new)
        for k in obsm_keys:
            a = root["obsm"][k]
            a.resize((n_new,) + a.shape[1:])
            a[n_t:n_new] = src["obsm"][k][:]

    warnings.append(
        "appended cells break any sorted-store contiguity; re-run `zarrsmith sort` if the "
        "store was sorted."
    )
    finalize()

    if gexp_params is not None:
        fmt, chunk_elems = gexp_params
        cfg_ow = replace(cfg, io=replace(cfg.io, overwrite=True))
        add_expr_layer(store, cfg_ow, fmt=fmt, chunk_elems=chunk_elems)
        warnings.append(f"layers/gexp re-derived ({fmt}) over the appended matrix.")
    return warnings


def _introspect_gexp(node) -> tuple[str, int]:
    if isinstance(node, zarr.Array):
        return "dense", node.chunks[0] * node.chunks[1]
    enc = node.attrs.get("encoding-type")
    fmt = {"csr_matrix": "csr", "csc_matrix": "csc"}.get(enc)
    if fmt is None:
        raise ConversionError(f"cannot refresh layers/gexp: unsupported encoding {enc!r}.")
    return fmt, int(node["data"].chunks[0])
