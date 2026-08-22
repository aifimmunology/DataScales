from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import zarr

from ..config import AppConfig
from ..engine import _run_parallel_threads, _stage, configure_runtime
from ..errors import ConversionError
from ..storage import open_input_group, open_store_rw
from .expr import _introspect_gexp, add_expr_layer

_SEG_BYTES = 256 * 1024 * 1024


def append_cells(
    store: str,
    cells: str,
    cfg: AppConfig,
    *,
    drop_obsp: bool = False,
    refresh_expr: bool = False,
    assume_yes: bool = False,
) -> list[str]:
    """Append the cells of another zarr store onto this one, in place."""
    import numpy as np
    import pandas as pd
    from anndata._io.specs import write_elem
    from anndata.io import read_elem

    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(cells)
    store_path = Path(store)
    root, finalize = open_store_rw(
        store_path, cfg, commit_message=f"zarrsmith append {Path(cells).name} → {store_path.name}"
    )
    warnings: list[str] = []

    for g, label in ((root, "store"), (src, "cells")):
        if "X" not in g:
            raise ConversionError(f"no X in {label} store — not an AnnData zarr store?")
        if g["X"].attrs.get("encoding-type") != "csr_matrix":
            raise ConversionError(
                f"append requires CSR X in {label}; got {g['X'].attrs.get('encoding-type')!r}."
            )
    if "raw" in root and len(list(root["raw"])) > 0:
        raise ConversionError("append does not extend raw (it is obs-aligned); drop raw first.")

    layer_keys = list(root["layers"]) if "layers" in root else []
    gexp_params = None
    if layer_keys and layer_keys != ["gexp"]:
        raise ConversionError(
            f"append cannot handle layers {layer_keys}; only a lone gexp layer is supported."
        )
    if layer_keys:
        gexp_params = _introspect_gexp(root["layers"]["gexp"])

    obsp_keys = list(root["obsp"]) if "obsp" in root else []

    var_t, var_s = read_elem(root["var"]), read_elem(src["var"])
    if len(var_t) != len(var_s) or not (var_t.index == var_s.index).all():
        raise ConversionError("var mismatch: names + order must be identical between stores.")

    obs_t, obs_s = read_elem(root["obs"]), read_elem(src["obs"])
    if list(obs_t.columns) != list(obs_s.columns):
        raise ConversionError(
            f"obs schema mismatch: store {list(obs_t.columns)} vs cells {list(obs_s.columns)}."
        )
    for c in obs_t.columns:
        is_cat = isinstance(obs_t[c].dtype, pd.CategoricalDtype) or isinstance(
            obs_s[c].dtype, pd.CategoricalDtype
        )
        # dtype equality covers categories, order (for ordered), and the ordered flag —
        # anything less degrades the column to a string array on concat
        if is_cat and obs_t[c].dtype != obs_s[c].dtype:
            raise ConversionError(
                f"obs column '{c}' categorical dtype mismatch "
                f"({obs_t[c].dtype} vs {obs_s[c].dtype}); reconcile before append."
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

    plan = []
    if obsp_keys:
        plan.append((f"drop obsp graphs {obsp_keys} (invalidated by new cells)", drop_obsp))
    if gexp_params is not None:
        plan.append(("re-derive layers/gexp (stale over the appended matrix)", refresh_expr))
    extras = []
    if "layers" in src and list(src["layers"]):
        extras.append(f"layers {list(src['layers'])}")
    if "raw" in src and len(list(src["raw"])) > 0:
        extras.append("raw")
    extra_obsm = [k for k in (list(src_obsm) if src_obsm is not None else []) if k not in obsm_keys]
    if extra_obsm:
        extras.append(f"obsm {extra_obsm}")
    if extras:
        plan.append(("leave behind (not carried from the cells store): " + ", ".join(extras), False))
    _confirm(plan, assume_yes)

    # mutations start here; order keeps the store readable as its old self until
    # indptr/shape flip (plain zarr has no rollback — icechunk discards on failure)
    try:
        if obsp_keys:
            for k in obsp_keys:
                del root["obsp"][k]
            warnings.append(f"dropped obsp graphs {obsp_keys} (invalidated by appended cells).")
        _append_arrays(root, x_t, x_s, obs_t, obs_s, obsm_keys, src_obsm,
                       indptr_t, indptr_s, n_t, n_s, n_vars, warnings, cfg)
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(
            f"append failed mid-mutation; {store_path} may be inconsistent "
            f"(plain zarr cannot roll back — icechunk discards uncommitted changes): {e}"
        ) from e

    warnings.append(
        "appended cells break any sorted-store contiguity; re-run `zarrsmith sort` if the "
        "store was sorted."
    )
    finalize()

    if gexp_params is not None:
        fmt, chunk_elems, target_sum = gexp_params
        cfg_ow = replace(cfg, io=replace(cfg.io, overwrite=True))
        if target_sum is None:
            warnings.append(
                "layers/gexp has no recorded target_sum (pre-zarrsmith layer); "
                "re-deriving at the default 1e4."
            )
            target_sum = 1e4
        add_expr_layer(store, cfg_ow, fmt=fmt, chunk_elems=chunk_elems, target_sum=target_sum)
        warnings.append(f"layers/gexp re-derived ({fmt}) over the appended matrix.")
    return warnings


def _append_arrays(root, x_t, x_s, obs_t, obs_s, obsm_keys, src_obsm,
                   indptr_t, indptr_s, n_t, n_s, n_vars, warnings, cfg):
    import numpy as np
    import pandas as pd
    from anndata._io.specs import write_elem

    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    nnz_new = nnz_t + nnz_s
    n_new = n_t + n_s

    with _stage(f"Appending X ({n_s} cells, nnz={nnz_s})"):
        for name in ("data", "indices"):
            dst_a, src_a = x_t[name], x_s[name]
            dst_a.resize((nnz_new,))
            # after the seam, cut on dst chunk multiples: disjoint whole-chunk
            # writes, so segments can run threaded with no RMW
            chunk0 = dst_a.chunks[0]
            step = max(chunk0, (_SEG_BYTES // max(1, chunk0 * dst_a.dtype.itemsize)) * chunk0)
            cuts = [0]
            seam = (-nnz_t) % chunk0
            if 0 < seam < nnz_s:
                cuts.append(seam)
            while cuts[-1] < nnz_s:
                cuts.append(min(nnz_s, cuts[-1] + step))
            jobs = [(src_a, dst_a, cuts[i], cuts[i + 1], nnz_t) for i in range(len(cuts) - 1)]
            _run_parallel_threads(_copy_shifted, jobs, cfg.chunks.cpus)
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
            a[n_t:n_new] = src_obsm[k][:]


def _copy_shifted(src, dst, s0, s1, off):
    dst[off + s0:off + s1] = src[s0:s1]


def _confirm(plan: list[tuple[str, bool]], assume_yes: bool) -> None:
    """Present the loss plan; proceed only with a flag, --yes, or an interactive yes."""
    if not plan or all(ok for _, ok in plan):
        return
    lines = "\n".join(f"  - {d}" for d, _ in plan)
    print(f"append will:\n{lines}", flush=True, file=sys.stderr)
    if assume_yes:
        return
    if sys.stdin.isatty():
        if input("Proceed? [y/N] ").strip().lower() in ("y", "yes"):
            return
        raise ConversionError("append cancelled.")
    raise ConversionError(
        f"append needs confirmation:\n{lines}\n"
        "Pass --yes (or --drop-obsp / --refresh-expr) to proceed non-interactively."
    )
