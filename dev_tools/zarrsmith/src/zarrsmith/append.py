from __future__ import annotations

import sys

import zarr

from convert_to_zarr.config import AppConfig
from convert_to_zarr.engine import _run_parallel_threads, _stage, configure_runtime
from convert_to_zarr.errors import ConversionError
from convert_to_zarr.storage import _store_name, open_input_group, open_store_rw

_SEG_BYTES = 256 * 1024 * 1024
_INDEX_SCAN_ROWS = 1 << 20
_NULLABLE_ENCODINGS = ("nullable-integer", "nullable-boolean", "nullable-string-array")


def append_cells(
    store: str,
    cells: str,
    cfg: AppConfig,
    *,
    drop_derived: bool = False,
    assume_yes: bool = False,
    extend_layers: bool = False,
) -> list[str]:
    """Append the cells of another zarr store onto this one, in place.

    Extends X and obs only. Derived obs-aligned elements on the store (obsm embeddings,
    obsp graphs, layers) are invalidated by new cells and are dropped with consent
    (``drop_derived``); re-derive layers afterwards with add-expr. With ``extend_layers``,
    CSR layers created by add-expr (recorded target_sum, X's exact sparsity) are extended
    in place instead — the lognorm transform runs on the appended cells only."""
    import numpy as np
    from anndata.io import read_elem

    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(cells)
    root, finalize = open_store_rw(
        store, cfg,
        commit_message=f"zarrsmith append {_store_name(cells)} → {_store_name(store)}",
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
    obsp_keys = list(root["obsp"]) if "obsp" in root else []

    var_t, var_s = read_elem(root["var"]), read_elem(src["var"])
    if len(var_t) != len(var_s) or not (var_t.index == var_s.index).all():
        raise ConversionError("var mismatch: names + order must be identical between stores.")

    _check_obs_schema(root["obs"], src["obs"])

    x_t, x_s = root["X"], src["X"]
    if x_t["data"].dtype != x_s["data"].dtype:
        raise ConversionError(
            f"X dtype mismatch: {x_t['data'].dtype} vs {x_s['data'].dtype}."
        )
    n_t, n_vars = (int(v) for v in x_t.attrs["shape"])
    n_s = int(x_s.attrs["shape"][0])

    obsm_keys = list(root["obsm"]) if "obsm" in root else []

    indptr_t = np.asarray(x_t["indptr"][:], dtype=np.int64)
    indptr_s = np.asarray(x_s["indptr"][:], dtype=np.int64)

    ext_layers, bad_layers = [], []
    if extend_layers and layer_keys:
        ext_layers, bad_layers = _extendable_layers(root["layers"], layer_keys, indptr_t)
    drop_layers = [k for k in layer_keys if k not in ext_layers]

    has_dup_names = _has_duplicate_names(root["obs"], src["obs"], n_t)

    plan = []
    derived = [
        f"{g} {keys}"
        for g, keys in (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers))
        if keys
    ]
    layer_hint = "; re-derive layers with add-expr" if drop_layers else ""
    if bad_layers:
        layer_hint += f" (layers {bad_layers}: sparsity differs from X, cannot extend)"
    if derived:
        plan.append((
            "drop derived elements (invalidated by appended cells): "
            + ", ".join(derived) + layer_hint,
            drop_derived,
        ))
    # not a loss on the store — append carries X + obs only — so report, don't gate
    extras = []
    if "layers" in src and list(src["layers"]):
        extras.append(f"layers {list(src['layers'])}")
    if "raw" in src and len(list(src["raw"])) > 0:
        extras.append("raw")
    if "obsm" in src and list(src["obsm"]):
        extras.append(f"obsm {list(src['obsm'])}")
    if extras:
        warnings.append("left behind (not carried from the cells store): " + ", ".join(extras))
    _confirm(plan, assume_yes)

    # mutations start here; order keeps the store readable as its old self until
    # indptr/shape flip (plain zarr has no rollback — icechunk discards on failure)
    try:
        for group_key, keys in (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers)):
            for k in keys:
                del root[group_key][k]
        if derived:
            warnings.append(
                "dropped " + ", ".join(derived)
                + f" (invalidated by appended cells{layer_hint})."
            )
        _append_arrays(root, src, x_t, x_s, indptr_t, indptr_s, n_t, n_s, n_vars, cfg)
        if ext_layers:
            _extend_lognorm_layers(root, x_s, ext_layers, indptr_t, indptr_s, n_t + n_s, n_vars, cfg)
            warnings.append(
                f"extended layers {ext_layers} in place (lognorm applied to the appended "
                "cells at each layer's recorded target_sum)."
            )
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(
            f"append failed mid-mutation; {store} may be inconsistent "
            f"(plain zarr cannot roll back — icechunk discards uncommitted changes): {e}"
        ) from e

    if has_dup_names:
        warnings.append("obs names contain duplicates after append.")
    warnings.append(
        "appended cells break any sorted-store contiguity; re-run `zarrsmith sort` if the "
        "store was sorted."
    )
    finalize()
    return warnings


def _check_obs_schema(obs_t, obs_s) -> None:
    """Column-level schema equality at the zarr encoding level — no full obs read."""
    import numpy as np

    cols_t = list(obs_t.attrs["column-order"])
    cols_s = list(obs_s.attrs["column-order"])
    if cols_t != cols_s:
        raise ConversionError(f"obs schema mismatch: store {cols_t} vs cells {cols_s}.")
    for g, label, cols in ((obs_t, "store", cols_t), (obs_s, "cells", cols_s)):
        stray = sorted(set(g) - set(cols) - {g.attrs["_index"]})
        if stray:
            raise ConversionError(f"obs in {label} has elements outside column-order: {stray}.")

    pairs = [(c, obs_t[c], obs_s[c]) for c in cols_t]
    pairs.append(("<index>", obs_t[obs_t.attrs["_index"]], obs_s[obs_s.attrs["_index"]]))
    for name, t, s in pairs:
        enc = t.attrs.get("encoding-type")
        if enc != s.attrs.get("encoding-type"):
            raise ConversionError(
                f"obs column '{name}' encoding mismatch "
                f"({enc!r} vs {s.attrs.get('encoding-type')!r}); reconcile before append."
            )
        if enc == "categorical":
            cat_t, cat_s = t["categories"][:], s["categories"][:]
            # codes are positional, so categories must match in value AND order —
            # anything less silently remaps the appended labels
            if (
                bool(t.attrs.get("ordered", False)) != bool(s.attrs.get("ordered", False))
                or len(cat_t) != len(cat_s)
                or not (np.asarray(cat_t) == np.asarray(cat_s)).all()
            ):
                raise ConversionError(
                    f"obs column '{name}' categorical dtype mismatch "
                    "(categories, order, and the ordered flag must be identical); "
                    "reconcile before append."
                )
        elif enc in _NULLABLE_ENCODINGS:
            if t["values"].dtype != s["values"].dtype:
                raise ConversionError(
                    f"obs column '{name}' dtype mismatch "
                    f"({t['values'].dtype} vs {s['values'].dtype})."
                )
        elif enc == "array":
            if t.dtype != s.dtype:
                raise ConversionError(
                    f"obs column '{name}' dtype mismatch ({t.dtype} vs {s.dtype}); "
                    "obs columns extend in place, so dtypes must match exactly."
                )
        elif enc != "string-array":
            raise ConversionError(
                f"obs column '{name}': unsupported encoding {enc!r} for in-place append."
            )


def _has_duplicate_names(obs_t, obs_s, n_t: int) -> bool:
    """Duplicate obs-name check involving the appended cells — streamed over the store
    index in chunk-aligned slices, so memory stays O(cells store)."""
    import numpy as np

    idx_s = np.asarray(obs_s[obs_s.attrs["_index"]][:])
    if len(np.unique(idx_s)) < len(idx_s):
        return True
    t_arr = obs_t[obs_t.attrs["_index"]]
    chunk0 = t_arr.chunks[0]
    step = max(chunk0, (_INDEX_SCAN_ROWS // max(1, chunk0)) * chunk0)
    for i0 in range(0, n_t, step):
        if np.isin(np.asarray(t_arr[i0:min(i0 + step, n_t)]), idx_s).any():
            return True
    return False


def _append_arrays(root, src, x_t, x_s, indptr_t, indptr_s, n_t, n_s, n_vars, cfg):
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    n_new = n_t + n_s

    with _stage(f"Appending X ({n_s} cells, nnz={nnz_s})"):
        for name in ("data", "indices"):
            _extend_flat(x_t[name], x_s[name], nnz_t, nnz_s, cfg.chunks.cpus)
        _rewrite_indptr(x_t, indptr_t, indptr_s, n_new)
        x_t.attrs["shape"] = [n_new, n_vars]

    with _stage(f"Appending obs ({n_s} cells)"):
        _append_obs(root["obs"], src["obs"], n_t, n_new)


def _extend_flat(dst_a, src_a, off, n_src, cpus):
    """Resize dst by n_src and copy src[:n_src] to dst[off:]. After the seam, cuts land
    on dst chunk multiples: disjoint whole-chunk writes, so segments run threaded with
    no RMW."""
    dst_a.resize((off + n_src,))
    chunk0 = dst_a.chunks[0]
    step = max(chunk0, (_SEG_BYTES // max(1, chunk0 * dst_a.dtype.itemsize)) * chunk0)
    cuts = [0]
    seam = (-off) % chunk0
    if 0 < seam < n_src:
        cuts.append(seam)
    while cuts[-1] < n_src:
        cuts.append(min(n_src, cuts[-1] + step))
    jobs = [(src_a, dst_a, cuts[i], cuts[i + 1], off) for i in range(len(cuts) - 1)]
    _run_parallel_threads(_copy_shifted, jobs, cpus)


def _rewrite_indptr(parent, indptr_t, indptr_s, n_new):
    import numpy as np

    nnz_t = int(indptr_t[-1])
    nnz_new = nnz_t + int(indptr_s[-1])
    indptr_dtype = np.int64 if nnz_new > np.iinfo(np.int32).max else parent["indptr"].dtype
    del parent["indptr"]
    ip = parent.require_array(
        "indptr", shape=(n_new + 1,), dtype=indptr_dtype, chunks=(n_new + 1,), overwrite=True
    )
    ip.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    ip[:] = np.concatenate([indptr_t, indptr_s[1:] + nnz_t]).astype(indptr_dtype)


def _extendable_layers(layers, keys, indptr_t):
    """Split layer keys into (extendable, mismatched). Extendable: CSR with add-expr's
    recorded target_sum and X's exact sparsity (indptr identical), so extension is a
    shifted copy of X's new indices + the lognorm transform on the new cells' data.
    Mismatched carry the attr but a different sparsity — extending would corrupt them."""
    import numpy as np

    ext, bad = [], []
    for k in keys:
        node = layers[k]
        if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") != "csr_matrix":
            continue
        if node.attrs.get("zarrsmith_target_sum") is None:
            continue
        lp = np.asarray(node["indptr"][:], dtype=np.int64)
        if len(lp) != len(indptr_t) or not (lp == indptr_t).all():
            bad.append(k)
            continue
        ext.append(k)
    return ext, bad


def _extend_lognorm_layers(root, x_s, keys, indptr_t, indptr_s, n_new, n_vars, cfg):
    """Extend add-expr CSR layers in place: indices shift-copy from the cells store's X
    (identical sparsity), indptr is value-identical to X's appended indptr, and data gets
    the lognorm transform over the new cells only — no old row is read or rewritten."""
    import numpy as np

    from .expr import _lognorm_band

    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    row_nnz_s = np.diff(indptr_s)
    n_s = len(row_nnz_s)
    row_step = max(1_000, min(200_000, _SEG_BYTES // (max(1, nnz_s // max(1, n_s)) * 12)))
    for k in keys:
        g = root["layers"][k]
        target_sum = float(g.attrs["zarrsmith_target_sum"])
        with _stage(f"Extending layers/{k} ({n_s} cells, nnz={nnz_s})"):
            _extend_flat(g["indices"], x_s["indices"], nnz_t, nnz_s, cfg.chunks.cpus)
            data = g["data"]
            data.resize((nnz_t + nnz_s,))
            for b0 in range(0, n_s, row_step):
                b1 = min(b0 + row_step, n_s)
                s0, s1, vals = _lognorm_band(x_s["data"], indptr_s, row_nnz_s, target_sum, b0, b1)
                data[nnz_t + s0:nnz_t + s1] = vals
            _rewrite_indptr(g, indptr_t, indptr_s, n_new)
            g.attrs["shape"] = [n_new, n_vars]


def _append_obs(obs_t, obs_s, n_t: int, n_new: int) -> None:
    """Extend each obs column in place — O(cells store) memory, no target rewrite."""
    pairs = [(c, c) for c in obs_t.attrs["column-order"]]
    pairs.append((obs_t.attrs["_index"], obs_s.attrs["_index"]))
    for name_t, name_s in pairs:
        t, s = obs_t[name_t], obs_s[name_s]
        if isinstance(t, zarr.Array):
            _extend_1d(t, s, n_t, n_new)
        elif t.attrs.get("encoding-type") == "categorical":
            _extend_1d(t["codes"], s["codes"], n_t, n_new)  # categories validated identical
        else:  # nullable-*: values + mask
            _extend_1d(t["values"], s["values"], n_t, n_new)
            _extend_1d(t["mask"], s["mask"], n_t, n_new)


def _extend_1d(dst, src, n_t: int, n_new: int) -> None:
    vals = src[:]
    if src.dtype != dst.dtype:  # e.g. categorical codes stored at different widths
        vals = vals.astype(dst.dtype)
    dst.resize((n_new,))
    dst[n_t:n_new] = vals


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
        "Pass --yes (or --drop-derived) to proceed non-interactively."
    )
