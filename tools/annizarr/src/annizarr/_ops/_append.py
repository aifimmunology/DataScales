from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from anndata.io import read_elem

from annizarr import _layout
from annizarr._config import load_config
from annizarr._ops._expr import lognorm_band, target_sum_attr
from annizarr._ops._result import AppendPlan, OpResult
from annizarr._runtime import configure_runtime, run_parallel, stage
from annizarr._storage import open_input_group, open_store_rw, store_name
from annizarr._zarr import as_array, as_group, get_array, get_group, shape_attr, str_attr, str_list_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from annizarr._config import AppConfig
    from annizarr.typing import PathLike

logger = logging.getLogger(__name__)

_INDEX_SCAN_ROWS = 1 << 20
_NULLABLE_ENCODINGS = ("nullable-integer", "nullable-boolean", "nullable-string-array")


def plan_append(store: PathLike, *, cells: PathLike) -> AppendPlan:
    """Validate an append and describe what it would do, without mutating anything.

    Reads metadata only (attrs, obs schema, small index/indptr arrays) from both stores.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store the cells would be appended onto.
    cells
        Another AnnData zarr store whose cells would be appended.

    Returns
    -------
    AppendPlan
        What appending would do: new-cell count, derived elements that would be
        dropped, layers eligible for in-place extension, and other notes.

    Raises
    ------
    ConversionError
        ``store``/``cells`` is not a CSR AnnData zarr store, ``var`` names/order
        mismatch, obs schema mismatch, an X dtype mismatch, or every appended cell's
        obs name is already present in the store (already appended?).
    """
    root = open_input_group(store)
    src = open_input_group(cells)

    for g, label in ((root, "store"), (src, "cells")):
        if "X" not in g:
            raise ConversionError(f"no X in {label} store — not an AnnData zarr store?")
        x = get_group(g, "X")
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise ConversionError(f"append requires CSR X in {label}; got {x.attrs.get('encoding-type')!r}.")
    if "raw" in root and len(list(get_group(root, "raw"))) > 0:
        raise ConversionError("append does not extend raw (it is obs-aligned); drop raw first.")

    layer_keys = list(get_group(root, "layers")) if "layers" in root else []
    obsp_keys = list(get_group(root, "obsp")) if "obsp" in root else []
    obsm_keys = list(get_group(root, "obsm")) if "obsm" in root else []

    var_t, var_s = read_elem(root["var"]), read_elem(src["var"])
    if len(var_t) != len(var_s) or not (var_t.index == var_s.index).all():
        raise ConversionError("var mismatch: names + order must be identical between stores.")

    _check_obs_schema(get_group(root, "obs"), get_group(src, "obs"))

    x_t, x_s = get_group(root, "X"), get_group(src, "X")
    data_t, data_s = get_array(x_t, "data"), get_array(x_s, "data")
    if data_t.dtype != data_s.dtype:
        raise ConversionError(f"X dtype mismatch: {data_t.dtype} vs {data_s.dtype}.")

    n_t, _ = shape_attr(x_t)
    n_s, _ = shape_attr(x_s)
    indptr_t = np.asarray(get_array(x_t, "indptr")[:], dtype=np.int64)

    ext_layers: list[str] = []
    bad_layers: list[str] = []
    if layer_keys:
        ext_layers, bad_layers = _extendable_layers(get_group(root, "layers"), layer_keys, indptr_t)
    drop_layers = [k for k in layer_keys if k not in ext_layers]

    notes: list[str] = []
    if bad_layers:
        notes.append(f"layers {bad_layers}: sparsity differs from X, cannot extend.")
    extras = []
    if "layers" in src and list(get_group(src, "layers")):
        extras.append(f"layers {list(get_group(src, 'layers'))}")
    if "raw" in src and len(list(get_group(src, "raw"))) > 0:
        extras.append("raw")
    if "obsm" in src and list(get_group(src, "obsm")):
        extras.append(f"obsm {list(get_group(src, 'obsm'))}")
    if extras:
        notes.append("left behind (not carried from the cells store): " + ", ".join(extras))

    n_duplicate_names = _count_duplicate_names(get_group(root, "obs"), get_group(src, "obs"), n_t)
    if n_s > 0 and n_duplicate_names == n_s:
        raise ConversionError(
            f"append would add no new cells: all {n_s} appended cells are already present "
            "in the store (already appended?)."
        )

    return AppendPlan(
        n_new=n_s,
        drop_obsm=tuple(obsm_keys),
        drop_obsp=tuple(obsp_keys),
        drop_layers=tuple(drop_layers),
        extendable_layers=tuple(ext_layers),
        n_duplicate_names=n_duplicate_names,
        notes=tuple(notes),
    )


def append(
    store: PathLike,
    *,
    cells: PathLike,
    drop_derived: bool = False,
    extend_layers: bool = False,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Append the cells of another zarr store onto this one, in place.

    Extends X and obs only. Derived obs-aligned elements on the store (obsm embeddings,
    obsp graphs, layers) are invalidated by new cells; :func:`plan_append` is called first,
    and if it would drop anything this raises unless ``drop_derived=True`` (re-derive
    layers afterwards with :func:`~annizarr._ops._expr.add_expr`). With ``extend_layers``,
    CSR layers created by add-expr (recorded target_sum, X's exact sparsity) are extended
    in place instead — the lognorm transform runs on the appended cells only.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store to append onto, in place.
    cells
        Another AnnData zarr store whose cells are appended.
    drop_derived
        Consent to dropping derived obs-aligned elements the plan says would drop.
    extend_layers
        Extend eligible add-expr CSR layers in place instead of dropping them.
    cfg
        Resolved configuration; ``None`` loads :func:`~annizarr.config.load_config` defaults.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet.
        Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the two stores involved.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        The plan (see :func:`plan_append`) would drop derived elements and
        ``drop_derived`` was not given, every appended cell is already present (see
        :func:`plan_append`), or the mutation fails partway through.
    """
    if cfg is None:
        cfg = load_config()

    plan = plan_append(store, cells=cells)
    drop_layers = list(plan.drop_layers)
    ext_layers = list(plan.extendable_layers) if extend_layers else []
    if not extend_layers:
        drop_layers += list(plan.extendable_layers)

    drops = plan.drops(extend_layers=extend_layers)
    if drops and not drop_derived:
        raise ConversionError(
            "append will drop derived elements (invalidated by appended cells): "
            + ", ".join(drops)
            + ". Pass drop_derived=True to proceed; re-derive layers with add-expr afterwards."
        )

    for note in plan.notes:
        logger.warning(note)

    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(cells)
    commit_message = message or f"annizarr append {store_name(cells)} → {store_name(store)}"
    root, finalize = open_store_rw(store, cfg, commit_message=commit_message, branch=branch)

    x_t, x_s = get_group(root, "X"), get_group(src, "X")
    n_t, n_vars = shape_attr(x_t)
    n_s, _ = shape_attr(x_s)
    indptr_t = np.asarray(get_array(x_t, "indptr")[:], dtype=np.int64)
    indptr_s = np.asarray(get_array(x_s, "indptr")[:], dtype=np.int64)

    obsm_keys, obsp_keys = list(plan.drop_obsm), list(plan.drop_obsp)
    try:
        for group_key, keys in (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers)):
            for k in keys:
                del get_group(root, group_key)[k]
        drop_groups = (("obsm", obsm_keys), ("obsp", obsp_keys), ("layers", drop_layers))
        dropped = [f"{g} {list(k)}" for g, k in drop_groups if k]
        if dropped:
            hint = "; re-derive layers with add-expr" if drop_layers else ""
            logger.warning("dropped " + ", ".join(dropped) + f" (invalidated by appended cells{hint}).")
        _append_arrays(root, src, x_t, x_s, indptr_t, indptr_s, n_t, n_s, n_vars, cfg)
        if ext_layers:
            _extend_lognorm_layers(root, x_s, ext_layers, indptr_t, indptr_s, n_t + n_s, n_vars, cfg)
            logger.warning(
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

    if plan.n_duplicate_names:
        logger.warning(
            f"appended cells introduce {plan.n_duplicate_names} duplicate obs name(s) "
            "(partial overlap with existing cells)."
        )
    logger.warning("appended cells break any sorted-store contiguity; re-run `annizarr sort` if the store was sorted.")
    snapshot_id = finalize()
    return OpResult(path=str(store), n_obs=n_t + n_s, n_vars=n_vars, snapshot_id=snapshot_id)


def _check_obs_schema(obs_t: zarr.Group, obs_s: zarr.Group) -> None:
    # column-level schema equality at the zarr encoding level — no full obs read
    cols_t = str_list_attr(obs_t, "column-order")
    cols_s = str_list_attr(obs_s, "column-order")
    if cols_t != cols_s:
        raise ConversionError(f"obs schema mismatch: store {cols_t} vs cells {cols_s}.")
    for g, label, cols in ((obs_t, "store", cols_t), (obs_s, "cells", cols_s)):
        stray = sorted(set(g) - set(cols) - {str_attr(g, "_index")})
        if stray:
            raise ConversionError(f"obs in {label} has elements outside column-order: {stray}.")

    pairs = [(c, obs_t[c], obs_s[c]) for c in cols_t]
    pairs.append(("<index>", obs_t[str_attr(obs_t, "_index")], obs_s[str_attr(obs_s, "_index")]))
    for name, t, s in pairs:
        enc = t.attrs.get("encoding-type")
        if enc != s.attrs.get("encoding-type"):
            raise ConversionError(
                f"obs column '{name}' encoding mismatch ({enc!r} vs {s.attrs.get('encoding-type')!r}); "
                "reconcile before append."
            )
        if enc == "categorical":
            t_grp, s_grp = as_group(t), as_group(s)
            cat_t = np.asarray(get_array(t_grp, "categories")[:])
            cat_s = np.asarray(get_array(s_grp, "categories")[:])
            # codes are positional, so categories must match in value AND order —
            # anything less silently remaps the appended labels
            if (
                bool(t_grp.attrs.get("ordered", False)) != bool(s_grp.attrs.get("ordered", False))
                or len(cat_t) != len(cat_s)
                or not (cat_t == cat_s).all()
            ):
                raise ConversionError(
                    f"obs column '{name}' categorical dtype mismatch "
                    "(categories, order, and the ordered flag must be identical); reconcile before append."
                )
        elif enc in _NULLABLE_ENCODINGS:
            t_grp, s_grp = as_group(t), as_group(s)
            t_values, s_values = get_array(t_grp, "values"), get_array(s_grp, "values")
            if t_values.dtype != s_values.dtype:
                raise ConversionError(f"obs column '{name}' dtype mismatch ({t_values.dtype} vs {s_values.dtype}).")
        elif enc == "array":
            t_arr, s_arr = as_array(t), as_array(s)
            if t_arr.dtype != s_arr.dtype:
                raise ConversionError(
                    f"obs column '{name}' dtype mismatch ({t_arr.dtype} vs {s_arr.dtype}); "
                    "obs columns extend in place, so dtypes must match exactly."
                )
        elif enc != "string-array":
            raise ConversionError(f"obs column '{name}': unsupported encoding {enc!r} for in-place append.")


def _count_duplicate_names(obs_t: zarr.Group, obs_s: zarr.Group, n_t: int) -> int:
    # a cells-store row is a duplicate if its name already occurs in the target OR it
    # repeats an earlier row of the cells store — counted once per row (not once per
    # colliding pair), so a name that is both already in the target AND repeated within
    # the cells store isn't counted twice. n_duplicate_names == n_new means every appended
    # cell is already present — plan_append raises on that.
    idx_s = np.asarray(get_array(obs_s, str_attr(obs_s, "_index"))[:])

    # stable sort groups equal names together in original-position order, so within each
    # group every position but the first (in the group) repeats an earlier cells-store row.
    order = np.argsort(idx_s, kind="stable")
    repeats_earlier = np.zeros(len(idx_s), dtype=bool)
    if len(idx_s) > 1:
        sorted_idx = idx_s[order]
        is_repeat_sorted = np.empty(len(idx_s), dtype=bool)
        is_repeat_sorted[0] = False
        is_repeat_sorted[1:] = sorted_idx[1:] == sorted_idx[:-1]
        repeats_earlier[order] = is_repeat_sorted

    t_arr = get_array(obs_t, str_attr(obs_t, "_index"))
    chunk0 = t_arr.chunks[0]
    step = max(chunk0, (_INDEX_SCAN_ROWS // max(1, chunk0)) * chunk0)
    in_target = np.zeros(len(idx_s), dtype=bool)
    for i0 in range(0, n_t, step):
        in_target |= np.isin(idx_s, np.asarray(t_arr[i0 : min(i0 + step, n_t)]))
    return int((in_target | repeats_earlier).sum())


def _append_arrays(
    root: Any,
    src: Any,
    x_t: Any,
    x_s: Any,
    indptr_t: Any,
    indptr_s: Any,
    n_t: int,
    n_s: int,
    n_vars: int,
    cfg: AppConfig,
) -> None:
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    n_new = n_t + n_s

    with stage(f"Appending X ({n_s} cells, nnz={nnz_s})"):
        for name in ("data", "indices"):
            _extend_flat(x_t[name], x_s[name], nnz_t, nnz_s, cfg.chunks.cpus)
        _rewrite_indptr(x_t, indptr_t, indptr_s, n_new)
        x_t.attrs["shape"] = [n_new, n_vars]

    with stage(f"Appending obs ({n_s} cells)"):
        _append_obs(root["obs"], src["obs"], n_t, n_new)


def _extend_flat(dst_a: Any, src_a: Any, off: int, n_src: int, cpus: int) -> None:
    # resizes dst by n_src and copies src[:n_src] to dst[off:]; after the seam, cuts land
    # on dst chunk multiples, so segments are disjoint whole-chunk writes (threaded, no RMW)
    dst_a.resize((off + n_src,))
    chunk0 = dst_a.chunks[0]
    step = max(chunk0, (_layout.BATCH_BYTES // max(1, chunk0 * dst_a.dtype.itemsize)) * chunk0)
    cuts = [0]
    seam = (-off) % chunk0
    if 0 < seam < n_src:
        cuts.append(seam)
    while cuts[-1] < n_src:
        cuts.append(min(n_src, cuts[-1] + step))
    jobs = [(src_a, dst_a, cuts[i], cuts[i + 1], off) for i in range(len(cuts) - 1)]
    run_parallel(_copy_shifted, jobs, cpus)


def _rewrite_indptr(parent: Any, indptr_t: Any, indptr_s: Any, n_new: int) -> None:
    nnz_t = int(indptr_t[-1])
    nnz_new = nnz_t + int(indptr_s[-1])
    indptr_dtype = np.int64 if nnz_new > np.iinfo(np.int32).max else parent["indptr"].dtype
    del parent["indptr"]
    ip = parent.require_array("indptr", shape=(n_new + 1,), dtype=indptr_dtype, chunks=(n_new + 1,), overwrite=True)
    ip.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    ip[:] = np.concatenate([indptr_t, indptr_s[1:] + nnz_t]).astype(indptr_dtype)


def _extendable_layers(layers: Any, keys: list[str], indptr_t: Any) -> tuple[list[str], list[str]]:
    # (extendable, mismatched); extendable = CSR with add-expr's recorded target_sum and
    # X's exact sparsity (indptr identical), so extension is a shifted copy of X's new
    # indices + the lognorm transform on the new cells' data. Mismatched carry the attr but
    # a different sparsity — extending would corrupt them.
    ext, bad = [], []
    for k in keys:
        node = layers[k]
        if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") != "csr_matrix":
            continue
        if target_sum_attr(node.attrs) is None:
            continue
        lp = np.asarray(node["indptr"][:], dtype=np.int64)
        if len(lp) != len(indptr_t) or not (lp == indptr_t).all():
            bad.append(k)
            continue
        ext.append(k)
    return ext, bad


def _extend_lognorm_layers(
    root: Any, x_s: Any, keys: list[str], indptr_t: Any, indptr_s: Any, n_new: int, n_vars: int, cfg: AppConfig
) -> None:
    # indices shift-copy from the cells store's X (identical sparsity), indptr is
    # value-identical to X's appended indptr, and data gets the lognorm transform over
    # the new cells only — no old row is read or rewritten
    nnz_t, nnz_s = int(indptr_t[-1]), int(indptr_s[-1])
    row_nnz_s = np.diff(indptr_s)
    n_s = len(row_nnz_s)
    row_step = max(1_000, min(200_000, _layout.BATCH_BYTES // (max(1, nnz_s // max(1, n_s)) * 12)))
    for k in keys:
        g = root["layers"][k]
        # eligibility (_extendable_layers) already checked this attr is present, under either key
        target_sum = target_sum_attr(g.attrs)
        assert target_sum is not None
        with stage(f"Extending layers/{k} ({n_s} cells, nnz={nnz_s})"):
            _extend_flat(g["indices"], x_s["indices"], nnz_t, nnz_s, cfg.chunks.cpus)
            data = g["data"]
            data.resize((nnz_t + nnz_s,))
            for b0 in range(0, n_s, row_step):
                b1 = min(b0 + row_step, n_s)
                s0, s1, vals = lognorm_band(x_s["data"], indptr_s, row_nnz_s, target_sum, b0, b1)
                data[nnz_t + s0 : nnz_t + s1] = vals
            _rewrite_indptr(g, indptr_t, indptr_s, n_new)
            g.attrs["shape"] = [n_new, n_vars]


def _append_obs(obs_t: Any, obs_s: Any, n_t: int, n_new: int) -> None:
    # extends each obs column in place — O(cells store) memory, no target rewrite
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


def _extend_1d(dst: Any, src: Any, n_t: int, n_new: int) -> None:
    vals = src[:]
    if src.dtype != dst.dtype:  # e.g. categorical codes stored at different widths
        vals = vals.astype(dst.dtype)
    dst.resize((n_new,))
    dst[n_t:n_new] = vals


def _copy_shifted(src: Any, dst: Any, s0: int, s1: int, off: int) -> None:
    dst[off + s0 : off + s1] = src[s0:s1]
