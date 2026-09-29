from __future__ import annotations

from typing import TYPE_CHECKING, Any

import zarr

from annizarr._config import AppConfig, load_config, resolve_backend_cfg
from annizarr._layout import dense_shards, write_grid
from annizarr._ops._result import OpResult
from annizarr._runtime import configure_runtime, run_parallel, stage
from annizarr._storage import check_output_target, open_input_group, open_output_store, store_name
from annizarr._writers._encoding import sparse_shards, suppress_autoshard_warning
from annizarr._zarr import get_group, shape_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from annizarr.typing import PathLike

_SMALL_ELEMS = ("obs", "var", "uns", "varm", "varp")


def rechunk(
    store: PathLike,
    *,
    output: PathLike,
    array: str = "X",
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Rewrite one matrix with the configured chunking; stream-copy everything else as-is.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store to read from.
    output
        Destination store path or URI.
    array
        The matrix element to rechunk: ``"X"``, ``"layers/<name>"``, or ``"raw/X"``.
    cfg
        Resolved configuration; ``None`` loads :func:`~annizarr.config.load_config` defaults.
    branch
        Icechunk branch to write ``output`` to; created off the current tip if it
        doesn't exist yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and ``array``.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``array`` is not a matrix element present on ``store``.
    """
    import anndata as ad
    from anndata.io import read_elem

    from annizarr._writers._encoding import autoshard_setting, write_elem

    if cfg is None:
        cfg = load_config()
    cfg = resolve_backend_cfg(cfg)
    check_output_target(output, cfg)
    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(store)

    # validate the target before touching (or overwriting) the output
    matrix_keys = ["X"]
    if "layers" in src:
        matrix_keys += [f"layers/{k}" for k in get_group(src, "layers")]
    if "raw" in src and "X" in get_group(src, "raw"):
        matrix_keys.append("raw/X")
    if array not in matrix_keys:
        raise ConversionError(f"array '{array}' is not a matrix element ({matrix_keys}).")

    ad.settings.zarr_write_format = 3
    commit_message = message or f"annizarr rechunk {array} → {store_name(output)}"
    out = open_output_store(output, cfg, commit_message=commit_message, branch=branch)
    try:
        dst = out.root
        dst.attrs.update(dict(src.attrs))

        with autoshard_setting(cfg.chunks.auto_shard):
            with stage("Copying metadata elements"):
                for key in _SMALL_ELEMS:
                    if key in src:
                        write_elem(dst, key, read_elem(src[key]))
                # obsm/obsp scale with n_obs — stream arrays and sparse groups, chunks preserved
                for key in ("obsm", "obsp"):
                    if key not in src:
                        continue
                    src_group = get_group(src, key)
                    g = dst.require_group(key)
                    g.attrs.update(dict(src_group.attrs))
                    for child in src_group:
                        node = src_group[child]
                        if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") in (
                            "csr_matrix",
                            "csc_matrix",
                        ):
                            _copy_matrix(node, dst, f"{key}/{child}", cfg, rechunk=False)
                        else:
                            write_elem(g, child, read_elem(node))
                if "raw" in src:
                    src_raw = get_group(src, "raw")
                    raw = dst.require_group("raw")
                    raw.attrs.update(dict(src_raw.attrs))
                    for key in ("var", "varm"):
                        if key in src_raw:
                            write_elem(raw, key, read_elem(src_raw[key]))

            if "layers" in src:
                src_layers = get_group(src, "layers")
                layers = dst.require_group("layers")
                layers.attrs.update(dict(src_layers.attrs))

            for key in matrix_keys:
                rechunked = key == array
                with stage(f"{'Rechunking' if rechunked else 'Copying'} {key}"):
                    _copy_matrix(src[key], dst, key, cfg, rechunk=rechunked)

        n_obs, n_vars = _matrix_shape(src["X"])
        snapshot_id = out.finalize()
    except BaseException:
        out.abort()
        raise
    return OpResult(path=str(output), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)


def _matrix_shape(node: Any) -> tuple[int, int]:
    if isinstance(node, zarr.Array):
        n_rows, n_cols = node.shape
        return n_rows, n_cols
    return shape_attr(node)


def _copy_matrix(node: Any, dst_root: Any, key: str, cfg: AppConfig, *, rechunk: bool) -> None:
    from annizarr import _layout

    parent_path, _, name = key.rpartition("/")
    parent = dst_root[parent_path] if parent_path else dst_root

    if isinstance(node, zarr.Array):
        n_rows, n_cols = node.shape
        if rechunk:
            row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
            col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
            layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor)
            out_chunks, shards = layout.chunks, layout.shards
        else:
            out_chunks = (node.chunks[0], node.chunks[1])
            node_shards = node.shards
            shards = (node_shards[0], node_shards[1]) if node_shards is not None else None
        out = parent.require_array(
            name,
            shape=node.shape,
            dtype=node.dtype,
            chunks=out_chunks,
            shards=shards,
            compressors=node.compressors,
            overwrite=True,
        )
        out.attrs.update(dict(node.attrs))
        # aligned to the array's write grid, read off `out` as created (not recomputed from
        # config): shards if sharded, else chunks
        block_row, block_col = write_grid(out)
        jobs = [
            (node, out, r0, min(r0 + block_row, n_rows), c0, min(c0 + block_col, n_cols))
            for r0 in range(0, n_rows, block_row)
            for c0 in range(0, n_cols, block_col)
        ]
        # cap in-flight blocks: peak RSS ~ workers x block, budgeted at ~2 GiB
        block_bytes = block_row * block_col * node.dtype.itemsize
        workers = max(1, min(cfg.chunks.cpus, (2 << 30) // max(1, block_bytes)))
        run_parallel(_copy_block, jobs, workers)
        return

    enc = node.attrs.get("encoding-type")
    if enc not in ("csr_matrix", "csc_matrix"):
        raise ConversionError(f"cannot copy '{key}': unsupported encoding {enc!r}.")
    g = parent.require_group(name)
    g.attrs.update(dict(node.attrs))
    nnz = int(node["data"].shape[0])
    # rechunk=True: a freshly-chosen flat chunk, auto-sharded per cfg like any array this op
    # creates. rechunk=False (copy-as-is, e.g. obsm/obsp or an untouched matrix_key): preserve
    # the source's own chunk AND shard shape exactly, so a copy never silently drops sharding.
    if rechunk:
        flat = min(cfg.chunks.sparse_flat_chunk, max(1, nnz))
        out_shards = sparse_shards(cfg.chunks.auto_shard)
    else:
        flat = node["data"].chunks[0]
        out_shards = node["data"].shards
    for arr_name in ("data", "indices"):
        src_a = node[arr_name]
        with suppress_autoshard_warning(rechunk and cfg.chunks.auto_shard):
            out = g.require_array(
                arr_name,
                shape=src_a.shape,
                dtype=src_a.dtype,
                chunks=(flat,),
                shards=out_shards,
                compressors=src_a.compressors,
                overwrite=True,
            )
        out.attrs.update(dict(src_a.attrs))
        # aligned to the OUTPUT array's write grid, read off `out` as created
        step = write_grid(out)[0]
        seg = max(1, _layout.BATCH_BYTES // (step * src_a.dtype.itemsize)) * step
        flat_jobs = [(src_a, out, s0, min(s0 + seg, nnz)) for s0 in range(0, nnz, seg)]
        run_parallel(_copy_flat, flat_jobs, cfg.chunks.cpus)
    ip = node["indptr"]
    out = g.require_array("indptr", shape=ip.shape, dtype=ip.dtype, chunks=ip.shape, overwrite=True)
    out.attrs.update(dict(ip.attrs))
    out[:] = ip[:]


def _copy_block(src: Any, dst: Any, r0: int, r1: int, c0: int, c1: int) -> None:
    dst[r0:r1, c0:c1] = src[r0:r1, c0:c1]


def _copy_flat(src: Any, dst: Any, s0: int, s1: int) -> None:
    dst[s0:s1] = src[s0:s1]
