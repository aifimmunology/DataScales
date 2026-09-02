from __future__ import annotations

from pathlib import Path

import zarr

from convert_to_zarr.config import AppConfig, _resolve_backend_cfg
from convert_to_zarr.engine import _run_parallel_threads, _stage, configure_runtime
from convert_to_zarr.errors import ConversionError
from convert_to_zarr.layout import _dense_shards
from convert_to_zarr.storage import open_input_group, open_output_store

_SMALL_ELEMS = ("obs", "var", "uns", "varm", "varp")
_SEG_BYTES = 256 * 1024 * 1024


def rechunk_store(
    input_store: str, output_zarr: str, cfg: AppConfig, *, array: str = "X"
) -> list[str]:
    """Rewrite one matrix with the configured chunking; stream-copy everything else as-is."""
    import anndata as ad
    from anndata._io.specs import write_elem
    from anndata.io import read_elem

    cfg = _resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)
    src = open_input_group(input_store)

    # validate the target before touching (or overwriting) the output
    matrix_keys = ["X"]
    if "layers" in src:
        matrix_keys += [f"layers/{k}" for k in src["layers"]]
    if "raw" in src and "X" in src["raw"]:
        matrix_keys.append("raw/X")
    if array not in matrix_keys:
        raise ConversionError(f"array '{array}' is not a matrix element ({matrix_keys}).")

    ad.settings.zarr_write_format = 3
    output_path = Path(output_zarr)
    dst, finalize = open_output_store(
        output_path, cfg, commit_message=f"zarrsmith rechunk {array} → {output_path.name}"
    )
    dst.attrs.update(dict(src.attrs))

    with _stage("Copying metadata elements"):
        for key in _SMALL_ELEMS:
            if key in src:
                write_elem(dst, key, read_elem(src[key]))
        # obsm/obsp scale with n_obs — stream arrays and sparse groups, chunks preserved
        for key in ("obsm", "obsp"):
            if key not in src:
                continue
            g = dst.require_group(key)
            g.attrs.update(dict(src[key].attrs))
            for child in src[key]:
                node = src[key][child]
                if isinstance(node, zarr.Array) or node.attrs.get("encoding-type") in (
                    "csr_matrix", "csc_matrix",
                ):
                    _copy_matrix(node, dst, f"{key}/{child}", cfg, rechunk=False)
                else:
                    write_elem(g, child, read_elem(node))
        if "raw" in src:
            raw = dst.require_group("raw")
            raw.attrs.update(dict(src["raw"].attrs))
            for key in ("var", "varm"):
                if key in src["raw"]:
                    write_elem(raw, key, read_elem(src["raw"][key]))

    if "layers" in src:
        layers = dst.require_group("layers")
        layers.attrs.update(dict(src["layers"].attrs))

    for key in matrix_keys:
        rechunked = key == array
        with _stage(f"{'Rechunking' if rechunked else 'Copying'} {key}"):
            _copy_matrix(src[key], dst, key, cfg, rechunk=rechunked)

    finalize()
    return []


def _copy_matrix(node, dst_root, key, cfg: AppConfig, *, rechunk: bool) -> None:
    parent_path, _, name = key.rpartition("/")
    parent = dst_root[parent_path] if parent_path else dst_root

    if isinstance(node, zarr.Array):
        n_rows, n_cols = node.shape
        if rechunk:
            row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
            col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
            shards, block_row, block_col = _dense_shards(
                row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor
            )
        else:
            row_chunk, col_chunk = node.chunks
            shards = node.shards
            block_row, block_col = shards or node.chunks
        out = parent.require_array(
            name, shape=node.shape, dtype=node.dtype, chunks=(row_chunk, col_chunk),
            shards=shards, compressors=node.compressors, overwrite=True,
        )
        out.attrs.update(dict(node.attrs))
        jobs = [
            (node, out, r0, min(r0 + block_row, n_rows), c0, min(c0 + block_col, n_cols))
            for r0 in range(0, n_rows, block_row)
            for c0 in range(0, n_cols, block_col)
        ]
        # cap in-flight blocks: peak RSS ~ workers x block, budgeted at ~2 GiB
        block_bytes = block_row * block_col * node.dtype.itemsize
        workers = max(1, min(cfg.chunks.cpus, (2 << 30) // max(1, block_bytes)))
        _run_parallel_threads(_copy_block, jobs, workers)
        return

    enc = node.attrs.get("encoding-type")
    if enc not in ("csr_matrix", "csc_matrix"):
        raise ConversionError(f"cannot copy '{key}': unsupported encoding {enc!r}.")
    g = parent.require_group(name)
    g.attrs.update(dict(node.attrs))
    nnz = int(node["data"].shape[0])
    flat = min(cfg.chunks.sparse_flat_chunk, max(1, nnz)) if rechunk else node["data"].chunks[0]
    for arr_name in ("data", "indices"):
        src_a = node[arr_name]
        out = g.require_array(
            arr_name, shape=src_a.shape, dtype=src_a.dtype, chunks=(flat,),
            compressors=src_a.compressors, overwrite=True,
        )
        out.attrs.update(dict(src_a.attrs))
        seg = max(1, _SEG_BYTES // (flat * src_a.dtype.itemsize)) * flat
        jobs = [(src_a, out, s0, min(s0 + seg, nnz)) for s0 in range(0, nnz, seg)]
        _run_parallel_threads(_copy_flat, jobs, cfg.chunks.cpus)
    ip = node["indptr"]
    out = g.require_array(
        "indptr", shape=ip.shape, dtype=ip.dtype, chunks=ip.shape, overwrite=True
    )
    out.attrs.update(dict(ip.attrs))
    out[:] = ip[:]


def _copy_block(src, dst, r0, r1, c0, c1):
    dst[r0:r1, c0:c1] = src[r0:r1, c0:c1]


def _copy_flat(src, dst, s0, s1):
    dst[s0:s1] = src[s0:s1]
