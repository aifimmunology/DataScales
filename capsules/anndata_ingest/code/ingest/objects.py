"""Existing AnnData objects staged out of core: annizarr streams each one into a staging zarr, then
only obs and var (small) are edited in memory, and X is rewritten block by block only when its genes
or dtype must change."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, sparse_dataset, write_elem
from annizarr import AppConfig, convert, rechunk

import contract
import hise

PER_CELL = ("obsm", "obsp", "layers", "varm", "uns")
ROWS_PER_BLOCK = 20_000


def left_behind(group: zarr.Group, keep_all: bool) -> list[str]:
    """What the store has no place for: `raw` always, per-cell elements unless the object is kept whole."""
    found = ["raw"] if "raw" in group else []
    return found + ([] if keep_all else [f"{key}/{k}" for key in PER_CELL if key in group for k in group[key]])


def matrix(node):
    return sparse_dataset(node) if isinstance(node, zarr.Group) else node


def copy_v2(src: zarr.Group, out: Path, keep_all: bool) -> None:
    """A zarr v2 AnnData (which annizarr can't rechunk) as v3: matrices in row blocks, the rest whole."""
    root = zarr.open_group(out, mode="w", zarr_format=3)
    root.attrs.update(src.attrs.asdict())
    for key in src:
        if key == "raw" or (key in PER_CELL and not keep_all):
            continue
        if key == "X":
            x = matrix(src["X"])
            write_csr(root, "X", blocks(x), x.shape[1], x.dtype)
        elif key == "layers":
            layers = root.create_group("layers", attributes=src["layers"].attrs.asdict())
            for k in src["layers"]:
                x = matrix(src["layers"][k])
                write_csr(layers, k, blocks(x), x.shape[1], x.dtype)
        else:
            write_elem(root, key, read_elem(src[key]))


def copy_in(path: Path, out: Path, cfg: AppConfig, keep_all: bool) -> list[str]:
    """Streams one existing AnnData into the staging zarr `out`; returns what was left out."""
    kind = hise.zarr_kind(path)
    source = path / hise.SPATIAL_TABLE if kind == "spatialdata" else path
    src = zarr.open_group(source, mode="r") if kind else None
    if src is not None and src.metadata.zarr_format == 2:
        copy_v2(src, out, keep_all)
        return left_behind(src, keep_all)
    if kind:
        rechunk(str(source), output=str(out), cfg=cfg)
    else:
        convert(str(path), output=str(out), cfg=cfg, fmt="h5ad")
    root = zarr.open_group(out, mode="r+", use_consolidated=False)
    left_out = left_behind(root, keep_all)
    for key in left_out:
        del root[key]
    return left_out


def blocks(x):
    for start in range(0, x.shape[0], ROWS_PER_BLOCK):
        block = x[start:start + ROWS_PER_BLOCK]
        yield block if sp.issparse(block) else sp.csr_matrix(block)


def x_stats(x) -> tuple[bool, float]:
    """Whether every stored value is a non-negative whole number, and the largest value."""
    counts, top = True, 0.0
    for block in blocks(x):
        v = block.data
        counts = counts and bool(np.all(v >= 0) and np.all(np.mod(v, 1) == 0))
        top = max(top, float(v.max()) if v.size else 0.0)
    return counts, top


def write_csr(parent: zarr.Group, name: str, rows, n_cols: int, dtype: np.dtype) -> None:
    """A CSR matrix written block by block from `rows`, an iterable of CSR row blocks."""
    group = parent.create_group(name)
    data = group.create_array("data", shape=(0,), dtype=dtype, chunks=(1_000_000,))
    indices = group.create_array("indices", shape=(0,), dtype=np.int64, chunks=(1_000_000,))
    indptr = [0]
    for block in rows:
        data.append(block.data.astype(dtype, copy=False))
        indices.append(block.indices.astype(np.int64))
        indptr.extend(indptr[-1] + block.indptr[1:])
    group.create_array("indptr", data=np.asarray(indptr, dtype=np.int64))
    group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": [len(indptr) - 1, n_cols]})


def rewrite_x(root: zarr.Group, out: Path, x, columns: np.ndarray, genes: np.ndarray, dtype: np.dtype) -> None:
    """X restricted and reordered to `genes` (missing ones zero-filled), in `dtype`, block by block."""
    pick = contract.gene_picker(columns, genes)
    write_csr(root, "X_fitted", (sp.csr_matrix(block @ pick, dtype=dtype) for block in blocks(x)), len(genes), dtype)
    del root["X"]
    os.replace(out / "X_fitted", out / "X")


def stage_object(path: Path, out: Path, column: str | None, sample: str, target, keep_all: bool,
                 cfg: AppConfig, name: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Streams one existing AnnData into `out` and fits it to `target` (the store's genes, X dtype,
    counts and obs kinds). Returns its obs, a note on gene fitting, and what was left out."""
    try:
        left_out = copy_in(path, out, cfg, keep_all)
    except TypeError as exc:
        raise contract.ContractError(f"{name}: annizarr can't copy {path.name} ({exc}).") from exc
    root = zarr.open_group(out, mode="r+", use_consolidated=False)
    var, obs = read_elem(root["var"]), read_elem(root["obs"])

    columns = np.asarray(var.index, dtype=object)
    genes = columns[hise.gene_mask(var)] if target.var is None else target.var
    diff = contract.panel_diff(genes, columns)
    x = matrix(root["X"])
    counts, top = x_stats(x)
    if target.counts and not counts:
        raise contract.ContractError(f"{name}: X holds non-count values, but the store's X holds counts; ingest "
                                     "raw counts, or start a new store for normalized data.")
    dtype = np.dtype(target.x_dtype or x.dtype)
    if np.issubdtype(dtype, np.integer) and top > np.iinfo(dtype).max:
        raise contract.ContractError(f"{name}: X values up to {top:g} don't fit the store's {dtype}.")
    if diff or dtype != x.dtype:
        rewrite_x(root, out, x, columns, genes, dtype)
    if diff:
        write_elem(root, "var", var.reindex(genes) if target.var is None else pd.DataFrame(index=pd.Index(genes)))

    obs["sample_id"] = obs[column].astype(str) if column else sample
    obs.index = [n if n.startswith(f"{sid}:") else f"{sid}:{n}" for sid, n in zip(obs["sample_id"], obs.index)]
    if target.obs is not None:
        left_out += [f"obs/{c}" for c in obs.columns if c not in target.obs]
        obs, failed = contract.match_obs(obs[[c for c in obs.columns if c in target.obs]], target.obs)
        left_out += [f"obs/{c} (values don't fit the store's type)" for c in failed]
    write_elem(root, "obs", obs)
    zarr.consolidate_metadata(str(out))
    note = contract.fit_note(diff) if diff and target.var is not None else ""
    if target.var is None:
        target.learn(genes, dtype, counts, contract.frame_kinds(obs))
    return obs, note, left_out
