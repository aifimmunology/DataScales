"""Apply: refuse early when the decisions or the target don't allow a write, then write the passing cells.

The copy is written by annizarr, with its default workers and compression: X and CSR layers stream from the source's
kept rows through its matrix writer, and layers annizarr's `add_layer` made are re-derived from the curated matrix.
obs, obsm and other layers are filtered to the kept rows; var, varm, varp and uns are copied unchanged; obsp is dropped.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import icechunk
import numpy as np
import pandas as pd
import zarr
import zarr.errors
from anndata.io import read_elem, write_elem
from annizarr import AppConfig
from annizarr._writers import write_adata, write_matrix
from annizarr.ops._layer import SOURCE_ATTR, _write_layer

import decisions as dec
from rows import KeptRows
from source import MAIN, Source, probe_writable


LAYOUTS = {"csr_matrix": "csr", "csc_matrix": "csc", "array": "dense"}


class ApplyError(RuntimeError):
    """Apply can't run with these decisions, or can't write where it was pointed."""


def output_path(target: str, source: Source, results_dir: str) -> str:
    return target or str(Path(results_dir) / f"{source.stem}.qc.zarr")


def _in_use(path: str) -> bool:
    """True when `path` holds anything but an empty zarr group (such as one left by the write probe)."""
    try:
        return any(True for _ in zarr.open_group(path, mode="r").keys())
    except (zarr.errors.GroupNotFoundError, FileNotFoundError):
        p = Path(path)
        return p.exists() and (not p.is_dir() or any(p.iterdir()))


def preflight(file: dict | None, source: Source, branch: str, target: str, results_dir: str) -> None:
    """The §4.9 checks, in order, before any report work."""
    if file is None:
        raise ApplyError("apply needs the qc_decisions.json of a report run; run in report mode first.")
    if not dec.is_sealed(file):
        raise ApplyError("Decisions edited after their report; rerun in report mode to preview them.")
    if file.get("input", {}).get("fingerprint") != source.fingerprint:
        raise ApplyError("Input changed since this report (e.g. new samples were ingested); rerun in report mode.")
    if file.get("pending"):
        raise ApplyError(f"Pending samples need a decision: {', '.join(file['pending'])}")
    if source.kind == "icechunk":
        if branch == MAIN or "/" in branch or ":" in branch:
            raise ApplyError(f"'{branch}' can't be a curation branch; give a branch name other than main.")
        if source.repo.origin is None:
            raise ApplyError(f"{source.location} is read-only; pass its writable origin (e.g. s3://bucket/prefix).")
        return
    output = output_path(target, source, results_dir)
    if "://" in output:
        raise ApplyError(f"{output}: the output copy must be a local path (e.g. under /results or /scratch).")
    if _in_use(output):
        raise ApplyError(f"{output} already holds data; give a new output path or remove it first.")
    try:
        probe_writable(output)
    except Exception as exc:
        raise ApplyError(str(exc)) from exc


def _take(value, keep: np.ndarray):
    if isinstance(value, pd.DataFrame):
        return value.iloc[keep]
    return value[keep]


def kept_obs(src: zarr.Group, keep: np.ndarray) -> pd.DataFrame:
    obs = read_elem(src).iloc[keep]
    for column in obs.columns[obs.dtypes == "category"]:
        obs[column] = obs[column].cat.remove_unused_categories()
    return obs


def _members(src: zarr.Group, key: str) -> dict:
    return dict(src[key].members()) if key in src else {}


def uns_entry(review, reviewer: str, created: str, cell_qc_sha256: str) -> dict:
    counts = review.table[["n_cells", "n_pass"]].set_axis(["before", "after"], axis=1).astype(np.int64)
    counts.index = counts.index.astype(str)
    entry = {
        "decisions": json.dumps(review.decisions, sort_keys=True),
        "cell_qc_sha256": cell_qc_sha256,
        "source": review.source.fingerprint,
        "source_kind": review.source.kind,
        "source_location": review.source.location,
        "created": created,
        "samples": np.asarray(counts.index[counts["after"] > 0], dtype=object),
        "counts": {"sample_id": np.asarray(counts.index, dtype=object),
                   "before": counts["before"].to_numpy(), "after": counts["after"].to_numpy()},
    }
    if reviewer:
        entry["reviewer"] = reviewer
    return entry


def copy_layers(src: zarr.Group, dst: zarr.Group, keep: np.ndarray, cfg: AppConfig) -> None:
    layers = _members(src, "layers")
    if not layers:
        return
    write_elem(dst, "layers", {})
    derived = {name: element for name, element in layers.items() if SOURCE_ATTR in element.attrs}
    for name, element in layers.items():
        if name in derived:
            continue
        if element.attrs.get("encoding-type") == "csr_matrix":
            write_matrix(dst["layers"], name, KeptRows(element, keep, cfg.chunks.cpus), cfg)
        else:
            write_elem(dst["layers"], name, _take(read_elem(element), keep))
    for name, element in derived.items():
        matrix = element.attrs[SOURCE_ATTR]
        if matrix in dst:
            _write_layer(dst, cfg, layer=name, matrix=matrix, layout=LAYOUTS[element.attrs["encoding-type"]],
                         overwrite=True)
        else:
            write_elem(dst["layers"], name, _take(read_elem(element), keep))


def copy_cells(src: zarr.Group, dst: zarr.Group, keep: np.ndarray, qc: dict) -> None:
    cfg = AppConfig()
    adata = ad.AnnData(
        obs=kept_obs(src["obs"], keep),
        var=read_elem(src["var"]),
        uns={**(read_elem(src["uns"]) if "uns" in src else {}), "qc": qc},
        obsm={name: _take(read_elem(element), keep) for name, element in _members(src, "obsm").items()},
        varm={name: read_elem(element) for name, element in _members(src, "varm").items()},
        varp={name: read_elem(element) for name, element in _members(src, "varp").items()},
    )
    dst.attrs.update(dict(src.attrs))
    write_adata(adata, dst, cfg, x=KeptRows(src["X"], keep, cfg.chunks.cpus))
    copy_layers(src, dst, keep, cfg)


def table_group(store: zarr.Group, source: Source) -> zarr.Group:
    """Where the curated AnnData goes in a new zarr: its root, or for SpatialData `tables/<name>`."""
    if not source.spatial:
        return store
    store.attrs.update(dict(source.store.attrs))
    tables, name = source.table.split("/")
    return store.require_group(tables).create_group(name)


def _metadata(review, qc: dict, created: str, n_pass: int) -> dict:
    meta = {"pipeline": "qc", "created": created, "source_snapshot": review.source.fingerprint,
            "samples": [str(s) for s in qc["samples"]], "decisions_sha256": review.decisions[dec.HASH_KEY],
            "n_pass": n_pass, "n_cells": len(review.obs)}
    if qc.get("reviewer"):
        meta["reviewer"] = qc["reviewer"]
    return meta


def apply(review, branch: str, target: str, reviewer: str, results_dir: str, cell_qc_sha256: str) -> dict:
    """Writes the passing cells to the curation branch (icechunk) or a new zarr; returns where they went."""
    created = datetime.now(timezone.utc).isoformat()
    qc = uns_entry(review, reviewer, created, cell_qc_sha256)
    keep = review.passed
    source = review.source
    if source.kind == "zarr":
        output = output_path(target, source, results_dir)
        copy_cells(source.root, table_group(zarr.open_group(output, mode="w"), source), keep, qc)
        return {"output": output, "n_pass": int(keep.sum()), "n_cells": len(keep)}

    snapshot = flush_curation(review, qc, created)
    move_branch(source.repo.origin, branch, snapshot)
    return {"commit": snapshot, "branch": branch, "n_pass": int(keep.sum()), "n_cells": len(keep)}


def writer(location: str) -> icechunk.Repository:
    if location.startswith("s3://"):
        bucket, _, prefix = location.removeprefix("s3://").partition("/")
        return icechunk.Repository.open(icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True))
    return icechunk.Repository.open(icechunk.local_filesystem_storage(location))


def flush_curation(review, qc: dict, created: str) -> str:
    """Writes the curation as a snapshot on top of the reviewed `main` snapshot, without moving any branch.

    Nothing is visible until the snapshot exists, so a failed or killed write leaves the repo as it was."""
    source, keep = review.source, review.passed
    session = writer(source.repo.origin).writable_session(MAIN)
    if session.snapshot_id != source.fingerprint:
        raise ApplyError("main moved during this run (e.g. a new ingest); rerun in report mode.")
    copy_cells(source.root, zarr.open_group(store=session.store, mode="w"), keep, qc)
    message = (f"QC curation: {len(qc['samples'])} samples, {int(keep.sum()):,} / {len(keep):,} cells pass "
               f"(source {source.fingerprint[:8]})")
    return session.flush(message, metadata=_metadata(review, qc, created, int(keep.sum())))


def move_branch(location: str, branch: str, snapshot: str) -> None:
    """Points `branch` at `snapshot`: creates it, or resets it only if no one moved it meanwhile."""
    repo = writer(location)
    if branch in repo.list_branches():
        repo.reset_branch(branch, snapshot, from_snapshot_id=repo.lookup_branch(branch))
    else:
        repo.create_branch(branch, snapshot)
