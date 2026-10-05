from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annizarr._core import _layout
from annizarr._core._layout import band_plan, dense_shards, write_grid, x_compressors
from annizarr._core._runtime import ArrayTarget, pipeline_parallel, progress, run_parallel, writer_parallel_mode
from annizarr._writers._encoding import set_array_attrs

if TYPE_CHECKING:
    from collections.abc import Callable

    import zarr
    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader

type _DensePayload = tuple[int, int, int, int, NDArray[Any]]


def _no_tick() -> None:
    pass


def _dense_job(
    target: ArrayTarget, reader: Reader, r0: int, r1: int, block_col: int, n_cols: int, tick: Callable[[], None]
) -> None:
    # one row band fetched once, written as disjoint write-grid blocks (no read-modify-write)
    try:
        arr = target.resolve()
        band = reader.dense_band(r0, r1)
        for c0, c1 in band_plan(n_cols, block_col):
            arr[r0:r1, c0:c1] = band.tile(c0, c1)
        tick()
    finally:
        if target.in_worker:
            reader.close()


def _pipeline_col_groups(
    col_bands: tuple[tuple[int, int], ...], bytes_per_col: int, budget: int
) -> list[tuple[int, int]]:
    # merge consecutive whole write-grid blocks into groups that stay within the per-job
    # payload budget, never below one block (a single oversized block is still one job).
    groups: list[tuple[int, int]] = []
    i, n = 0, len(col_bands)
    while i < n:
        c0, c1 = col_bands[i]
        j = i + 1
        while j < n and (col_bands[j][1] - c0) * bytes_per_col <= budget:
            c1 = col_bands[j][1]
            j += 1
        groups.append((c0, c1))
        i = j
    return groups


def _dense_pipeline_produce(reader: Reader, r0: int, r1: int, c0: int, c1: int) -> _DensePayload:
    try:
        return r0, r1, c0, c1, reader.dense_block(r0, r1, c0, c1)
    finally:
        reader.close()


def write_dense(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    n_rows, n_cols = reader.shape
    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor)

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=reader.dtype,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=x_compressors(),
        overwrite=True,
    )
    set_array_attrs(zarr_arr)

    block_row, block_col = write_grid(zarr_arr)
    row_bands = band_plan(n_rows, block_row)
    mode = writer_parallel_mode(reader, zarr_arr.store_path.store)

    if mode == "processes":
        target = ArrayTarget.detached(zarr_arr)
        jobs = [(target, reader, r0, r1, block_col, n_cols, _no_tick) for r0, r1 in row_bands]
        run_parallel(_dense_job, jobs, cfg.chunks.cpus, mode="processes")
        return

    if mode == "pipeline":
        budget = min(_layout.BATCH_BYTES, _layout.PIPELINE_BATCH_BYTES)
        col_bands = band_plan(n_cols, block_col)
        itemsize = reader.dtype.itemsize
        pipeline_jobs = [
            (reader, r0, r1, c0, c1)
            for r0, r1 in row_bands
            for c0, c1 in _pipeline_col_groups(col_bands, (r1 - r0) * itemsize, budget)
        ]

        def consume(payload: _DensePayload) -> None:
            pr0, pr1, pc0, pc1, data = payload
            zarr_arr[pr0:pr1, pc0:pc1] = data

        pipeline_parallel(_dense_pipeline_produce, consume, pipeline_jobs, cfg.chunks.cpus)
        return

    tick = progress(len(row_bands), f"Writing {key}")
    target = ArrayTarget.attached(zarr_arr)
    cpus = cfg.chunks.cpus if mode == "threads" else 1
    jobs = [(target, reader, r0, r1, block_col, n_cols, tick) for r0, r1 in row_bands]
    run_parallel(_dense_job, jobs, cpus, mode="threads")
