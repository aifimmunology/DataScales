from __future__ import annotations

import os
import threading

import pytest

from annizarr._core._runtime import configure_runtime, map_parallel, pipeline_parallel


def _worker_zarr_threads(_: int) -> object:
    import zarr

    return zarr.config.get("threading.max_workers")


def test_configure_runtime_reapplies_zarr_thread_sizing_every_call() -> None:
    """A second op in the same process must not keep the first call's thread sizing."""
    import zarr

    configure_runtime(2)
    configure_runtime(4)
    assert zarr.config.get("threading.max_workers") == max(4, os.cpu_count() or 1)


def test_process_pool_workers_cap_their_own_zarr_threads() -> None:
    """Each worker process re-pins zarr's pool so cpus workers do not each spawn a full pool."""
    assert map_parallel(_worker_zarr_threads, [(0,), (1,)], 2, mode="processes") == [2, 2]


def _pipeline_double(x: int) -> int:
    return x * 2


def _pipeline_identity(x: int) -> int:
    return x


def _pipeline_raise_on(bad: int, x: int) -> int:
    if x == bad:
        raise ValueError(f"boom-produce-{x}")
    return x


def test_pipeline_parallel_consumes_every_job_exactly_once_with_correct_results() -> None:
    results: list[int] = []
    lock = threading.Lock()

    def consume(payload: int) -> None:
        with lock:
            results.append(payload)

    jobs = [(i,) for i in range(20)]
    pipeline_parallel(_pipeline_double, consume, jobs, cpus=4)
    assert sorted(results) == [i * 2 for i in range(20)]


def test_pipeline_parallel_propagates_a_produce_exception_and_stops() -> None:
    consumed: list[int] = []

    def consume(payload: int) -> None:
        consumed.append(payload)

    jobs = [(5, i) for i in range(10)]
    with pytest.raises(ValueError, match="boom-produce-5"):
        pipeline_parallel(_pipeline_raise_on, consume, jobs, cpus=2)


def test_pipeline_parallel_propagates_a_consume_exception_and_stops() -> None:
    def consume(payload: int) -> None:
        if payload == 5:
            raise RuntimeError("boom-consume")

    jobs = [(i,) for i in range(10)]
    with pytest.raises(RuntimeError, match="boom-consume"):
        pipeline_parallel(_pipeline_identity, consume, jobs, cpus=2)
