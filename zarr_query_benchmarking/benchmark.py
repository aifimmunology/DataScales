"""Timing harness around a validated query.

Validation runs once, up front and untimed. Then the full read-and-convert
unit (``run_query``) is timed over one or more repeats. Min and median are
reported because they are the most stable summary for I/O microbenchmarks.

Note on caching: the OS page cache and zarr's own caches mean a second read of
the same chunks is often warm. ``warmup`` runs are timed but excluded from the
summary; for true cold-cache numbers, drop the page cache between runs (see the
README) and use ``--repeats 1``.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from .query import QueryRequest, StoreInfo, run_query, validate_request


@dataclass
class BenchmarkResult:
    request: QueryRequest
    info: StoreInfo
    timings_s: list[float] = field(default_factory=list)
    result_shape: tuple[int, ...] = ()
    result_nbytes: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def min_s(self) -> float:
        return min(self.timings_s) if self.timings_s else float("nan")

    @property
    def median_s(self) -> float:
        return statistics.median(self.timings_s) if self.timings_s else float("nan")

    def as_dict(self) -> dict:
        r, i = self.request, self.info
        return {
            "store": str(r.store),
            "array_path": r.array_path,
            "storage_format": i.storage_format if i else None,
            "store_shape": list(i.shape) if i else None,
            "store_chunks": list(i.chunks) if i and i.chunks else None,
            "axis": r.axis,
            "count": r.count,
            "mode": r.mode,
            "final_format": r.final_format,
            "result_shape": list(self.result_shape),
            "result_mb": round(self.result_nbytes / 1e6, 3),
            "timings_s": [round(t, 6) for t in self.timings_s],
            "min_s": round(self.min_s, 6) if self.timings_s else None,
            "median_s": round(self.median_s, 6) if self.timings_s else None,
            "error": self.error,
        }


def benchmark_request(
    req: QueryRequest, repeats: int = 5, warmup: int = 1
) -> BenchmarkResult:
    """Validate, then time ``run_query`` ``repeats`` times (after ``warmup`` runs).

    Never raises for an invalid/failing request — the problem is captured in
    ``result.error`` so a batch of stores can be compared even if one fails.
    """
    try:
        info = validate_request(req)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the batch
        return BenchmarkResult(request=req, info=None, error=str(exc))  # type: ignore[arg-type]

    result = BenchmarkResult(request=req, info=info)
    try:
        for _ in range(max(0, warmup)):
            run_query(req, info)
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            arr = run_query(req, info)
            result.timings_s.append(time.perf_counter() - t0)
        result.result_shape = tuple(int(s) for s in np.shape(arr))
        result.result_nbytes = int(np.asarray(arr).nbytes)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result
