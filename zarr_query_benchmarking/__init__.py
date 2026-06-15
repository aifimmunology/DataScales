"""Zarr query benchmarking — compare read speed across zarr storage strategies.

Pulls a requested amount of data (cells or genes) from a single-cell AnnData
zarr store using direct zarr slicing, converts to a final dense format, and
times the operation. Supports dense and sparse-CSR (and CSC) ``X`` encodings.
"""

from .query import (
    QueryRequest,
    QueryResult,
    StoreInfo,
    inspect_store,
    validate_request,
    run_query,
)
from .benchmark import BenchmarkResult, benchmark_request

__all__ = [
    "QueryRequest",
    "QueryResult",
    "StoreInfo",
    "inspect_store",
    "validate_request",
    "run_query",
    "BenchmarkResult",
    "benchmark_request",
]
