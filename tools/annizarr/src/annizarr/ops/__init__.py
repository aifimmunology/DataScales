from __future__ import annotations

from annizarr.ops._append import append, plan_append
from annizarr.ops._concat import concat as concat
from annizarr.ops._convert import convert
from annizarr.ops._convert import convert_10x_h5 as convert_10x_h5
from annizarr.ops._convert import convert_adata as convert_adata
from annizarr.ops._convert import convert_h5ad as convert_h5ad
from annizarr.ops._expr import add_expr
from annizarr.ops._rechunk import rechunk
from annizarr.ops._result import AppendPlan, OpResult
from annizarr.ops._sort import sort

# concat/convert_10x_h5/convert_adata/convert_h5ad stay importable but out of __all__: the
# CLI/tests use them directly, but convert() is the one public entry point for the op.
__all__ = [
    "AppendPlan",
    "OpResult",
    "add_expr",
    "append",
    "convert",
    "plan_append",
    "rechunk",
    "sort",
]
