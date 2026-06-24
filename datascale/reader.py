"""Importable reader for sorted/partitioned DataScale stores (Feature B).

A store written with ``--sort-by`` has its rows physically sorted by one or more obs
columns, so every distinct key tuple is a contiguous row range recorded in the ``sort_index``
group. This reader turns a key selection into a minimal slice read:

    from datascale import open_sorted
    store = open_sorted("atlas.zarr")                 # or open_sorted("repo", icechunk=True)
    store.groups()                                    # the (cell_type[, demographic, …]) table
    adata = store.select(cell_type="Tcell")           # contiguous block → in-memory AnnData
    adata = store.select(cell_type="Tcell", demographic="adult")   # sub-range
    adata = store.select(demographic="adult")         # cross-cut: gathered + concatenated

Only ``X[start:end]`` (plus the matching obs/var/obsm rows) is read, so a subset of a large
store comes back fast without materialising the full matrix.
"""
from __future__ import annotations

from typing import Any

from .storage import open_input_group


class QueryError(RuntimeError):
    """Raised when a sorted store cannot be opened or a selection cannot be served."""


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge sorted, possibly-adjacent half-open [start, end) spans into minimal runs."""
    spans = sorted(spans)
    merged: list[list[int]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


class SortedStore:
    """Read-only handle to a sorted DataScale store, queried by sort-key values."""

    _INDEX_KEY = "datascale_sort_index"

    def __init__(self, group: Any):
        if "uns" not in group or self._INDEX_KEY not in group["uns"]:
            raise QueryError(
                "Store has no 'uns/datascale_sort_index' — it was not written with grouping "
                "(--sort-by). Open it with anndata directly, or re-convert with --sort-by."
            )
        from anndata.io import read_elem

        self._group = group
        si = group["uns"][self._INDEX_KEY]
        self._ranges = read_elem(si["ranges"])  # DataFrame: sort keys + start + end
        # Sort keys are the range-table columns minus the start/end offsets, in order.
        self.sort_keys: list[str] = [
            c for c in self._ranges.columns if c not in ("start", "end")
        ]

    # ── Introspection ────────────────────────────────────────────────────────
    def groups(self):
        """Return the range table (one row per distinct key tuple, with start/end)."""
        return self._ranges.copy()

    def __repr__(self) -> str:
        return f"SortedStore(sort_keys={self.sort_keys}, n_groups={len(self._ranges)})"

    # ── Selection ──────────────────────────────────────────────────────────────
    def _resolve_spans(self, keys: dict[str, Any]) -> list[tuple[int, int]]:
        unknown = [k for k in keys if k not in self.sort_keys]
        if unknown:
            raise QueryError(
                f"Unknown sort key(s) {unknown}; this store is sorted by {self.sort_keys}."
            )
        sel = self._ranges
        for k, v in keys.items():
            sel = sel[sel[k] == v]
        if len(sel) == 0:
            raise QueryError(f"No rows match selection {keys}.")
        spans = list(zip(sel["start"].astype(int), sel["end"].astype(int)))
        return _merge_spans(spans)

    def select(self, with_obsm: bool = True, **keys: Any):
        """Return an in-memory ``AnnData`` for the rows matching ``keys``.

        Partial keys are allowed: omitted keys span all their values. A selection that maps
        to one contiguous run is a single slice read; a cross-cutting selection (e.g. only a
        non-primary key) is gathered from several runs and concatenated.
        """
        import anndata as ad
        import numpy as np
        import scipy.sparse as sp
        from anndata.io import read_elem, sparse_dataset

        if not keys:
            raise QueryError("select() needs at least one sort-key filter.")

        spans = self._resolve_spans(keys)

        x_ds = sparse_dataset(self._group["X"])
        x_parts = [x_ds[s:e] for s, e in spans]
        X = x_parts[0] if len(x_parts) == 1 else sp.vstack(x_parts, format="csr")

        # obs is small relative to X; read once and slice the selected spans.
        obs_full = read_elem(self._group["obs"])
        obs = obs_full.iloc[np.concatenate([np.arange(s, e) for s, e in spans])]
        var = read_elem(self._group["var"])

        adata = ad.AnnData(X=X, obs=obs, var=var)

        if with_obsm and "obsm" in self._group:
            import zarr
            obsm_group = self._group["obsm"]
            for name in obsm_group:
                arr = obsm_group[name]
                # Slice array-type obsm directly (reads only needed chunks). Skip encoded
                # groups (e.g. dataframe obsm) for now — they'd need a full read_elem.
                if isinstance(arr, zarr.Array):
                    adata.obsm[name] = np.concatenate([arr[s:e] for s, e in spans])
        return adata


def open_sorted(path: str, *, icechunk: bool = False, branch: str = "main") -> SortedStore:
    """Open a sorted DataScale store for subset queries.

    Parameters
    ----------
    path: str
        Path to the zarr directory, or to the icechunk repository when ``icechunk=True``.
    icechunk: bool
        Open through an icechunk read-only session instead of a plain zarr directory.
    branch: str
        icechunk branch to read (ignored for plain zarr).
    """
    group = open_input_group(path, icechunk=icechunk, branch=branch)
    return SortedStore(group)
