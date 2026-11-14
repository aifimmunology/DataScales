"""Gene panels: every table is indexed by gene symbol with Gene Expression features only (spec §3S.4);
panel comparisons and count checks. (The joint table takes the union of panels: see joint.py.)"""
from __future__ import annotations

import re

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

GEX = "Gene Expression"
MITO_PREFIX = "MT-"
SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_name")
ID_COLUMNS = ("gene_ids", "feature_id", "gene_id")
TYPE_COLUMNS = ("feature_types", "feature_type")
_ENSEMBL = re.compile(r"^ENS[A-Z]*[GT]\d+")
SPECIES = {"ENSG": "human", "ENST": "human", "ENSMUSG": "mouse", "ENSMUST": "mouse"}


class PanelError(ValueError):
    """A gene panel the store can't take."""


def gene_mask(var: pd.DataFrame) -> np.ndarray:
    """True for the Gene Expression features; everything is a gene when var has no feature type."""
    for col in TYPE_COLUMNS:
        if col in var:
            return np.asarray(var[col].astype(str) == GEX)
    return np.ones(len(var), dtype=bool)


def _ensembl(values) -> bool:
    values = [str(v) for v in values]
    return bool(values) and sum(bool(_ENSEMBL.match(v)) for v in values) > len(values) / 2


def symbol_var(var: pd.DataFrame, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """The Gene Expression rows of `var` (as a boolean mask over var) and their new var: indexed by
    unique gene symbol, with `gene_ids` (Ensembl, when known) and `feature_types` only."""
    keep = gene_mask(var)
    genes = var[keep]
    if not _ensembl(genes.index):
        symbols = genes.index.astype(str)
    else:
        column = next((c for c in SYMBOL_COLUMNS if c in genes), None)
        if column is None:
            raise PanelError(f"{name}: the table is indexed by Ensembl IDs and has no symbol column "
                             f"({', '.join(SYMBOL_COLUMNS)}).")
        symbols = genes[column].astype(str)
    ids = next((genes[c].astype(str).to_numpy() for c in ID_COLUMNS if c in genes), None)
    if ids is None and _ensembl(genes.index):
        ids = genes.index.astype(str).to_numpy()
    out = pd.DataFrame(index=pd.Index(ad.utils.make_index_unique(pd.Index(np.asarray(symbols, dtype=object)))))
    out["gene_ids"] = ids if ids is not None else ""
    out["feature_types"] = GEX
    return keep, out


def species(var: pd.DataFrame) -> str | None:
    """human or mouse from the Ensembl prefixes of `gene_ids`, or None when they don't say."""
    found = {SPECIES.get(re.sub(r"\d.*$", "", str(g))) for g in var.get("gene_ids", [])} - {None}
    return found.pop() if len(found) == 1 else None


def is_mito(var_names) -> np.ndarray:
    return np.char.startswith(np.asarray(var_names).astype(str), MITO_PREFIX)


def panel_diff(store_genes, sample_genes, n_examples: int = 5) -> dict | None:
    """How a sample's genes differ from the store's, or None when names and order match."""
    store_genes, sample_genes = list(store_genes), list(sample_genes)
    if store_genes == sample_genes:
        return None
    in_sample, in_store = set(sample_genes), set(store_genes)
    only_store = [g for g in store_genes if g not in in_sample]
    only_sample = [g for g in sample_genes if g not in in_store]
    return {"n_shared": len(in_store & in_sample), "n_only_in_store": len(only_store),
            "n_only_in_sample": len(only_sample), "only_in_store": only_store[:n_examples],
            "only_in_sample": only_sample[:n_examples]}


def describe_diff(sid: str, diff: dict) -> str:
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return f"{sid}: the same {diff['n_shared']} genes in a different order"
    return (f"{sid}: {diff['n_shared']} shared, {diff['n_only_in_store']} only in the store "
            f"(e.g. {', '.join(diff['only_in_store']) or '-'}), {diff['n_only_in_sample']} only in the sample "
            f"(e.g. {', '.join(diff['only_in_sample']) or '-'})")


def fit_note(diff: dict | None) -> str:
    if not diff:
        return ""
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return "genes reordered to the store's"
    return f"fitted to the store's genes: {diff['n_only_in_store']} zero-filled, {diff['n_only_in_sample']} dropped"


def gene_picker(sample_genes, store_genes) -> sp.csr_matrix:
    """A 0/1 matrix mapping a sample's genes onto the store's: X @ picker gives the store's genes,
    zero for those the sample lacks."""
    src = pd.Index(np.asarray(sample_genes, dtype=object)).get_indexer(pd.Index(np.asarray(store_genes, dtype=object)))
    present = src >= 0
    return sp.csr_matrix((np.ones(present.sum(), dtype=np.float32), (src[present], np.flatnonzero(present))),
                         shape=(len(sample_genes), len(store_genes)))


def fit_counts(x: sp.csr_matrix, sample_genes, store_genes, dtype: np.dtype) -> sp.csr_matrix:
    """`x` on the store's genes (missing ones zero-filled, extra ones dropped), in `dtype`."""
    if list(sample_genes) != list(store_genes):
        x = sp.csr_matrix(x @ gene_picker(sample_genes, store_genes))
    return as_dtype(x, dtype)


def holds_counts(x: sp.csr_matrix) -> bool:
    return bool(np.all(x.data >= 0) and np.all(np.mod(x.data, 1) == 0))


def as_dtype(x: sp.csr_matrix, dtype: np.dtype) -> sp.csr_matrix:
    dtype = np.dtype(dtype)
    if x.dtype == dtype:
        return x
    if np.issubdtype(dtype, np.integer) and x.nnz and x.data.max() > np.iinfo(dtype).max:
        raise PanelError(f"counts up to {x.data.max():g} don't fit the joint table's {dtype}.")
    return x.astype(dtype)
