"""The gene policy: how a sample's genes compare with the store's, and how a sample is fitted to them.

A store's genes are fixed when it is created. A sample with other genes is fitted: store genes it
lacks are zero-filled and its other genes dropped, with a warning and a note in the commit. A sample
sharing no gene is refused. Supporting stores that grow new genes is a change to this module."""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from contracts import ContractError, ObjectInput, RawInput

logger = logging.getLogger("ingest")


def check_same(var_names: np.ndarray, expected: np.ndarray | None) -> None:
    """A last guard before a sample is staged: its genes are exactly the store's, in order."""
    if expected is None or np.array_equal(var_names, np.asarray(expected, dtype=object)):
        return
    if len(var_names) == len(expected) and set(var_names) == set(expected):
        found = "the same genes in a different order"
    else:
        found = f"{len(var_names)} genes, the store {len(expected)}, {len(set(var_names) ^ set(expected))} not shared"
    raise ContractError(f"var mismatch: the sample has {found}; gene names and order must match the store's.")


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


def require_shared(diff: dict | None, n_genes: int, name: str) -> None:
    """Refuses a sample that shares no gene with the store (or, for a new store, no gene-expression
    feature at all): fitting it would store all-zero cells."""
    if n_genes == 0:
        raise ContractError(f"{name} has no Gene Expression features to store.")
    if diff and diff["n_shared"] == 0:
        raise ContractError(f"{name} shares no gene names with the store ({n_genes} genes, e.g. "
                            f"{', '.join(diff['only_in_store'][:3])}; the sample has e.g. "
                            f"{', '.join(diff['only_in_sample'][:3])}); its cells would be stored as all zeros. "
                            "Check that both use the same gene identifiers (symbols vs Ensembl IDs).")


def describe_diff(sid: str, diff: dict) -> str:
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return f"{sid}: the same {diff['n_shared']} genes in a different order"
    return (f"{sid}: {diff['n_shared']} shared, {diff['n_only_in_store']} only in the store "
            f"(e.g. {', '.join(diff['only_in_store']) or '-'}), {diff['n_only_in_sample']} only in the sample "
            f"(e.g. {', '.join(diff['only_in_sample']) or '-'})")


def fit_note(diff: dict) -> str:
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return "genes reordered to the store's"
    return (f"fitted to the store's genes: {diff['n_only_in_store']} zero-filled, "
            f"{diff['n_only_in_sample']} dropped")


def gene_picker(sample_genes, store_genes) -> sp.csr_matrix:
    """A 0/1 matrix that maps a sample's genes onto the store's: X @ picker gives the store's genes,
    zero for those the sample lacks. Duplicate sample genes map by their first occurrence."""
    src = pd.Index(ad.utils.make_index_unique(pd.Index(np.asarray(sample_genes, dtype=object)))).get_indexer(
        pd.Index(np.asarray(store_genes, dtype=object)))
    present = src >= 0
    return sp.csr_matrix((np.ones(present.sum(), dtype=np.float32), (src[present], np.flatnonzero(present))),
                         shape=(len(sample_genes), len(store_genes)))


def conform_var(adata: ad.AnnData, store_genes) -> ad.AnnData:
    """The sample's counts on the store's genes: store genes the sample lacks are zero-filled and its
    genes the store lacks are dropped. Annotation must run before this, on the sample's own genes."""
    x = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    fitted = sp.csr_matrix(x @ gene_picker(adata.var_names, store_genes)).astype(x.dtype)
    return ad.AnnData(fitted, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(np.asarray(store_genes, dtype=object))))


def fit(adata: ad.AnnData, store_genes: np.ndarray | None) -> tuple[ad.AnnData, str]:
    """The sample on the store's genes (as it is for a new store), and a note on how they were fitted."""
    diff = None if store_genes is None else panel_diff(store_genes, adata.var_names)
    return (conform_var(adata, store_genes), fit_note(diff)) if diff else (adata, "")


def check_panels(new: list[RawInput | ObjectInput], store_genes: np.ndarray | None, gene_names) -> None:
    """Warns, from var alone and before any sample is read, about each new sample whose genes differ
    from the store's (or, for a new store, the first sample's); refuses one sharing none.
    `gene_names(path)` reads a sample's genes."""
    reference = store_genes
    diffs = {}
    for item in new:
        names = gene_names(item.path)
        if reference is None:
            require_shared(None, len(names), item.name)
            reference = names
            continue
        diff = panel_diff(reference, names)
        require_shared(diff, len(reference), item.name)
        if diff:
            diffs[item.name] = diff
    if diffs:
        against = ("the store's" if store_genes is not None
                   else f"the first sample's ({new[0].name}, {len(reference)} genes)")
        logger.warning("Gene panel differs from %s for %d sample(s); they are fitted to its genes (missing genes "
                       "zero-filled, extra genes dropped): %s", against, len(diffs),
                       "; ".join(describe_diff(sid, d) for sid, d in diffs.items()))
