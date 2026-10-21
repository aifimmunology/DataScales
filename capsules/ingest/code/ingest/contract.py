"""The store contract: a fixed obs schema, CSR uint32 counts, and the projection every sample passes through."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

X_DTYPE = np.uint32
OBS_INDEX = "barcodes"
MITO_PREFIX = "MT-"

STR, INT, FLOAT = "str", "Int64", "float64"

HISE_STR_COLS = (
    "adt_qc_flag", "batch_id", "cell_name", "cell_uuid", "chip_id", "hto_barcode",
    "hto_category", "original_barcodes", "pbmc_sample_id", "pool_id", "well_id",
)
HISE_INT_COLS = ("adt_umis", "n_genes", "n_mito_umis", "n_reads", "n_umis")
QC_COLS = {"n_genes_by_counts": INT, "total_counts": FLOAT, "total_counts_mt": FLOAT, "pct_counts_mt": FLOAT}
SCRUBLET_COLS = {"scrublet_score": FLOAT, "predicted_doublets": INT}

Schema = dict[str, str]


class ContractError(ValueError):
    """A sample the store cannot take as it is."""


def label_columns(model: str) -> Schema:
    return {f"{model}_label": STR, f"{model}_prob": FLOAT}


def requested_schema(run_scrublet: bool, models: list[str]) -> Schema:
    """The obs columns, in order; each CellTypist model adds `<model>_label` and `<model>_prob`."""
    schema: Schema = {"sample_id": STR}
    schema.update(dict.fromkeys(HISE_STR_COLS, STR))
    schema.update(dict.fromkeys(HISE_INT_COLS, INT))
    schema.update(QC_COLS)
    if run_scrublet:
        schema.update(SCRUBLET_COLS)
    for model in models:
        schema.update(label_columns(model))
    return schema


def check_schema(store_columns: list[str], schema: Schema) -> None:
    """Appends need identical obs columns; a store's schema is fixed when it is created."""
    if store_columns == list(schema):
        return
    missing = sorted(set(store_columns) - set(schema))
    extra = sorted(set(schema) - set(store_columns))
    raise ContractError(
        "This run's obs schema differs from the store's, so its cells cannot be appended. "
        f"Store-only columns: {missing}; run-only columns: {extra}. Match the store's Scrublet setting "
        "and CellTypist models, or start a new store."
    )


def is_mito(var_names) -> np.ndarray:
    return np.char.startswith(np.asarray(var_names).astype(str), MITO_PREFIX)


def _empty_column(kind: str, n: int):
    if kind == STR:
        return np.full(n, "", dtype=object)
    if kind == FLOAT:
        return np.full(n, np.nan, dtype=np.float64)
    return pd.array([pd.NA] * n, dtype="Int64")


def _project_column(values: pd.Series | None, kind: str, n: int):
    if values is None:
        return _empty_column(kind, n)
    if kind == STR:
        return np.array(["" if pd.isna(v) else str(v) for v in values.to_numpy(dtype=object)], dtype=object)
    numeric = pd.to_numeric(values, errors="raise")
    if kind == FLOAT:
        return numeric.astype(np.float64).to_numpy()
    return pd.array(numeric, dtype="Int64")


def _as_counts(matrix) -> sp.csr_matrix:
    x = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    if x.data.size:
        if np.issubdtype(x.data.dtype, np.floating) and ((x.data < 0).any() or (x.data != np.round(x.data)).any()):
            raise ContractError("X holds negative or fractional values; the store takes raw counts only.")
        if int(x.data.max()) > np.iinfo(X_DTYPE).max:
            raise ContractError(f"X holds a count larger than {X_DTYPE.__name__} can store.")
    return x.astype(X_DTYPE)


def _check_var(var_names: np.ndarray, expected: np.ndarray | None) -> None:
    if expected is None or np.array_equal(var_names, np.asarray(expected, dtype=object)):
        return
    if len(var_names) == len(expected) and set(var_names) == set(expected):
        found = "the same genes in a different order"
    else:
        found = f"{len(var_names)} genes, the store {len(expected)}, {len(set(var_names) ^ set(expected))} not shared"
    raise ContractError(f"var mismatch: the sample has {found}; gene names and order must match the store's.")


def project(adata: ad.AnnData, schema: Schema, expected_var_names: np.ndarray | None = None) -> ad.AnnData:
    """A new AnnData holding exactly what the store accepts: CSR uint32 X, the schema's obs, and var."""
    var_names = np.asarray(adata.var_names, dtype=object)
    _check_var(var_names, expected_var_names)
    obs = pd.DataFrame(index=pd.Index(np.asarray(adata.obs_names, dtype=object), name=OBS_INDEX))
    for col, kind in schema.items():
        obs[col] = _project_column(adata.obs[col] if col in adata.obs else None, kind, adata.n_obs)
    var = pd.DataFrame(index=pd.Index(var_names))
    var["mt"] = is_mito(var_names)
    return ad.AnnData(X=_as_counts(adata.X), obs=obs, var=var)
