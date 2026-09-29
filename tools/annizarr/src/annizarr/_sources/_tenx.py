from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os

    import anndata as ad
    import h5py
    from numpy.typing import NDArray

_FEATURE_EXTRA_EXCLUDE = frozenset({"name", "feature_type", "id", "gene_id", "_all_tag_keys"})


def load_10x_h5(path: str | os.PathLike[str]) -> ad.AnnData:
    import h5py

    with h5py.File(str(path), "r") as f:
        if "matrix" in f:
            return _read_v3_10x_h5(f)
        return _read_legacy_10x_h5(f)


def _as_float32_if_int32(data: NDArray[Any]) -> NDArray[Any]:
    import numpy as np

    # some Cell Ranger versions tag float32 counts as int32; widen to match
    return data.astype(np.float32) if data.dtype == np.dtype("int32") else data


def _read_v3_10x_h5(f: h5py.File) -> ad.AnnData:
    import anndata as ad
    import h5py

    m = f["matrix"]
    try:
        data = _as_float32_if_int32(m["data"][()])
        indices = m["indices"][()]
        indptr = m["indptr"][()]
        shape = m["shape"][()]  # on-disk shape is (n_features, n_barcodes)
        n_vars, n_obs = int(shape[0]), int(shape[1])
        barcodes = m["barcodes"][()].astype(str)
        if "features" not in m:
            raise ValueError("10x v3 h5 file has no features group")
        feat = m["features"]
        names = feat["name"][()].astype(str)
        feature_types = feat["feature_type"][()].astype(str)
        if "gene_id" in feat:
            # probe-barcode matrix: "id" is the probe id, "gene_id" the underlying gene
            gene_ids = feat["gene_id"][()].astype(str)
            probe_ids: NDArray[Any] | None = feat["id"][()].astype(str)
        else:
            gene_ids = feat["id"][()].astype(str)
            probe_ids = None
    except KeyError as exc:
        raise ValueError(f"10x v3 h5 file is missing a required dataset: {exc}") from exc

    var: dict[str, Any] = {"var_names": names, "gene_ids": gene_ids}
    if probe_ids is not None:
        var["probe_ids"] = probe_ids
    var["feature_types"] = feature_types
    for key, dset in feat.items():
        if isinstance(dset, h5py.Dataset) and key not in _FEATURE_EXTRA_EXCLUDE:
            values = dset[()]
            var[key] = values.astype(bool) if values.dtype.kind == "b" else values.astype(str)

    obs: dict[str, Any] = {"obs_names": barcodes}
    if "filtered_barcodes" in m:
        obs["filtered_barcodes"] = m["filtered_barcodes"][()].astype(bool)

    from scipy.sparse import csr_matrix

    matrix = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    adata = ad.AnnData(matrix, obs=obs, var=var)
    return adata[:, adata.var["feature_types"] == "Gene Expression"].copy()


def _read_legacy_10x_h5(f: h5py.File) -> ad.AnnData:
    import anndata as ad

    children = list(f.keys())
    if not children:
        raise ValueError(f"{f.filename} has no top-level groups; not a recognised 10x v2 layout.")
    if len(children) > 1:
        raise ValueError(
            f"{f.filename} contains more than one genome. Legacy 10x h5 files with multiple "
            f"genomes are not supported here. Available genomes: {children}"
        )
    g = f[children[0]]
    try:
        data = _as_float32_if_int32(g["data"][()])
        indices = g["indices"][()]
        indptr = g["indptr"][()]
        shape = g["shape"][()]  # on-disk shape is (n_genes, n_barcodes)
        n_vars, n_obs = int(shape[0]), int(shape[1])
        barcodes = g["barcodes"][()].astype(str)
        gene_names = g["gene_names"][()].astype(str)
        gene_ids = g["genes"][()].astype(str)
    except KeyError as exc:
        raise ValueError(f"10x v2 h5 file is missing a required dataset: {exc}") from exc

    from scipy.sparse import csr_matrix

    matrix = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    return ad.AnnData(matrix, obs={"obs_names": barcodes}, var={"var_names": gene_names, "gene_ids": gene_ids})
