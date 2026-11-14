import ast
import json
import os
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from annizarr import Repo

import store
from conftest import N_CELLS, N_GENES

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


class Gpu(sp.csr_matrix):
    def get(self):
        return sp.csr_matrix(self)


def results(adata, leiden=("leiden",), obsm=("X_pca", "X_umap")) -> store.Results:
    x = da.from_array(adata.X.astype(np.float32), chunks=(1000, N_GENES), asarray=False)
    lognorm = x.map_blocks(lambda b: Gpu(b), dtype=np.float32, meta=Gpu((0, 0), dtype=np.float32))
    rng = np.random.default_rng(0)
    return store.Results(
        lognorm=lognorm, target_sum=1e4, highly_variable=np.arange(N_GENES) < 50,
        obsm={k: rng.normal(size=(N_CELLS, 2)) for k in obsm},
        obs={k: pd.Categorical(rng.integers(0, 3, N_CELLS).astype(str)) for k in leiden},
        uns={"method": "pca", "leiden": {k: 1.0 for k in leiden}},
    )


def check(group, adata, leiden=("leiden",), obsm=("X_pca", "X_umap")):
    out = ad.io.read_elem(group)
    assert set(obsm) == set(out.obsm) & set(store.PROCESS_OBSM)
    assert set(leiden) == {c for c in out.obs if store.LEIDEN_COLUMN.match(c)}
    assert out.var["highly_variable"].sum() == 50 and out.uns["process"]["method"] == "pca"
    assert group["layers/lognorm"].attrs["target_sum"] == 1e4
    assert (out.X != adata.X).nnz == 0 and (out.layers["lognorm"] != adata.X).nnz == 0
    return out


def test_unreadable_input_is_one_line(tmp_path, stores):
    with pytest.raises(store.InputError, match="can't open .*missing.zarr as an AnnData"):
        store.open_source(str(tmp_path / "missing.zarr"), "table", "main", "")
    with pytest.raises(store.InputError, match="has no X and no tables/other"):
        store.open_source(str(stores["tissue"]), "other", "main", "")


def test_new_output_never_reuses_a_name(tmp_path):
    assert store.new_output("/data/a/pbmc.zarr/", "", str(tmp_path)) == f"{tmp_path}/pbmc.processed.zarr"
    (tmp_path / "pbmc.processed.zarr").mkdir()
    (tmp_path / "pbmc.processed-1.zarr.partial").mkdir()
    assert store.new_output("/data/pbmc.zarr", "", str(tmp_path)) == f"{tmp_path}/pbmc.processed-2.zarr"
    assert store.new_output("x", str(tmp_path / "out"), "/r") == f"{tmp_path}/out.zarr"
    assert store.free_path(str(tmp_path / "scvi_model")) == str(tmp_path / "scvi_model")


def _functions(source: str, names) -> dict[str, str]:
    tree = ast.parse(source)
    return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


def test_helpers_match_the_notebooks():
    """store.py's copies of the notebooks' helper cell must stay word for word the same."""
    names = ("child", "put", "put_column", "free_path", "new_output", "save_copy", "save_commit")
    module = _functions(Path(store.__file__).read_text(), names)
    assert set(module) == set(names)
    for nb in NOTEBOOKS.glob("*.ipynb"):
        cell = next("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                    if "def save_commit" in "".join(c["source"]))
        assert _functions(cell, names) == module, nb.name
