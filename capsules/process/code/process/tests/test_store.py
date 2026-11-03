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


def test_zarr_input_gets_a_full_copy(stores, adata, tmp_path):
    src = store.open_source(str(stores["pbmc"]), "table", "main", "")
    assert (src.table, src.snapshot, src.repo) == ("", "zarr", None)
    first = store.save(src, results(adata), "m", "main", "", str(tmp_path), {})
    second = store.save(src, results(adata), "m", "main", "", str(tmp_path), {})
    assert first == f"saved to {tmp_path}/pbmc.processed.zarr" and second.endswith("pbmc.processed-1.zarr")
    out = check(zarr.open_group(tmp_path / "pbmc.processed.zarr", mode="r"), adata)
    assert list(out.obs.columns[:3]) == ["sample_id", "chip_id", "cell_type"]
    assert not list(tmp_path.glob("*.partial"))


def test_ingest_store_without_layers(stores, adata, tmp_path):
    src = store.open_source(str(stores["ingest_no_sample"]), "table", "main", "")
    store.save(src, results(adata), "m", "main", str(tmp_path / "out"), str(tmp_path), {})
    check(zarr.open_group(tmp_path / "out.zarr", mode="r"), adata)


def test_spatialdata_copy_keeps_every_element(stores, adata, tmp_path):
    import spatialdata as sd

    src = store.open_source(str(stores["tissue"]), "table", "main", "")
    assert src.table == "tables/table"
    written = store.save(src, results(adata, leiden=("leiden_0.5", "leiden_1.0")), "m", "main", "", str(tmp_path), {})
    out = Path(written.removeprefix("saved to "))
    sdata = sd.read_zarr(out)
    assert set(sdata.images) == {"image"} and set(sdata.shapes) == {"cells"}
    check(zarr.open_group(out, mode="r")["tables/table"], adata, leiden=("leiden_0.5", "leiden_1.0"))
    for f in (stores["tissue"] / "images").rglob("*"):
        if f.is_file():
            assert f.read_bytes() == (out / f.relative_to(stores["tissue"])).read_bytes()


def test_rerun_drops_stale_results(stores, adata, tmp_path):
    src = store.open_source(str(stores["pbmc"]), "table", "main", "")
    first = Path(store.save(src, results(adata, ("leiden_0.5", "leiden_2.0"), ("X_pca", "X_pca_harmony", "X_umap")),
                            "m", "main", "", str(tmp_path), {}).removeprefix("saved to "))
    again = store.open_source(str(first), "table", "main", "")
    out = Path(store.save(again, results(adata, ("leiden",), ("X_scVI", "X_umap")), "m", "main", "", str(tmp_path), {})
               .removeprefix("saved to "))
    check(zarr.open_group(out, mode="r"), adata, leiden=("leiden",), obsm=("X_scVI", "X_umap"))


def test_icechunk_through_a_read_only_mount_and_its_origin_twice(icechunk, adata, tmp_path):
    mount, origin = str(icechunk["mount"]), str(icechunk["origin"])
    before = sorted((p, p.stat().st_mtime_ns) for p in icechunk["mount"].rglob("*"))
    n = len(Repo(origin).log())
    for i in range(2):
        src = store.open_source(mount, "table", "main", origin)
        assert src.snapshot == Repo(origin).log()[0].id
        written = store.save(src, results(adata), f"run {i}", "main", str(tmp_path / "ignored"), str(tmp_path),
                             {"method": "pca"})
        assert written.startswith("committed") and written.endswith("on main")
    log = Repo(origin).log()
    assert len(log) == n + 2 and log[0].metadata["source_snapshot"] == log[1].id and log[0].metadata["method"] == "pca"
    check(Repo(origin).open_zarr("r"), adata)
    assert sorted((p, p.stat().st_mtime_ns) for p in icechunk["mount"].rglob("*")) == before
    assert not (tmp_path / "ignored.zarr").exists()


def test_branch_moved_writes_nothing(icechunk, adata, tmp_path):
    origin = str(icechunk["origin"])
    src = store.open_source(origin, "table", "main", "")
    other = Repo(origin, branch="main")
    other.open_zarr("w").attrs["touched"] = "yes"
    tip = other.commit("moved meanwhile")
    with pytest.raises(RuntimeError, match="moved since the store was opened"):
        store.save(src, results(adata), "m", "main", "", str(tmp_path), {})
    assert Repo(origin).log()[0].id == tip


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
