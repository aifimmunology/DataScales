"""sample.prepare: the per-sample table written band by band — counts, obs (sample_id, QC, labels), uns
record, spatialdata attributes for the renamed region — with QC, labels and fingerprint independent of
how the rows are chunked."""
import hashlib
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import annotate
import fixtures_bundles as fb
import inventory
import matrix
import readers
import sample
from contracts import Loaded, Sample, TableSource


def loaded_sample(folder, tmp_path, kind="xenium"):
    cand = fb.candidate(folder, kind)
    s = Sample(cand.sid, kind, cand, inventory=inventory.inspect(cand, tmp_path / "s"))
    return s, readers.load(s, tmp_path / "s", 1)


def old_fingerprint(x: sp.csr_matrix, ids) -> str:
    """The in-memory fingerprint of earlier versions (stored in existing stores' uns records)."""
    digest = hashlib.sha256(f"{x.shape[0]}:{x.shape[1]}".encode())
    for part in (x.indptr.astype(np.int64), x.indices.astype(np.int64), x.data.astype(np.float64)):
        digest.update(np.ascontiguousarray(part).tobytes())
    digest.update("\n".join(map(str, ids)).encode())
    return digest.hexdigest()


@pytest.fixture
def full(tmp_path):
    return loaded_sample(fb.xenium_full(tmp_path / "full"), tmp_path)


def test_prepared_table_is_complete(full, tmp_path):
    s, loaded = full
    source_x = sp.vstack([b for _, b in matrix.iter_rows(loaded.table.path)]).tocsr()
    prep = sample.prepare(loaded, s, None, tmp_path / "s")
    assert prep.table_path == tmp_path / "s" / "prepared" / "full" / "full_table"
    table = ad.read_zarr(prep.table_path)
    assert (table.X != source_x).nnz == 0 and sp.issparse(table.X)
    assert list(table.var_names) == fb.GENES and table.obsm["spatial"].shape == (30, 2)
    assert set(table.obs["sample_id"]) == {"full"} and set(table.obs["region"]) == {"full_cell_boundaries"}
    expected_qc = annotate.qc_metrics(ad.AnnData(source_x, obs=pd.DataFrame(index=table.obs_names),
                                                 var=pd.DataFrame(index=table.var_names)))
    for col in annotate.QC_COLUMNS:
        assert np.allclose(table.obs[col].to_numpy(float), expected_qc[col].to_numpy(float))
    record = table.uns["spatial_ingest"]
    assert record["fingerprint"] == prep.fingerprint == old_fingerprint(source_x, table.obs["cell_id"])
    assert record["genes_in_panel"] == 42 and record["celltypist_model"] == ""
    assert table.uns["spatialdata_attrs"] == {"region": "full_cell_boundaries", "region_key": "region",
                                              "instance_key": "cell_id"}
    attrs = dict(zarr.open_group(str(prep.table_path), mode="r").attrs)
    assert attrs["spatialdata-encoding-type"] == "ngff:regions_table" and attrs["region"] == "full_cell_boundaries"
    assert prep.stats["n_cells"] == 30 and prep.stats["mean_total_counts"] > 0
    assert sorted(prep.elements["shapes"]) == ["full_cell_boundaries", "full_nucleus_boundaries"]


def test_prepared_sample_reads_back_as_spatialdata(full, tmp_path):
    """Elements written plus the table moved under tables/: spatialdata reads it and joins table to region."""
    import spatialdata as sd

    s, loaded = full
    prep = sample.prepare(loaded, s, None, tmp_path / "s")
    store = tmp_path / "store.zarr"
    sd.SpatialData(**prep.elements).write(store)
    zarr.open_group(str(store / "tables"), mode="a")
    shutil.move(str(prep.table_path), store / "tables" / "full_table")
    zarr.consolidate_metadata(str(store))
    back = sd.read_zarr(store)
    table = back.tables["full_table"]
    assert table.n_obs == 30
    from spatialdata.transformations import get_transformation
    assert list(get_transformation(back.shapes["full_cell_boundaries"], get_all=True)) == ["full"]
    _, joined = sd.join_spatialelement_table(back, spatial_element_names="full_cell_boundaries",
                                             table_name="full_table", how="inner")
    assert joined.n_obs == 30


def test_fingerprint_and_qc_do_not_depend_on_chunking(full, tmp_path, monkeypatch):
    s, loaded = full
    whole = sample.prepare(loaded, s, None, tmp_path / "a")
    monkeypatch.setattr(matrix, "BAND_NNZ", 17)
    monkeypatch.setattr(sample, "FINGERPRINT_CHUNK", 5)
    banded = sample.prepare(loaded, s, None, tmp_path / "b")
    assert banded.fingerprint == whole.fingerprint
    pd.testing.assert_frame_equal(banded.obs, whole.obs)


def test_celltypist_labels_per_band_equal_whole(tmp_path, tiny_model, monkeypatch):
    s, loaded = loaded_sample(fb.xenium_full(tmp_path / "ct", n=40), tmp_path)
    whole = sample.prepare(loaded, s, tiny_model, tmp_path / "a")
    monkeypatch.setattr(matrix, "BAND_NNZ", 60)
    monkeypatch.setattr(sample, "LABEL_ROWS", 3)
    banded = sample.prepare(loaded, s, tiny_model, tmp_path / "b")
    label, prob = annotate.label_columns(tiny_model)
    assert list(banded.obs[label].astype(str)) == list(whole.obs[label].astype(str))
    assert np.allclose(banded.obs[prob], whole.obs[prob])
    assert isinstance(banded.obs[label].dtype, pd.CategoricalDtype)
    assert whole.stats["n_celltypist_labels"] >= 1
    assert ad.read_zarr(whole.table_path).uns["spatial_ingest"]["celltypist_model"] == tiny_model.stem


def test_circles_region_and_atera_ids(tmp_path):
    s, loaded = loaded_sample(fb.atera(tmp_path / "atera"), tmp_path, "atera")
    prep = sample.prepare(loaded, s, None, tmp_path / "s")
    table = ad.read_zarr(prep.table_path)
    assert table.uns["spatialdata_attrs"]["region"] == "atera_cell_circles"
    assert list(prep.elements["shapes"]) == ["atera_cell_circles"]
    assert table.obs.index.name is None and "barcode" in table.obs     # index/column name clash resolved


def test_non_count_values_refused(tmp_path):
    group = zarr.open_group(str(tmp_path / "t.zarr"), mode="w")
    group.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    matrix.write_csr(group, [sp.csr_matrix(np.array([[0.5, 1.0], [2.0, 0.0]]))], 2)
    obs = pd.DataFrame({"cell_id": ["a", "b"]}, index=["a", "b"])
    ad.io.write_elem(group, "obs", obs)
    ad.io.write_elem(group, "var", pd.DataFrame({"gene_ids": ["", ""], "feature_types": "Gene Expression"},
                                                index=["G1", "G2"]))
    loaded = Loaded({}, TableSource(tmp_path / "t.zarr", obs, 2, 2), [], None, None)
    cand = fb.candidate(tmp_path, "spatialdata", "x", files={})
    with pytest.raises(sample.SampleError, match="non-count"):
        sample.prepare(loaded, Sample("x", "spatialdata", cand), None, tmp_path / "s")


def test_real_breast_fingerprint_matches_old_store_record(tmp_path):
    """The streamed fingerprint equals the one earlier versions stored for the same bundle."""
    from conftest import bundle

    s, loaded = loaded_sample(bundle("Xenium_V1_human_Breast_2fov"), tmp_path)
    prep = sample.prepare(loaded, s, None, tmp_path / "s")
    assert prep.stats["n_cells"] == 7275 and prep.stats["genes_in_panel"] == 280
    assert prep.fingerprint == "d0027ed437be862790672ccfdbf2dfad8a316cbaf87c4da2397b1260c863999c"


def test_obs_name_clashes_renamed(tmp_path):
    group = zarr.open_group(str(tmp_path / "t.zarr"), mode="w")
    group.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    matrix.write_csr(group, [sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))], 2)
    obs = pd.DataFrame({"cell_id": ["a", "b"], "cell_ID": ["x", "y"], "bad name": [1, 2],
                        "region": ["cells", "cells"]}, index=["a", "b"])
    ad.io.write_elem(group, "var", pd.DataFrame({"gene_ids": ["", ""], "feature_types": "Gene Expression"},
                                                index=["G1", "G2"]))
    loaded = Loaded({}, TableSource(tmp_path / "t.zarr", obs, 2, 2), ["cells"], "region", "cell_id")
    cand = fb.candidate(tmp_path, "xenium", "x", files={})
    prep = sample.prepare(loaded, Sample("x", "xenium", cand), None, tmp_path / "s")
    assert {"source_cell_ID", "bad_name", "cell_id"} <= set(prep.obs.columns) and "cell_ID" not in prep.obs
    assert zarr.open_group(str(prep.table_path), mode="r").metadata.zarr_format == 3
    assert ad.read_zarr(prep.table_path).uns["spatialdata_attrs"]["region"] == "x_cells"
