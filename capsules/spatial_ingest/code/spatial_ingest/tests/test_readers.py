"""readers.load on bundles: the streamed table equals the source matrix (Gene Expression columns, cells with a
position), elements are built from whatever exists, missing parts said once (INFO; the precheck warns)."""
import logging

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import fixtures_bundles as fb
import inventory
import matrix
import readers
from contracts import Sample


def load(cand, scratch):
    inv = inventory.inspect(cand, scratch)
    return readers.load(Sample(cand.sid, cand.kind, cand, inventory=inv), scratch, 1)


def table_x(loaded) -> sp.csr_matrix:
    return sp.vstack([band for _, band in matrix.iter_rows(loaded.table.path)]).tocsr()


def h5_gex(path) -> tuple[sp.csr_matrix, list[str]]:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        n_feat, n_cells = m["shape"][:]
        x = sp.csr_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(n_cells, n_feat))
        keep = m["features/feature_type"][:].astype(str) == "Gene Expression"
        return x[:, keep], list(m["barcodes"][:].astype(str))


def written(elements, tmp_path):
    """The elements written as a SpatialData and read back (proves they're valid and writable)."""
    import spatialdata as sd

    sd.SpatialData(**elements).write(tmp_path / "check.zarr")
    return sd.read_zarr(tmp_path / "check.zarr")


def test_complete_xenium(tmp_path):
    folder = fb.xenium_full(tmp_path / "full")
    loaded = load(fb.candidate(folder), tmp_path / "s")
    x, barcodes = h5_gex(folder / "cell_feature_matrix.h5")
    assert (table_x(loaded) != x).nnz == 0
    assert list(loaded.table.obs.index) == barcodes and list(loaded.table.obs["cell_id"]) == barcodes
    assert loaded.regions == ["cell_boundaries"] and loaded.region_key == "region" and loaded.instance_key == "cell_id"
    assert set(loaded.table.obs["region"]) == {"cell_boundaries"}
    assert {"cell_labels", "z_level", "nucleus_count"} <= set(loaded.table.obs.columns)
    assert "x_centroid" not in loaded.table.obs
    group = zarr.open_group(str(loaded.table.path), mode="r")
    var = ad.io.read_elem(group["var"])
    assert list(var.index) == fb.GENES and list(var.columns) == ["gene_ids", "feature_types"]
    assert group["obsm"]["spatial"].shape == (30, 2)
    assert {k: sorted(v) for k, v in loaded.elements.items() if v} == {
        "labels": ["cell_labels", "nucleus_labels"], "shapes": ["cell_boundaries", "nucleus_boundaries"],
        "points": ["transcripts"]}
    back = written(loaded.elements, tmp_path)
    assert set(back.shapes["cell_boundaries"].index) == set(barcodes)
    assert len(back.points["transcripts"].compute()) == 30 * 3 + 5


def test_geo_mtx_partial_gives_circles(tmp_path, caplog):
    folder, cand = fb.geo_partial(tmp_path / "geo")
    with caplog.at_level(logging.INFO):
        loaded = load(cand, tmp_path / "s")
    x = fb.counts(25, len(fb.features()), 1)[:, :len(fb.GENES)]
    assert (table_x(loaded) != x).nnz == 0
    assert loaded.regions == ["cell_circles"] and list(loaded.elements["shapes"]) == ["cell_circles"]
    assert sorted(loaded.elements["shapes"]["cell_circles"].index) == sorted(loaded.table.obs["cell_id"])
    # the precheck warns about missing parts; the reader says it once more at INFO only
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("readers")]
    lines = [r.getMessage() for r in caplog.records if r.name == "readers"]
    assert len(lines) == 1 and lines[0].startswith("GSM1_s: not available:") and "cell boundaries" in lines[0]


def test_atera_without_cells_zarr_raises_no_warning(tmp_path, recwarn):
    loaded = load(fb.candidate(fb.atera(tmp_path / "atera"), "atera"), tmp_path / "s")
    assert loaded.regions == ["cell_circles"] and "cell_circles" in loaded.elements["shapes"]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning) and "cell_labels" in str(w.message)]


def test_missing_positions_refused(tmp_path):
    _, cand = fb.geo_partial(tmp_path / "nopos", positions_ok=False)
    with pytest.raises(readers.ReadError, match="cell positions"):
        load(cand, tmp_path / "s")


def test_cells_without_position_are_left_out(tmp_path):
    folder = fb.xenium_full(tmp_path / "drop", labels=False)
    cells = pd.read_parquet(folder / "cells.parquet")
    fb.write_parquet(cells.iloc[3:], folder / "cells.parquet")
    loaded = load(fb.candidate(folder), tmp_path / "s")
    x, barcodes = h5_gex(folder / "cell_feature_matrix.h5")
    assert loaded.table.n_obs == 27 and list(loaded.table.obs.index) == barcodes[3:]
    assert (table_x(loaded) != x[3:]).nnz == 0


@pytest.mark.parametrize("cell_major", [True, False])
def test_explorer_zarr_matrix(tmp_path, cell_major):
    folder = fb.xenium_explorer(tmp_path / "xe", cell_major=cell_major)
    loaded = load(fb.candidate(folder), tmp_path / "s")
    x = fb.counts(20, len(fb.features()), 2)[:, :len(fb.GENES)]
    assert (table_x(loaded) != x).nnz == 0
    assert list(loaded.table.obs["cell_id"]) == list(fb.cell_ids(20))
    assert loaded.regions == ["cell_boundaries"]
    assert sorted(loaded.elements["shapes"]) == ["cell_boundaries", "nucleus_boundaries"]
    assert sorted(loaded.elements["labels"]) == ["cell_labels", "nucleus_labels"]


def test_reseg_points_match_the_table(tmp_path):
    loaded = load(fb.candidate(fb.xenium_reseg(tmp_path / "reseg")), tmp_path / "s")
    points = loaded.elements["points"]["transcripts"].compute()
    assigned = set(points["cell_id"]) - {"UNASSIGNED"}
    assert assigned <= set(loaded.table.obs["cell_id"])


def test_xoa1_bytes_transcripts_and_nan_label_id(tmp_path):
    loaded = load(fb.candidate(fb.xenium_bytes(tmp_path / "bytes")), tmp_path / "s")
    shapes = loaded.elements["shapes"]["cell_boundaries"]
    assert len(shapes) == 20                                   # grouped by cell_id, not one polygon per vertex
    back = written(loaded.elements, tmp_path)                  # bytes columns would fail to write
    points = back.points["transcripts"].compute()
    assert set(points["fov_name"]) == {"A1"} and "UNASSIGNED" in set(points["cell_id"])
    assert str(points["feature_name"].iloc[0]).startswith("GENE") or str(points["feature_name"].iloc[0]).startswith("MT-")


def test_atera(tmp_path):
    folder = fb.atera(tmp_path / "atera")
    loaded = load(fb.candidate(folder, "atera"), tmp_path / "s")
    x = fb.counts(15, len(fb.features()), 5)[:, :len(fb.GENES)]
    assert (table_x(loaded) != x).nnz == 0
    obs = loaded.table.obs
    assert list(obs["cell_id"]) == list(range(1000, 1015)) and list(obs["label_id"]) == list(range(1, 16))
    assert loaded.regions == ["cell_circles"]
    circles = loaded.elements["shapes"]["cell_circles"]
    assert list(circles.index) == list(obs["cell_id"])
    var = ad.io.read_elem(zarr.open_group(str(loaded.table.path), mode="r")["var"])
    assert list(var.index) == fb.GENES and var["gene_ids"].iloc[0].startswith("ENSG")


def test_real_breast_2fov(tmp_path):
    from conftest import bundle

    folder = bundle("Xenium_V1_human_Breast_2fov")
    loaded = load(fb.candidate(folder), tmp_path / "s")
    x, barcodes = h5_gex(folder / "cell_feature_matrix.h5")
    assert (table_x(loaded) != x).nnz == 0 and list(loaded.table.obs.index) == barcodes
    assert {k: sorted(v) for k, v in loaded.elements.items()} == {
        "images": ["morphology", "morphology_focus"], "labels": ["cell_labels", "nucleus_labels"],
        "shapes": ["cell_boundaries", "nucleus_boundaries"], "points": ["transcripts"]}
    stack = loaded.elements["images"]["morphology"]                     # the z-stack, lazy, z in µm
    from spatialdata.transformations import get_transformation
    assert tuple(stack["scale0"].ds.dims)[:2] == ("c", "z")
    assert get_transformation(stack).to_affine_matrix(("z",), ("z",))[0, 0] == 3.0
    from readers import elements
    tiles = elements.tiled_imread(folder / "morphology_focus" / "morphology_focus_0000.ome.tif")
    assert tiles.shape[0] == 4 and max(tiles.chunksize[1:]) <= 1024    # read by tiles, not whole planes


def test_real_tiny_atera(tmp_path):
    from conftest import bundle

    loaded = load(fb.candidate(bundle("tiny_atera_dataset"), "atera"), tmp_path / "s")
    assert loaded.table.n_obs == 13 and loaded.table.n_vars == 53 and loaded.regions == ["cell_boundaries"]
    assert {"morphology", "morphology_3d"} == set(loaded.elements["images"])


def test_reading_a_bundle_streams_the_matrix(tmp_path, monkeypatch):
    """Peak traced (numpy/pandas) memory of readers.load on an h5 bundle stays well below the matrix's own
    in-memory size: bands (BAND_NNZ) and zarr chunks (ARRAY_CHUNK) are shrunk so a test-sized matrix shows it."""
    import tracemalloc

    monkeypatch.setattr(matrix, "BAND_NNZ", 50_000)
    monkeypatch.setattr(matrix, "ARRAY_CHUNK", 50_000)
    n, n_genes = 10_000, 1_000                                       # 3 M nonzeros: the matrix dominates obs
    folder = tmp_path / "big"
    folder.mkdir()
    ids = np.array([f"c{i:06d}-1" for i in range(n)], dtype=object)
    feats = pd.DataFrame({"id": [f"ENSG{i:011d}" for i in range(n_genes)], "name": [f"G{i}" for i in range(n_genes)],
                          "feature_type": "Gene Expression"})
    x = fb.counts(n, n_genes, seed=7, density=0.3)
    full_bytes = x.data.astype(np.float32).nbytes + x.indices.astype(np.int32).nbytes
    fb.write_h5(folder / "cell_feature_matrix.h5", x, ids, feats)
    fb.write_parquet(fb.cells_frame(ids, fb.positions(n)), folder / "cells.parquet")
    del x
    cand = fb.candidate(folder)
    inv = inventory.inspect(cand, tmp_path / "s")
    load(fb.geo_partial(tmp_path / "warm")[1], tmp_path / "w")        # imports done before measuring
    tracemalloc.start()
    loaded = readers.load(Sample(cand.sid, "xenium", cand, inventory=inv), tmp_path / "s", 1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert loaded.table.n_obs == n
    assert peak < full_bytes / 2, (peak, full_bytes)
