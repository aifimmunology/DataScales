"""Fix round 1 (FIXES.md B-1..B-5): integer cell IDs stay integers end to end (fast spatialdata joins),
the 3D morphology.ome.tif z-stack is read lazily (plain or .gz) with z in µm, Explorer transcripts.zarr.zip
is read, invalid polygons are counted, protein features come first in the dropped-features line, and
transcripts are read with bounded memory (categories by a one-column scan, small partitions)."""
import logging
import shutil
import time

import numpy as np
import pandas as pd
import pytest
import zarr

import fixtures_bundles as fb
import inventory
import readers
import sample
from contracts import Sample
from readers import elements, table as tables


def load(cand, scratch):
    inv = inventory.inspect(cand, scratch)
    return inv, readers.load(Sample(cand.sid, cand.kind, cand, inventory=inv), scratch, 1)


def test_xoa10_integer_ids_stay_integers_and_join_fast(tmp_path):
    import spatialdata as sd

    n = 20_000                      # with text keys, spatialdata's join (np.isin on objects) takes ~6 s here
    cand = fb.candidate(fb.xenium_xoa10(tmp_path / "rep1", n=n))
    _, loaded = load(cand, tmp_path / "s")
    assert pd.api.types.is_integer_dtype(loaded.table.obs["cell_id"])
    shapes = loaded.elements["shapes"]["cell_boundaries"]
    assert pd.api.types.is_integer_dtype(shapes.index) and loaded.regions == ["cell_boundaries"]
    prep = sample.prepare(loaded, Sample(cand.sid, "xenium", cand), None, tmp_path / "s")
    store = tmp_path / "store.zarr"
    sd.SpatialData(**prep.elements).write(store)
    zarr.open_group(str(store / "tables"), mode="a")
    shutil.move(str(prep.table_path), store / "tables" / "rep1_table")
    zarr.consolidate_metadata(str(store))
    back = sd.read_zarr(store)
    t0 = time.time()
    _, joined = sd.join_spatialelement_table(back, spatial_element_names="rep1_cell_boundaries",
                                             table_name="rep1_table", how="left")
    assert joined.n_obs == n and time.time() - t0 < 3


def test_text_ids_stay_text_on_both_sides(tmp_path):
    _, loaded = load(fb.candidate(fb.xenium_full(tmp_path / "full")), tmp_path / "s")
    obs_ids = loaded.table.obs["cell_id"]
    for name in ("cell_boundaries", "nucleus_boundaries"):
        shapes = loaded.elements["shapes"][name]
        ids = shapes.index if name == "cell_boundaries" else shapes["cell_id"]
        assert pd.api.types.is_string_dtype(ids) and set(ids) <= set(obs_ids)


@pytest.mark.parametrize("gz", [False, True])
def test_zstack_is_read_lazily_as_3d_with_z_in_microns(tmp_path, gz):
    from spatialdata.transformations import get_transformation

    folder = fb.xenium_full(tmp_path / "z", labels=False)
    fb.write_zstack(folder / "morphology.ome.tif", gz=gz)
    files = None
    if gz:   # GEO: `morphology.ome.tif.gz` under the canonical name, gz noted (discovery's convention)
        files = {p.name: p for p in folder.iterdir()}
        files["morphology.ome.tif"] = files.pop("morphology.ome.tif.gz")
    cand = fb.candidate(folder, files=files)
    inv, loaded = load(cand, tmp_path / "s")
    assert inv.chosen["morphology_3d"] == "morphology.ome.tif" and "morphology images" not in inv.missing_optional
    image = loaded.elements["images"]["morphology"]
    scale0 = image["scale0"]["image"]
    assert scale0.dims == ("c", "z", "y", "x") and scale0.shape == (1, 3, 128, 128)
    assert list(scale0.coords["c"].values) == ["DAPI"]
    assert get_transformation(image).to_affine_matrix(("z", "y", "x"), ("z", "y", "x"))[0, 0] == 3.0
    tiles = elements.tiled_imread(tables.plain(loaded_path(tmp_path, gz), tmp_path / "s"))
    assert tiles.chunksize[-2:] == (64, 64)                       # tile reads, never whole planes


def loaded_path(tmp_path, gz):
    staged = tmp_path / "s" / "stage" / "z" / "morphology.ome.tif"
    return staged if staged.exists() else tmp_path / "z" / ("morphology.ome.tif.gz" if gz else "morphology.ome.tif")


def test_explorer_transcripts_zarr_read_when_no_parquet(tmp_path):
    folder = fb.xenium_explorer(tmp_path / "xe")
    frame = fb.transcripts_frame(fb.cell_ids(20), fb.positions(20, 2))
    fb.write_explorer_transcripts(folder / "transcripts.zarr.zip", frame)
    inv, loaded = load(fb.candidate(folder), tmp_path / "s")
    assert inv.chosen["transcripts"] == "transcripts.zarr.zip" and "transcripts" not in inv.missing_optional
    points = loaded.elements["points"]["transcripts"].compute()
    assert len(points) == len(frame)
    assert points["feature_name"].astype(str).value_counts().sort_index().equals(
        frame["feature_name"].value_counts().sort_index().rename_axis("feature_name"))
    assert np.isclose(points["x"].sum(), frame["x_location"].sum(), rtol=1e-5)


def test_atera_3d_image_z_in_microns(tmp_path):
    from conftest import bundle
    from spatialdata.transformations import get_transformation

    _, loaded = load(fb.candidate(bundle("tiny_atera_dataset"), "atera"), tmp_path / "s")
    image = loaded.elements["images"]["morphology_3d"]
    assert get_transformation(image).to_affine_matrix(("z", "y", "x"), ("z", "y", "x"))[0, 0] == 3.0


def test_invalid_polygons_counted_in_one_info_line(tmp_path, caplog):
    folder = fb.xenium_full(tmp_path / "bad", labels=False)
    bounds = pd.read_parquet(folder / "nucleus_boundaries.parquet")
    first = bounds["label_id"] == 1                       # a self-intersecting bow-tie for cell 1
    bounds.loc[first, ["vertex_x", "vertex_y"]] = [[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]]
    fb.write_parquet(bounds, folder / "nucleus_boundaries.parquet")
    with caplog.at_level(logging.INFO):
        load(fb.candidate(folder), tmp_path / "s")
    lines = [r.getMessage() for r in caplog.records if r.name == "readers"]
    assert len(lines) == 1 and "nucleus_boundaries: 1 of 30 polygons are invalid geometries" in lines[0]


def test_protein_listed_first_with_names(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        load(fb.candidate(fb.xenium_full(tmp_path / "p", labels=False)), tmp_path / "s")
    line = next(r.getMessage() for r in caplog.records if r.name == "readers.table")
    assert line.split("left out: ")[1].startswith("2 Protein Expression (PD-1, CD45-prot)")


def test_transcripts_categories_and_partitions_bounded(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(elements, "POINTS_PARTITION_ROWS", 40)
    frame = fb.transcripts_frame(fb.cell_ids(60), fb.positions(60), per_cell=3)
    path = tmp_path / "transcripts.parquet"
    pq.write_table(__import__("pyarrow").Table.from_pandas(frame, preserve_index=False), path, row_group_size=20)
    points = elements.transcripts(path, 0.2125, tmp_path)
    assert points.npartitions >= len(frame) // 40
    assert list(points["feature_name"].dtype.categories) == list(dict.fromkeys(frame["feature_name"]))
    assert len(points.compute()) == len(frame)
