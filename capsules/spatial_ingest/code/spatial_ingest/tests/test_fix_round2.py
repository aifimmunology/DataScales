"""Fix round 2 (FIXES2.md Dev-B): boundary polygons built batch by batch (M7), one INFO progress line before each
long element read (L7), all of a sample's files staged in ONE stage.fetch call (L5). H5 (cells without a
position kept) is covered in test_readers / test_inventory."""
import logging

import numpy as np
import pandas as pd
import pytest

import fixtures_bundles as fb
import inventory
import readers
import stage
from contracts import Sample
from readers import elements


def boundaries(tmp_path, n=40, label_id=True, csv=False):
    ids = fb.cell_ids(n)
    xy = fb.positions(n)
    frame = fb.boundaries_frame(ids, xy, 2.0, label_id=label_id)
    frame = frame.drop(index=frame.index[5:7])                    # one 3-vertex ring (cell 1): still a polygon
    frame = frame.drop(index=frame.index[10:13])                  # one 2-vertex "ring" (cell 2): dropped
    if csv:
        path = tmp_path / "cell_boundaries.csv.gz"
        import gzip
        with gzip.open(path, "wt") as f:
            frame.to_csv(f, index=False)
        return path, ids
    return fb.write_parquet(frame, tmp_path / "cell_boundaries.parquet"), ids


@pytest.mark.parametrize("label_id,csv", [(True, False), (False, False), (False, True)])
def test_batched_polygons_equal_one_batch(tmp_path, monkeypatch, label_id, csv):
    path, ids = boundaries(tmp_path, label_id=label_id, csv=csv)
    whole = elements.polygons_from_table(path, 0.2125, tmp_path, False)
    monkeypatch.setattr(elements, "BOUNDARY_BATCH_ROWS", 7)       # polygons straddle every batch boundary
    batched = elements.polygons_from_table(path, 0.2125, tmp_path, False)
    assert list(batched.index) == list(whole.index) == [i for k, i in enumerate(ids) if k != 2]
    assert batched.geometry.geom_equals(whole.geometry).all()
    assert np.isclose(batched.geometry.iloc[0].area, 16.0)           # 4×4 square around the centroid


def test_progress_line_before_each_long_read(tmp_path, caplog):
    folder = fb.xenium_full(tmp_path / "full", labels=False)
    fb.write_zstack(folder / "morphology.ome.tif")
    cand = fb.candidate(folder)
    inv = inventory.inspect(cand, tmp_path / "s")
    with caplog.at_level(logging.INFO, logger="readers.elements"):
        readers.load(Sample(cand.sid, "xenium", cand, inventory=inv), tmp_path / "s", 1)
    lines = [r.getMessage() for r in caplog.records if r.name == "readers.elements"]
    for what in ("nucleus boundaries from nucleus_boundaries.parquet", "cell boundaries from cell_boundaries.parquet",
                 "transcripts from transcripts.parquet", "z-stack image morphology.ome.tif"):
        assert any(line.startswith(f"full: reading {what} (") and line.endswith("GB)") for line in lines), what


def test_readers_stage_every_file_in_one_fetch(tmp_path, monkeypatch):
    import tarfile

    folder = fb.xenium_full(tmp_path / "full", labels=False)
    archive = tmp_path / "full.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(folder, arcname="full")
    files = {p.relative_to(folder).as_posix(): fb.Member(archive, inner=f"full/{p.relative_to(folder).as_posix()}",
                                                          size=p.stat().st_size)
             for p in folder.rglob("*") if p.is_file()}
    cand = fb.Candidate("full", "xenium", archive, files)
    inv = inventory.inspect(cand, tmp_path / "s")
    calls, passes = [], []
    real_fetch, real_tar = stage.fetch, stage._fetch_tar
    monkeypatch.setattr(stage, "fetch", lambda c, names, dest: calls.append(list(names)) or real_fetch(c, names, dest))
    monkeypatch.setattr(stage, "_fetch_tar", lambda c, w: passes.append(c) or real_tar(c, w))
    readers.stage_bundle(Sample("full", "xenium", cand, inventory=inv), inv, tmp_path / "s2")
    assert len(calls) == 1 and set(calls[0]) == set(inv.chosen.values())
    assert passes == [archive]                                     # one pass over the .tar.gz


def test_prepared_table_keeps_unplaced_cells(tmp_path):
    """H5 end to end: the per-sample table holds every matrix cell, has_position, NaN centroids; stats count them."""
    import anndata as ad

    import sample

    folder = fb.xenium_full(tmp_path / "part", labels=False)
    cells = pd.read_parquet(folder / "cells.parquet")
    fb.write_parquet(cells.iloc[:12], folder / "cells.parquet")
    cand = fb.candidate(folder)
    s = Sample(cand.sid, "xenium", cand, inventory=inventory.inspect(cand, tmp_path / "s"))
    prep = sample.prepare(readers.load(s, tmp_path / "s", 1), s, None, tmp_path / "s")
    table = ad.read_zarr(prep.table_path)
    assert table.n_obs == 30 and table.obs["has_position"].sum() == 12
    assert np.isnan(table.obsm["spatial"][~table.obs["has_position"].to_numpy()]).all()
    assert prep.stats["n_cells_without_position"] == 18 and prep.stats["n_cells"] == 30


def open_files_under(folder) -> list[str]:
    import os

    out = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(str(folder)):
            out.append(target)
    return out


def test_readers_close_releases_every_staged_file(tmp_path):
    """Lazy elements keep their image / label files open until written; readers.close(sid) closes them, so
    deleting the staged files leaves no `.nfs…` placeholders on NFS scratch. Parquet handles never stay open."""
    import spatialdata as sd

    folder = fb.xenium_full(tmp_path / "full")
    fb.write_zstack(folder / "morphology.ome.tif", gz=True)            # staged as a gunzipped copy
    files = {p.name: p for p in folder.iterdir()}
    files["morphology.ome.tif"] = files.pop("morphology.ome.tif.gz")
    files["transcripts.parquet"] = fb.gzip_file(folder / "transcripts.parquet")   # GEO-style .parquet.gz
    cand = fb.candidate(folder, files=files)
    scratch = tmp_path / "s"
    inv = inventory.inspect(cand, scratch)
    assert not open_files_under(scratch)
    s = Sample(cand.sid, "xenium", cand, inventory=inv)
    loaded = readers.load(s, scratch, 1)
    sd.SpatialData(**loaded.elements).write(tmp_path / "out.zarr")
    assert open_files_under(scratch)                                  # the z-stack TiffFile (lazy image)
    readers.close("full")
    assert not open_files_under(scratch)
