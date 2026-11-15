"""store: installing a prepared sample (one commit), the repair of a killed install, and verify_tables."""
import pandas as pd
import pytest

import fixtures_sdata as F
import store as stores
from contracts import Prepared


def prepared(tmp_path, sid="S1", **table_kwargs):
    raw = F.elements("a", 6, sid)
    elements = {"images": {f"{sid}_image": raw["images"]["image-a"]},
                "shapes": {f"{sid}_cells": raw["shapes"]["cells-a"]},
                "points": {f"{sid}_points": raw["points"]["points-a"]}}
    table = F.sample_table(tmp_path / "scratch" / f"{sid}.zarr", sid, F.GENES, 6, 0, **table_kwargs)
    return Prepared(sid, elements, table, pd.DataFrame(), pd.DataFrame(), "fp", "region")


@pytest.fixture
def local(tmp_path):
    path = tmp_path / "store.zarr"
    stores.create(path)
    return path


def test_install_sample_is_one_commit(tmp_path, local):
    import spatialdata as sd

    moved = stores.install_sample(local, prepared(tmp_path))
    assert sorted(moved) == ["images/S1_image", "points/S1_points", "shapes/S1_cells", "tables/S1_table"]
    assert stores.committed(local)["tables"] == set()          # nothing counts before the caller consolidates
    stores.consolidate(local)
    sdata = sd.read_zarr(local)
    assert set(sdata.tables) == {"S1_table"} and set(sdata.shapes) == {"S1_cells"}
    assert sdata.tables["S1_table"].n_obs == 6 and stores.verify_tables(local) == 1
    with pytest.raises(stores.StoreError, match="already exists"):
        stores.install_sample(local, prepared(tmp_path / "again"))


def test_killed_install_is_removed_by_the_repair(tmp_path, local):
    stores.install_sample(local, prepared(tmp_path))
    actions = stores.remove_uncommitted(local)
    assert "deleted uncommitted tables/S1_table" in actions and "deleted uncommitted shapes/S1_cells" in actions
    assert not (local / "tables" / "S1_table").exists()


def test_table_from_another_filesystem_is_copied(tmp_path, local, monkeypatch):
    import os

    real = os.rename

    p = prepared(tmp_path)

    def cross_device(src, dst):
        if str(src) == str(p.table_path):
            raise OSError(18, "Invalid cross-device link")
        return real(src, dst)

    monkeypatch.setattr(stores.os, "rename", cross_device)
    stores.install_sample(local, p)
    assert (local / "tables" / "S1_table" / "X").is_dir() and p.table_path.exists()


def test_verify_tables_catches_a_case_clash(tmp_path, local):
    p = prepared(tmp_path, extra_obs={"cell_ID": range(6)})          # beside cell_id: spatialdata refuses it
    stores.install_sample(local, p)
    stores.consolidate(local)
    with pytest.raises(stores.StoreError, match="S1_table.*cell_ID"):
        stores.verify_tables(local)


def test_verify_tables_catches_a_dangling_region(tmp_path, local):
    p = prepared(tmp_path)
    p.elements["shapes"] = {}
    stores.install_sample(local, p)
    stores.consolidate(local)
    with pytest.raises(stores.StoreError, match="S1_cells"):
        stores.verify_tables(local)


def test_store_kinds(tmp_path, local):
    assert stores.describe(local) == "ours" and stores.describe(tmp_path / "nope") == "missing"
    v2 = F.write(tmp_path / "other.zarr", {"a": {"n": 4, "seed": 0}})
    assert stores.describe(v2) == "spatialdata"
    with pytest.raises(stores.StoreError, match="put it in List of paths to samples and give a new Storage Location"):
        stores.check_usable(v2, may_create=True)
    bundle = tmp_path / "Xenium_outs"
    bundle.mkdir()
    (bundle / "experiment.xenium").write_text("{}")
    with pytest.raises(stores.StoreError, match="is a Xenium bundle: put it in List of paths to samples"):
        stores.check_usable(bundle, may_create=True)
    (tmp_path / "x.zip").write_bytes(b"PK")
    assert stores.describe(tmp_path / "x.zip") == "file"


def test_failed_element_write_is_retried_once(tmp_path, local, monkeypatch):
    import os

    import spatialdata as sd

    real, left = os.replace, {"n": 1}

    def replace(src, dst, *a, **k):
        if left["n"] and str(src).endswith(".partial") and f"{stores.STAGING}/S1.zarr" in str(src):
            left["n"] -= 1
            raise FileNotFoundError(2, "No such file or directory", str(src))
        return real(src, dst, *a, **k)

    p = prepared(tmp_path)
    monkeypatch.setattr(os, "replace", replace)
    stores.install_sample(local, p)
    stores.consolidate(local)
    assert left["n"] == 0 and set(sd.read_zarr(local).images) == {"S1_image"}


# ---------------------------------------------------------------- round 2: uninstall, create, locks, progress

def test_uninstall_sample_removes_exactly_its_elements(tmp_path, local):
    import spatialdata as sd

    stores.install_sample(local, prepared(tmp_path / "a", "S1"))
    stores.install_sample(local, prepared(tmp_path / "b", "S1_x"))        # a longer ID sharing the prefix
    stores.consolidate(local)
    removed = stores.uninstall_sample(local, "S1")
    stores.consolidate(local)
    assert sorted(removed) == ["images/S1_image", "points/S1_points", "shapes/S1_cells", "tables/S1_table"]
    sdata = sd.read_zarr(local)
    assert set(sdata.tables) == {"S1_x_table"} and set(sdata.shapes) == {"S1_x_cells"}


def test_uninstall_without_a_record_uses_names(tmp_path, local):
    import zarr

    stores.install_sample(local, prepared(tmp_path / "a", "S1"))
    stores.install_sample(local, prepared(tmp_path / "b", "S1_x"))
    for sid in ("S1", "S1_x"):                                               # a store from before the record
        del zarr.open_group(local / "tables" / f"{sid}_table", mode="r+").attrs[stores.ELEMENTS_ATTR]
    assert sorted(stores.uninstall_sample(local, "S1")) == ["images/S1_image", "points/S1_points",
                                                            "shapes/S1_cells", "tables/S1_table"]
    assert (local / "shapes" / "S1_x_cells").is_dir()


def test_install_logs_one_progress_line_per_element(tmp_path, local, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="store"):
        stores.install_sample(local, prepared(tmp_path))
    lines = [r.getMessage() for r in caplog.records if "writing" in r.getMessage()]
    assert len(lines) == 3 and any("image S1_image (1 x 32 x 32" in line for line in lines)


def test_killed_creation_is_recreated(tmp_path):
    """Critic L1: an older version's killed creation (store written, marker not yet) holds nothing."""
    from spatialdata import SpatialData

    path = tmp_path / "s.zarr"
    SpatialData().write(path)
    assert stores.describe(path) == "empty"
    assert stores.check_usable(path, may_create=True) is True
    stores.create(path)
    assert stores.describe(path) == "ours"
    (tmp_path / ".s2.zarr.creating").mkdir()                                 # a killed new-style creation
    stores.create(tmp_path / "s2.zarr")
    assert not (tmp_path / ".s2.zarr.creating").exists() and stores.describe(tmp_path / "s2.zarr") == "ours"


def test_store_with_data_is_never_taken_for_a_killed_creation(tmp_path):
    v2 = F.write(tmp_path / "other.zarr", {"a": {"n": 4, "seed": 0}})
    assert stores.describe(v2) == "spatialdata"


def test_local_lock_refuses_a_second_run(tmp_path, local):
    """Critic M4: two runs on one local store."""
    import json
    import subprocess
    import sys

    lock = stores.acquire_lock(local)
    with pytest.raises(stores.StoreError, match="Another run is adding to"):
        stores.acquire_lock(local)
    stores.release_lock(lock)
    assert not lock.exists()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stores.lock_path(local).write_text(json.dumps({"host": __import__("socket").gethostname(), "pid": dead.pid,
                                                   "started": 0}))
    lock = stores.acquire_lock(local)                                        # its run is gone: taken over
    assert json.loads(lock.read_text())["pid"] == __import__("os").getpid()
    stores.release_lock(lock)
