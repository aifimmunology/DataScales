import json
import os
import socket
import subprocess
import sys
import uuid

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import conftest as fx
import inputs
import prepare
import store as stores
from conftest import GENES
from config import Prepared, Sample


@pytest.fixture
def local(tmp_path):
    path = tmp_path / "store.zarr"
    stores.create(path)
    return path


@pytest.fixture
def bucket(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    for key, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                       "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(key, value)
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bucket")
        yield "s3://bucket/stores"


def install(local, tmp_path, sid):
    raw = fx.sd_elements("a", 6, sid)
    elements = {"images": {f"{sid}_image": raw["images"]["image-a"]},
                "shapes": {f"{sid}_cells": raw["shapes"]["cells-a"]},
                "points": {f"{sid}_points": raw["points"]["points-a"]}}
    table = fx.sample_table(tmp_path / sid / f"{sid}.zarr", sid, GENES, 6, 0)
    stores.install_sample(local, Prepared(sid, elements, table, pd.DataFrame(), pd.DataFrame(), "fp", "region"))
    stores.consolidate(local)
    return local / "tables" / f"{sid}_table"


def add(local, sid, path):
    note = stores.add_joint_rows(local, sid, path, stores.staging(local), 2)
    stores.consolidate(local)
    return note


def joint(local):
    return ad.read_zarr(stores.joint_path(local))


def test_installed_sample_reads_back_as_spatialdata(tmp_path, local):
    import spatialdata as sd

    install(local, tmp_path, "S1")
    sdata = sd.read_zarr(local)
    assert set(sdata.tables) == {"S1_table"} and set(sdata.shapes) == {"S1_cells"}
    assert sdata.tables["S1_table"].n_obs == 6 and stores.verify_tables(local) == 1


def test_a_store_this_capsule_did_not_create_is_refused(tmp_path, local):
    other = fx.write_sdata(tmp_path / "other.zarr", {"a": {"n": 4, "seed": 0}})
    assert stores.describe(local) == "ours" and stores.describe(other) == "spatialdata"
    with pytest.raises(stores.StoreError, match="put it in List of paths to samples and give a new Storage Location"):
        stores.check_usable(other, may_create=True)


def test_joint_table_is_the_union_of_genes_with_the_source_values(tmp_path, local):
    a = fx.sample_table(tmp_path / "a.zarr", "A", GENES[:20], 8, 0)
    b = fx.sample_table(tmp_path / "b.zarr", "B", GENES[10:] + ["NEW1"], 6, 1)
    add(local, "A", a)
    assert add(local, "B", b).startswith("23 new gene(s) added to the joint table (now 43)")
    t = joint(local)
    assert list(t.var_names) == GENES + ["NEW1"] and t.n_obs == 14
    for sid, path in (("A", a), ("B", b)):
        source = ad.read_zarr(path)
        rows = t[t.obs["sample_id"] == sid]
        assert np.array_equal(rows[:, list(source.var_names)].X.toarray(), source.X.toarray())
        assert rows.X.sum() == source.X.sum()
    assert list(t.var["n_samples_measured"]) == [1] * 10 + [2] * 10 + [1] * 23


def test_joint_table_flags_cells_without_position(tmp_path, local):
    folder = fx.xenium_full(tmp_path / "part")
    cells = pd.read_parquet(folder / "cells.parquet")
    fx.write_parquet(cells.iloc[:12], folder / "cells.parquet")
    cand = fx.candidate(folder)
    sample = Sample(cand.sid, "xenium", cand, inventory=inputs.inspect(cand, tmp_path / "s"))
    prep = prepare.prepare(prepare.load(sample, tmp_path / "s"), sample, None, tmp_path / "s")
    add(local, "part", prep.table_path)
    t = joint(local)
    assert t.n_obs == 30 and t.obs["has_position"].astype(bool).sum() == 12
    assert np.isnan(ad.read_zarr(prep.table_path).obsm["spatial"][12:]).all()


def test_same_sample_twice_is_refused(tmp_path, local):
    a = fx.sample_table(tmp_path / "a.zarr", "A", GENES, 5, 0)
    add(local, "A", a)
    with pytest.raises(stores.JointError, match="already in the joint table"):
        add(local, "A", a)


def test_append_killed_midway_is_repaired_by_the_next_run(tmp_path, local, monkeypatch):
    import annizarr.ops._append as append_ops

    for sid in ("S1", "S2"):
        install(local, tmp_path, sid)
    add(local, "S1", local / "tables" / "S1_table")

    class Killed(BaseException):
        pass

    real = append_ops._append_obs

    def killed(*args, **kwargs):
        real(*args, **kwargs)
        raise Killed()

    monkeypatch.setattr(append_ops, "_append_obs", killed)
    with pytest.raises(Killed):
        stores.add_joint_rows(local, "S2", local / "tables" / "S2_table", stores.staging(local), 2)
    monkeypatch.undo()
    assert stores.sample_counts(local) is None
    assert stores.repair_joint(local, ["S1", "S2"], stores.staging(local), 2)
    assert dict(stores.sample_counts(local)) == {"S1": 6, "S2": 6}


def test_local_lock_refuses_a_second_run_and_is_taken_over_when_its_run_is_gone(local):
    lock = stores.acquire_lock(local)
    with pytest.raises(stores.StoreError, match="Another run is adding to"):
        stores.acquire_lock(local)
    stores.release_lock(lock)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stores.lock_path(local).write_text(json.dumps({"host": socket.gethostname(), "pid": dead.pid, "started": 0}))
    lock = stores.acquire_lock(local)
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    stores.release_lock(lock)


def test_s3_lock_is_exclusive_and_released(bucket):
    url = f"{bucket}/locked.zarr"
    lock = stores.acquire_s3_lock(url, stale_minutes=60)
    with pytest.raises(stores.S3Error, match="Another run is adding to"):
        stores.acquire_s3_lock(url, stale_minutes=60)
    stores.release_s3_lock(lock)
    assert stores.LOCK not in stores.remote_files(stores.client(), *stores.split(url), control=True)


def grow(local, tmp_path, sid):
    add(local, sid, fx.sample_table(tmp_path / f"{sid}.zarr", sid, GENES, 5, len(sid)))


def round_trip(url, tmp_path):
    assert stores.root_meta(url) is None
    first = tmp_path / "first.zarr"
    stores.create(first)
    grow(first, tmp_path, "A")
    assert stores.sync(first, url, 4)["uploaded"] > 0
    copy = tmp_path / "copy.zarr"
    downloaded = stores.download(url, copy, 4)
    assert stores.sync(copy, url, 4, downloaded)["uploaded"] == 0
    grow(copy, tmp_path, "B")
    assert stores.sync(copy, url, 4, downloaded)["uploaded"] > 0
    final = tmp_path / "final.zarr"
    stores.download(url, final, 4)
    t = ad.read_zarr(final / "tables" / "table")
    assert list(t.obs["sample_id"].unique()) == ["A", "B"] and t.n_obs == 10


def test_s3_sync_round_trip_uploads_only_changes(bucket, tmp_path):
    round_trip(f"{bucket}/demo.zarr", tmp_path)


@pytest.mark.skipif(not os.environ.get("SPATIAL_S3_TEST_PREFIX"), reason="SPATIAL_S3_TEST_PREFIX not set")
def test_real_s3_lock_and_round_trip(tmp_path):
    url = f"{os.environ['SPATIAL_S3_TEST_PREFIX'].rstrip('/')}/run-{uuid.uuid4().hex[:12]}/store.zarr"
    bucket, prefix = stores.split(url)
    owned = prefix.rsplit("/", 1)[0]
    client = stores.client()
    try:
        lock = stores.acquire_s3_lock(url)
        with pytest.raises(stores.S3Error, match="Another run is adding to"):
            stores.acquire_s3_lock(url)
        round_trip(url, tmp_path)
        stores.release_s3_lock(lock)
    finally:
        keys = list(stores.remote_files(client, bucket, owned, control=True))
        for start in range(0, len(keys), 1000):
            batch = [{"Key": f"{owned}/{k}"} for k in keys[start:start + 1000]]
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
    assert not stores.remote_files(client, bucket, owned, control=True)
