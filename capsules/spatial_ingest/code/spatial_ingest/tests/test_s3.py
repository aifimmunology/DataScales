"""s3: a store copied from a prefix, grown locally and synced back (moto), plus one real-bucket round
trip that runs only when SPATIAL_S3_TEST_PREFIX is set and deletes only its own unique sub-prefix."""
import os
import uuid

import pytest

import s3
import store as stores
import fixtures_sdata as F


def grow(local, tmp_path, sid):
    """Adds one sample's table to the local store and commits it (what a run does between download and sync)."""
    import joint

    table = F.sample_table(tmp_path / f"{sid}.zarr", sid, F.GENES, 5, len(sid))
    joint.add(local, sid, table, stores.staging(local), 2)
    stores.consolidate(local)


def round_trip(url, tmp_path):
    """Create -> sync -> download -> grow -> sync -> download again; returns the last copy's joint table."""
    import anndata as ad

    assert s3.root_meta(url) is None
    first = tmp_path / "first.zarr"
    stores.create(first)
    grow(first, tmp_path, "A")
    synced = s3.sync(first, url, 4)
    assert synced["uploaded"] > 0 and synced["deleted"] == 0
    assert stores.kind_of(s3.root_meta(url)) == "ours"
    copy = tmp_path / "copy.zarr"
    downloaded = s3.download(url, copy, 4)
    assert s3.sync(copy, url, 4, downloaded)["uploaded"] == 0              # nothing changed, nothing sent
    grow(copy, tmp_path, "B")
    again = s3.sync(copy, url, 4, downloaded)
    assert again["uploaded"] > 0
    final = tmp_path / "final.zarr"
    s3.download(url, final, 4)
    return ad.read_zarr(final / "tables" / "table")


def test_moto_round_trip(tmp_path):
    moto = pytest.importorskip("moto")
    import boto3

    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bucket")
        t = round_trip("s3://bucket/stores/demo.zarr", tmp_path)
    assert list(t.obs["sample_id"].unique()) == ["A", "B"] and t.n_obs == 10


def test_split_needs_a_bucket():
    assert s3.split("s3://b/p/q/") == ("b", "p/q")
    with pytest.raises(s3.S3Error, match="names no bucket"):
        s3.split("s3:///x")


def test_etag_matches_multipart_form(tmp_path, monkeypatch):
    monkeypatch.setattr(s3, "PART_SIZE", 4)
    f = tmp_path / "f"
    f.write_bytes(b"0123456789")
    assert s3.etag(f).endswith("-3")


@pytest.mark.skipif(not os.environ.get("SPATIAL_S3_TEST_PREFIX"), reason="SPATIAL_S3_TEST_PREFIX not set")
def test_real_s3(tmp_path):
    base = os.environ["SPATIAL_S3_TEST_PREFIX"].rstrip("/")
    url = f"{base}/run-{uuid.uuid4().hex[:12]}/store.zarr"
    bucket, prefix = s3.split(url)
    owned = prefix.rsplit("/", 1)[0] + "/"                                    # this run's unique sub-prefix only
    client = s3.client()
    try:
        lock = s3.acquire_lock(url)
        with pytest.raises(s3.S3Error, match="Another run is adding to"):
            s3.acquire_lock(url)
        t = round_trip(url, tmp_path)
        assert list(t.obs["sample_id"].unique()) == ["A", "B"] and t.n_obs == 10
        changed_meanwhile(url, tmp_path)
        s3.release_lock(lock)
        assert s3.LOCK not in s3.remote_files(client, *s3.split(url), control=True)
    finally:
        keys = list(s3.remote_files(client, bucket, owned.rstrip("/"), control=True))
        for start in range(0, len(keys), 1000):
            batch = [{"Key": f"{owned}{k}"} for k in keys[start:start + 1000]]
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
    assert not s3.remote_files(client, bucket, owned.rstrip("/"), control=True)


# ---------------------------------------------------------------- round 2: lock (K2), changed store, order (K3), H1

@pytest.fixture
def bucket(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bucket")
        yield "s3://bucket/stores"


def changed_meanwhile(url, tmp_path):
    """Two runs download the same store; the second to sync is refused instead of deleting the first's sample."""
    a, b = tmp_path / "ra.zarr", tmp_path / "rb.zarr"
    got_a, got_b = s3.download(url, a, 4), s3.download(url, b, 4)
    grow(a, tmp_path, "RUNA")
    s3.sync(a, url, 4, got_a)
    grow(b, tmp_path, "RUNB")
    with pytest.raises(s3.S3Error, match="changed during this run"):
        s3.sync(b, url, 4, got_b)
    check = tmp_path / "check.zarr"
    s3.download(url, check, 4)
    import joint
    assert "RUNA" in joint.read_schema(check).samples


def test_lock_is_exclusive_stale_taken_over_and_released(bucket, caplog):
    import json
    import logging

    import boto3

    url = f"{bucket}/locked.zarr"
    lock = s3.acquire_lock(url, stale_minutes=60)
    with pytest.raises(s3.S3Error, match="Another run is adding to .*last heartbeat 0 min ago"):
        s3.acquire_lock(url, stale_minutes=60)
    s3.release_lock(lock)
    client = boto3.client("s3", region_name="us-east-1")
    assert "stores/locked.zarr/.spatial_ingest_lock" not in [o["Key"] for o in
                                                            client.list_objects_v2(Bucket="bucket").get("Contents", [])]
    client.put_object(Bucket="bucket", Key="stores/locked.zarr/.spatial_ingest_lock",
                      Body=json.dumps({"host": "gone", "pid": 1, "started": 0, "heartbeat": 0}).encode())
    with caplog.at_level(logging.WARNING, logger="s3"):
        lock = s3.acquire_lock(url, stale_minutes=60)
    assert "Taking over the lock" in caplog.text
    s3.release_lock(lock)


def test_lock_of_a_killed_run_on_this_host_is_taken_over_at_once(bucket, caplog):
    """A run killed mid-sync and retried in the same container: its lock's heartbeat is fresh, but its pid is gone."""
    import json
    import logging
    import socket
    import subprocess
    import time

    import boto3

    url = f"{bucket}/killed.zarr"
    gone = subprocess.Popen(["true"])
    gone.wait()
    now = time.time()
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket="bucket", Key="stores/killed.zarr/.spatial_ingest_lock",
        Body=json.dumps({"host": socket.gethostname(), "pid": gone.pid, "started": now, "heartbeat": now}).encode())
    with caplog.at_level(logging.WARNING, logger="s3"):
        lock = s3.acquire_lock(url, stale_minutes=60)
    assert "no longer running on this host" in caplog.text
    s3.release_lock(lock)


def test_heartbeat_refreshes_and_a_lost_lock_stops_the_sync(bucket, tmp_path):
    import json
    import time

    import boto3

    url = f"{bucket}/hb.zarr"
    lock = s3.acquire_lock(url, stale_minutes=0.05)                          # heartbeat every 5 s
    client = boto3.client("s3", region_name="us-east-1")
    key = "stores/hb.zarr/.spatial_ingest_lock"
    first = json.loads(client.get_object(Bucket="bucket", Key=key)["Body"].read())["heartbeat"]
    deadline = time.time() + 20
    while time.time() < deadline:
        if json.loads(client.get_object(Bucket="bucket", Key=key)["Body"].read())["heartbeat"] > first:
            break
        time.sleep(0.5)
    else:
        pytest.fail("no heartbeat")
    client.put_object(Bucket="bucket", Key=key, Body=b"{}")                  # another run took it over
    deadline = time.time() + 20
    while not lock.lost and time.time() < deadline:
        time.sleep(0.5)
    assert lock.lost
    local = tmp_path / "s.zarr"
    stores.create(local)
    with pytest.raises(s3.S3Error, match="lost its lock"):
        s3.sync(local, url, 2)
    s3.release_lock(lock)


def test_two_runs_on_one_store_do_not_lose_a_sample(bucket, tmp_path):
    """Critic K2 (operations #1): the second sync used to delete the first run's sample."""
    url = f"{bucket}/conc.zarr"
    first = tmp_path / "first.zarr"
    stores.create(first)
    grow(first, tmp_path, "BASE")
    s3.sync(first, url, 4)
    changed_meanwhile(url, tmp_path)


def build(tmp_path, url):
    """A remote store with one sample, and a local copy grown by a second sample with new genes and
    new obs columns (var, obs, indptr and the record all grow): what an append sync uploads."""
    from test_joint import install
    import joint

    first = tmp_path / "base.zarr"
    stores.create(first)
    path = install(first, tmp_path, "S1")
    joint.add(first, "S1", path, stores.staging(first), 2)
    stores.consolidate(first)
    s3.sync(first, url, 4)
    copy = tmp_path / "copy.zarr"
    downloaded = s3.download(url, copy, 4)
    from test_store import prepared
    p = prepared(tmp_path / "S2", "S2", extra_obs={"batch": ["x"] * 6})
    import fixtures_sdata
    fixtures_sdata.sample_table(p.table_path, "S2", fixtures_sdata.GENES + ["NEWGENE"], 6, 5,
                                extra_obs={"batch": ["x"] * 6})
    stores.install_sample(copy, p)
    stores.consolidate(copy)
    joint.add(copy, "S2", copy / "tables" / "S2_table", stores.staging(copy), 2)
    stores.consolidate(copy)
    return copy, downloaded


@pytest.mark.parametrize("phase", ["new data", "changed data", "metadata", "root", "deletes"])
def test_killed_append_sync_leaves_a_readable_store_and_the_next_run_repairs(bucket, tmp_path, monkeypatch, phase):
    """Critic K3 (operations #2): a kill mid-sync left the remote unreadable and the repair wedged."""
    import anndata as ad
    import spatialdata as sd

    import joint

    url = f"{bucket}/k3.zarr"
    copy, downloaded = build(tmp_path, url)

    class Killed(BaseException):
        pass

    def kill(name):
        if name == phase:
            raise Killed()

    monkeypatch.setattr(s3, "_phase", kill)
    with pytest.raises(Killed):
        s3.sync(copy, url, 4, downloaded)
    monkeypatch.undo()
    after = tmp_path / "after.zarr"
    got = s3.download(url, after, 4)
    sdata = sd.read_zarr(after)                                             # readable whatever the phase
    t = sdata.tables["table"]
    # the joint table flips with its own zarr.json (uploaded last of its files, just before the root): after the
    # metadata phase it already holds S2's rows while the root doesn't list S2 yet (a window of two uploads)
    expected_samples = {"S1"} if phase in ("new data", "changed data") else {"S1", "S2"}
    assert set(t.obs["sample_id"].astype(str)) == expected_samples
    assert t.X.shape == (6 * len(expected_samples), len(t.var_names))
    s1 = ad.read_zarr(after / "tables" / "S1_table")
    rows = t[t.obs["sample_id"] == "S1"]
    assert (rows[:, list(s1.var_names)].X != s1.X).nnz == 0
    # the next run: repair what the kill left, sync, and the store is complete and consistent
    stores.remove_uncommitted(after)
    present = sorted(n.removesuffix("_table") for n in stores.committed(after)["tables"] if n != "table")
    joint.repair(after, present, stores.staging(after), 2)
    stores.consolidate(after)
    s3.sync(after, url, 4, got)
    final = tmp_path / "final.zarr"
    s3.download(url, final, 4)
    assert set(sd.read_zarr(final).tables["table"].obs["sample_id"].astype(str)) == set(present)
    assert stores.verify_tables(final) >= 2


def test_killed_first_sync_is_cleared_and_created_again(bucket, tmp_path, monkeypatch, caplog):
    """Critic H1 (operations #3): a prefix left by a killed first sync was refused forever."""
    import logging

    url = f"{bucket}/new.zarr"
    first = tmp_path / "first.zarr"
    stores.create(first)
    grow(first, tmp_path, "A")

    class Killed(BaseException):
        pass

    monkeypatch.setattr(s3, "_phase", lambda name: (_ for _ in ()).throw(Killed()) if name == "changed data" else None)
    with pytest.raises(Killed):
        s3.sync(first, url, 4)
    monkeypatch.undo()
    with caplog.at_level(logging.WARNING, logger="s3"):
        assert s3.root_meta(url) is None
    assert "creation was killed" in caplog.text
    s3.sync(first, url, 4)
    assert stores.kind_of(s3.root_meta(url)) == "ours"


def test_unmarked_partial_prefix_is_still_refused(bucket):
    import boto3

    boto3.client("s3", region_name="us-east-1").put_object(Bucket="bucket", Key="stores/x.zarr/notes.txt", Body=b"x")
    assert s3.root_meta(f"{bucket}/x.zarr") == {}
