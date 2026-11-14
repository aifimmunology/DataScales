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
        t = round_trip(url, tmp_path)
        assert list(t.obs["sample_id"].unique()) == ["A", "B"] and t.n_obs == 10
    finally:
        keys = list(s3.remote_files(client, bucket, owned.rstrip("/")))
        for start in range(0, len(keys), 1000):
            batch = [{"Key": f"{owned}{k}"} for k in keys[start:start + 1000]]
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
    assert not s3.remote_files(client, bucket, owned.rstrip("/"))
