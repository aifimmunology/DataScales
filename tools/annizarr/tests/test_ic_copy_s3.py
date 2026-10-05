"""copy_repo through a real S3 endpoint (moto), all three directions.

Uses moto's ThreadedMotoServer (a real HTTP server, not response-mocking) because
icechunk's Rust core makes its own HTTP calls outside Python's request machinery.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest
import zarr

pytest.importorskip("icechunk")
boto3 = pytest.importorskip("boto3")
pytest.importorskip("moto")

from moto.server import ThreadedMotoServer  # noqa: E402

from annizarr.ic import Repo, copy_repo  # noqa: E402


@pytest.fixture
def moto_s3(monkeypatch: pytest.MonkeyPatch):
    """A live moto S3 server + client, with two fresh, uniquely-named buckets.

    moto's backend state is a process-global singleton that outlives any one
    ThreadedMotoServer instance, so bucket names are randomised per test — reusing a
    fixed name would silently resurrect another test's objects.
    """
    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)  # boto3 reads this directly

    client = boto3.client("s3")
    bucket_a, bucket_b = f"bucket-a-{uuid.uuid4().hex[:10]}", f"bucket-b-{uuid.uuid4().hex[:10]}"
    for bucket in (bucket_a, bucket_b):
        client.create_bucket(Bucket=bucket)
    try:
        yield client, endpoint, bucket_a, bucket_b
    finally:
        server.stop()


def _make_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src.zarr"
    root = zarr.open_group(str(src), mode="w")
    x = root.create_array("X", shape=(10, 4), dtype="float32", chunks=(5, 4))
    x[...] = np.arange(40, dtype="float32").reshape(10, 4)
    repo_path = tmp_path / "repo.icechunk"
    Repo.init(src, repo_path)
    return repo_path


def _local_objects(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _s3_objects(client: object, bucket: str, prefix: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            out[key[len(prefix) + 1 :]] = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return out


def _open_on_moto(bucket: str, prefix: str, endpoint: str):
    # icechunk's own storage constructor, pointed at moto's real HTTP endpoint;
    # `storage_for`/`from_env=True` has no endpoint override, so this bypasses it
    import icechunk

    storage = icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint,
        allow_http=True,
        region="us-east-1",
        access_key_id="testing",
        secret_access_key="testing",
        force_path_style=True,
    )
    return icechunk.Repository.open(storage)


def test_copy_repo_round_trip_local_to_s3_to_s3_to_local(moto_s3, tmp_path: Path) -> None:
    """copy_repo through a real S3 endpoint (moto), chained through all three directions:
    local -> s3, s3 -> s3 (a different bucket), then s3 -> local -- each hop's objects
    (and its 'main' branch, once reopened directly against moto) must match the original."""
    client, endpoint, bucket_a, bucket_b = moto_s3
    repo_path = _make_repo(tmp_path)
    local = _local_objects(repo_path)

    copy_repo(str(repo_path), f"s3://{bucket_a}/p1")
    assert _s3_objects(client, bucket_a, "p1") == local
    assert _open_on_moto(bucket_a, "p1", endpoint).list_branches() == {"main"}

    copy_repo(f"s3://{bucket_a}/p1", f"s3://{bucket_b}/p2")
    assert _s3_objects(client, bucket_b, "p2") == local
    assert _open_on_moto(bucket_b, "p2", endpoint).list_branches() == {"main"}

    dest = tmp_path / "from_s3.icechunk"
    copy_repo(f"s3://{bucket_b}/p2", str(dest))
    assert _local_objects(dest) == local
    assert Repo(dest).branches() == ["main"]
