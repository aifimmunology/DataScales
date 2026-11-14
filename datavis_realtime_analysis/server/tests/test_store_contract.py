import asyncio
import threading

import pytest
import zarr
from fastapi import HTTPException

from server.config import Location
from server.store import GcsStore, LocalStore, Mirror, MirrorError

from .conftest import request
from .fakes import FakeGcs

FILES = {"g/zarr.json": b'{"attributes": {"k": 1}}', "g/data/c/0": b"0123456789", "g/sub/zarr.json": b"{}",
         "g/other/zarr.json": b"{}"}


@pytest.fixture(params=["local", "gcs"])
def store(request, tmp_path):
    if request.param == "local":
        for rel, data in FILES.items():
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(data)
        return LocalStore(tmp_path)
    client = FakeGcs()
    client.bucket("b").objects.update({f"pre/{k}": v for k, v in FILES.items()})
    return GcsStore(Location("gs://b/pre"), client)


def serve_bytes(store, path, **headers) -> tuple[int, bytes]:
    req = request(**headers)
    resp = store.serve(path, req)
    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.disconnect"}

    asyncio.run(resp(req.scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent)


def test_serve_whole_object(store):
    assert serve_bytes(store, "g/data/c/0") == (200, b"0123456789")


def test_serve_range(store):
    assert serve_bytes(store, "g/data/c/0", range="bytes=2-4") == (206, b"234")


def test_serve_missing_is_404(store):
    with pytest.raises(HTTPException) as e:
        store.serve("g/data/c/9", request())
    assert e.value.status_code == 404


def test_list_children(store):
    assert store.list_children("g") == ["data", "other", "sub"]
    assert store.list_children("nope") == []


def test_read_attrs(store):
    assert store.read_attrs("g") == {"k": 1}
    assert store.read_attrs("g/missing") is None


def test_delete_prefix(store):
    assert store.delete_prefix("g/sub") == 1
    assert store.list_children("g") == ["data", "other"]
    assert store.delete_prefix("g/sub") == 0


def test_local_rejects_traversal(tmp_path):
    (tmp_path / "secret").write_text("x")
    s = LocalStore(tmp_path / "sub")
    (tmp_path / "sub").mkdir()
    for bad in ("../secret", "a/../../secret", ".hidden/x"):
        with pytest.raises(HTTPException):
            s.serve(bad, request())


def test_local_etag_revalidates(tmp_path):
    (tmp_path / "f.json").write_text("{}")
    s = LocalStore(tmp_path)
    etag = s.serve("f.json", request()).headers["etag"]
    assert s.serve("f.json", request(if_none_match=etag)).status_code == 304


def _fill(value):
    def write(g):
        g.update_attributes({"v": value})
        g.create_array("a", shape=(3,), dtype="int8", fill_value=-1)[:] = [value] * 3
    return write


def test_replace_group_swaps_content(tmp_path):
    s = LocalStore(tmp_path)
    s.replace_group("obs/col", _fill(1))
    s.replace_group("obs/col", _fill(2))
    g = zarr.open_group(str(tmp_path / "obs/col"), mode="r")
    assert g.attrs["v"] == 2 and list(g["a"][:]) == [2, 2, 2]
    assert not list((tmp_path / ".datavis_staging").iterdir())


def test_replace_group_never_exposes_a_missing_column(tmp_path):
    s = LocalStore(tmp_path)
    s.replace_group("obs/col", _fill(0))
    stop, misses = threading.Event(), []

    def reader():
        while not stop.is_set():
            if not (tmp_path / "obs/col/zarr.json").is_file():
                misses.append(1)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(30):
        s.replace_group("obs/col", _fill(i))
    stop.set()
    t.join()
    assert not misses


def test_mirror_copies_and_prunes(tmp_path):
    local = LocalStore(tmp_path)
    local.replace_group("obs/col", _fill(1))
    client = FakeGcs()
    m = Mirror(Location("gs://b/pre"), local, client)
    m.copy("obs/col")
    objs = client.bucket("b").objects
    assert "pre/obs/col/zarr.json" in objs and "pre/obs/col/a/c/0" in objs
    objs["pre/obs/col/stale"] = b"x"
    m.replace("obs/col")
    assert "pre/obs/col/stale" not in objs and "pre/obs/col/zarr.json" in objs
    assert m.delete_prefix("obs/col") >= 2 and not [k for k in objs if "obs/col" in k]


def test_mirror_failure_is_reported(tmp_path):
    local = LocalStore(tmp_path)
    local.replace_group("v", _fill(1))

    class Broken(FakeGcs):
        def bucket(self, name):
            raise RuntimeError("no creds")

    with pytest.raises(MirrorError):
        Mirror(Location("gs://b/pre"), local, Broken()).copy("v")
