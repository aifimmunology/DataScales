import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import embedding, gpu, store
from server.main import app
from server.store import LocalStore

from .conftest import write_adata, write_view


@pytest.fixture
def client(root, monkeypatch):
    embedding.describe.cache_clear()
    gpu.JOBS.clear()
    monkeypatch.setattr(gpu, "_ensure_worker", lambda: None)
    while not gpu.JOB_QUEUE.empty():
        gpu.JOB_QUEUE.get()
    return TestClient(app)


def submit(client, **over):
    body = {"group": "", "n": 40, "indices": list(range(10)), "name": "sub", "params": {"n_neighbors": 5}, **over}
    return client.post("/api/submit", json=body)


def test_config_reports_latent(client):
    cfg = client.get("/api/config").json()
    assert cfg["embedding"]["latent"] == "lat" and "error" not in cfg


def test_config_survives_an_unreadable_store(client, monkeypatch):
    embedding.describe.cache_clear()
    monkeypatch.setattr(store, "STORE", LocalStore("/nonexistent/store.zarr"))
    res = client.get("/api/config")
    assert res.status_code == 200 and "unreadable" in res.json()["error"]


def test_validate_endpoint(client):
    assert client.get("/api/validate").json()["ok"] is True


def test_views_lists_only_complete_views(root, client):
    write_view(root, "done_1", [1, 2, 3], label="Done")
    (root / "umap_views" / "partial_2").mkdir()
    ids = [v["id"] for v in client.get("/api/views").json()]
    assert ids == ["main", "done_1"]


def test_views_surface_store_errors_instead_of_an_empty_list(root, client, monkeypatch):
    write_view(root, "done_1", [1, 2, 3])

    def boom(rel):
        raise HTTPException(503, "credentials expired")

    monkeypatch.setattr(store.STORE, "read_attrs", boom)
    res = client.get("/api/views")
    assert res.status_code == 503 and "credentials" in res.json()["detail"]


def test_submit_queues_root_rows(client):
    res = submit(client)
    assert res.status_code == 200
    _, rows = gpu.JOB_QUEUE.get_nowait()
    assert rows == list(range(10))


def test_submit_from_a_view_maps_to_root_rows(root, client):
    write_view(root, "v_1", [30, 31, 32, 33, 34, 35, 36])
    assert submit(client, group="umap_views/v_1", n=7, indices=[0, 1, 2, 3, 4, 5], params={"n_neighbors": 3}).status_code == 200
    assert gpu.JOB_QUEUE.get_nowait()[1] == [30, 31, 32, 33, 34, 35]


@pytest.mark.parametrize("over,code,fragment", [
    ({"indices": []}, 400, "empty"),
    ({"n": 41}, 409, "reload"),
    ({"indices": [0, 99]}, 400, "out of range"),
    ({"group": "umap_views/ghost", "n": 3}, 404, "unknown view"),
    ({"indices": list(range(5)), "params": {"n_neighbors": 5}}, 400, "smaller than the selection"),
    ({"params": {"n_neighbors": 1}}, 400, "at least 2"),
    ({"params": {"batch_key": "nope", "n_neighbors": 5}}, 400, "categorical obs column"),
    ({"params": {"batch_key": "score", "n_neighbors": 5}}, 400, "categorical obs column"),
    ({"params": {"method": "bogus"}}, 400, "method"),
])
def test_submit_rejects_bad_requests_up_front(client, over, code, fragment):
    res = submit(client, **over)
    assert res.status_code == code and fragment in res.json()["detail"]
    assert gpu.JOB_QUEUE.empty() and not gpu.JOBS


def test_submit_accepts_a_categorical_batch_key(client):
    assert submit(client, params={"batch_key": "batch", "n_neighbors": 5}).status_code == 200


def test_failed_job_leaves_no_view_even_if_the_mirror_delete_fails(root, monkeypatch):
    write_view(root, "partial_1", [1, 2, 3])

    class BrokenMirror:
        uri = "gs://b/p"

        def delete_prefix(self, rel):
            raise RuntimeError("no creds")

    monkeypatch.setattr(store, "MIRROR", BrokenMirror())
    gpu._discard("umap_views/partial_1")
    assert not (root / "umap_views" / "partial_1").exists()


def test_job_failing_to_mirror_is_discarded(root, monkeypatch):
    write_view(root, "v_9", [1, 2, 3])
    calls = []

    class Mirror:
        uri = "gs://b/p"

        def copy(self, rel):
            raise store.MirrorError("bucket down")

        def delete_prefix(self, rel):
            calls.append(rel)

    monkeypatch.setattr(store, "MIRROR", Mirror())
    monkeypatch.setattr(gpu, "_send", lambda req: None)
    monkeypatch.setattr(gpu, "_follow", lambda proc, job: None)
    with pytest.raises(RuntimeError, match="not mirrored"):
        gpu._run_gpu_job({"id": "j", "slug": "v_9", "name": "n", "params": {}}, [1, 2])
    assert not (root / "umap_views" / "v_9").exists() and calls == ["umap_views/v_9"]


def test_delete_view_removes_local_and_mirror(root, client, monkeypatch):
    write_view(root, "v_1", [1, 2, 3])
    gone = []

    class Mirror:
        uri = "gs://b/p"

        def delete_prefix(self, rel):
            gone.append(rel)
            return 1

    monkeypatch.setattr(store, "MIRROR", Mirror())
    assert client.delete("/api/views/v_1").json()["deleted"] == "v_1"
    assert gone == ["umap_views/v_1"] and not (root / "umap_views" / "v_1").exists()
    assert client.delete("/api/views/v_1").status_code == 404


def test_data_proxy_serves_and_404s(client):
    assert client.get("/api/data/obsm/X_umap/zarr.json").status_code == 200
    assert client.get("/api/data/obsm/X_umap/c/99/99").status_code == 404
    assert client.get("/api/data/../etc/passwd").status_code in (404, 400)
