import json

import pytest
import zarr

from server.validate import inspect_store

from .conftest import write_adata


def report(path, **kw):
    return inspect_store(str(path), **kw)


def test_good_store(tmp_path):
    r = report(write_adata(tmp_path / "s.zarr"))
    assert r["ok"] and not r["errors"]
    assert r["info"]["n_obs"] == 40 and r["info"]["latent"] == "lat"
    assert r["capabilities"] == {"view": True, "label": True, "reembed": True, "latent": True, "genes": True}
    assert r["info"]["categorical"]["cell_type"] == 4


def test_missing_umap_is_an_error(tmp_path):
    r = report(write_adata(tmp_path / "s.zarr", umap=False))
    assert not r["ok"] and any("X_umap" in e for e in r["errors"])


def test_not_a_zarr_group(tmp_path):
    r = report(tmp_path / "nothing")
    assert not r["ok"] and "cannot open" in r["errors"][0]


def test_no_latent_no_x_disables_reembedding(tmp_path):
    r = report(write_adata(tmp_path / "s.zarr", latent=False, X=None))
    assert r["ok"] and not r["capabilities"]["reembed"] and not r["capabilities"]["genes"]
    assert any("Generate New UMAP" in w for w in r["warnings"])


def test_csc_x_is_slow_warning(tmp_path):
    r = report(write_adata(tmp_path / "s.zarr", latent=False, X="csc"))
    assert r["capabilities"]["reembed"] and any("CSC" in w for w in r["warnings"])


def test_obs_column_with_wrong_length_is_an_error(tmp_path):
    path = write_adata(tmp_path / "s.zarr")
    meta = path / "obs" / "cell_type" / "codes" / "zarr.json"
    m = json.loads(meta.read_text())
    m["shape"] = [39]
    meta.write_text(json.dumps(m))
    r = report(path)
    assert not r["ok"] and any("cell_type" in e for e in r["errors"])


def test_duplicate_barcodes_only_found_when_deep(tmp_path):
    path = write_adata(tmp_path / "s.zarr", barcodes=["same"] * 40)
    assert report(path)["ok"]
    deep = report(path, deep=True)
    assert not deep["ok"] and any("duplicate" in e for e in deep["errors"])


def test_selection_larger_than_the_body_limit_is_an_error(tmp_path):
    r = report(write_adata(tmp_path / "s.zarr"), max_body=100)
    assert not r["ok"] and any("body limit" in e for e in r["errors"])


@pytest.mark.parametrize("n", [1_000, 99_999, 1_000_000, 2_720_000, 12_000_000])
def test_worst_case_request_estimate_covers_the_real_payload(n):
    """Select-all must fit the body limit (the 2026-11-11 413)."""
    from server.config import MAX_BODY_BYTES

    real = len(json.dumps({"group": "", "n": n, "indices": list(range(n)), "name": "x" * 60, "params": {}}))
    estimate = n * (len(str(n)) + 2) + 1024
    assert estimate >= real and real < MAX_BODY_BYTES


def test_cli_exit_codes(tmp_path, capsys):
    from server.validate import main

    assert main([str(write_adata(tmp_path / "ok.zarr"))]) == 0
    assert main([str(write_adata(tmp_path / "bad.zarr", umap=False))]) == 1
    assert main([]) == 2
