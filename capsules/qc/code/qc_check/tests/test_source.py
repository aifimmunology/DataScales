import anndata as ad
import pytest
import zarr
from anndata.io import write_elem
from annizarr import Repo

import source
from conftest import build_store, cohort, sample_adata


def test_opens_both_kinds_at_main(store, kind):
    src = source.open_source(store)
    assert src.kind == kind
    assert int(src.root["X"].attrs["shape"][0]) == 720
    if kind == "icechunk":
        assert src.fingerprint == Repo(store).log()[0].id
    else:
        assert len(src.fingerprint) == 64


def test_reads_only_the_listed_columns(store, monkeypatch):
    src = source.open_source(store)
    seen = []
    real = source.read_elem
    monkeypatch.setattr(source, "read_elem", lambda elem: seen.append(elem.basename) or real(elem))
    obs = src.read_obs(["sample_id", "pct_counts_mt"], categorical=("sample_id",))
    assert seen == ["sample_id", "pct_counts_mt"]
    assert list(obs.columns) == ["sample_id", "pct_counts_mt"] and obs["sample_id"].dtype == "category"


def test_zarr_fingerprint_is_stable_and_tracks_changes(tmp_path):
    path = build_store(tmp_path / "s.zarr", cohort(2), "zarr")
    first = source.zarr_fingerprint(zarr.open_group(path, mode="r"))
    assert first == source.zarr_fingerprint(zarr.open_group(path, mode="r"))

    from annizarr import append, convert
    from conftest import cfg
    convert(sample_adata("SP0009", seed=9), output=str(tmp_path / "new.zarr"), cfg=cfg())
    append(path, cells=str(tmp_path / "new.zarr"), cfg=cfg())
    grown = source.zarr_fingerprint(zarr.open_group(path, mode="r"))
    assert grown != first

    root = zarr.open_group(path, mode="r+")
    obs = ad.io.read_elem(root["obs"])
    obs["extra"] = 1.0
    write_elem(root, "obs", obs)
    assert source.zarr_fingerprint(zarr.open_group(path, mode="r")) != grown


def test_refuses_dense_x_and_missing_store(tmp_path):
    adata = sample_adata("S1")
    adata.X = adata.X.toarray()
    adata.write_zarr(tmp_path / "dense.zarr")
    with pytest.raises(source.SourceError, match="X must be CSR"):
        source.open_source(str(tmp_path / "dense.zarr"))
    with pytest.raises(source.SourceError, match="No icechunk repo or AnnData zarr"):
        source.open_source(str(tmp_path / "nothing.zarr"))
    with pytest.raises(source.SourceError, match="No input given"):
        source.open_source("")


def test_origin_is_icechunk_only(tmp_path):
    path = build_store(tmp_path / "s.zarr", cohort(1), "zarr")
    with pytest.raises(source.SourceError, match="icechunk inputs only"):
        source.open_source(path, origin="s3://bucket/x")


def test_curation_is_none_without_a_branch(tmp_path):
    src = source.open_source(build_store(tmp_path / "ic", cohort(2), "icechunk"))
    assert src.curation("qc") is None


def test_plain_zarr_on_s3_is_refused():
    with pytest.raises(source.SourceError, match="only icechunk repos can be on S3"):
        source._open_zarr("s3://bucket/store.zarr", "", "")
