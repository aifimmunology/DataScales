"""outputs.py: which destination a run gets (every rule, no GPU) and each writer, locally and over memory://."""
from pathlib import Path

import fsspec
import pytest
import zarr
from annizarr import Repo

import outputs
from config import Config, ConfigError, load_config
from outputs import MARKER, OutputError
from store import open_source
from test_store import check, results


def cfg(input, tmp_path, **kw) -> Config:
    return Config(input=str(input), results_dir=str(tmp_path / "res"), **kw).validate()


def run(c: Config, adata, leiden=("leiden",)):
    dest = outputs.plan(c)
    src = open_source(c.input, c.table, c.branch, dest.location if dest.mode == "commit" else "")
    outputs.check(dest, src)
    return outputs.save(dest, src, results(adata, leiden).write, "m", c.branch, {"method": "pca"}), dest


def zarr_at(written: str) -> zarr.Group:
    return zarr.open_group(written.removeprefix("saved to "), mode="r")


def memory(name: str) -> str:
    fs = fsspec.filesystem("memory")
    if fs.exists(f"/{name}"):
        fs.rm(f"/{name}", recursive=True)
    return f"memory://{name}"


# --- plan: every rule ------------------------------------------------------------------------------------------

def test_results_copy_follows_the_input_unless_a_format_is_picked(stores, icechunk, tmp_path):
    res = f"{tmp_path}/res"
    assert outputs.plan(cfg(stores["pbmc"], tmp_path)) == outputs.Destination("copy", "zarr", f"{res}/pbmc.processed")
    assert outputs.plan(cfg(icechunk["mount"], tmp_path)).kind == "icechunk"
    assert outputs.plan(cfg(icechunk["mount"], tmp_path, results_format="zarr")).kind == "zarr"
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, results_copy="mine.zarr")).location == f"{res}/mine"
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, results_copy=f"{tmp_path}/abs")).location == f"{tmp_path}/abs"


def test_results_copy_settings_with_an_external_location_are_an_error(tmp_path):
    for extra in ({"results_copy": "x"}, {"results_format": "zarr"}):
        with pytest.raises(ConfigError, match="leave them empty"):
            Config(external="s3://b/p", **extra).validate()
    with pytest.raises(ConfigError, match="external_format must be one of"):
        load_config(["--external_format", "h5ad"], env={})
    assert load_config(["--results_format", "IceChunk"], env={}).results_format == "icechunk"


def test_zarr_input_external_new_existing_and_mismatch(stores, icechunk, tmp_path):
    new = tmp_path / "new.zarr"
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(new))) == outputs.Destination("new", "zarr", str(new))
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(new), external_format="icechunk")).kind == "icechunk"
    (tmp_path / "empty").mkdir()
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "empty"))).mode == "new"
    existing = outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(stores["ingest_no_sample"])))
    assert (existing.mode, existing.kind) == ("in_place", "zarr")
    with pytest.raises(OutputError, match="is an existing zarr, but external_format is icechunk"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(stores["ingest_no_sample"]), external_format="icechunk"))
    with pytest.raises(OutputError, match="only an icechunk input of that same repo"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(icechunk["origin"])))
    with pytest.raises(OutputError, match="is an existing icechunk, but external_format is zarr"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(icechunk["origin"]), external_format="zarr"))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "notes.txt").write_text("x")
    with pytest.raises(OutputError, match="isn't a zarr or an icechunk repo"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "junk")))


def test_icechunk_input_only_commits_onto_its_own_repo(stores, icechunk, tmp_path):
    mount, origin = icechunk["mount"], icechunk["origin"]
    assert outputs.plan(cfg(mount, tmp_path, external=str(origin))) == outputs.Destination("commit", "icechunk", str(origin))
    for location, message in [(tmp_path / "typo", "nothing there"), (stores["pbmc"], "a zarr there")]:
        with pytest.raises(OutputError, match=f"{message}: check .* for a typo"):
            outputs.plan(cfg(mount, tmp_path, external=str(location)))
    other = Repo.init(stores["ingest_no_sample"], tmp_path / "other_repo")
    with pytest.raises(OutputError, match="doesn't hold the input's snapshot"):
        outputs.plan(cfg(mount, tmp_path, external=other.path))
    with pytest.raises(OutputError, match="can't use branch dev"):
        outputs.plan(cfg(mount, tmp_path, external=str(origin), branch="dev"))


def test_half_written_marker(stores, tmp_path):
    out = tmp_path / "out.zarr"
    zarr.open_group(out, mode="w")
    (out / MARKER).write_text("new")
    with pytest.raises(OutputError, match="half-written"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(out)))
    (out / MARKER).write_text("results")
    assert outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(out))).pending == "results"


def test_spatialdata_never_becomes_icechunk(stores, tmp_path):
    for c in (cfg(stores["tissue"], tmp_path, results_format="icechunk"),
              cfg(stores["tissue"], tmp_path, external=str(tmp_path / "new"), external_format="icechunk")):
        dest = outputs.plan(c)
        with pytest.raises(OutputError, match="SpatialData zarr can't be stored as icechunk"):
            outputs.check(dest, open_source(c.input, c.table, c.branch, ""))


def test_in_place_needs_the_same_cells(stores, adata, tmp_path):
    import annizarr

    other = tmp_path / "other.zarr"
    annizarr.convert(adata[:100].copy(), output=str(other))
    c = cfg(stores["pbmc"], tmp_path, external=str(other))
    with pytest.raises(OutputError, match="doesn't have the input's cells and genes"):
        outputs.check(outputs.plan(c), open_source(c.input, c.table, c.branch, ""))
    c = cfg(stores["pbmc"], tmp_path, external=str(stores["tissue"]))
    with pytest.raises(OutputError, match="isn't an AnnData store like the input"):
        outputs.check(outputs.plan(c), open_source(c.input, c.table, c.branch, ""))


# --- save: each writer -----------------------------------------------------------------------------------------

def test_zarr_copy_never_reuses_a_name(stores, adata, tmp_path):
    first, _ = run(cfg(stores["pbmc"], tmp_path), adata)
    second, _ = run(cfg(stores["pbmc"], tmp_path), adata)
    assert first == f"saved to {tmp_path}/res/pbmc.processed.zarr" and second.endswith("pbmc.processed-1.zarr")
    out = check(zarr_at(first), adata)
    assert list(out.obs.columns[:3]) == ["sample_id", "chip_id", "cell_type"]
    assert not list((tmp_path / "res").glob("*.partial"))


def test_ingest_store_without_layers_named_copy(stores, adata, tmp_path):
    written, _ = run(cfg(stores["ingest_no_sample"], tmp_path, results_copy="out"), adata)
    assert written == f"saved to {tmp_path}/res/out.zarr"
    check(zarr_at(written), adata)


def test_spatialdata_copy_keeps_every_element(stores, adata, tmp_path):
    import spatialdata as sd

    written, _ = run(cfg(stores["tissue"], tmp_path), adata, ("leiden_0.5", "leiden_1.0"))
    out = Path(written.removeprefix("saved to "))
    sdata = sd.read_zarr(out)
    assert set(sdata.images) == {"image"} and set(sdata.shapes) == {"cells"}
    check(zarr.open_group(out, mode="r")["tables/table"], adata, leiden=("leiden_0.5", "leiden_1.0"))
    for f in (stores["tissue"] / "images").rglob("*"):
        if f.is_file():
            assert f.read_bytes() == (out / f.relative_to(stores["tissue"])).read_bytes()


def test_rerun_drops_stale_results(stores, adata, tmp_path):
    first, _ = run(cfg(stores["pbmc"], tmp_path), adata, ("leiden_0.5", "leiden_2.0"))
    again, _ = run(cfg(first.removeprefix("saved to "), tmp_path), adata)
    check(zarr_at(again), adata)


def test_icechunk_copy_keeps_history_and_never_touches_the_mount(icechunk, adata, tmp_path):
    mount = icechunk["mount"]
    before = sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*"))
    n = len(Repo(str(mount)).log())
    outs = [run(cfg(mount, tmp_path), adata)[0].split(" at ")[1] for _ in range(2)]
    assert outs == [f"{tmp_path}/res/mount_icechunk.processed", f"{tmp_path}/res/mount_icechunk.processed-1"]
    for out in outs:
        log = Repo(out).log()
        assert len(log) == n + 1 and log[0].metadata["source_snapshot"] == log[1].id == Repo(str(mount)).log()[0].id
        check(Repo(out).open_zarr("r"), adata)
    assert sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*")) == before


def test_conversions_in_the_copy(stores, icechunk, adata, tmp_path):
    written, _ = run(cfg(icechunk["mount"], tmp_path, results_format="zarr"), adata)
    assert written == f"saved to {tmp_path}/res/mount_icechunk.processed.zarr"
    check(zarr_at(written), adata)
    written, _ = run(cfg(stores["pbmc"], tmp_path, results_format="icechunk", branch="dev"), adata)
    out = written.split(" at ")[1]
    assert out == f"{tmp_path}/res/pbmc.processed" and " on dev " in written
    assert Repo(out, branch="dev").log()[0].metadata["source"] == str(stores["pbmc"])
    check(Repo(out, branch="dev").open_zarr("r"), adata)


def test_commit_through_the_external_repo_twice(icechunk, adata, tmp_path):
    mount, origin = icechunk["mount"], str(icechunk["origin"])
    before = sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*"))
    n = len(Repo(origin).log())
    for _ in range(2):
        written, _ = run(cfg(mount, tmp_path, external=origin), adata)
        assert written.startswith("committed") and written.endswith(f"on main at {origin}")
    log = Repo(origin).log()
    assert len(log) == n + 2 and log[0].metadata["source_snapshot"] == log[1].id and log[0].metadata["method"] == "pca"
    check(Repo(origin).open_zarr("r"), adata)
    assert sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*")) == before
    assert not (tmp_path / "res").exists()


def test_branch_moved_writes_nothing(icechunk, adata, tmp_path):
    origin = str(icechunk["origin"])
    c = cfg(icechunk["mount"], tmp_path, external=origin)
    dest = outputs.plan(c)
    src = open_source(c.input, c.table, c.branch, origin)
    other = Repo(origin, branch="main")
    other.open_zarr("w").attrs["touched"] = "yes"
    tip = other.commit("moved meanwhile")
    with pytest.raises(RuntimeError, match="moved since the store was opened"):
        outputs.save(dest, src, results(adata).write, "m", "main", {})
    assert Repo(origin).log()[0].id == tip


def test_new_external_zarr_local_and_remote(stores, adata, tmp_path):
    local, _ = run(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "ext.zarr")), adata)
    assert local == f"saved to {tmp_path}/ext.zarr" and not (tmp_path / "ext.zarr.partial").exists()
    check(zarr_at(local), adata)
    remote = memory("bucket/ext.zarr")
    written, _ = run(cfg(stores["tissue"], tmp_path, external=remote), adata)
    assert written == f"saved to {remote}"
    assert not fsspec.filesystem("memory").exists(f"/bucket/ext.zarr/{MARKER}")
    check(zarr.open_group(remote, mode="r")["tables/table"], adata)
    assert outputs.detect(remote) == ("zarr", "")


def test_new_external_icechunk_from_a_zarr(stores, adata, tmp_path):
    written, _ = run(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "repo"), external_format="icechunk"), adata)
    assert written.startswith("committed") and written.endswith(f"of a new repo at {tmp_path}/repo")
    assert len(Repo(str(tmp_path / "repo")).log()) == 2
    check(Repo(str(tmp_path / "repo")).open_zarr("r"), adata)


def test_in_place_into_an_existing_zarr_twice(stores, adata, tmp_path):
    first, _ = run(cfg(stores["pbmc"], tmp_path), adata, ("leiden_0.5", "leiden_2.0"))
    target = first.removeprefix("saved to ")
    for leiden in (("leiden",), ("leiden_1.0",)):
        written, dest = run(cfg(stores["pbmc"], tmp_path, external=target), adata, leiden)
        assert dest.mode == "in_place" and written == f"written into {target}"
        check(zarr.open_group(target, mode="r"), adata, leiden=leiden)
    assert not (Path(target) / MARKER).exists()


def test_in_place_remote_with_a_leftover_results_marker(stores, adata, tmp_path):
    remote = memory("bucket/inplace.zarr")
    run(cfg(stores["pbmc"], tmp_path, external=remote), adata)
    fsspec.filesystem("memory").pipe_file(f"/bucket/inplace.zarr/{MARKER}", b"results")
    written, dest = run(cfg(stores["pbmc"], tmp_path, external=remote), adata)
    assert dest.pending == "results" and written == f"written into {remote}"
    assert outputs.detect(remote) == ("zarr", "")
    check(zarr.open_group(remote, mode="r"), adata)


def test_location_changed_during_the_run(stores, adata, tmp_path):
    c = cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "late.zarr"))
    dest = outputs.plan(c)
    src = open_source(c.input, c.table, c.branch, "")
    zarr.open_group(tmp_path / "late.zarr", mode="w").create_group("obs")
    with pytest.raises(OutputError, match="changed during the run"):
        outputs.save(dest, src, results(adata).write, "m", "main", {})


# --- review findings -------------------------------------------------------------------------------------------

def test_the_input_itself_is_never_the_external_location(stores, icechunk, tmp_path):
    for input in (stores["pbmc"], icechunk["mount"]):
        with pytest.raises(OutputError, match="is the input itself"):
            outputs.plan(cfg(input, tmp_path, external=str(input) + "/"))


def test_an_input_branch_without_data_matches_no_repo(stores, icechunk, tmp_path):
    empty = Repo.create(tmp_path / "empty_repo")
    empty.checkout("main")
    with pytest.raises(OutputError, match="has no commits with data"):
        outputs.plan(cfg(empty.path, tmp_path, external=str(icechunk["origin"])))
    Repo(str(icechunk["origin"])).checkout("dev", create=True)
    with pytest.raises(OutputError, match="the input has no branch dev"):
        outputs.plan(cfg(icechunk["mount"], tmp_path, external=str(icechunk["origin"]), branch="dev"))


def test_in_place_needs_the_same_genes(stores, adata, tmp_path):
    import annizarr

    other = adata.copy()
    other.var_names = [f"X{g}" for g in other.var_names]
    annizarr.convert(other, output=str(tmp_path / "genes.zarr"))
    c = cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "genes.zarr"))
    with pytest.raises(OutputError, match="cells and genes"):
        outputs.check(outputs.plan(c), open_source(c.input, c.table, c.branch, ""))


def test_leftovers_and_read_only_locations_are_refused_up_front(stores, tmp_path, monkeypatch):
    (tmp_path / "ext.zarr.partial").mkdir()
    with pytest.raises(OutputError, match="partial was left by an earlier run"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "ext.zarr")))
    monkeypatch.setattr(outputs.os, "access", lambda path, mode: False)  # root ignores modes; /data is a ro mount
    with pytest.raises(OutputError, match="can't be written"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "new.zarr")))
    monkeypatch.undo()
    with pytest.raises(ConfigError, match="results_copy is a local path"):
        Config(results_copy="s3://b/x").validate()


def test_a_marker_only_location_says_half_written(stores, tmp_path):
    remote = memory("bucket/half.zarr")
    fsspec.filesystem("memory").pipe_file(f"/bucket/half.zarr/{MARKER}", b"new")
    with pytest.raises(OutputError, match="half-written"):
        outputs.plan(cfg(stores["pbmc"], tmp_path, external=remote))


def test_new_repo_on_a_branch_keeps_the_data_on_main(stores, adata, tmp_path):
    run(cfg(stores["pbmc"], tmp_path, external=str(tmp_path / "repo"), external_format="icechunk", branch="dev"), adata)
    main = ad_read(Repo(str(tmp_path / "repo")).open_zarr("r"))
    assert main.n_obs == adata.n_obs and "process" not in main.uns
    check(Repo(str(tmp_path / "repo"), branch="dev").open_zarr("r"), adata)


def test_write_failures_are_one_line(icechunk, adata, tmp_path):
    origin = str(icechunk["origin"])
    c = cfg(icechunk["mount"], tmp_path, external=origin)
    dest = outputs.plan(c)
    src = open_source(c.input, c.table, c.branch, origin)
    other = Repo(origin, branch="main")
    other.open_zarr("w").attrs["touched"] = "yes"
    other.commit("moved")
    with pytest.raises(OutputError, match="nothing was committed .*moved since the store was opened"):
        outputs.save(dest, src, results(adata).write, "m", "main", {})


def ad_read(group):
    import anndata as ad

    return ad.io.read_elem(group)
