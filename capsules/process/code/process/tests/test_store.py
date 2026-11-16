import ast
import json
import os
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import fsspec
import zarr
from annizarr import Repo

import store
from conftest import N_CELLS, N_GENES
from config import Config, ConfigError, load_config
from store import MARKER, OutputError, open_source

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


class Gpu(sp.csr_matrix):
    def get(self):
        return sp.csr_matrix(self)


def results(adata, leiden=("leiden",), obsm=("X_pca", "X_umap"), leiden_key="leiden") -> store.Results:
    x = da.from_array(adata.X.astype(np.float32), chunks=(1000, N_GENES), asarray=False)
    lognorm = x.map_blocks(lambda b: Gpu(b), dtype=np.float32, meta=Gpu((0, 0), dtype=np.float32))
    rng = np.random.default_rng(0)
    return store.Results(
        lognorm=lognorm, target_sum=1e4, highly_variable=np.arange(N_GENES) < 50,
        obsm={k: rng.normal(size=(N_CELLS, 2)) for k in obsm},
        obs={k: pd.Categorical(rng.integers(0, 3, N_CELLS).astype(str)) for k in leiden},
        uns={"method": "pca", "leiden": {k: 1.0 for k in leiden}}, leiden_key=leiden_key,
    )


def check(group, adata, leiden=("leiden",), obsm=("X_pca", "X_umap")):
    out = ad.io.read_elem(group)
    assert set(obsm) == set(out.obsm) & {*store.EMBEDDING_KEYS, "X_umap"}
    assert set(leiden) == {c for c in out.obs if store.LEIDEN_COLUMN.match(c)}
    assert out.var["highly_variable"].sum() == 50 and out.uns["process"]["method"] == "pca"
    assert group["layers/lognorm"].attrs["target_sum"] == 1e4
    assert (out.X != adata.X).nnz == 0 and (out.layers["lognorm"] != adata.X).nnz == 0
    return out


def test_unreadable_input_is_one_line(tmp_path, stores):
    with pytest.raises(store.InputError, match="can't open .*missing.zarr as an AnnData"):
        store.open_source(str(tmp_path / "missing.zarr"), "table", "main", "")
    with pytest.raises(store.InputError, match="has no X and no tables/other"):
        store.open_source(str(stores["tissue"]), "other", "main", "")


def test_new_output_never_reuses_a_name(tmp_path):
    assert store.new_output("/data/a/pbmc.zarr/", "", str(tmp_path)) == f"{tmp_path}/pbmc.processed.zarr"
    (tmp_path / "pbmc.processed.zarr").mkdir()
    (tmp_path / "pbmc.processed-1.zarr.partial").mkdir()
    assert store.new_output("/data/pbmc.zarr", "", str(tmp_path)) == f"{tmp_path}/pbmc.processed-2.zarr"
    assert store.new_output("x", str(tmp_path / "out"), "/r") == f"{tmp_path}/out.zarr"
    assert store.free_path(str(tmp_path / "scvi_model")) == str(tmp_path / "scvi_model")


def _functions(source: str, names) -> dict[str, str]:
    tree = ast.parse(source)
    return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}


def test_helpers_match_the_notebooks():
    """store.py's copies of the notebooks' helper cell must stay word for word the same."""
    names = ("child", "put", "put_column", "free_path", "new_output", "save_copy", "save_commit")
    module = _functions(Path(store.__file__).read_text(), names)
    assert set(module) == set(names)
    for nb in NOTEBOOKS.glob("*.ipynb"):
        cell = next("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                    if "def save_commit" in "".join(c["source"]))
        assert _functions(cell, names) == module, nb.name


def test_custom_names_keep_the_default_results_and_replace_only_their_own(stores, adata, tmp_path):
    import shutil

    out = str(tmp_path / "x.zarr")
    shutil.copytree(stores["pbmc"], out)
    group = zarr.open_group(out, mode="r+", use_consolidated=False)
    results(adata, ("leiden",)).write(group)
    results(adata, ("clusters_0.5", "clusters_1.0"), ("X_pca", "X_umap_b"), "clusters").write(group)
    done = ad.io.read_elem(group)
    assert {"leiden", "clusters_0.5", "clusters_1.0"} <= set(done.obs) and {"X_umap", "X_umap_b"} <= set(done.obsm)
    results(adata, ("clusters",), ("X_pca", "X_umap_b"), "clusters").write(group)
    done = ad.io.read_elem(group)
    assert {"leiden", "clusters"} <= set(done.obs) and not {"clusters_0.5", "clusters_1.0"} & set(done.obs)
    assert "X_umap" in done.obsm


def test_access_errors_are_recognized():
    import botocore.exceptions as be

    denied = be.ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
    wrapped = PermissionError("denied")
    wrapped.__cause__ = denied
    assert store.is_access_error(denied) and store.is_access_error(wrapped)
    assert store.is_access_error(RuntimeError("Unable to locate credentials"))
    assert not store.is_access_error(FileNotFoundError("nope"))
    outer = RuntimeError("write failed")
    middle = ValueError("icechunk error")
    outer.__context__, middle.__cause__ = middle, denied
    assert store.is_access_error(outer)
    assert "AWS secret with write permission" in store.access_hint("s3://b/p", writing=True)


# --- output: which destination a run gets, and each writer (locally and over memory://) ---


def cfg(input, tmp_path, **kw) -> Config:
    return Config(input=str(input), results_dir=str(tmp_path / "res"), **kw).validate()


def run(c: Config, adata, leiden=("leiden",)):
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, dest.location if dest.mode == "commit" else "")
    store.check(dest, src)
    return store.save(dest, src, results(adata, leiden).write, "m", c.branch, {"method": "pca"}), dest


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
    assert store.plan(cfg(stores["pbmc"], tmp_path)) == store.Destination("copy", "zarr", f"{res}/pbmc.processed")
    assert store.plan(cfg(icechunk["mount"], tmp_path)).kind == "icechunk"
    assert store.plan(cfg(icechunk["mount"], tmp_path, output_format="zarr")).kind == "zarr"
    assert store.plan(cfg(stores["pbmc"], tmp_path, output="mine.zarr")).location == f"{res}/mine"
    assert store.plan(cfg(stores["pbmc"], tmp_path, output=f"{res}/deep/abs")).location == f"{res}/deep/abs"


def test_one_output_box_is_a_copy_under_results_and_an_external_location_elsewhere(stores, tmp_path):
    res = tmp_path / "res"
    for output, mode in (("", "copy"), ("name", "copy"), (f"{res}/sub/x", "copy"), (str(tmp_path / "elsewhere.zarr"), "new"),
                         (str(res), "new"), ("memory://b/x.zarr", "new")):
        assert store.plan(cfg(stores["pbmc"], tmp_path, output=output)).mode == mode, output
    with pytest.raises(ConfigError, match="output_format must be one of"):
        load_config(["--output_format", "h5ad"], env={})
    assert load_config(["--output_format", "IceChunk"], env={}).output_format == "icechunk"


def test_zarr_input_external_new_existing_and_mismatch(stores, icechunk, tmp_path):
    new = tmp_path / "new.zarr"
    assert store.plan(cfg(stores["pbmc"], tmp_path, output=str(new))) == store.Destination("new", "zarr", str(new))
    assert store.plan(cfg(stores["pbmc"], tmp_path, output=str(new), output_format="icechunk")).kind == "icechunk"
    (tmp_path / "empty").mkdir()
    assert store.plan(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "empty"))).mode == "new"
    existing = store.plan(cfg(stores["pbmc"], tmp_path, output=str(stores["ingest_no_sample"])))
    assert (existing.mode, existing.kind) == ("in_place", "zarr")
    with pytest.raises(OutputError, match="already holds a zarr store, but Output format is icechunk"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(stores["ingest_no_sample"]), output_format="icechunk"))
    with pytest.raises(OutputError, match="is an icechunk repo but the input is a zarr"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(icechunk["origin"])))
    with pytest.raises(OutputError, match="already holds a icechunk store, but Output format is zarr"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(icechunk["origin"]), output_format="zarr"))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "notes.txt").write_text("x")
    with pytest.raises(OutputError, match="isn't a zarr or an icechunk repo"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "junk")))


def test_icechunk_input_only_commits_onto_its_own_repo(stores, icechunk, tmp_path):
    mount, origin = icechunk["mount"], icechunk["origin"]
    assert store.plan(cfg(mount, tmp_path, output=str(origin))) == store.Destination("commit", "icechunk", str(origin))
    for location, message in [(tmp_path / "typo", "nothing there"), (stores["pbmc"], "a zarr there")]:
        with pytest.raises(OutputError, match=f"{message} at .*: check it for typos"):
            store.plan(cfg(mount, tmp_path, output=str(location)))
    other = Repo.init(stores["ingest_no_sample"], tmp_path / "other_repo")
    with pytest.raises(OutputError, match="is not the repo the input was read from"):
        store.plan(cfg(mount, tmp_path, output=other.path))
    with pytest.raises(OutputError, match="can't use branch dev"):
        store.plan(cfg(mount, tmp_path, output=str(origin), branch="dev"))


def test_half_written_marker(stores, tmp_path):
    out = tmp_path / "out.zarr"
    zarr.open_group(out, mode="w")
    (out / MARKER).write_text("new")
    with pytest.raises(OutputError, match="left unfinished"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(out)))
    (out / MARKER).write_text("results")
    assert store.plan(cfg(stores["pbmc"], tmp_path, output=str(out))).pending == "results"


def test_spatialdata_never_becomes_icechunk(stores, tmp_path):
    for c in (cfg(stores["tissue"], tmp_path, output_format="icechunk"),
              cfg(stores["tissue"], tmp_path, output=str(tmp_path / "new"), output_format="icechunk")):
        dest = store.plan(c)
        with pytest.raises(OutputError, match="SpatialData zarr can't be stored as icechunk"):
            store.check(dest, open_source(c.input, c.table, c.branch, ""))


def test_in_place_needs_the_same_cells(stores, adata, tmp_path):
    import annizarr

    other = tmp_path / "other.zarr"
    annizarr.convert(adata[:100].copy(), output=str(other))
    c = cfg(stores["pbmc"], tmp_path, output=str(other))
    with pytest.raises(OutputError, match="not the input's, so the results would not line up"):
        store.check(store.plan(c), open_source(c.input, c.table, c.branch, ""))
    c = cfg(stores["pbmc"], tmp_path, output=str(stores["tissue"]))
    with pytest.raises(OutputError, match="isn't an AnnData store like the input"):
        store.check(store.plan(c), open_source(c.input, c.table, c.branch, ""))


# --- save: each writer -----------------------------------------------------------------------------------------

def test_zarr_copy_never_reuses_a_name(stores, adata, tmp_path):
    first, _ = run(cfg(stores["pbmc"], tmp_path), adata)
    second, _ = run(cfg(stores["pbmc"], tmp_path), adata)
    assert first == f"saved to {tmp_path}/res/pbmc.processed.zarr" and second.endswith("pbmc.processed-1.zarr")
    out = check(zarr_at(first), adata)
    assert list(out.obs.columns[:3]) == ["sample_id", "chip_id", "cell_type"]
    assert not list((tmp_path / "res").glob("*.partial"))


def test_ingest_store_without_layers_named_copy(stores, adata, tmp_path):
    written, _ = run(cfg(stores["ingest_no_sample"], tmp_path, output="out"), adata)
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
    written, _ = run(cfg(icechunk["mount"], tmp_path, output_format="zarr"), adata)
    assert written == f"saved to {tmp_path}/res/mount_icechunk.processed.zarr"
    check(zarr_at(written), adata)
    written, _ = run(cfg(stores["pbmc"], tmp_path, output_format="icechunk", branch="dev"), adata)
    out = written.split(" at ")[1]
    assert out == f"{tmp_path}/res/pbmc.processed" and " on branch dev " in written
    assert Repo(out, branch="dev").log()[0].metadata["source"] == str(stores["pbmc"])
    check(Repo(out, branch="dev").open_zarr("r"), adata)


def test_commit_through_the_external_repo_twice(icechunk, adata, tmp_path):
    mount, origin = icechunk["mount"], str(icechunk["origin"])
    before = sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*"))
    n = len(Repo(origin).log())
    for _ in range(2):
        written, _ = run(cfg(mount, tmp_path, output=origin), adata)
        assert written.startswith("saved as a new version") and written.endswith(f"on branch main at {origin}")
    log = Repo(origin).log()
    assert len(log) == n + 2 and log[0].metadata["source_snapshot"] == log[1].id and log[0].metadata["method"] == "pca"
    check(Repo(origin).open_zarr("r"), adata)
    assert sorted((p, p.stat().st_mtime_ns) for p in mount.rglob("*")) == before
    assert not (tmp_path / "res").exists()


def test_branch_moved_writes_nothing(icechunk, adata, tmp_path):
    origin = str(icechunk["origin"])
    c = cfg(icechunk["mount"], tmp_path, output=origin)
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, origin)
    other = Repo(origin, branch="main")
    other.open_zarr("w").attrs["touched"] = "yes"
    tip = other.commit("moved meanwhile")
    with pytest.raises(store.BranchMoved, match="someone else saved to branch"):
        store.save(dest, src, results(adata).write, "m", "main", {})
    assert Repo(origin).log()[0].id == tip


def test_new_external_zarr_local_and_remote(stores, adata, tmp_path):
    local, _ = run(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "ext.zarr")), adata)
    assert local == f"saved to {tmp_path}/ext.zarr" and not (tmp_path / "ext.zarr.partial").exists()
    check(zarr_at(local), adata)
    remote = memory("bucket/ext.zarr")
    written, _ = run(cfg(stores["tissue"], tmp_path, output=remote), adata)
    assert written == f"saved to {remote}"
    assert not fsspec.filesystem("memory").exists(f"/bucket/ext.zarr/{MARKER}")
    check(zarr.open_group(remote, mode="r")["tables/table"], adata)
    assert store.detect(remote) == ("zarr", "")


def test_new_external_icechunk_from_a_zarr(stores, adata, tmp_path):
    written, _ = run(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "repo"), output_format="icechunk"), adata)
    assert written.startswith("saved as a new version") and written.endswith(f"of a new repo at {tmp_path}/repo")
    assert len(Repo(str(tmp_path / "repo")).log()) == 2
    check(Repo(str(tmp_path / "repo")).open_zarr("r"), adata)


def test_in_place_into_an_existing_zarr_twice(stores, adata, tmp_path):
    import shutil

    first, _ = run(cfg(stores["pbmc"], tmp_path), adata, ("leiden_0.5", "leiden_2.0"))
    target = str(tmp_path / "elsewhere.zarr")
    shutil.copytree(first.removeprefix("saved to "), target)
    for leiden in (("leiden",), ("leiden_1.0",)):
        written, dest = run(cfg(stores["pbmc"], tmp_path, output=target), adata, leiden)
        assert dest.mode == "in_place" and written == f"written into {target}"
        check(zarr.open_group(target, mode="r"), adata, leiden=leiden)
    assert not (Path(target) / MARKER).exists()


def test_in_place_remote_with_a_leftover_results_marker(stores, adata, tmp_path):
    remote = memory("bucket/inplace.zarr")
    run(cfg(stores["pbmc"], tmp_path, output=remote), adata)
    fsspec.filesystem("memory").pipe_file(f"/bucket/inplace.zarr/{MARKER}", b"results")
    written, dest = run(cfg(stores["pbmc"], tmp_path, output=remote), adata)
    assert dest.pending == "results" and written == f"written into {remote}"
    assert store.detect(remote) == ("zarr", "")
    check(zarr.open_group(remote, mode="r"), adata)


def test_location_changed_during_the_run(stores, adata, tmp_path):
    c = cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "late.zarr"))
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, "")
    zarr.open_group(tmp_path / "late.zarr", mode="w").create_group("obs")
    with pytest.raises(OutputError, match="changed during the run"):
        store.save(dest, src, results(adata).write, "m", "main", {})


# --- review findings -------------------------------------------------------------------------------------------

def test_the_input_itself_is_never_the_external_location(stores, icechunk, tmp_path):
    for input in (stores["pbmc"], icechunk["mount"]):
        with pytest.raises(OutputError, match="same as the input"):
            store.plan(cfg(input, tmp_path, output=str(input) + "/"))


def test_an_input_branch_without_data_matches_no_repo(stores, icechunk, tmp_path):
    empty = Repo.create(tmp_path / "empty_repo")
    empty.checkout("main")
    with pytest.raises(OutputError, match="has no commits with data"):
        store.plan(cfg(empty.path, tmp_path, output=str(icechunk["origin"])))
    Repo(str(icechunk["origin"])).checkout("dev", create=True)
    with pytest.raises(OutputError, match="the input has no branch dev"):
        store.plan(cfg(icechunk["mount"], tmp_path, output=str(icechunk["origin"]), branch="dev"))


def test_in_place_needs_the_same_genes(stores, adata, tmp_path):
    import annizarr

    other = adata.copy()
    other.var_names = [f"X{g}" for g in other.var_names]
    annizarr.convert(other, output=str(tmp_path / "genes.zarr"))
    c = cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "genes.zarr"))
    with pytest.raises(OutputError, match="cells x .* genes"):
        store.check(store.plan(c), open_source(c.input, c.table, c.branch, ""))


def test_leftovers_and_read_only_locations_are_refused_up_front(stores, tmp_path, monkeypatch):
    (tmp_path / "ext.zarr.partial").mkdir()
    with pytest.raises(OutputError, match="partial was left by an earlier run"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "ext.zarr")))
    monkeypatch.setattr(store.os, "access", lambda path, mode: False)  # root ignores modes; /data is a ro mount
    with pytest.raises(OutputError, match="can't be written"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "new.zarr")))
    monkeypatch.undo()
    with pytest.raises(ConfigError, match="results_dir is a local path"):
        Config(results_dir="s3://b/x").validate()


def test_a_marker_only_location_says_half_written(stores, tmp_path):
    remote = memory("bucket/half.zarr")
    fsspec.filesystem("memory").pipe_file(f"/bucket/half.zarr/{MARKER}", b"new")
    with pytest.raises(OutputError, match="left unfinished"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=remote))


def test_new_repo_on_a_branch_keeps_the_data_on_main(stores, adata, tmp_path):
    run(cfg(stores["pbmc"], tmp_path, output=str(tmp_path / "repo"), output_format="icechunk", branch="dev"), adata)
    main = ad_read(Repo(str(tmp_path / "repo")).open_zarr("r"))
    assert main.n_obs == adata.n_obs and "process" not in main.uns
    check(Repo(str(tmp_path / "repo"), branch="dev").open_zarr("r"), adata)


def test_write_failures_are_one_line(icechunk, adata, tmp_path):
    origin = str(icechunk["origin"])
    c = cfg(icechunk["mount"], tmp_path, output=origin)
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, origin)
    other = Repo(origin, branch="main")
    other.open_zarr("w").attrs["touched"] = "yes"
    other.commit("moved")
    with pytest.raises(store.BranchMoved, match="were not saved"):
        store.save(dest, src, results(adata).write, "m", "main", {})


def ad_read(group):
    import anndata as ad

    return ad.io.read_elem(group)


# --- no access to S3: one line saying what to attach, before any GPU work -------------------------------------

def denied(monkeypatch, *methods):
    from fsspec.implementations.memory import MemoryFileSystem

    def refuse(self, *args, **kwargs):
        raise PermissionError("An error occurred (403) when calling the HeadObject operation: Forbidden")

    for method in methods:
        monkeypatch.setattr(MemoryFileSystem, method, refuse)


@pytest.mark.parametrize("method", ["pipe_file", "cat_file", "rm_file"])
def test_remote_without_access_is_an_aws_secret_hint(stores, tmp_path, monkeypatch, method):
    denied(monkeypatch, method)
    with pytest.raises(OutputError, match="attach an AWS secret with write permission"):
        store.plan(cfg(stores["pbmc"], tmp_path, output=memory("bucket/denied.zarr")))


def test_write_probe_leaves_nothing_behind(stores, tmp_path):
    remote = memory("bucket/probe.zarr")
    store.plan(cfg(stores["pbmc"], tmp_path, output=remote))
    assert not fsspec.filesystem("memory").exists("/bucket/probe.zarr/.process-write-check")


def test_write_refused_after_the_run_is_the_same_hint(stores, adata, tmp_path, monkeypatch):
    remote = memory("bucket/late-denied.zarr")
    c = cfg(stores["pbmc"], tmp_path, output=remote)
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, "")
    denied(monkeypatch, "pipe_file")
    with pytest.raises(OutputError, match="attach an AWS secret with write permission"):
        store.save(dest, src, results(adata).write, "m", c.branch, {})


def test_unreadable_remote_input_is_the_read_hint(monkeypatch):
    import annizarr

    def refuse(path):
        raise PermissionError("403 Forbidden")

    monkeypatch.setattr(annizarr.Repo, "exists", staticmethod(refuse))
    with pytest.raises(store.InputError, match="no read access to s3://b/in.zarr: attach an AWS secret with read"):
        store.input_kind("s3://b/in.zarr")


def test_private_annizarr_and_experimental_anndata_apis_are_present():
    from anndata.experimental import read_elem_lazy

    assert callable(store.copy_group) and callable(read_elem_lazy)


def test_a_failed_copy_leaves_no_partial_and_says_so(stores, adata, tmp_path):
    c = cfg(stores["pbmc"], tmp_path)
    dest = store.plan(c)
    src = open_source(c.input, c.table, c.branch, "")

    def boom(group):
        raise RuntimeError("disk full")

    with pytest.raises(OutputError, match="nothing was left there"):
        store.save(dest, src, boom, "m", c.branch, {})
    assert not list((tmp_path / "res").glob("*.partial"))


def test_remote_output_equal_to_the_input_is_refused(tmp_path):
    with pytest.raises(OutputError, match="same as the input"):
        store.plan(Config(input="memory://b/in.zarr", output="memory://b/in.zarr/", results_dir=str(tmp_path)).validate())


def test_a_users_own_numbered_column_is_not_dropped_by_a_custom_leiden_key(stores, adata, tmp_path):
    import shutil

    out = str(tmp_path / "x.zarr")
    shutil.copytree(stores["pbmc"], out)
    group = zarr.open_group(out, mode="r+", use_consolidated=False)
    results(adata, ("cluster",), leiden_key="cluster").write(group)
    store.put_column(group["obs"], "cluster_2", pd.Categorical(["a"] * N_CELLS))
    results(adata, ("cluster",), leiden_key="cluster").write(group)
    assert "cluster_2" in ad.io.read_elem(group).obs
