"""The joint table: gene and obs union, measured-gene records, streamed rows, and crash safety."""
import logging

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import fixtures_sdata as F
import joint
import store as stores


@pytest.fixture
def local(tmp_path):
    path = tmp_path / "store.zarr"
    stores.create(path)
    return path


def add(local, sid, path, model=""):
    note = joint.add(local, sid, path, stores.staging(local), 2, model)
    stores.consolidate(local)
    return note


def read(local):
    return ad.read_zarr(joint.table_path(local))


def rows_of(t, sid, genes):
    sub = t[t.obs["sample_id"] == sid]
    return sub[:, list(genes)].X.toarray(), sub


def test_union_stats():
    genes, stats = joint.union(["A", "B", "C"], {"s1": ["B", "C", "D"], "s2": ["E"]})
    assert genes == ["A", "B", "C", "D", "E"]
    assert stats["s1"] == {"genes_in_panel": 3, "genes_shared": 2, "pct_shared": 66.7, "genes_added": 1,
                           "genes_missing": 2}
    assert stats["s2"]["genes_shared"] == 0 and stats["s2"]["genes_missing"] == 4
    genes, stats = joint.union([], {"only": ["X", "Y"]})
    assert genes == ["X", "Y"] and stats["only"]["pct_shared"] == 100.0


def test_differing_panels_union_with_measured_record(tmp_path, local):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES[:20], 8, 0)
    b = F.sample_table(tmp_path / "b.zarr", "B", F.GENES[10:] + ["NEW1"], 6, 1)
    assert add(local, "A", a, "model") == ""
    note = add(local, "B", b)
    assert note.startswith("12 new gene(s) added to the joint table (now 32); 10 joint gene(s) not in its panel")
    t = read(local)
    assert list(t.var_names) == F.GENES[:20] + F.GENES[20:] + ["NEW1"] and t.n_obs == 14
    for sid, path in (("A", a), ("B", b)):
        source = ad.read_zarr(path)
        x, sub = rows_of(t, sid, source.var_names)
        assert np.array_equal(x, source.X.toarray()) and sub.X.sum() == source.X.sum()   # filled genes are zero
    assert list(t.var["n_samples_measured"]) == [1] * 10 + [2] * 10 + [1] * 12
    schema = joint.read_schema(local)
    assert schema.samples == ["A", "B"] and schema.measured["B"] == F.GENES[10:] + ["NEW1"]
    assert schema.celltypist_model == "model" and schema.species == "human" and schema.panel_source == "A"
    assert t.X.dtype == np.int32 and dict(joint.sample_counts(local)) == {"A": 8, "B": 6}


def test_obs_union_and_type_fitting(tmp_path, local, caplog):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 5, 0, extra_obs={"batch": ["x"] * 5, "score": [1.5] * 5,
                                                                         "flag": [True] * 5})
    b = F.sample_table(tmp_path / "b.zarr", "B", F.GENES, 4, 1, extra_obs={"score": [2, 3, 4, 5], "flag": ["y"] * 4,
                                                                         "cell_ID": [9, 8, 7, 6]})
    add(local, "A", a)
    with caplog.at_level(logging.WARNING, logger="joint"):
        add(local, "B", b)
    t = read(local)
    assert list(t.obs["batch"].astype(object).where(t.obs["batch"].notna(), None)) == ["x"] * 5 + [None] * 4
    assert list(t.obs["score"]) == [1.5] * 5 + [2.0, 3.0, 4.0, 5.0]
    assert "source_cell_ID" in t.obs and "cell_ID" not in t.obs
    assert "flag are left out of the joint table" in caplog.text
    assert t.obs["flag"].isna().sum() == 4


def test_joint_after_union_reads_with_spatialdata_and_anndata(tmp_path, local):
    import spatialdata as sd

    add(local, "A", F.sample_table(tmp_path / "a.zarr", "A", F.GENES[:12], 5, 0, extra_obs={"cell_ID": range(5)}))
    add(local, "B", F.sample_table(tmp_path / "b.zarr", "B", F.GENES[5:], 7, 1, extra_obs={"kind": ["q"] * 7}))
    sdata = sd.read_zarr(local)
    assert sdata.tables["table"].shape == (12, len(F.GENES)) and stores.verify_tables(local) == 1
    assert read(local).obs_names[0] == "A:cell0"


def test_dtype_follows_the_first_sample(tmp_path, local):
    add(local, "A", F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 5, 0, dtype=np.int32))
    add(local, "B", F.sample_table(tmp_path / "b.zarr", "B", F.GENES, 5, 1, dtype=np.float32))
    assert read(local).X.dtype == np.int32


def test_same_sample_twice_is_refused(tmp_path, local):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 5, 0)
    add(local, "A", a)
    with pytest.raises(joint.JointError, match="already in the joint table"):
        add(local, "A", a)


def test_rows_stream_in_chunks(tmp_path, monkeypatch, local):
    import matrix
    monkeypatch.setattr(matrix, "BAND_NNZ", 11)
    monkeypatch.setattr(joint, "FLAT_CHUNK", 16)
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 40, 0)
    chunks = list(joint.iter_rows(a))
    assert len(chunks) > 5 and chunks[1][0] == chunks[0][1].shape[0]
    assert np.array_equal(sp.vstack([c for _, c in chunks]).toarray(), ad.read_zarr(a).X.toarray())
    add(local, "A", a)
    add(local, "B", F.sample_table(tmp_path / "b.zarr", "B", ["Z1"] + F.GENES[:3], 33, 1))
    t = read(local)
    assert np.array_equal(t[:40].X.toarray()[:, :31], ad.read_zarr(a).X.toarray())
    import zarr
    assert zarr.open_group(joint.table_path(local) / "X")["indptr"].dtype == np.int64


# ---------------------------------------------------------------- crash safety

def tables(tmp_path):
    return {"A": F.sample_table(tmp_path / "a.zarr", "A", F.GENES[:20], 8, 0, extra_obs={"batch": ["x"] * 8}),
            "B": F.sample_table(tmp_path / "b.zarr", "B", F.GENES[10:] + ["NEW1"], 6, 1)}


def rebuild(local, paths):
    """What the repair does: drop the joint table and add every per-sample table again."""
    joint.drop(local)
    for sid, path in paths.items():
        add(local, sid, path)


def expected(tmp_path):
    clean = tmp_path / "clean.zarr"
    stores.create(clean)
    for sid, path in tables(tmp_path).items():
        add(clean, sid, path)
    return read(clean)


def same(a, b):
    assert list(a.var_names) == list(b.var_names) and list(a.obs_names) == list(b.obs_names)
    assert np.array_equal(a.X.toarray(), b.X.toarray())
    assert list(a.var["n_samples_measured"]) == list(b.var["n_samples_measured"])


@pytest.mark.parametrize("kill", ["after_var", "mid_append", "after_append"])
def test_kill_during_union_is_repairable(tmp_path, local, monkeypatch, kill):
    import annizarr.ops._append as append_ops

    paths = tables(tmp_path)
    add(local, "A", paths["A"])

    class Killed(BaseException):
        pass

    def boom(*a, **k):
        raise Killed()

    if kill == "after_var":
        monkeypatch.setattr("annizarr.append", boom)
    elif kill == "mid_append":
        real = append_ops._append_obs
        monkeypatch.setattr(append_ops, "_append_obs", lambda *a, **k: (real(*a, **k), boom()))
    else:
        monkeypatch.setattr(joint, "write_schema", boom)
    with pytest.raises(Killed):
        joint.add(local, "B", paths["B"], stores.staging(local), 2)
    monkeypatch.undo()
    counts = joint.sample_counts(local)
    if kill == "after_var":
        # a consistent table with unmeasured new genes: the sample is simply added again
        assert dict(counts) == {"A": 8} and read(local).n_vars == 32
        add(local, "B", paths["B"])
    else:
        assert counts is None
        rebuild(local, paths)
    same(read(local), expected(tmp_path))
    assert dict(joint.sample_counts(local)) == {"A": 8, "B": 6}


def test_missing_record_reads_as_broken(tmp_path, local):
    import shutil

    add(local, "A", tables(tmp_path)["A"])
    shutil.rmtree(joint.table_path(local) / "uns" / joint.UNS_KEY)
    schema = joint.read_schema(local)
    assert not schema.intact and schema.samples == [] and joint.sample_counts(local) is None


# ---------------------------------------------------------------- shared-filesystem faults (C-1) and append cost (C-6)

def flaky_replace(monkeypatch, match: str, times: int, done: bool):
    """os.replace that fails `times` times with ENOENT for `.partial` chunk renames under `match`: after doing
    the rename (`done`, an NFS RENAME replayed after it succeeded) or without doing it (a real failure)."""
    import os

    real, left = os.replace, {"n": times}

    def replace(src, dst, *a, **k):
        if left["n"] > 0 and str(src).endswith(".partial") and match in str(src):
            left["n"] -= 1
            if done:
                real(src, dst, *a, **k)
            raise FileNotFoundError(2, "No such file or directory", str(src))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", replace)
    return left


def test_replayed_chunk_rename_counts_as_done(tmp_path, local, monkeypatch):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 8, 0)
    left = flaky_replace(monkeypatch, ".joint.zarr", 5, done=True)
    add(local, "A", a)
    assert left["n"] == 0
    assert np.array_equal(read(local).X.toarray(), ad.read_zarr(a).X.toarray())


def test_failed_staged_write_is_retried_once(tmp_path, local, monkeypatch, caplog):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 8, 0)
    left = flaky_replace(monkeypatch, ".joint.zarr", 1, done=False)
    with caplog.at_level(logging.INFO, logger="store"):
        add(local, "A", a)
    assert left["n"] == 0 and "writing it again" in caplog.text
    assert np.array_equal(read(local).X.toarray(), ad.read_zarr(a).X.toarray())


def test_twice_failed_staged_write_is_one_line(tmp_path, local, monkeypatch):
    a = F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 8, 0)
    flaky_replace(monkeypatch, ".joint.zarr", 1000, done=False)
    with pytest.raises(stores.StoreError, match="Writing A's joint rows failed twice .*store is unchanged") as err:
        joint.add(local, "A", a, stores.staging(local), 2)
    assert "\n" not in str(err.value) and joint.read_schema(local) is None


def test_append_onto_a_small_first_sample_widens_obs_chunks(tmp_path, local):
    import zarr

    add(local, "A", F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 3, 0, extra_obs={"flag": [True] * 3}))
    obs = zarr.open_group(joint.table_path(local) / "obs", mode="r")
    assert all(n.chunks[0] >= joint.OBS_CHUNK for _, n in obs.members(max_depth=None) if isinstance(n, zarr.Array))
    add(local, "B", F.sample_table(tmp_path / "b.zarr", "B", F.GENES, 500, 1, extra_obs={"new": ["q"] * 500}))
    obs = zarr.open_group(joint.table_path(local) / "obs", mode="r")
    assert all(n.chunks[0] >= joint.OBS_CHUNK for _, n in obs.members(max_depth=None) if isinstance(n, zarr.Array))
    files = sum(1 for p in (joint.table_path(local) / "obs").rglob("*") if p.is_file())
    assert files < 120                                    # a few files per column, not one per few rows
    add(local, "C", F.sample_table(tmp_path / "c.zarr", "C", F.GENES, 4, 2))   # annizarr's new columns widened too
    t = read(local)
    assert t.n_obs == 507 and list(t.obs["new"][3:503].astype(str).unique()) == ["q"]
    assert dict(joint.sample_counts(local)) == {"A": 3, "B": 500, "C": 4}


def test_missing_obs_column_reads_as_broken(tmp_path, local):
    import shutil

    add(local, "A", F.sample_table(tmp_path / "a.zarr", "A", F.GENES, 3, 0, extra_obs={"batch": ["x"] * 3}))
    shutil.rmtree(joint.table_path(local) / "obs" / "batch")
    assert joint.sample_counts(local) is None
