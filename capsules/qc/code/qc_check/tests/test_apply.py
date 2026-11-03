import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from annizarr import Repo, add_layer

import samples
from conftest import build_store, append_samples, cohort, sample_adata
from helpers import accept_all, cell_qc, decisions, exit_code, run, summary, tree_digest, write_json


@pytest.fixture(autouse=True)
def small_cohort(monkeypatch):
    monkeypatch.setattr(samples, "MIN_CELLS", 50)


def reviewed(store: str, results, **overrides) -> str:
    """A report run, every pending sample accepted (plus `overrides`), and the preview run; returns its file."""
    run(results / "first", input=store)
    edited = accept_all(results / "first", results / "edited.json", **overrides)
    run(results / "preview", input=store, decisions=edited)
    return str(results / "preview" / "qc_decisions.json")


def output_for(kind: str, tmp_path) -> str:
    return "" if kind == "icechunk" else str(tmp_path / "curated.zarr")


def curated(store: str, kind: str, output: str, branch: str = "qc") -> ad.AnnData:
    root = Repo(store, branch=branch).open_zarr("r") if kind == "icechunk" else zarr.open_group(output, mode="r")
    return ad.read_zarr(root)


def raw(store: str, kind: str) -> ad.AnnData:
    return ad.read_zarr(Repo(store).open_zarr("r") if kind == "icechunk" else zarr.open_group(store, mode="r"))


def test_apply_writes_exactly_the_passing_cells(store, kind, results, tmp_path):
    file = reviewed(store, results, SP0006={"exclude": True})
    before = tree_digest(store) if kind == "zarr" else Repo(store).log()[0].id
    output = output_for(kind, tmp_path)
    run(results / "apply", input=store, decisions=file, mode="apply", target=output, reviewer="tester")

    pd.testing.assert_frame_equal(cell_qc(results / "preview"), cell_qc(results / "apply"))
    keep = cell_qc(results / "apply")["qc_pass"].to_numpy()
    src, out = raw(store, kind), curated(store, kind, output)
    assert out.n_obs == keep.sum() and "SP0006" not in set(out.obs["sample_id"])
    assert (out.X != src.X[keep]).nnz == 0
    pd.testing.assert_frame_equal(out.obs, src.obs.iloc[keep].assign(
        sample_id=src.obs["sample_id"].iloc[keep].cat.remove_unused_categories()), check_categorical=False)
    assert list(out.obs_names) == list(src.obs_names[keep])
    pd.testing.assert_frame_equal(out.var, src.var)
    root = Repo(store).open_zarr("r") if kind == "icechunk" else zarr.open_group(output, mode="r")
    if kind == "icechunk":
        root = Repo(store, branch="qc").open_zarr("r")
    assert len(list(root["obsp"].keys())) == 0
    qc = out.uns["qc"]
    assert json.loads(qc["decisions"]) == decisions(results / "preview")
    assert qc["reviewer"] == "tester" and "SP0006" not in list(qc["samples"])
    assert list(qc["counts"]["after"]) == pd.read_csv(results / "apply" / "apply_summary.csv")["after"].tolist()
    assert summary(results / "apply").startswith("Mode: apply\nApplied: ")

    if kind == "zarr":
        assert tree_digest(store) == before
    else:
        repo = Repo(store)
        assert repo.log()[0].id == before
        tip = Repo(store, branch="qc").log()[0]
        assert tip.parent_id == before
        assert {"pipeline", "created", "source_snapshot", "samples", "decisions_sha256", "reviewer", "n_pass",
                "n_cells"} <= set(tip.metadata)
        assert tip.metadata["n_pass"] == int(keep.sum()) and tip.metadata["source_snapshot"] == before


def test_layers_and_obsm_are_filtered_and_obsp_dropped(tmp_path, results):
    adata = ad.concat(cohort(5), merge="same")
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["X_test"] = np.arange(adata.n_obs * 2, dtype=float).reshape(-1, 2)
    adata.obsp["graph"] = sp.identity(adata.n_obs, format="csr")
    adata.uns["note"] = "kept"
    path = str(tmp_path / "plain.zarr")
    adata.write_zarr(path)
    add_layer(path, layer="by_gene", matrix="X", layout="csc")
    file = reviewed(path, results)
    run(results / "apply", input=path, decisions=file, mode="apply", target=str(tmp_path / "out.zarr"))
    keep = cell_qc(results / "apply")["qc_pass"].to_numpy()
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert (out.layers["counts"] != adata.layers["counts"][keep]).nnz == 0
    assert (out.layers["by_gene"] != adata.X[keep]).nnz == 0 and out.layers["by_gene"].format == "csc"
    assert dict(zarr.open_group(str(tmp_path / "out.zarr"))["layers/by_gene"].attrs)["annizarr-source"] == "X"
    np.testing.assert_array_equal(out.obsm["X_test"], adata.obsm["X_test"][keep])
    assert len(out.obsp) == 0 and out.uns["note"] == "kept"


@pytest.mark.parametrize("problem, message", [
    ("no_file", "apply needs the qc_decisions.json of a report run"),
    ("edited", "Decisions edited after their report"),
    ("pending", "Pending samples need a decision"),
])
def test_refuses_before_any_report_work(store, results, problem, message, caplog, monkeypatch):
    monkeypatch.setattr(samples, "MIN_CELLS", 500)
    run(results / "first", input=store)
    d = decisions(results / "first")
    if problem == "edited":
        d["note"] = "changed by hand"
    file = "" if problem == "no_file" else write_json(results / "d.json", d)
    if problem == "pending":
        assert d["pending"]
    out = results / "apply"
    assert exit_code(out, input=store, decisions=file, mode="apply", target=str(results / "o.zarr")) == 1
    assert message in caplog.text
    assert list(out.iterdir()) == []


def test_refuses_read_only_icechunk_and_used_output(tmp_path, results, monkeypatch, caplog):
    ic = build_store(tmp_path / "ic", cohort(5), "icechunk")
    plain = build_store(tmp_path / "p.zarr", cohort(5), "zarr")
    ic_file, plain_file = reviewed(ic, results / "ic"), reviewed(plain, results / "p")
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "file.txt").write_text("x")
    assert exit_code(results / "a1", input=plain, decisions=plain_file, mode="apply", target=str(tmp_path / "used")) == 1
    assert "already holds data" in caplog.text
    assert exit_code(results / "a2", input=plain, decisions=plain_file, mode="apply", target=plain) == 1
    assert f"{plain} already holds data" in caplog.text
    assert exit_code(results / "a5", input=plain, decisions=plain_file, mode="apply", target="s3://b/out.zarr") == 1
    assert "must be a local path" in caplog.text
    assert exit_code(results / "a3", input=ic, decisions=ic_file, mode="apply", target="main") == 1
    assert "other than main" in caplog.text
    import annizarr.ic._repo as repo_module
    monkeypatch.setattr(repo_module, "is_readonly_path", lambda p: True)
    assert exit_code(results / "a4", input=ic, decisions=ic_file, mode="apply") == 1
    assert "is read-only; pass its writable origin" in caplog.text
    assert all(list((results / f"a{i}").iterdir()) == [] for i in range(1, 6))


def test_curation_cycle_on_icechunk(tmp_path, results, caplog):
    store = build_store(tmp_path / "ic", cohort(5), "icechunk")
    file = reviewed(store, results / "1", SP0002={"exclude": True})
    run(results / "1" / "apply", input=store, decisions=file, mode="apply")
    first_tip = Repo(store, branch="qc").log()[0].id

    append_samples(store, [sample_adata("SP0007", seed=7, batch="B2", chip="C3")], tmp_path / "staging")
    new_main = Repo(store).log()[0].id
    assert exit_code(results / "stale", input=store, decisions=file, mode="apply") == 1
    assert "Input changed since this report" in caplog.text

    run(results / "2", input=store)
    d = decisions(results / "2")
    assert d["samples"]["SP0002"] == {"exclude": True}
    assert "SP0007" in d["samples"] or d["pending"] == ["SP0007"]
    assert summary(results / "2").splitlines()[2].startswith("qc covers 5 of 6 samples on main (curated ")
    file2 = accept_all(results / "2", results / "2.json")
    run(results / "2p", input=store, decisions=file2)
    run(results / "2a", input=store, decisions=str(results / "2p" / "qc_decisions.json"), mode="apply")

    qc_log = [s.id for s in Repo(store, branch="qc").log()]
    assert qc_log[1] == new_main and first_tip not in qc_log
    assert len(qc_log) == len(Repo(store).log()) + 1
    assert Repo(store).branches() == ["main", "qc"]
    out = curated(store, "icechunk", "")
    assert set(out.obs["sample_id"]) == {"SP0001", "SP0003", "SP0004", "SP0005", "SP0007"}


def test_custom_branch_is_independent(tmp_path, results):
    store = build_store(tmp_path / "ic", cohort(5), "icechunk")
    file = reviewed(store, results)
    run(results / "qc", input=store, decisions=file, mode="apply")
    qc_tip = Repo(store, branch="qc").log()[0].id
    strict = reviewed(store, results / "s", SP0001={"exclude": True})
    run(results / "strict", input=store, decisions=strict, mode="apply", target="strict")
    assert Repo(store, branch="qc").log()[0].id == qc_tip
    assert "SP0001" not in set(curated(store, "icechunk", "", branch="strict").obs["sample_id"])
    assert "SP0001" in set(curated(store, "icechunk", "", branch="qc").obs["sample_id"])
