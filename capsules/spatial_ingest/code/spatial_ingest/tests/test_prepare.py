import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

import conftest as fx
import inputs
import prepare
from conftest import GENES
from config import Candidate, Inventory, LeftOut, Member, Part, Sample


def var(names):
    return pd.DataFrame({"gene_ids": [fx.gene_id(n) for n in names], "feature_types": "Gene Expression"},
                        index=pd.Index(list(names)))


def bundle(sid, names=GENES, *, missing=(), optional=(), species="human"):
    cand = Candidate(sid, "xenium", Path(f"/data/{sid}"), files={"experiment.xenium": Member(Path(f"/data/{sid}/e"))})
    v = None if missing else var(names)
    inv = Inventory(sid, "xenium", {}, list(missing), list(optional), v, n_cells=10)
    return Sample(sid, "xenium", cand, inventory=inv, var=v, species=None if missing else species)


def from_object(sid, obj):
    origin = Path(f"/data/{obj}.zarr")
    cand = Candidate(obj, "spatialdata", origin, zarr=Member(origin))
    return Sample(sid, "spatialdata", cand, part=Part(sid, "table", (f"cells-{sid}",), (), ()), path=origin,
                  var=var(GENES), species="human")


def store_state(**kw):
    return prepare.StoreState(**{"present": ["s0"], "genes": GENES, "species": "human", "model": "", **kw})


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_incomplete_bundle_is_one_concise_warning(caplog):
    left = [LeftOut("/data/notes.txt", "not a sample")]
    with caplog.at_level(logging.INFO):
        plan = prepare.check([bundle("a", optional=["morphology images", "transcripts"]), bundle("b")], left, [],
                             prepare.StoreState(), None)
    assert [s.sid for s in plan.samples] == ["a", "b"]
    assert warnings_of(caplog) == ["Left out 1 item(s) in the inputs:\n  - /data/notes.txt: not a sample",
                                   "a: no morphology images, transcripts in the bundle; it is added without them."]


def test_missing_essentials_stop_the_precheck_naming_every_sample():
    samples = [bundle("ok"), bundle("gse1", missing=["count matrix"]), bundle("gse2", missing=["cells table"])]
    with pytest.raises(prepare.IngestError) as err:
        prepare.check(samples, [], [], prepare.StoreState(), None)
    text = str(err.value)
    assert "nothing was written" in text and "gse1 (/data/gse1): missing count matrix" in text
    assert "gse2 (/data/gse2): missing cells table" in text and "ok (" not in text


def test_mixed_species_are_refused():
    mouse = [g.capitalize() for g in GENES]
    with pytest.raises(prepare.IngestError, match="Mouse_Ileum is mouse; the store and the other 2 samples are human"):
        prepare.check([bundle("h0"), bundle("h1"), bundle("Mouse_Ileum", mouse, species="mouse")], [], [],
                      store_state(), None)
    with pytest.raises(prepare.IngestError, match="different species: m is mouse; the other 1 sample is human"):
        prepare.check([bundle("h0"), bundle("m", mouse, species="mouse")], [], [], prepare.StoreState(), None)


def test_celltypist_model_mismatch_is_refused(tiny_model):
    with pytest.raises(prepare.IngestError, match="shares 0 of its 42 genes with low"):
        prepare.check([bundle("low", [g.lower() for g in GENES])], [], [], prepare.StoreState(), tiny_model)
    with pytest.raises(prepare.IngestError, match="built without CellTypist labels, and this run gives model tiny_panel"):
        prepare.check([bundle("b")], [], [], store_state(), tiny_model)


def test_differing_panels_are_one_warning(caplog):
    with caplog.at_level(logging.INFO):
        plan = prepare.check([bundle("a", GENES), bundle("b", GENES[:30] + ["NEW1", "NEW2"])], [], [],
                             store_state(genes=GENES[:30]), None)
    assert plan.union_genes == GENES + ["NEW1", "NEW2"]
    assert plan.panels["b"]["genes_added"] == 2 and plan.panels["b"]["genes_missing"] == 12
    (warned,) = warnings_of(caplog)
    assert warned.startswith("Gene panels differ") and "union of 44 genes" in warned


def test_colliding_spatialdata_ids_get_the_object_prefix(caplog):
    state = store_state(present=["rep1"], records={"rep1": {"source": "/data/cohortA.zarr"}})
    samples = [from_object("rep1", "cohortB"), from_object("rep2", "cohortB"), from_object("rep2", "cohortC")]
    with caplog.at_level(logging.INFO):
        plan = prepare.check(samples, [], [], state, None)
    assert [s.sid for s in plan.samples] == ["cohortB_rep1", "rep2", "cohortC_rep2"]
    assert "rep1 from /data/cohortB.zarr is added as cohortB_rep1" in caplog.text


def test_spatialdata_object_without_a_table_is_refused(tmp_path):
    path = fx.drop_tables(fx.write_sdata(tmp_path / "toy.zarr", {"a": {"n": 6, "seed": 0}}))
    with pytest.raises(prepare.SplitError, match="has no table; spatial_ingest needs a table of cells x genes"):
        prepare.plan(path, "toy", "sample_id")


def test_spatialdata_object_splits_into_samples_with_their_own_rows(tmp_path):
    path = fx.write_sdata(tmp_path / "two.zarr", {"a": {"n": 10, "seed": 0}, "b": {"n": 12, "seed": 1}})
    parts = {p.sid: p for p in prepare.plan(path, "two", "sample_id")}
    assert set(parts) == {"two_a", "two_b"}
    loaded = prepare.load_part(path, parts["two_b"], tmp_path / "scratch")
    source = read_elem(zarr.open_group(path / "tables" / "table", mode="r", use_consolidated=False))
    rows = np.flatnonzero(source.obs["region"].astype(str) == "cells-b")
    assert np.array_equal(ad.read_zarr(loaded.table.path).X.toarray(), source.X[rows].toarray())
    assert set(loaded.elements["shapes"]) == {"cells-b"}


def load(cand, scratch):
    inv = inputs.inspect(cand, scratch)
    sample = Sample(cand.sid, cand.kind, cand, inventory=inv)
    return sample, prepare.load(sample, scratch)


@pytest.mark.parametrize("kind,n,seed", [("xenium", 30, 0), ("atera", 15, 5)])
def test_count_matrix_streams_in_bands_equal_to_the_source(tmp_path, monkeypatch, kind, n, seed):
    monkeypatch.setattr(inputs, "BAND_NNZ", 17)
    folder = (fx.xenium_full if kind == "xenium" else fx.atera)(tmp_path / kind, n=n, seed=seed)
    _, loaded = load(fx.candidate(folder, kind), tmp_path / "s")
    bands = [band for _, band in inputs.iter_rows(loaded.table.path)]
    expected = fx.sparse_counts(n, len(fx.features()), seed)[:, :len(GENES)]
    assert len(bands) > 1 and (sp.vstack(bands).tocsr() != expected).nnz == 0


def test_prepared_table_has_qc_labels_and_its_region(tmp_path, tiny_model):
    sample, loaded = load(fx.candidate(fx.xenium_full(tmp_path / "full")), tmp_path / "s")
    prep = prepare.prepare(loaded, sample, tiny_model, tmp_path / "s")
    table = ad.read_zarr(prep.table_path)
    assert table.n_obs == 30 and list(table.var_names) == GENES
    assert set(table.obs["region"]) == {"full_cell_boundaries"}
    assert sorted(prep.elements["shapes"]) == ["full_cell_boundaries", "full_nucleus_boundaries"]
    assert {"total_counts", "pct_counts_mt", "tiny_panel_label", "tiny_panel_prob"} <= set(table.obs)
    assert set(table.obs["tiny_panel_label"]) <= {"Epithelial", "Immune"}
    assert table.uns["spatial_ingest"]["celltypist_model"] == "tiny_panel"
