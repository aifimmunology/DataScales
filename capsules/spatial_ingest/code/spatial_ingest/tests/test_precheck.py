"""The precheck's decisions on hand-made samples (no files): what stops a run, what is only reported."""
import logging
from pathlib import Path

import pandas as pd
import pytest

import precheck
from conftest import HUMAN
from contracts import Candidate, Inventory, LeftOut, Member, Part, Sample


def var(names, prefix="ENSG"):
    return pd.DataFrame({"gene_ids": [f"{prefix}{i:011d}" for i in range(len(names))],
                         "feature_types": "Gene Expression"}, index=pd.Index(list(names)))


def bundle(sid, names=HUMAN, *, missing=(), optional=(), prefix="ENSG", kind="xenium", n_cells=10, notes=()):
    cand = Candidate(sid, kind, Path(f"/data/{sid}"), files={"experiment.xenium": Member(Path(f"/data/{sid}/e"))},
                     notes=tuple(notes))
    inv = Inventory(sid, kind, {}, list(missing), list(optional), None if missing else var(names, prefix),
                    n_cells=n_cells)
    v = None if missing else var(names, prefix)
    return Sample(sid, kind, cand, inventory=inv, var=v, species=None if missing else
                  {"ENSG": "human", "ENSMUSG": "mouse"}[prefix])


def sdata(sid, names=HUMAN, origin="/data/objects/obj.zarr"):
    cand = Candidate("obj", "spatialdata", Path(origin), zarr=Member(Path(origin)))
    part = Part(sid, "table", ("cell_boundaries",), (), ())
    return Sample(sid, "spatialdata", cand, part=part, path=Path(origin), var=var(names), species="human")


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_clean_samples_pass_with_no_warning(caplog):
    with caplog.at_level(logging.INFO):
        plan = precheck.check([bundle("a"), sdata("b")], [], [], precheck.StoreState(), None)
    assert [s.sid for s in plan.samples] == ["a", "b"] and plan.present == [] and plan.union_genes == HUMAN
    assert plan.panels["a"] == {"genes_in_panel": 42, "genes_shared": 42, "genes_added": 42, "pct_shared": 100.0,
                                "genes_missing": 0, "species": "human"}
    assert plan.panels["b"]["genes_added"] == 0
    assert warnings_of(caplog) == []
    assert "Precheck: 2 sample(s) to add; the store is new or empty." in caplog.text and "a: Xenium bundle from /data/a, 10 cells, 42 genes" in caplog.text


def test_missing_essentials_stop_the_run_listing_every_sample():
    samples = [bundle("ok"), bundle("gse1", missing=["count matrix (cell_feature_matrix.h5/.zarr.zip or matrix.mtx)"]),
               bundle("gse2", missing=["cells table with cell IDs and x/y positions (cells.parquet/.csv.gz)"])]
    with pytest.raises(precheck.IngestError) as err:
        precheck.check(samples, [], [], precheck.StoreState(), None)
    text = str(err.value)
    assert "nothing was written" in text
    assert "gse1 (/data/gse1): missing count matrix" in text and "gse2 (/data/gse2): missing cells table" in text
    assert "ok (" not in text and "Every bundle needs a count matrix" in text


def test_every_problem_is_listed_in_one_error():
    problems = ["broken.zarr (/data/broken.zarr): has no table; spatial_ingest needs a table of cells × genes"]
    samples = [bundle("a"), bundle("a"), bundle("m", [g.capitalize() for g in HUMAN], prefix="ENSMUSG"),
               bundle("x", missing=["count matrix"])]
    with pytest.raises(precheck.IngestError) as err:
        precheck.check(samples, [], problems, precheck.StoreState(), None)
    text = str(err.value)
    assert "has no table" in text and "Two samples have the ID a" in text
    assert "different species: a (human), m (mouse)" in text and "x (/data/x): missing count matrix" in text


def test_species_differing_from_the_store_is_refused():
    state = precheck.StoreState(present=["old"], genes=HUMAN, species="human", model="")
    with pytest.raises(precheck.IngestError, match="the store holds human genes"):
        precheck.check([bundle("m", prefix="ENSMUSG")], [], [], state, None)


def test_present_samples_are_skipped_and_a_different_source_is_warned(caplog):
    state = precheck.StoreState(present=["a", "b"], records={"a": {"source": "/data/a"}, "b": {"source": "/elsewhere/b"}},
                                genes=HUMAN, species="human", model="")
    with caplog.at_level(logging.INFO):
        plan = precheck.check([bundle("a"), bundle("b"), bundle("c")], [], [], state, None)
    assert plan.present == ["a", "b"] and [s.sid for s in plan.samples] == ["c"]
    assert warnings_of(caplog) == ["b is already in the store (added from /elsewhere/b); /data/b is skipped. If it is "
                                   "a different sample, rename it."]


def test_panels_differ_union_and_one_strong_warning(caplog):
    state = precheck.StoreState(present=["s0"], genes=HUMAN[:30], species="human", model="")
    with caplog.at_level(logging.INFO):
        plan = precheck.check([bundle("a", HUMAN), bundle("b", HUMAN[:30] + ["NEW1", "NEW2"])], [], [], state, None)
    assert plan.union_genes == HUMAN[:30] + HUMAN[30:] + ["NEW1", "NEW2"]
    assert plan.panels["a"]["genes_added"] == 12 and plan.panels["b"]["genes_added"] == 2
    assert plan.panels["b"]["genes_shared"] == 30 and plan.panels["b"]["pct_shared"] == pytest.approx(93.8)
    assert plan.panels["b"]["genes_missing"] == 12
    warned = warnings_of(caplog)
    assert len(warned) == 1 and warned[0].startswith("GENE PANELS DIFFER") and "union of 44 genes" in warned[0]
    assert "b: 32 genes = 30 also in other panels (94%) + 2 only in this panel; 12 of the 44 joint genes not measured" \
        in warned[0]
    assert "the store holds 1 sample(s)" in caplog.text


def test_identical_panels_in_a_new_store_are_not_warned(caplog):
    with caplog.at_level(logging.INFO):
        precheck.check([bundle("a"), bundle("b", list(reversed(HUMAN)))], [], [], precheck.StoreState(), None)
    assert warnings_of(caplog) == []


def test_left_out_items_and_missing_optional_parts_are_one_warning_each(caplog):
    left = [LeftOut("/data/qc_report.zip", "no sample in it"), LeftOut("/data/notes.txt", "not a sample")]
    with caplog.at_level(logging.INFO):
        precheck.check([bundle("a", optional=["morphology images", "transcripts"]), bundle("b", notes=["prefix GSM1_"])],
                       left, [], precheck.StoreState(), None)
    warned = warnings_of(caplog)
    assert len(warned) == 2
    assert warned[0].startswith("Left out 2 item(s)") and "/data/notes.txt: not a sample" in warned[0]
    assert warned[1] == "a: no morphology images, transcripts in the bundle; it is added without them."
    assert "b: prefix GSM1_" not in caplog.text   # discovery logs the candidate's notes, not the precheck


def test_celltypist_overlap_refused_low_warned_and_counted(tiny_model, caplog):
    with pytest.raises(precheck.IngestError, match="shares 0 of its 42 genes with low"):
        precheck.check([bundle("ok"), bundle("low", [g.lower() for g in HUMAN])], [], [], precheck.StoreState(),
                       tiny_model)
    with caplog.at_level(logging.INFO):
        plan = precheck.check([bundle("ok"), bundle("half", HUMAN[:15])], [], [], precheck.StoreState(), tiny_model)
    assert plan.panels["ok"]["celltypist_genes_shared"] == 42 and plan.panels["half"]["celltypist_genes_shared"] == 15
    warned = [w for w in warnings_of(caplog) if "CellTypist" in w]
    assert len(warned) == 1 and "half: 15 of the model's 42 genes (36%)" in warned[0]


def test_store_model_mismatch_only_matters_when_something_is_added(tiny_model):
    state = precheck.StoreState(present=["a"], genes=HUMAN, species="human", model="")
    with pytest.raises(precheck.IngestError, match="built without CellTypist labels, and this run gives model tiny_panel"):
        precheck.check([bundle("b")], [], [], state, tiny_model)
    assert precheck.check([bundle("a")], [], [], state, tiny_model).samples == []


def test_missing_default_samples_folder_names_what_data_holds(tmp_path, monkeypatch):
    import config
    (tmp_path / "my_asset").mkdir()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_SAMPLES", str(tmp_path / "samples"))
    settings = config.Settings(samples=str(tmp_path / "samples"))
    with pytest.raises(precheck.IngestError, match=r"No samples given, and .*samples doesn't exist.*holds: my_asset"):
        precheck.gather(settings, tmp_path / "scratch", set())


def test_store_model_messages_never_send_the_user_in_a_circle(tiny_model):
    state = precheck.StoreState(present=["a"], genes=HUMAN, species="human", model="tiny_panel")
    with pytest.raises(precheck.IngestError, match="labels every sample with CellTypist model tiny_panel, and this run "
                                                   "gives no model.*separate store"):
        precheck.check([bundle("b")], [], [], state, None)
    with pytest.raises(precheck.IngestError) as err:
        precheck.check([bundle("low", HUMAN[:5])], [], [], state, tiny_model)
    assert "add low to a separate store" in str(err.value) and "leave the model field empty" not in str(err.value)


def test_missing_essentials_name_empty_files_and_how_to_add_the_others():
    a, b = bundle("GSM1_a", missing=["cell positions"]), bundle("GSM2_b")
    a.inventory.notes.append("0-byte files ignored: GSM1_a_cells.csv.gz")
    for s, name in ((a, "GSM1_a"), (b, "GSM2_b")):
        object.__setattr__(s.candidate, "origin", Path("/data/flat"))
        object.__setattr__(s.candidate, "files", {"cells.parquet": Member(Path(f"/data/flat/{name}_cells.parquet")),
                                                  "x.h5": Member(Path(f"/data/flat/{name}_x.h5"))})
    with pytest.raises(precheck.IngestError) as err:
        precheck.check([a, b], [], [], precheck.StoreState(), None)
    text = str(err.value)
    assert "0-byte files ignored: GSM1_a_cells.csv.gz: a broken upload" in text
    assert "To add the other samples now, list only them in List of paths to samples: /data/flat/GSM2_b_*" in text


def test_a_bundle_and_a_spatialdata_object_alone_in_one_folder_are_flagged(caplog):
    xen = bundle("raw_section")
    object.__setattr__(xen.candidate, "origin", Path("/data/section1/xenium_output"))
    with caplog.at_level(logging.INFO):
        precheck.check([xen, sdata("1417642334", origin="/data/section1/combined_spatialdata_zarr")], [], [],
                       precheck.StoreState(), None)
    warned = warnings_of(caplog)
    assert len(warned) == 1 and "may be the same tissue" in warned[0] and "/data/section1 " in warned[0]
    caplog.clear()
    with caplog.at_level(logging.INFO):   # a download folder with several bundles beside the object: no warning
        precheck.check([bundle("a"), bundle("b"), sdata("z", origin="/data/z.zarr")], [], [], precheck.StoreState(), None)
    assert warnings_of(caplog) == []
