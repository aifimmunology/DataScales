import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import annotate
from conftest import HUMAN, counts

MODELS = Path("/scratch/spatial_dev/robustness/downloads/celltypist_models")


def chunk(n: int, seed: int, genes=HUMAN) -> ad.AnnData:
    return ad.AnnData(counts(n, len(genes), seed), obs=pd.DataFrame(index=[f"c{seed}_{i}" for i in range(n)]),
                      var=pd.DataFrame(index=list(genes)))


def test_qc_metrics_are_per_cell_so_chunks_agree():
    whole = chunk(30, 1)
    qc = annotate.qc_metrics(whole)
    parts = pd.concat([annotate.qc_metrics(whole[:12]), annotate.qc_metrics(whole[12:])])
    assert list(qc.columns) == list(annotate.QC_COLUMNS)
    pd.testing.assert_frame_equal(qc, parts)
    assert (qc["pct_counts_mt"] > 0).any()


def test_labels_are_per_cell_so_chunks_agree(tiny_model):
    whole = chunk(40, 3)
    labels = annotate.celltypist_labels(whole, tiny_model)
    parts = pd.concat([annotate.celltypist_labels(whole[:15], tiny_model),
                       annotate.celltypist_labels(whole[15:], tiny_model)])
    label, prob = annotate.label_columns(tiny_model)
    assert list(labels.columns) == [label, prob]
    assert (labels[label].astype(str) == parts[label].astype(str)).all()
    np.testing.assert_allclose(labels[prob], parts[prob], rtol=1e-5)
    assert set(labels[label]) <= {"Epithelial", "Immune"} and labels[prob].between(0, 1).all()


def test_model_overlap_and_thresholds(tiny_model):
    assert annotate.model_overlap(tiny_model, HUMAN) == (len(HUMAN), len(HUMAN))
    assert annotate.model_overlap(tiny_model, HUMAN[:5] + ["OTHER"]) == (5, len(HUMAN))
    assert annotate.overlap_floor(100) == 10 and annotate.overlap_floor(6639) == 332
    shared, n, note = annotate.check_overlap(tiny_model, "s1", HUMAN)
    assert (shared, n, note) == (len(HUMAN), len(HUMAN), None)
    shared, n, note = annotate.check_overlap(tiny_model, "s1", HUMAN[:15])
    assert shared == 15 and "s1: 15 of the model's 42 genes (36%)" in note
    with pytest.raises(annotate.AnnotateError, match="shares 0 of its 42 genes with s2's panel.*another species"):
        annotate.check_overlap(tiny_model, "s2", [g.lower() for g in HUMAN])
    with pytest.raises(annotate.AnnotateError, match="shares 9 of its 42.*leave the model field empty"):
        annotate.check_overlap(tiny_model, "s3", HUMAN[:9])
    with pytest.raises(annotate.AnnotateError, match="This store labels every sample with it, so add s3 to a separate"):
        annotate.check_overlap(tiny_model, "s3", HUMAN[:9], store_model=True)


def test_bad_or_missing_model_files_are_one_clear_error(tmp_path):
    with pytest.raises(annotate.AnnotateError, match="not found"):
        annotate.load_model(tmp_path / "none.pkl")
    junk = tmp_path / "junk.pkl"
    junk.write_bytes(b"not a model")
    with pytest.raises(annotate.AnnotateError, match="isn't a CellTypist model") as err:
        annotate.load_model(junk)
    assert "🛑" not in str(err.value) and "celltypist.train" in str(err.value) and "'Model'" not in str(err.value)
    import pickle
    other = tmp_path / "dict.pkl"
    other.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(annotate.AnnotateError, match="isn't a CellTypist model"):
        annotate.load_model(other)


def test_a_celltypist_failure_is_a_user_error(tiny_model):
    with pytest.raises(annotate.AnnotateError, match="could not label"):
        annotate.celltypist_labels(chunk(5, 2, [f"x{i}" for i in range(42)]), tiny_model)


@pytest.mark.skipif(not MODELS.is_dir(), reason="public CellTypist models not downloaded")
def test_public_mouse_model_on_a_human_panel_is_refused():
    model = MODELS / "Adult_Mouse_Gut.pkl"
    with pytest.raises(annotate.AnnotateError, match=r"shares \d+ of its \d+ genes with human"):
        annotate.check_overlap(model, "human", HUMAN + ["CD3E", "EPCAM", "PTPRC"])
    shared, n = annotate.model_overlap(MODELS / "Immune_All_Low.pkl", ["CD3E", "PTPRC", "MS4A1"])
    assert shared == 3 and n > 1000


def test_importing_celltypist_keeps_the_warning_filters():
    code = ("import sys, warnings; sys.path.insert(0, sys.argv[1]); import annotate; "
            "before = list(warnings.filters); annotate._celltypist(); import celltypist.classifier; "
            "print(warnings.filters == before)")
    done = subprocess.run([sys.executable, "-c", code, str(Path(annotate.__file__).parent)], capture_output=True,
                          text=True)
    assert done.stdout.strip().splitlines()[-1] == "True", done.stderr[-2000:]


def test_zero_count_cells_raise_no_warning(tiny_model):
    import warnings as w
    adata = chunk(10, 4)
    adata.X[:3] = 0
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        annotate.celltypist_labels(adata, tiny_model)
    assert not [c for c in caught if "zero counts" in str(c.message)]
