import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import genes


def test_xenium_var_is_kept_by_symbol():
    var = pd.DataFrame({"gene_ids": ["ENSG1", "ENSG2"], "feature_types": ["Gene Expression"] * 2, "genome": "x"},
                       index=["ABCC11", "ACTA2"])
    keep, out = genes.symbol_var(var, "s")
    assert keep.all() and list(out.index) == ["ABCC11", "ACTA2"]
    assert list(out.columns) == ["gene_ids", "feature_types"] and list(out["gene_ids"]) == ["ENSG1", "ENSG2"]


def test_atera_var_is_reindexed_to_symbols_and_controls_dropped():
    var = pd.DataFrame({"feature_name": ["ATOX1", "NegControlCodeword_0035", "C2CD4B"],
                        "feature_type": ["Gene Expression", "Negative Control Codeword", "Gene Expression"]},
                       index=pd.Index(["ENSG00000177556", "NegControlCodeword_0035", "ENSG00000205502"], name="feature_id"))
    keep, out = genes.symbol_var(var, "s")
    assert list(keep) == [True, False, True]
    assert list(out.index) == ["ATOX1", "C2CD4B"] and list(out["gene_ids"]) == ["ENSG00000177556", "ENSG00000205502"]


def test_duplicates_made_unique_and_ensembl_without_symbols_refused():
    _, out = genes.symbol_var(pd.DataFrame(index=["X", "Y", "X"]), "s")
    assert list(out.index) == ["X", "Y", "X-1"]
    with pytest.raises(genes.PanelError, match="Ensembl"):
        genes.symbol_var(pd.DataFrame(index=["ENSG00000177556", "ENSG00000205502"]), "s")
    _, out = genes.symbol_var(pd.DataFrame({"gene_symbols": ["A", "B"]}, index=["ENSG01", "ENSG02"]), "s")
    assert list(out.index) == ["A", "B"]


def test_species():
    assert genes.species(pd.DataFrame({"gene_ids": ["ENSG01", "ENST02"]})) == "human"
    assert genes.species(pd.DataFrame({"gene_ids": ["ENSMUSG01"]})) == "mouse"
    assert genes.species(pd.DataFrame({"gene_ids": ["", "x"]})) is None
    assert genes.species(pd.DataFrame(index=["A"])) is None


def test_fit_counts_zero_fills_and_drops():
    x = sp.csr_matrix(np.array([[1, 2, 3]], dtype=np.int32))
    fitted = genes.fit_counts(x, ["A", "B", "C"], ["C", "Z", "A"], np.dtype("float32"))
    assert fitted.dtype == np.float32 and fitted.toarray().tolist() == [[3, 0, 1]]
    assert genes.fit_note(genes.panel_diff(["C", "Z", "A"], ["A", "B", "C"])) == \
        "fitted to the store's genes: 1 zero-filled, 1 dropped"


def test_as_dtype_refuses_overflow_and_counts_check():
    with pytest.raises(genes.PanelError, match="don't fit"):
        genes.as_dtype(sp.csr_matrix(np.array([[70000.0]])), np.dtype("int16"))
    assert genes.holds_counts(sp.csr_matrix(np.array([[1.0, 2.0]])))
    assert not genes.holds_counts(sp.csr_matrix(np.array([[0.5]])))
