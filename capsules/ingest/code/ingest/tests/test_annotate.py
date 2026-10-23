import anndata as ad
import numpy as np
import scipy.sparse as sp
import scrublet

import annotate
import contract


def test_scrublet_without_threshold_keeps_scores_and_leaves_calls_na(monkeypatch):
    n = 50
    monkeypatch.setattr(scrublet.Scrublet, "scrub_doublets", lambda self, **kw: (np.linspace(0, 1, n), None))
    counts = sp.random(n, 20, density=0.3, format="csr", random_state=0)
    counts.data = np.ceil(counts.data * 10)
    adata = ad.AnnData(counts)
    annotate.run_scrublet(adata)
    assert adata.obs["scrublet_score"].notna().all()
    assert adata.obs["predicted_doublets"].isna().all()
    schema = {"scrublet_score": contract.FLOAT, "predicted_doublets": contract.INT}
    projected = contract.project(adata, schema, None)
    assert projected.obs["predicted_doublets"].isna().all()
    assert str(projected.obs["predicted_doublets"].dtype) == "Int64"
