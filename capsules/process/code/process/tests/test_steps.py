import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr

import steps
from conftest import needs_gpu
from store import InputError


def test_spans_are_contiguous_runs():
    assert steps.spans(np.array([3, 4, 5, 9, 10, 20])) == [(3, 6), (9, 11), (20, 21)]
    assert steps.spans(np.array([0])) == [(0, 1)]


def test_subset_rows_compare_as_strings():
    obs = pd.DataFrame({"sample_id": pd.Categorical(["A", "B", "A", "C", "B"]), "n": [1, 2, 1, 3, 2]})
    assert steps.subset_rows(obs, "sample_id", ["A", "C"]).tolist() == [0, 2, 3]
    assert steps.subset_rows(obs, "n", ["2"]).tolist() == [1, 4]
    with pytest.raises(InputError, match="no cells have sample_id in \\['Z'\\]; it has A, B, C"):
        steps.subset_rows(obs, "sample_id", ["Z"])


def test_slice_rows_reads_spans():
    x = da.from_array(np.arange(40).reshape(20, 2), chunks=(5, 2))
    rows = np.array([1, 2, 3, 12, 13])
    out = steps.slice_rows(x, rows)
    assert np.array_equal(out.compute(), np.arange(40).reshape(20, 2)[rows])
    assert not np.isnan(out.chunks[0]).any()
    assert steps.slice_rows(x, None) is x


def test_full_rows_and_column_fill_cells_outside_the_subset():
    rows = np.array([1, 3])
    assert np.isnan(steps.full_rows(np.ones((2, 2), np.float32), rows, 4)[[0, 2]]).all()
    col = steps.full_column(pd.Series(pd.Categorical(["0", "1"])), rows, 4)
    assert list(col.isna()) == [True, False, True, False] and list(col.categories) == ["0", "1"]


def test_batch_keys_with_missing_values_are_an_error():
    obs = pd.DataFrame({"chip": pd.Categorical(["C0", "C1", "C1"]), "pool": pd.Categorical(["C0", "", None])})
    steps.check_batch(obs, ["chip"])
    assert list(obs["chip"].cat.categories) == ["C0", "C1"]
    with pytest.raises(InputError, match='2 of 3 cells have no pool \\(missing or ""\\)'):
        steps.check_batch(obs, ["pool"])


def test_read_obs_reads_optional_columns_it_has_and_needs_required_ones(stores):
    group = zarr.open_group(stores["ingest_no_sample"], mode="r")
    obs = steps.read_obs(group, ["sample_id"], ["batch"])
    assert list(obs.columns) == ["batch"] and obs.index[0] == "cell0"
    with pytest.raises(InputError, match="no column chip_id"):
        steps.read_obs(group, [], ["chip_id"])


@needs_gpu
def test_the_gpu_libraries_take_our_arguments():
    import inspect

    import rapids_singlecell as rsc
    import scvi

    def params(fn):
        return set(inspect.signature(fn).parameters)

    assert {"rng", "svd_solver", "n_comps"} <= params(rsc.pp.pca)
    assert {"rng", "use_rep", "n_pcs", "algorithm"} <= params(rsc.pp.neighbors)
    assert {"rng", "min_dist", "init_pos", "n_components"} <= params(rsc.tl.umap)
    assert {"rng", "resolution", "key_added", "n_iterations"} <= params(rsc.tl.leiden)
    assert {"rng", "key", "basis", "adjusted_basis", "max_iter_harmony", "theta", "alpha"} <= params(rsc.pp.harmony_integrate)
    assert {"max_epochs", "batch_size", "train_size", "early_stopping", "plan_kwargs"} <= params(scvi.model.SCVI.train)
