import anndata as ad
import h5py
import numpy as np
import pytest
from annizarr.sources import detect_format

import hise
from conftest import write_hise_h5


def test_detects_and_loads_hise(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09999-01.h5", "SP09999", 50, seed=3)
    assert detect_format(path) == "hise"
    adata = hise.read_sample(path)
    assert adata.shape == (50, 500)
    assert adata.X.format == "csr" and adata.X.dtype == np.uint16
    assert list(adata.var_names[:2]) == ["MT-0", "MT-1"]
    assert adata.obs["pbmc_sample_id"].iloc[0] == "SP09999"
    assert adata.obs["n_umis"].to_numpy().sum() == adata.X.sum()


@pytest.mark.parametrize("name,expected", [
    ("B092-P1_SP02425-01.h5", "SP02425"), ("SP02430.h5", "SP02430"), ("B092-P1_BMC01671-05.h5", "BMC01671"),
])
def test_sample_id(name, expected):
    assert hise.sample_id(name) == expected


def test_hise_variable_length_strings_and_adt_features(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09998-01.h5", "SP09998", 30, seed=4)
    with h5py.File(path, "r+") as f:
        m = f["matrix"]
        for key in ("barcodes", "features/name", "features/feature_type", "observations/cell_name", "observations/batch_id"):
            values = [v.decode() for v in m[key][:]]
            if key == "features/feature_type":
                values[-1] = "Antibody Capture"
            del m[key]
            m[key] = np.array(values, dtype=h5py.string_dtype())
    adata = hise.read_sample(path)
    assert adata.n_vars == 499 and "GENE494" not in adata.var_names
    assert adata.obs_names[0] == "BC00000-1"
    assert adata.obs["cell_name"].iloc[0] == "SP09998_0" and adata.obs["batch_id"].iloc[0] == "B092"


def test_plain_10x_and_h5ad_load_in_memory(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09997-01.h5", "SP09997", 40, seed=5)
    with h5py.File(path, "r+") as f:
        del f["matrix/observations"]
    assert detect_format(path) == "10x"
    tenx = hise.read_sample(path)
    assert tenx.shape == (40, 500) and tenx.X.format == "csr"

    h5ad = tmp_path / "B092-P1_SP09996-01.h5ad"
    ad.AnnData(tenx.X, obs=tenx.obs, var=tenx.var).write_h5ad(h5ad)
    back = hise.read_sample(h5ad)
    assert not back.isbacked and back.X.format == "csr"
    assert (back.X != tenx.X).nnz == 0


def test_new_pipeline_layout_maps_to_the_hise_columns(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09995-01.h5", "SP09995", 20, seed=6)
    renames = {"batch_id": "Batch", "pool_id": "Pool", "chip_id": "Chip", "cell_uuid": "uuid",
               "pbmc_sample_id": "sample_name", "n_mito_umis": "n_mitochondrial_counts", "hto_barcode": "hashtag"}
    with h5py.File(path, "r+") as f:
        obs = f["matrix/observations"]
        for old, new in renames.items():
            obs.move(old, new)
        obs["prop_mitochondrial_counts"] = np.zeros(20)
    adata = hise.read_sample(path)
    assert {"batch_id", "pool_id", "chip_id", "cell_uuid", "pbmc_sample_id", "n_mito_umis", "hto_barcode"} <= set(adata.obs)
    assert not set(renames.values()) & set(adata.obs)
    assert adata.obs["batch_id"].iloc[0] == "B092" and adata.obs["pbmc_sample_id"].iloc[0] == "SP09995"
