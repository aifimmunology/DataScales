import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from annizarr import AppConfig, Repo, append, convert
from annizarr.config import ChunkConfig, IOConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MARKERS = ["LYZ", "S100A8", "S100A9", "HBB", "HBA1", "HBA2", "IGKC", "IGHM", "IGLC2"]
GENES = [f"MT-{i}" for i in range(5)] + MARKERS + [f"GENE{i}" for i in range(586)]
LABELS = ["T cell", "B cell", "NK cell", "Monocyte"]
LABEL = "pbmc_clean_AIFI_L1_2024_label"
SAMPLES = {"SP0001": ("B1", "C1"), "SP0002": ("B1", "C1"), "SP0003": ("B1", "C2"), "SP0004": ("B2", "C2"),
           "SP0005": ("B2", "C3"), "SP0006": ("B2", "C3")}
CELLS = 120
CHUNK = 50


def sample_adata(sid: str, n_cells: int = CELLS, seed: int = 0, ambient: float = 0.0, batch: str = "B1",
                 chip: str = "C1", with_metrics: bool = True) -> ad.AnnData:
    """One sample with contract columns; `ambient` adds monocyte-marker counts to every cell."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, len(GENES), density=0.45, format="csr", random_state=seed, dtype=np.float64)
    X.data = rng.integers(1, 30, size=X.nnz).astype(np.float64)
    X = X.tolil()
    X[:, :5] = rng.integers(0, 4, size=(n_cells, 5))
    if ambient:
        X[:, 5:8] = rng.poisson(ambient, size=(n_cells, 3))
    X = sp.csr_matrix(X, dtype=np.uint32)
    X.eliminate_zeros()
    labels = pd.Categorical(rng.choice(LABELS, size=n_cells), categories=LABELS)
    doublets = pd.array(rng.choice([0, 1], size=n_cells, p=[0.9, 0.1]), dtype="Int64")
    obs = pd.DataFrame({
        "sample_id": pd.Categorical([sid] * n_cells),
        "batch_id": pd.Categorical([batch] * n_cells), "pool_id": pd.Categorical(["P1"] * n_cells),
        "chip_id": pd.Categorical([chip] * n_cells),
        "scrublet_score": rng.random(n_cells), "predicted_doublets": doublets,
        LABEL: labels, LABEL.replace("_label", "_prob"): rng.random(n_cells),
        "hto_category": pd.Categorical(rng.choice(["singlet", "doublet"], size=n_cells)),
    }, index=[f"{sid}:BC{i:05d}" for i in range(n_cells)])
    if with_metrics:
        total = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        mt = np.asarray(X[:, :5].sum(axis=1)).ravel()
        obs.insert(4, "n_genes_by_counts", pd.array(np.diff(X.indptr), dtype="Int64"))
        obs.insert(5, "total_counts", total)
        obs.insert(6, "pct_counts_mt", mt / total * 100)
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=GENES))


def cohort(n_samples: int = len(SAMPLES), ambient_sample: str | None = "SP0006", **kwargs) -> list[ad.AnnData]:
    out = []
    for i, (sid, (batch, chip)) in enumerate(list(SAMPLES.items())[:n_samples]):
        out.append(sample_adata(sid, seed=i, batch=batch, chip=chip, ambient=40.0 if sid == ambient_sample else 0.0,
                                **kwargs))
    return out


def cfg() -> AppConfig:
    return AppConfig(io=IOConfig(overwrite=True), chunks=ChunkConfig(row_chunk=CHUNK, cpus=2))


def build_store(path: Path, adatas: list[ad.AnnData], kind: str) -> str:
    """A plain zarr (`kind="zarr"`) or an icechunk repo with one commit per sample after the first."""
    staged = path.parent / f"{path.name}_staging"
    staged.mkdir(exist_ok=True)
    if kind == "zarr":
        convert(adatas[0], output=str(path), cfg=cfg())
        for i, adata in enumerate(adatas[1:]):
            convert(adata, output=str(staged / f"{i}.zarr"), cfg=cfg())
            append(str(path), cells=str(staged / f"{i}.zarr"), cfg=cfg())
        return str(path)
    repo = Repo.create(str(path))
    convert(adatas[0], output=repo, cfg=cfg(), message="seed")
    append_samples(str(path), adatas[1:], staged)
    return str(path)


def append_samples(path: str, adatas: list[ad.AnnData], staged: Path) -> None:
    repo = Repo(path)
    for i, adata in enumerate(adatas):
        out = staged / f"{adata.obs['sample_id'].iloc[0]}.zarr"
        convert(adata, output=str(out), cfg=cfg())
        append(repo, cells=str(out), cfg=cfg(), commit=False)
    if adatas:
        repo.commit("ingest", metadata={"pipeline": "ingest"})


@pytest.fixture
def results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture(params=["icechunk", "zarr"])
def kind(request):
    return request.param


@pytest.fixture
def store(tmp_path, kind):
    return build_store(tmp_path / ("store" if kind == "icechunk" else "store.zarr"), cohort(), kind)


def env(results: Path, **extra) -> dict:
    return {"RESULTS_DIR": str(results), "BAND_CELLS": "100", "EMBED_PER_SAMPLE": "80",
            **{k.upper(): str(v) for k, v in extra.items()}}


def spatial_table(n_samples: int = 5) -> ad.AnnData:
    """A Xenium-like cell table: no sample_id (each sample is a region), nucleus counts and centroids."""
    table = ad.concat(cohort(n_samples, with_metrics=False), merge="same")
    rng = np.random.default_rng(5)
    regions = "cells_" + table.obs["sample_id"].astype(str)
    obs = pd.DataFrame({
        "cell_id": np.arange(table.n_obs),
        "region": pd.Categorical(regions),
        "nucleus_count": rng.choice([0, 1, 1, 1, 2], size=table.n_obs),
        "transcript_counts": np.asarray(table.X.sum(axis=1)).ravel().astype(np.float64),
        "total_counts": np.asarray(table.X.sum(axis=1)).ravel().astype(np.float64) + 7,
    }, index=table.obs_names)
    out = ad.AnnData(table.X, obs=obs, var=table.var)
    out.obsm["spatial"] = rng.random((out.n_obs, 2)) * 1000
    out.uns["spatialdata_attrs"] = {"region": sorted(set(regions)), "region_key": "region", "instance_key": "cell_id"}
    return out


def write_spatialdata(root, table: ad.AnnData) -> None:
    from anndata.io import write_elem
    root.attrs["spatialdata_attrs"] = {"version": "0.2"}
    image = root.require_group("images").create_group("morphology")
    image.create_array("s0", data=np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    write_elem(root.require_group("tables"), "table", table)


def build_spatial_store(path: Path, kind: str) -> str:
    table = spatial_table()
    if kind == "zarr":
        import zarr
        write_spatialdata(zarr.open_group(str(path), mode="w"), table)
        return str(path)
    repo = Repo.create(str(path))
    write_spatialdata(repo.open_zarr("w"), table)
    repo.commit("spatial seed")
    return str(path)
