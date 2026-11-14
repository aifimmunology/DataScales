import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import logs  # noqa: E402

logs.setup()   # the run's logging: library noise silenced, as in main.main

BUNDLES = Path(os.environ.get("SPATIAL_TEST_DATA", "/scratch/spatial_dev/data"))
HUMAN = [f"GENE{i}" for i in range(40)] + ["MT-CO1", "MT-ND1"]


def bundle(name: str, zipped: bool = False) -> Path:
    path = BUNDLES / ("zip" if zipped else "extract") / (f"{name}_outs.zip" if zipped else name)
    if not path.exists():
        pytest.skip(f"public test bundle {path} not present (set SPATIAL_TEST_DATA)")
    return path


def counts(n_cells: int, n_genes: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    x = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=seed, dtype=np.float32)
    x.data = rng.integers(1, 20, size=x.nnz).astype(np.float32)
    return x


def make_table(n_cells: int, genes: list[str], seed: int, region: str, *, ensembl: bool = False,
               human: bool = True, controls: int = 0) -> ad.AnnData:
    names = list(genes) + [f"NegControlProbe_{i}" for i in range(controls)]
    prefix = "ENSG" if human else "ENSMUSG"
    ids = [f"{prefix}{i:011d}" for i in range(len(names))]
    var = pd.DataFrame({"gene_ids": ids, "feature_types": ["Gene Expression"] * len(genes) + ["Negative Control Probe"] * controls,
                        "feature_name": names}, index=pd.Index(ids if ensembl else names))
    if not ensembl:
        var = var.drop(columns="feature_name")
    cell_ids = [f"cell{seed}_{i}" for i in range(n_cells)]
    obs = pd.DataFrame({"cell_id": cell_ids, "cell_area": np.linspace(10, 20, n_cells),
                        "transcript_counts": np.arange(n_cells), "region": pd.Categorical([region] * n_cells),
                        "leiden": pd.Categorical(["0"] * n_cells)},
                       index=pd.Index([f"bc{seed}_{i}" for i in range(n_cells)]))
    table = ad.AnnData(counts(n_cells, len(names), seed), obs=obs, var=var)
    table.obsm["spatial"] = np.random.default_rng(seed).uniform(0, 100, size=(n_cells, 2))
    return table


def make_elements(tag: str, cell_ids: list[str], system: str = "global", sep: str = "-") -> dict:
    """Image, labels, cell/nucleus shapes and points of one synthetic sample, named `<name><sep><tag>`
    (or plain names when tag is empty), in coordinate system `system`."""
    import geopandas as gpd
    from shapely.geometry import Point
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel
    from spatialdata.transformations import Identity

    t = {system: Identity()}
    name = (lambda n: f"{n}{sep}{tag}") if tag else (lambda n: n)
    rng = np.random.default_rng(len(cell_ids))
    centers = rng.uniform(0, 60, (len(cell_ids), 2))

    def circles(radius: float):
        return gpd.GeoDataFrame({"geometry": [Point(x, y).buffer(radius) for x, y in centers]},
                                index=pd.Index(cell_ids, name="cell_id"))

    points = pd.DataFrame({"x": rng.uniform(0, 60, 50), "y": rng.uniform(0, 60, 50),
                           "feature_name": pd.Categorical(rng.choice(HUMAN[:5], 50))})
    return {
        "images": {name("morphology_focus"): Image2DModel.parse(rng.integers(0, 255, (1, 64, 64)).astype(np.uint16),
                                                                dims=("c", "y", "x"), transformations=t)},
        "labels": {name("cell_labels"): Labels2DModel.parse(rng.integers(0, 5, (64, 64)).astype(np.uint32),
                                                            dims=("y", "x"), transformations=t)},
        "shapes": {name("cell_boundaries"): ShapesModel.parse(circles(2), transformations=t),
                   name("nucleus_boundaries"): ShapesModel.parse(circles(1), transformations=t)},
        "points": {name("transcripts"): PointsModel.parse(points, coordinates={"x": "x", "y": "y"},
                                                          feature_key="feature_name", transformations=t)},
    }


def write_spatialdata(path: Path, samples: dict[str, dict], *, sample_column: bool = False, shared_system: bool = False,
                      one_table: bool = True) -> Path:
    """A SpatialData with one or more synthetic samples. `samples` maps a tag to
    {"genes", "n", "seed", ...make_table kwargs}. Each sample's elements carry its tag; with several
    samples each gets its own coordinate system `global-<tag>` unless `shared_system`."""
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    single = len(samples) == 1
    elements: dict[str, dict] = {k: {} for k in ("images", "labels", "shapes", "points")}
    tables, regions = [], []
    for tag, spec in samples.items():
        spec = dict(spec)
        n, seed, gene_list = spec.pop("n"), spec.pop("seed"), spec.pop("genes")
        region = "cell_boundaries" if single else f"cell_boundaries-{tag}"
        table = make_table(n, gene_list, seed, region, **spec)
        if sample_column:
            table.obs["sample_id"] = tag
        system = "global" if single or shared_system else f"global-{tag}"
        for kind, items in make_elements("" if single else tag, list(table.obs["cell_id"]), system).items():
            elements[kind].update(items)
        tables.append(table)
        regions.append(region)
    if one_table:
        joined = ad.concat(tables, join="inner", merge="first") if len(tables) > 1 else tables[0]
        joined.obs_names_make_unique()
        joined.obs["region"] = pd.Categorical(joined.obs["region"].astype(str))
        out_tables = {"table": TableModel.parse(joined, region=regions if len(regions) > 1 else regions[0],
                                                region_key="region", instance_key="cell_id")}
    else:
        out_tables = {f"table_{tag}": TableModel.parse(t, region=r, region_key="region", instance_key="cell_id")
                      for tag, t, r in zip(samples, tables, regions)}
    SpatialData(**elements, tables=out_tables).write(path)
    return path


@pytest.fixture
def env(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return {"RESULTS_DIR": str(results), "SCRATCH_DIR": str(tmp_path / "scratch"), "WRITE_CPUS": "2",
            "N_JOBS": "1", "CREATE_ROOTS": str(tmp_path)}


@pytest.fixture
def run_capsule(env):
    """Runs the capsule in process with app-panel style positional values; returns the run report."""
    import config
    import main

    def _run(*values: str, **extra: str) -> dict:
        return main.run(config.load_settings(list(values), {**env, **extra}))

    return _run


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    """A CellTypist model with two labels, trained on synthetic counts over HUMAN genes."""
    import annotate

    celltypist, _ = annotate._celltypist()
    x = sp.vstack([counts(60, len(HUMAN), seed) for seed in range(2)]).astype(np.float32)
    x = sp.csr_matrix(np.log1p(x.multiply(1e4 / np.maximum(x.sum(axis=1), 1)).toarray()))
    train = ad.AnnData(x, var=pd.DataFrame(index=HUMAN))
    path = tmp_path_factory.mktemp("models") / "tiny_panel.pkl"
    labels = np.resize(["Epithelial", "Immune"], train.n_obs)
    celltypist.train(train, labels=labels, use_SGD=True, max_iter=50, check_expression=False).write(str(path))
    return path
