#!/usr/bin/env bash
# Runs both notebooks headless on tiny synthetic stores (AnnData .zarr, ingest-style .zarr, SpatialData .zarr,
# icechunk, a read-only icechunk mount with a writable origin) and
# checks that each saved object or commit holds the results and reads back. Needs the GPU capsule and internet
# (OmniPath). Usage: ./test_notebooks.sh [case ...]; WORK=<dir> keeps the stores and executed notebooks there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK="${WORK:-$(mktemp -d)}"
echo "Working in $WORK"
exec python -u - "$HERE" "$@" <<'PY'
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import zarr
from annizarr import Repo, convert

HERE, WORK = Path(sys.argv[1]), Path(os.environ["WORK"])
SCRNA, SPATIAL = HERE / "rapids_scrna_out_of_core_dask.ipynb", HERE / "rapids_spatial_atera_breast_cancer.ipynb"
N_CELLS, N_TYPES, N_GENES = 6000, 8, 2500
DROPPED = 60  # cells with no counts, which the spatial QC removes


def cells(nb, text):
    return [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code" and text in "".join(c["source"])]


def gene_names():
    import omnipath as op

    scope = {}
    exec("".join(cells(SPATIAL, "CELL_TYPE_MARKERS: dict")[0]["source"]).split("celltype_net")[0], scope)
    markers = scope["CELL_TYPE_MARKERS"]
    lr = op.interactions.import_intercell_network(transmitter_params={"categories": "ligand"},
                                                  receiver_params={"categories": "receptor"})
    pairs = lr[["genesymbol_intercell_source", "genesymbol_intercell_target"]].drop_duplicates().head(80).values
    progeny = op.requests.Annotations.get(resources="PROGENy")["genesymbol"].drop_duplicates().head(800)
    genes = list(dict.fromkeys([g for gs in markers.values() for g in gs] + [g for p in pairs for g in p]
                               + [g for g in progeny if "_" not in g]))
    genes += [f"FILLER{i}" for i in range(N_GENES - len(genes))]
    return markers, pairs, genes


def synthetic():
    """Marker-driven cell types in spatial blobs, ligands per type with broadly expressed receptors."""
    rng = np.random.default_rng(0)
    markers, pairs, genes = gene_names()
    types = ["Invasive_Tumor", "DCIS_Basal_like", "T_Cell", "Macrophage", "CAF_Neoplastic", "Endothelial",
             "B_Cell", "Pericyte"][:N_TYPES]
    col = {g: i for i, g in enumerate(genes)}
    profile = np.full((N_TYPES, len(genes)), 0.3)
    for t, name in enumerate(types):
        profile[t, [col[g] for g in markers[name]]] += 6
    for i, (lig, rec) in enumerate(pairs):
        profile[i % N_TYPES, col[lig]] += 4
        profile[:, col[rec]] += 2
    centers = rng.uniform(0, 1000, (N_TYPES, 2))
    blob = rng.integers(0, N_TYPES, N_CELLS)
    cell_type = np.where(rng.random(N_CELLS) < 0.8, blob, rng.integers(0, N_TYPES, N_CELLS))
    xy = centers[blob] + rng.normal(0, 60, (N_CELLS, 2))
    counts = rng.poisson(profile[cell_type]).astype(np.int32)
    counts[:DROPPED] = 0
    sample = rng.integers(0, 3, N_CELLS)
    chip = np.array(["C0", "C1", ""])[np.where(rng.random(N_CELLS) < 0.02, 2, sample % 2)]
    obs = pd.DataFrame({
        "sample_id": pd.Categorical([f"SP{s}" for s in sample]),
        "chip_id": pd.Categorical(chip),
        "nucleus_count": np.ones(N_CELLS, dtype=np.int32),
        "region": pd.Categorical(["cells"] * N_CELLS),
        "cell_id": np.arange(N_CELLS),
    }, index=[f"cell{i}" for i in range(N_CELLS)])
    return ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes), obsm={"spatial": xy})


def write_spatialdata(table, path):
    import dask.dataframe as dd
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel

    rng = np.random.default_rng(1)
    points = pd.DataFrame({"x": rng.uniform(0, 1000, 500), "y": rng.uniform(0, 1000, 500),
                           "gene": pd.Categorical(rng.choice(table.var_names[:20], 500))})
    sd.SpatialData(
        images={"image": Image2DModel.parse(rng.integers(0, 255, (1, 64, 64), dtype=np.uint8), dims=("c", "y", "x"))},
        labels={"labels": Labels2DModel.parse(rng.integers(0, 5, (64, 64), dtype=np.uint8), dims=("y", "x"))},
        points={"transcripts": PointsModel.parse(dd.from_pandas(points, npartitions=1))},
        shapes={"cells": ShapesModel.parse(table.obsm["spatial"], geometry=0, radius=np.ones(table.n_obs))},
        tables={"table": TableModel.parse(table, region="cells", region_key="region", instance_key="cell_id")},
    ).write(path)


def ingest_like(table, sample_id):
    """As anndata_ingest writes it: uint32 CSR counts, categorical obs, "" for missing strings, no layers."""
    rng = np.random.default_rng(2)
    missing = rng.random(table.n_obs) < 0.05
    obs = pd.DataFrame({"chemistry": np.where(missing, "", "v3"),
                        "batch": pd.Categorical(np.where(missing, "", "B1"))}, index=table.obs_names)
    if sample_id:
        obs.insert(0, "sample_id", pd.Categorical(np.where(missing, "", table.obs["sample_id"].astype(str))))
    return ad.AnnData(X=sp.csr_matrix(table.X, dtype=np.uint32), obs=obs, var=table.var.copy())


def make_stores():
    stores = WORK / "stores"
    if (stores / "done").exists():
        return stores
    stores.mkdir(parents=True, exist_ok=True)
    table = synthetic()
    adata = table.copy()
    del adata.obsm["spatial"]
    adata.write_zarr(stores / "pbmc.zarr")
    write_spatialdata(table, stores / "tissue.zarr")
    convert(ingest_like(table, sample_id=True), output=str(stores / "ingest.zarr"))
    convert(ingest_like(table, sample_id=False), output=str(stores / "ingest_no_sample.zarr"))
    Repo.init(stores / "pbmc.zarr", stores / "pbmc_icechunk")
    Repo.init(stores / "tissue.zarr", stores / "tissue_icechunk")
    Repo.init(stores / "ingest.zarr", stores / "origin_icechunk")
    shutil.copytree(stores / "origin_icechunk", stores / "mount_icechunk")
    read_only = ["pbmc.zarr", "tissue.zarr", "ingest.zarr", "ingest_no_sample.zarr", "mount_icechunk"]
    subprocess.run(["chmod", "-R", "a-w", *(stores / s for s in read_only)], check=True)
    (stores / "done").touch()
    return stores


def toggle(source, name):
    """Uncomments the block of `# ` lines under a `# <name>...` marker."""
    lines, out = source.splitlines(keepends=True), []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if line.strip().lstrip("# -").startswith(name):
            indent = line[:len(line) - len(line.lstrip())]
            while i < len(lines) and lines[i].startswith(indent + "# "):
                out.append(lines[i].replace(indent + "# ", indent, 1))
                i += 1
    return "".join(out)


def set_params(source, params):
    for key, value in params.items():
        source, n = re.subn(rf"^{key} = .*$", lambda _: f"{key} = {value!r}", source, flags=re.M)
        assert n == 1, f"{key} is not a setting"
    return source


def run(name, nb, params, enable=(), before_save=None, expect_error=None, expect_output=()):
    doc = json.loads(nb.read_text())
    for c in doc["cells"]:
        if c["cell_type"] == "code":
            source = "".join(c["source"])
            for marker in enable:
                source = toggle(source, marker)
            if "parameters" in c["metadata"].get("tags", []):
                source = set_params(source, params)
            c["source"] = source
    if before_save:
        save = next(i for i, c in enumerate(doc["cells"]) if "".join(c["source"]).startswith("if SAVE"))
        doc["cells"].insert(save, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                                   "source": before_save})
    path = WORK / "notebooks" / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(doc))
    cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=3600", str(path)]
    if expect_error:
        cmd.insert(-1, "--allow-errors")
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"{path} failed:\n{done.stderr[-3000:]}")
    text = json.dumps([c.get("outputs", []) for c in json.loads(path.read_text())["cells"]])
    for expected in filter(None, [expect_error, *expect_output]):
        assert expected in text, f"{name}: expected {expected!r} in the output"


def tree_hash(path, parts=None):
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        rel = f.relative_to(path)
        if f.is_file() and (parts is None or rel.parts[0] in parts):
            h.update(str(rel).encode() + f.read_bytes())
    return h.hexdigest()


def check_scrna(group, source, harmony=False):
    a = ad.io.read_elem(group)
    expected = {"X_pca", "X_umap"} | ({"X_pca_harmony"} if harmony else set())
    assert expected <= set(a.obsm), a.obsm.keys()
    assert a.obs["leiden"].notna().all() and a.var["highly_variable"].sum() == 2000
    assert a.uns["process"]["basis"] == ("X_pca_harmony" if harmony else "X_pca")
    assert group["layers/lognorm"].attrs["target_sum"] == 1e4
    ref = sc.AnnData(X=source.X.astype(np.float32))
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    assert (a.X != source.X).nnz == 0, "X changed"
    assert np.allclose(a.layers["lognorm"].toarray(), ref.X.toarray(), rtol=1e-5, atol=1e-5), "lognorm differs from scanpy"


def check_spatial(table):
    removed = table.obs_names[:DROPPED]
    for col in ("spatial_cell_type", "spatial_leiden", "spatial_niche"):
        assert table.obs.loc[removed, col].isna().all() and table.obs[col].drop(removed).notna().all(), col
    for key in ("X_pca_spatial", "X_umap_spatial"):
        values = pd.DataFrame(table.obsm[key], index=table.obs_names)
        assert values.loc[removed].isna().all().all() and values.drop(removed).notna().all().all(), key
    assert table.uns["spatial_process"]["n_cells_kept"] == table.n_obs - DROPPED


def check_sdata_copy(src, out):
    import spatialdata as sd

    sdata = sd.read_zarr(out)
    for part in ("images", "labels", "points", "shapes"):
        assert tree_hash(src, [part]) == tree_hash(out, [part]), f"{part} changed in the copy"
    return sdata.tables["table"]


def case_helpers_identical(stores):
    scrna, spatial = (cells(nb, "def save_commit") for nb in (SCRNA, SPATIAL))
    assert len(scrna) == len(spatial) == 1 and scrna[0]["source"] == spatial[0]["source"], "the helper cells differ"


def case_scrna_zarr(stores):
    src, before = stores / "pbmc.zarr", tree_hash(stores / "pbmc.zarr")
    results = WORK / "results_scrna_zarr"
    params = {"STORE": str(src), "SAVE": True, "RESULTS": str(results)}
    run("scrna_zarr", SCRNA, params)
    out = results / "pbmc.processed.zarr"
    check_scrna(zarr.open_group(out, mode="r"), ad.read_zarr(src))
    assert (results / "umap.png").exists() and (results / "cluster_summary.csv").exists()
    first = tree_hash(out)
    run("scrna_zarr_again", SCRNA, params)
    assert tree_hash(out) == first, "an existing output was overwritten"
    check_scrna(zarr.open_group(results / "pbmc.processed-1.zarr", mode="r"), ad.read_zarr(src))
    assert tree_hash(src) == before, "the input changed"


def case_scrna_ingest(stores):
    src = stores / "ingest.zarr"
    results = WORK / "results_scrna_ingest"
    run("scrna_ingest", SCRNA, {"STORE": str(src), "SAVE": True, "RESULTS": str(results)})
    check_scrna(zarr.open_group(results / "ingest.processed.zarr", mode="r"), ad.read_zarr(src))
    assert "SP0" in (results / "cluster_summary.csv").read_text()


def case_scrna_ingest_no_sample(stores):
    src = stores / "ingest_no_sample.zarr"
    results = WORK / "results_scrna_no_sample"
    params = {"STORE": str(src), "SAVE": True, "RESULTS": str(results), "OUTPUT": str(results / "out")}
    run("scrna_no_sample", SCRNA, params)
    run("scrna_no_sample_again", SCRNA, params)
    for name in ("out.zarr", "out-1.zarr"):
        check_scrna(zarr.open_group(results / name, mode="r"), ad.read_zarr(src))
    assert not (results / "out").exists()


def case_scrna_sdata(stores):
    src = stores / "tissue.zarr"
    results = WORK / "results_scrna_sdata"
    run("scrna_sdata", SCRNA, {"STORE": str(src), "SAVE": True, "RESULTS": str(results)})
    out = results / "tissue.processed.zarr"
    check_sdata_copy(src, out)
    check_scrna(zarr.open_group(out, mode="r")["tables/table"], ad.io.read_elem(zarr.open_group(src, mode="r")["tables/table"]))


def case_scrna_icechunk(stores):
    src = stores / "pbmc_icechunk"
    n = len(Repo(src).log())
    run("scrna_icechunk", SCRNA, {"STORE": str(src), "SAVE": True, "RESULTS": str(WORK / "results_scrna_ic"),
                                  "BATCH_KEYS": ["sample_id"], "BRANCH": "main", "ORIGIN": None},
        enable=("icechunk", "harmony"))
    repo = Repo(src)
    assert len(repo.log()) == n + 1, "expected one commit"
    check_scrna(repo.open_zarr("r"), ad.read_zarr(stores / "pbmc.zarr"), harmony=True)


def case_branch_moved(stores):
    src = stores / "pbmc_icechunk"
    tip = Repo(src).log()[0].id
    moved = ("_other = Repo(STORE, branch=BRANCH)\n_other.open_zarr('w').attrs['touched'] = str(os.urandom(8))\n"
             "_other.commit('moved meanwhile')")
    run("scrna_branch_moved", SCRNA, {"STORE": str(src), "SAVE": True, "RESULTS": str(WORK / "results_moved"),
                                      "BATCH_KEYS": ["chip_id"], "BRANCH": "main", "ORIGIN": None},
        enable=("icechunk",), before_save=moved, expect_error="moved since the store was opened",
        expect_output=["cells have no chip_id"])
    log = Repo(src).log()
    assert log[1].id == tip and log[0].message == "moved meanwhile", "a process commit landed"


def case_icechunk_origin(stores):
    mount, origin = stores / "mount_icechunk", stores / "origin_icechunk"
    before, n = tree_hash(mount), len(Repo(origin).log())
    params = {"STORE": str(mount), "SAVE": True, "RESULTS": str(WORK / "results_origin"), "BRANCH": "main",
              "ORIGIN": str(origin)}
    run("scrna_origin", SCRNA, params, enable=("icechunk",))
    run("scrna_origin_again", SCRNA, params, enable=("icechunk",))
    log = Repo(origin).log()
    assert len(log) == n + 2, "expected two commits on the origin"
    assert log[0].metadata["source_snapshot"] == log[1].id
    assert tree_hash(mount) == before, "the mount changed"
    check_scrna(Repo(origin).open_zarr("r"), ad.read_zarr(stores / "ingest.zarr"))


def case_spatial_sdata(stores):
    src = stores / "tissue.zarr"
    results = WORK / "results_spatial_sdata"
    run("spatial_sdata", SPATIAL, {"STORE": str(src), "SAVE": True, "RESULTS": str(results)})
    check_spatial(check_sdata_copy(src, results / "tissue.processed.zarr"))


def case_spatial_icechunk(stores):
    src = stores / "tissue_icechunk"
    n = len(Repo(src).log())
    run("spatial_icechunk", SPATIAL, {"STORE": str(src), "SAVE": True, "RESULTS": str(WORK / "results_spatial_ic"),
                                      "BRANCH": "main", "ORIGIN": None}, enable=("icechunk",))
    repo = Repo(src)
    assert len(repo.log()) == n + 1, "expected one commit"
    check_spatial(ad.io.read_elem(repo.open_zarr("r")["tables/table"]))


CASES = {k.removeprefix("case_"): v for k, v in dict(globals()).items() if k.startswith("case_")}
stores = make_stores()
failed = []
for name in sys.argv[2:] or CASES:
    try:
        CASES[name](stores)
        print(f"PASS {name}")
    except Exception as exc:
        failed.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
sys.exit(1 if failed else 0)
PY
