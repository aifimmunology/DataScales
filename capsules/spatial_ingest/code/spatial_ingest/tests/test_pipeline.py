import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import zarr
from anndata.io import read_elem, write_elem

import annotate
import joint
import main
import precheck
import store as stores
from conftest import HUMAN, write_spatialdata

READBACK = """
import json, sys, warnings
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import spatialdata as sd
from spatialdata import join_spatialelement_table
from spatialdata.transformations import get_transformation
s = sd.read_zarr(sys.argv[1])
out = {"systems": sorted(s.coordinate_systems), "tables": {k: list(t.shape) for k, t in s.tables.items()},
       "elements": {k: sorted(getattr(s, k)) for k in ("images", "labels", "shapes", "points")}}
for sid in json.loads(sys.argv[2]):
    region = s.tables[f"{sid}_table"].uns["spatialdata_attrs"]["region"]
    el, t = join_spatialelement_table(s, spatial_element_names=region, table_name=f"{sid}_table", how="inner")
    out[sid] = {"joined": [len(el[region]), t.n_obs],
                "systems": sorted({cs for k in ("images", "labels", "shapes", "points")
                                   for n, e in getattr(s, k).items() if n.startswith(sid + "_")
                                   for cs in get_transformation(e, get_all=True)})}
j = s.tables["table"]
out["joint"] = {"names": list(j.obs_names[:2]), "obs": list(j.obs.columns), "var": list(j.var.columns),
                "uns": sorted(j.uns), "obsm": list(j.obsm)}
print(json.dumps(out))
"""


def readback(store: Path, sids: list[str]) -> dict:
    """The store read in a fresh process with stock spatialdata."""
    done = subprocess.run([sys.executable, "-c", READBACK, str(store), json.dumps(sids)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def snapshot(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*"))
            if p.is_file()}


@pytest.fixture
def inputs_dir(tmp_path):
    d = tmp_path / "inputs"
    d.mkdir()
    write_spatialdata(d / "single.zarr", {"x": {"genes": HUMAN, "n": 25, "seed": 0}})
    write_spatialdata(d / "pair.zarr", {"a": {"genes": HUMAN, "n": 20, "seed": 1},
                                        "b": {"genes": HUMAN[:30] + ["EXTRA1"], "n": 15, "seed": 2}})
    return d


def joint_obs(store: Path) -> pd.DataFrame:
    return read_elem(zarr.open_group(store / "tables" / "table" / "obs", mode="r"))


def test_existing_objects_build_one_joint_store(tmp_path, inputs_dir, run_capsule, caplog):
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(inputs_dir))
    assert report["appended_samples"] == ["pair_a", "pair_b", "single"] and report["n_cells"] == 60
    assert report["fitted"] == {"single": "12 new gene(s) added to the joint table (now 42)"}
    assert report["joint_genes"] == len(HUMAN) and report["kinds"]["single"] == "spatialdata"
    assert "GENE PANELS DIFFER" in caplog.text
    back = readback(store, ["pair_a", "pair_b", "single"])
    assert back["systems"] == ["pair_a", "pair_b", "single"]
    assert back["tables"]["table"] == [60, len(HUMAN)]
    assert back["tables"]["pair_b_table"] == [15, 30] and back["tables"]["single_table"] == [25, len(HUMAN)]
    assert back["elements"]["shapes"] == sorted(f"{s}_{n}" for s in ("pair_a", "pair_b", "single")
                                                for n in ("cell_boundaries", "nucleus_boundaries"))
    for sid, n in {"pair_a": 20, "pair_b": 15, "single": 25}.items():
        assert back[sid]["joined"] == [n, n] and back[sid]["systems"] == [sid]
    assert back["joint"]["names"] == ["pair_a:cell1_0", "pair_a:cell1_1"]
    assert back["joint"]["obs"][:5] == list(joint.LEADING) and "leiden" not in back["joint"]["obs"]
    assert back["joint"]["var"] == ["gene_ids", "feature_types", "n_samples_measured"] and back["joint"]["obsm"] == []
    summary = pd.read_csv(Path(report["settings"]["results_dir"]) / "spatial_ingest_summary.csv")
    assert list(summary["status"]) == ["appended"] * 3 and list(summary["genes_shared"]) == [30, 30, 30]
    assert list(summary["kind"]) == ["spatialdata"] * 3 and list(summary["genes_added"]) == [30, 0, 12]
    var = read_elem(zarr.open_group(store / "tables" / "table" / "var", mode="r"))
    assert var["n_samples_measured"].tolist() == [3] * 30 + [1] * 12
    obs = joint_obs(store)
    assert obs["sample_id"].value_counts().to_dict() == {"single": 25, "pair_a": 20, "pair_b": 15}
    assert (obs.loc[obs["sample_id"] == "pair_b", "pct_counts_mt"] == 0).all()
    assert (obs.loc[obs["sample_id"] == "single", "pct_counts_mt"] > 0).any()


def test_rerun_skips_present_samples_and_changes_nothing(tmp_path, inputs_dir, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir))
    before = snapshot(store)
    report = run_capsule(str(store), str(inputs_dir))
    assert report["appended_samples"] == [] and report["skipped_samples"] == ["pair_a", "pair_b", "single"]
    assert snapshot(store) == before


def test_a_later_run_appends_and_the_capsule_reads_its_own_store(tmp_path, inputs_dir, run_capsule):
    first = tmp_path / "first.zarr"
    run_capsule(str(first), str(inputs_dir / "pair.zarr"))
    run_capsule(str(first), str(inputs_dir / "single.zarr"))
    assert joint_obs(first)["sample_id"].tolist() == ["pair_a"] * 20 + ["pair_b"] * 15 + ["single"] * 25
    merged = tmp_path / "merged.zarr"
    report = run_capsule(str(merged), str(first))
    assert sorted(report["appended_samples"]) == ["pair_a", "pair_b", "single"] and report["n_cells"] == 60
    back = readback(merged, ["pair_a", "pair_b", "single"])
    assert "pair_a_cell_boundaries" in back["elements"]["shapes"] and "pair_a_pair_a_cell_boundaries" not in back["elements"]["shapes"]


def test_celltypist_labels_and_a_different_model_is_refused(tmp_path, inputs_dir, run_capsule, tiny_model):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "single.zarr"), str(tiny_model))
    obs = joint_obs(store)
    assert set(obs["tiny_panel_label"]) <= {"Epithelial", "Immune"} and obs["tiny_panel_prob"].between(0, 1).all()
    with pytest.raises(main.IngestError, match="CellTypist model"):
        run_capsule(str(store), str(inputs_dir / "pair.zarr"))


def test_mouse_sample_into_a_human_store_is_refused_before_any_write(tmp_path, inputs_dir, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "single.zarr"))
    write_spatialdata(tmp_path / "mouse.zarr", {"m": {"genes": [g.capitalize() for g in HUMAN], "n": 10, "seed": 3,
                                                       "human": False}})
    before = snapshot(store)
    with pytest.raises(main.IngestError, match=r"different species: mouse \(mouse\); the store holds human genes"):
        run_capsule(str(store), str(tmp_path / "mouse.zarr"))
    assert snapshot(store) == before


def test_atera_like_ensembl_index_and_controls(tmp_path, run_capsule):
    write_spatialdata(tmp_path / "ens.zarr", {"e": {"genes": HUMAN, "n": 12, "seed": 4, "ensembl": True, "controls": 3}})
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(tmp_path / "ens.zarr"))
    var = read_elem(zarr.open_group(store / "tables" / "ens_table" / "var", mode="r"))
    assert list(var.index) == HUMAN and var["gene_ids"].str.startswith("ENSG").all()


@pytest.mark.parametrize("kind", ["anndata", "other", "readonly"])
def test_unusable_locations_are_refused(tmp_path, inputs_dir, run_capsule, kind, monkeypatch):
    target = tmp_path / "target.zarr"
    if kind == "anndata":
        import anndata as ad
        ad.AnnData(pd.DataFrame({"g": [1.0]})).write_zarr(target)
    elif kind == "other":
        target.mkdir()
        (target / "notes.txt").write_text("x")
    else:
        run_capsule(str(target), str(inputs_dir / "single.zarr"))

        def denied(self, data):
            raise PermissionError(30, "Read-only file system")

        monkeypatch.setattr(Path, "write_bytes", denied)
    with pytest.raises(stores.StoreError, match={"anndata": "AnnData", "other": "isn't a SpatialData",
                                                 "readonly": "read-only"}[kind]):
        run_capsule(str(target), str(inputs_dir / "pair.zarr"))


def test_missing_store_outside_create_roots_needs_create(tmp_path, inputs_dir, env):
    import config
    elsewhere = tmp_path / "elsewhere" / "s.zarr"
    settings = config.load_settings([str(elsewhere), str(inputs_dir / "single.zarr")],
                                    {**env, "CREATE_ROOTS": str(tmp_path / "roots")})
    with pytest.raises(main.IngestError, match="No store at .*CREATE=true allows"):
        main.run(settings)
    main.run(config.load_settings([str(elsewhere), str(inputs_dir / "single.zarr")],
                                  {**env, "CREATE_ROOTS": str(tmp_path / "roots"), "CREATE": "true"}))
    assert stores.describe(elsewhere) == "ours"


def test_repair_removes_leftovers_and_appends_a_sample_missing_from_the_joint_table(tmp_path, inputs_dir, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    (store / stores.STAGING / "x.zarr").mkdir(parents=True)
    (store / "shapes" / "half_written").mkdir()
    joint.drop(store)
    joint.add(store, "pair_b", store / "tables" / "pair_b_table", stores.staging(store), 2)
    stores.consolidate(store)
    report = run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    assert "deleted leftover .spatial_ingest_staging/" in report["repairs"]
    assert "deleted uncommitted shapes/half_written" in report["repairs"]
    assert "appended present sample pair_a, missing from the joint table" in report["repairs"]
    assert joint_obs(store)["sample_id"].value_counts().to_dict() == {"pair_a": 20, "pair_b": 15}


def test_repair_rebuilds_a_joint_table_broken_by_a_killed_append(tmp_path, inputs_dir, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    obs = zarr.open_group(store / "tables" / "table" / "obs", mode="r+", use_consolidated=False)
    write_elem(obs, "sample_id", pd.Categorical(["pair_a"] * 20))
    report = run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    assert any("rebuilt" in r for r in report["repairs"])
    assert joint_obs(store)["sample_id"].tolist() == ["pair_a"] * 20 + ["pair_b"] * 15
    assert joint.read_schema(store).samples == ["pair_a", "pair_b"]


KILLER = """
import os, signal, sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
watch = Path(sys.argv[2])
def killer():
    while not watch.exists():
        time.sleep(0.002)
    os.kill(os.getpid(), signal.SIGKILL)
threading.Thread(target=killer, daemon=True).start()
import main
sys.exit(main.main(sys.argv[3:]))
"""


@pytest.mark.parametrize("watch", ["{staging}/pair_b.zarr/shapes", "{staging}/pair_b.joint.zarr", "{store}/tables/pair_b_table"])
def test_sigkill_mid_run_is_repaired_by_the_next_run(tmp_path, inputs_dir, env, watch, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "single.zarr"))
    path = Path(watch.format(staging=store / stores.STAGING, store=store))
    args = [sys.executable, "-c", KILLER, str(Path(main.__file__).parent), str(path), str(store),
            str(inputs_dir / "pair.zarr")]
    done = subprocess.run(args, env={**__import__("os").environ, **env}, capture_output=True, text=True, timeout=600)
    assert done.returncode == -9, done.stdout[-2000:] + done.stderr[-2000:]
    report = run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    counts = joint_obs(store)["sample_id"].value_counts().to_dict()
    assert counts == {"single": 25, "pair_a": 20, "pair_b": 15}, (report["repairs"], counts)
    assert readback(store, ["single", "pair_a", "pair_b"])["tables"]["table"][0] == 60


def test_s3_round_trip(tmp_path, inputs_dir, run_capsule, monkeypatch):
    moto = pytest.importorskip("moto")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    import boto3
    import s3

    uploads: list[str] = []
    with moto.mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="bucket")
        real = s3.client

        def recording():
            c = real()
            original = c.upload_file
            c.upload_file = lambda f, b, k, **kw: (uploads.append(k), original(f, b, k, **kw))[1]
            return c

        monkeypatch.setattr(s3, "client", recording)
        url = "s3://bucket/stores/demo.zarr"
        run_capsule(url, str(inputs_dir / "single.zarr"))
        assert uploads[-1] == "stores/demo.zarr/zarr.json"
        uploads.clear()
        report = run_capsule(url, str(inputs_dir / "pair.zarr"))
        assert report["appended_samples"] == ["pair_a", "pair_b"] and uploads[-1] == "stores/demo.zarr/zarr.json"
        assert report["s3_sync"]["unchanged"] > 0
        uploads.clear()
        report = run_capsule(url, str(inputs_dir))
        assert report["appended_samples"] == [] and report["s3_sync"]["uploaded"] == 0 and uploads == []
        copy = tmp_path / "copy.zarr"
        s3.download(url, copy, 4)
        assert readback(copy, ["single", "pair_a", "pair_b"])["tables"]["table"] == [60, len(HUMAN)]
        client.put_object(Bucket="bucket", Key="foreign/readme.txt", Body=b"x")
        with pytest.raises(main.IngestError, match="no root zarr.json"):
            run_capsule("s3://bucket/foreign", str(inputs_dir / "single.zarr"))


def test_a_store_this_capsule_did_not_create_is_refused_as_target(tmp_path, inputs_dir, run_capsule):
    before = snapshot(inputs_dir / "single.zarr")
    with pytest.raises(stores.StoreError, match="didn't create"):
        run_capsule(str(inputs_dir / "single.zarr"), str(inputs_dir / "pair.zarr"))
    assert snapshot(inputs_dir / "single.zarr") == before


def test_a_bad_model_or_bad_inputs_stop_before_a_store_is_created(tmp_path, inputs_dir, run_capsule):
    junk = tmp_path / "junk.pkl"
    junk.write_bytes(b"not a model")
    store = tmp_path / "new.zarr"
    with pytest.raises(annotate.AnnotateError, match="isn't a CellTypist model"):
        run_capsule(str(store), str(inputs_dir), str(junk))
    with pytest.raises(main.USER_ERRORS, match="found in the inputs"):
        run_capsule(str(store), str(tmp_path / "results"))
    with pytest.raises(precheck.IngestError, match="No samples given, and /data/samples doesn't exist"):
        run_capsule(str(store), "")
    assert not store.exists()


def test_the_same_cells_under_another_name_are_skipped_and_the_run_goes_on(tmp_path, inputs_dir, run_capsule, caplog):
    import shutil
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "single.zarr"))
    shutil.copytree(inputs_dir / "single.zarr", tmp_path / "again.zarr")
    report = run_capsule(str(store), str(tmp_path / "again.zarr"))
    assert report["same_cells_skipped"] == {"again": "single"} and report["appended_samples"] == []
    assert joint_obs(store)["sample_id"].value_counts().to_dict() == {"single": 25}
    # in one run: a copy given before another sample is skipped, and the later sample still goes in
    other = tmp_path / "one_run"
    other.mkdir()
    shutil.copytree(inputs_dir / "single.zarr", other / "a_first.zarr")
    shutil.copytree(inputs_dir / "single.zarr", other / "b_copy.zarr")
    shutil.copytree(inputs_dir / "pair.zarr", other / "c_pair.zarr")
    caplog.clear()
    report = run_capsule(str(tmp_path / "fresh.zarr"), str(other))
    assert report["same_cells_skipped"] == {"b_copy": "a_first"}
    assert sorted(report["appended_samples"]) == ["a_first", "c_pair_a", "c_pair_b"]
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "same cells" in r.getMessage()]
    assert len(warned) == 1 and warned[0].startswith("b_copy holds the same cells as a_first")
    summary = summary_of(report).set_index("sample_id")
    assert summary.loc["b_copy", "status"] == "skipped (same cells as a_first)"


def test_skipped_present_samples_keep_their_kind_in_the_summary(tmp_path, inputs_dir, run_capsule):
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(inputs_dir / "single.zarr"))
    report = run_capsule(str(store), str(inputs_dir / "single.zarr"))
    row = summary_of(report).set_index("sample_id").loc["single"]
    assert row["status"] == "skipped (present)" and row["kind"] == "spatialdata"


def test_a_failing_sample_keeps_the_samples_before_it(tmp_path, inputs_dir, run_capsule, monkeypatch):
    import sample
    real = sample.prepare

    def failing(loaded, item, model, scratch):
        if item.sid == "pair_b":
            raise OSError("disk full")
        return real(loaded, item, model, scratch)

    monkeypatch.setattr(sample, "prepare", failing)
    store = tmp_path / "out.zarr"
    with pytest.raises(RuntimeError, match="pair_b"):
        run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    assert joint_obs(store)["sample_id"].value_counts().to_dict() == {"pair_a": 20}
    monkeypatch.setattr(sample, "prepare", real)
    report = run_capsule(str(store), str(inputs_dir / "pair.zarr"))
    assert report["appended_samples"] == ["pair_b"] and report["repairs"] == []


def test_a_sample_spanning_regions_with_repeated_cell_ids_gets_unique_names(tmp_path, run_capsule):
    path = write_spatialdata(tmp_path / "multi.zarr", {"r1": {"genes": HUMAN, "n": 10, "seed": 5},
                                                       "r2": {"genes": HUMAN, "n": 10, "seed": 5}})
    obs = zarr.open_group(path / "tables" / "table" / "obs", mode="r+")
    write_elem(obs, "sample_id", pd.Categorical(["P1"] * 20))
    store = tmp_path / "out.zarr"
    run_capsule(str(store), str(path))
    names = read_elem(zarr.open_group(store / "tables" / "table" / "obs", mode="r"))
    assert len(names) == 20 and names.index.is_unique and names.index[0] == "P1:P1_cell_boundaries-r1:cell5_0"


# ---------------------------------------------------------------- end to end over every kind of input

def summary_of(report: dict) -> pd.DataFrame:
    return pd.read_csv(Path(report["settings"]["results_dir"]) / "spatial_ingest_summary.csv")


def warnings_in(caplog) -> list[str]:
    import logging
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_mixed_kinds_in_one_parent_folder_and_junk_left_out(tmp_path, run_capsule, caplog):
    import zipfile
    import fixtures_bundles as fb
    import fixtures_sdata as fs

    parent = tmp_path / "download"
    fb.xenium_full(parent / "PatientA_run1" / "outs", protein=True)
    fb.atera(parent / "AteraKidney")
    fs.write(parent / "atlas.zarr", {"s1": {"n": 12, "genes": HUMAN, "seed": 3}})
    (parent / "notes.txt").write_text("hello")
    (parent / ".DS_Store").write_bytes(b"x")
    with zipfile.ZipFile(parent / "qc_report.zip", "w") as zf:
        zf.writestr("report/index.html", "<html></html>")
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(parent))
    assert sorted(report["appended_samples"]) == ["AteraKidney", "PatientA_run1", "atlas"]
    assert report["kinds"] == {"AteraKidney": "atera", "PatientA_run1": "xenium", "atlas": "spatialdata"}
    left = {Path(x["path"]).name for x in report["left_out"]}
    assert {"notes.txt", "qc_report.zip"} <= left and ".DS_Store" not in left
    left_warnings = [w for w in warnings_in(caplog) if w.startswith("Left out")]
    assert len(left_warnings) == 1
    back = readback(store, ["PatientA_run1", "AteraKidney", "atlas"])
    assert back["tables"]["table"][0] == 30 + 15 + 12
    summary = summary_of(report).set_index("sample_id")
    assert summary.loc["PatientA_run1", "kind"] == "xenium" and summary.loc["PatientA_run1", "n_cells"] == 30


def test_two_bundles_in_one_zip_are_two_samples_and_a_rerun_skips_both(tmp_path, run_capsule):
    import fixtures_bundles as fb
    from fixtures_archives import zip_folder

    src = tmp_path / "src"
    fb.xenium_full(src / "RunA", seed=1, protein=False)
    fb.xenium_full(src / "RunB", seed=2, protein=False)
    archive = zip_folder(src, tmp_path / "two_runs.zip", arcroot="")
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(archive))
    assert sorted(report["appended_samples"]) == ["RunA", "RunB"] and report["n_cells"] == 60
    before = snapshot(store)
    again = run_capsule(str(store), str(archive))
    assert again["appended_samples"] == [] and sorted(again["skipped_samples"]) == ["RunA", "RunB"]
    assert snapshot(store) == before
    assert not any((tmp_path / "scratch").glob("stage/*"))


def test_geo_partial_bundle_goes_in_with_one_warning(tmp_path, run_capsule, caplog):
    import fixtures_bundles as fb

    folder, _ = fb.geo_partial(tmp_path / "GSE1")
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(folder))
    assert report["appended_samples"] == ["GSM1_s"] and report["n_cells"] == 25
    missing = [w for w in warnings_in(caplog) if w.startswith("GSM1_s: no ")]
    assert len(missing) == 1
    back = readback(store, ["GSM1_s"])
    assert back["GSM1_s"]["joined"] == [25, 25]


def test_a_bundle_missing_essentials_stops_the_run_before_anything_is_written(tmp_path, run_capsule):
    import fixtures_bundles as fb

    good = fb.xenium_full(tmp_path / "in" / "Good", protein=False)
    bad, _ = fb.geo_partial(tmp_path / "in" / "GSE2", positions_ok=False)
    store = tmp_path / "out.zarr"
    with pytest.raises(precheck.IngestError) as err:
        run_capsule(str(store), f"{good}, {bad}")
    text = str(err.value)
    problems, hint = text.split("again:", 1)[1].split("Every bundle needs", 1)
    assert "nothing was written" in text and "GSM1_s" in problems and "Good" not in problems
    assert hint.rstrip().endswith(f"list only them in List of paths to samples: {good}")
    assert not store.exists()


def test_different_panels_go_into_the_union_with_one_strong_warning(tmp_path, run_capsule, caplog):
    import fixtures_sdata as fs

    fs.write(tmp_path / "in" / "one.zarr", {"s": {"n": 10, "genes": HUMAN[:30], "seed": 1}})
    fs.write(tmp_path / "in" / "two.zarr", {"s": {"n": 8, "genes": HUMAN[20:] + ["NEW1"], "seed": 2}})
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(tmp_path / "in"))
    assert report["joint_genes"] == len(HUMAN) + 1
    panel = [w for w in warnings_in(caplog) if w.startswith("GENE PANELS DIFFER")]
    assert len(panel) == 1 and f"union of {len(HUMAN) + 1} genes" in panel[0]
    var = read_elem(zarr.open_group(store / "tables" / "table" / "var", mode="r"))
    assert var.loc["GENE0", "n_samples_measured"] == 1 and var.loc["GENE25", "n_samples_measured"] == 2


def test_a_model_sharing_no_genes_is_refused_before_anything_is_written(tmp_path, run_capsule, tiny_model):
    import fixtures_sdata as fs

    fs.write(tmp_path / "mouse.zarr", {"s": {"n": 10, "genes": [g.lower() for g in HUMAN], "seed": 1}})
    store = tmp_path / "out.zarr"
    with pytest.raises(precheck.IngestError, match="shares 0 of its 42 genes"):
        run_capsule(str(store), str(tmp_path / "mouse.zarr"), str(tiny_model))
    assert not store.exists()


def test_old_spatialdata_goes_in_and_one_without_a_table_is_refused(tmp_path, run_capsule):
    import fixtures_sdata as fs

    fs.pre01(tmp_path / "old.zarr", {"s": {"n": 10, "genes": HUMAN, "seed": 1}})
    store = tmp_path / "out.zarr"
    report = run_capsule(str(store), str(tmp_path / "old.zarr"))
    assert report["n_cells"] == 10
    fs.write(tmp_path / "notable.zarr", {"t": {"n": 5, "genes": HUMAN, "seed": 2}})
    fs.drop_tables(tmp_path / "notable.zarr")
    with pytest.raises(precheck.IngestError, match="table"):
        run_capsule(str(store), str(tmp_path / "notable.zarr"))


def test_store_location_refusals_are_one_clear_line(tmp_path, inputs_dir, run_capsule, env):
    import config
    roots = tmp_path / "roots"
    roots.mkdir()
    with pytest.raises(precheck.IngestError, match=r"is a folder for stores, not a store: give a \.zarr path inside"):
        main.run(config.load_settings([str(roots), str(inputs_dir / "single.zarr")], {**env, "CREATE_ROOTS": str(roots)}))
    (tmp_path / "afile.txt").write_text("x")
    with pytest.raises(precheck.IngestError, match="is a file"):
        run_capsule(str(tmp_path / "afile.txt"), str(inputs_dir / "single.zarr"))
    with pytest.raises(precheck.IngestError, match="absolute path"):
        run_capsule("rel/store.zarr", str(inputs_dir / "single.zarr"))
