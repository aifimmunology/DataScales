import json
import re
from pathlib import Path

import pytest

from config import Config, ConfigError, _cast, load_config

PANEL = Path(__file__).resolve().parents[3] / ".codeocean" / "app-panel.json"


def test_defaults_follow_the_notebook():
    c = load_config([], env={})
    assert (c.input, c.method, c.branch, c.table, c.output) == ("", "pca", "main", "table", "")
    assert (c.n_top_genes, c.n_comps, c.n_pcs, c.n_neighbors, c.umap_min_dist) == (2000, 50, 30, 20, 0.45)
    assert (c.target_sum, c.scale_max_value, c.leiden_keys) == (1e4, 10.0, {"leiden": 1.1})
    assert (c.rmm_mode, c.gpus) == ("managed", "") and c.model_dir == "/results/scvi_model"


def test_env_then_flags_and_empty_values_keep_the_previous():
    env = {"INPUT": "/data/a.zarr", "N_TOP_GENES": "3000", "SCVI_N_LATENT": "20", "BATCH_KEY": "chip_id"}
    c = load_config(["--input", "", "--n-top-genes", "1000", "--batch_key", "  "], env=env)
    assert (c.input, c.n_top_genes, c.scvi_n_latent, c.batch_keys) == ("/data/a.zarr", 1000, 20, ["chip_id"])


def test_every_field_is_a_flag_with_underscores_or_dashes():
    c = load_config(["--neighbors_algorithm", "brute", "--scvi-max-epochs", "5", "--scvi_early_stopping", "yes",
                     "--rmm-mode", "pool"], env={})
    assert (c.neighbors_algorithm, c.scvi_max_epochs, c.scvi_early_stopping, c.rmm_mode) == ("brute", 5, True, "pool")


def test_empty_panel_fields_mean_the_default():
    env = {"N_COMPS": "40"}
    for argv in (["--subset_column", "--n_comps", "--method", "scVI (train)", "--output"],
                 ["--subset_column", "", "--n_comps", "", "--method", "scVI (train)", "--output", ""],
                 ["--method", "scVI (train)"]):
        c = load_config(argv, env=env)
        assert (c.subset_column, c.n_comps, c.method, c.output) == ("", 40, "scvi", ""), argv


@pytest.mark.parametrize("raw, method", [("PCA", "pca"), ("scVI (train)", "scvi"), ("scvi", "scvi"), (" pca ", "pca")])
def test_method_accepts_the_panel_labels(raw, method):
    assert load_config(["--method", raw], env={}).method == method


@pytest.mark.parametrize("argv, message", [
    (["--method", "harmony"], "not a valid pca or scvi"),
    (["--n_comps", "many"], "not a valid int"),
    (["--leiden_resolutions", "0.5,high"], "numbers separated by commas"),
    (["--leiden_resolutions", "0"], "positive"),
    (["--subset_column", "sample_id"], "needs subset_values"),
    (["--rmm_mode", "ucx"], "rmm_mode must be"),
    (["--plots", "maybe"], "not a valid bool"),
])
def test_bad_values(argv, message):
    with pytest.raises(ConfigError, match=message):
        load_config(argv, env={})


@pytest.mark.parametrize("res, keys", [
    ("1.1", {"leiden": 1.1}),
    ("0.5,1.0,2.0", {"leiden_0.5": 0.5, "leiden_1.0": 1.0, "leiden_2.0": 2.0}),
    ("0.5, 1, 2", {"leiden_0.5": 0.5, "leiden_1.0": 1.0, "leiden_2.0": 2.0}),
    (" 2 ,", {"leiden": 2.0}),
])
def test_leiden_keys(res, keys):
    assert Config(leiden_resolutions=res).leiden_keys == keys


def test_subset_and_batch_lists():
    c = Config(subset_column="sample_id", subset_values="SP0, SP2,", batch_key="chip_id,sample_id")
    assert c.subset == ["SP0", "SP2"] and c.batch_keys == ["chip_id", "sample_id"]
    assert Config(subset_values="SP0").subset == []


def test_relative_paths_resolve_from_where_run_started(tmp_path):
    env = {"PROCESS_CWD": str(tmp_path / "sub")}
    c = load_config(["--input", "../data/x.zarr", "--output", "out", "--results_dir", "res", "--origin", "s3://b/p"],
                    env=env)
    expected = (tmp_path.resolve() / p for p in ("data/x.zarr", "sub/out", "sub/res"))
    assert (c.input, c.output, c.results_dir) == tuple(map(str, expected))
    assert c.origin == "s3://b/p" and load_config(["--input", "/abs.zarr"], env=env).input == "/abs.zarr"


def test_changed_lists_only_overrides():
    assert Config(n_comps=30, method="scvi").changed() == {"n_comps": 30, "method": "scvi"}


def test_panel_matches_config():
    panel = json.loads(PANEL.read_text())
    assert panel["named_parameters"] is True
    categories = {c["id"] for c in panel["categories"]}
    names = [p["param_name"] for p in panel["parameters"]]
    assert len(names) == len(set(names)) and len({p["id"] for p in panel["parameters"]}) == len(names)
    expected = {"input", "table", "branch", "origin", "subset_column", "subset_values", "method", "batch_key",
                "n_top_genes", "n_comps", "n_neighbors", "umap_min_dist", "leiden_resolutions", "output"}
    assert set(names) == expected
    for p in panel["parameters"]:
        assert p["category"] in categories
        if "pattern" in p:
            assert re.fullmatch(p["pattern"], ""), f"{p['param_name']}: an empty field must be allowed"
        if p.get("default_value"):
            assert _cast(p["param_name"], p["default_value"]) == getattr(Config(), p["param_name"]), p["param_name"]
        for option in p.get("extra_data", []):
            _cast(p["param_name"], option)
    assert all("optional" in c["description"].lower() for c in panel["categories"][:1])
    assert re.fullmatch(next(p["pattern"] for p in panel["parameters"] if p["param_name"] == "leiden_resolutions"),
                        "0.5, 1.0,2")
