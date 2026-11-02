import pytest

from config import ConfigError, Settings, load_settings


def test_defaults_follow_the_notebook():
    s = load_settings(["--input", "/data/x.zarr"], env={})
    assert (s.branch, s.integration, s.output_mode, s.leiden_resolution) == ("qc", "none", "full", 1.1)
    assert (s.n_top_genes, s.n_comps, s.n_pcs, s.n_neighbors, s.random_seed) == (2000, 50, 30, 20, 4242)


def test_flags_override_env_and_empty_flags_keep_env():
    env = {"INPUT": "/data/a.zarr", "INTEGRATION": "harmony", "BATCH_KEYS": "chip_id"}
    s = load_settings(["--input", "", "--batch-keys", "chip_id, pool_id"], env=env)
    assert s.input == "/data/a.zarr"
    assert s.batch_key_list == ["chip_id", "pool_id"]


@pytest.mark.parametrize("argv, message", [
    ([], "No input"),
    (["--input", "x", "--integration", "harmony"], "needs batch keys"),
    (["--input", "x", "--integration", "scvi", "--batch-keys", "b"], "integration must be"),
    (["--input", "x", "--output-mode", "copy"], "output mode must be"),
    (["--input", "x", "--leiden-resolution", "high"], "not a valid float"),
])
def test_refused(argv, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(argv, env={})


def test_output_path_defaults_to_results_stem():
    s = Settings(input="/data/curated.qc.zarr/", results_dir="/results")
    assert s.output_path() == "/results/curated.qc.processed.zarr"
    assert Settings(input="x", output="/scratch/o.zarr").output_path() == "/scratch/o.zarr"


def test_every_setting_is_a_flag():
    s = load_settings(["--input", "x", "--n-top-genes", "3000", "--neighbors-algorithm", "brute"], env={})
    assert (s.n_top_genes, s.neighbors_algorithm) == (3000, "brute")
