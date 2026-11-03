import pytest

from config import ConfigError, load_settings


def test_defaults_are_a_report_run():
    s = load_settings([], env={})
    assert s.mode == "report" and not s.apply
    assert s.branch() == "qc" and s.groups() == ["batch_id", "pool_id", "chip_id"]


def test_named_app_panel_flags_in_either_spelling():
    s = load_settings(["--input=/data/store", "--mode", "apply", "--group_columns", "batch_id, chip_id",
                       "--celltype-qc", "yes", "--compute_metrics=on", "--target", "strict"], env={})
    assert s.input == "/data/store" and s.apply
    assert s.groups() == ["batch_id", "chip_id"]
    assert s.celltype_qc and s.compute_metrics and s.branch() == "strict"


def test_flags_override_env_and_empty_values_fall_back():
    s = load_settings(["--input", "", "--label_column", "x_label"], env={"INPUT": "/env/store", "SEED": "8"})
    assert s.input == "/env/store" and s.label_column == "x_label" and s.seed == 8


@pytest.mark.parametrize("argv, message", [
    (["--mode", "preview"], "mode must be one of report, apply"),
    (["--celltype_qc", "maybe"], "CELLTYPE_QC='maybe' is not a valid bool"),
])
def test_bad_values(argv, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(argv, env={})


def test_bad_env_value():
    with pytest.raises(ConfigError, match="SEED='four' is not a valid int"):
        load_settings([], env={"SEED": "four"})
