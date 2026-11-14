"""inventory.inspect: roles chosen by content and cell-ID consistency, essentials vs optional parts."""

import pytest

import fixtures_bundles as fb
import inventory


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "scratch"


def test_complete_xenium_bundle(tmp_path, scratch):
    inv = inventory.inspect(fb.candidate(fb.xenium_full(tmp_path / "full")), scratch)
    assert inv.missing_essential == []
    assert inv.chosen == {"matrix": "cell_feature_matrix.h5", "cells": "cells.parquet", "experiment": "experiment.xenium",
                          "cells_zarr": "cells.zarr.zip", "cell_boundaries": "cell_boundaries.parquet",
                          "nucleus_boundaries": "nucleus_boundaries.parquet", "transcripts": "transcripts.parquet"}
    assert inv.missing_optional == ["morphology images"]
    assert inv.n_cells == 30
    # var is the matrix's raw features, protein and controls included (dropped later, with one INFO line)
    assert len(inv.var) == len(fb.features(protein=True))
    assert set(inv.var["feature_types"]) == {"Gene Expression", "Negative Control Probe", "Unassigned Codeword",
                                             "Protein Expression"}


def test_geo_partial_with_mtx_and_csv_cells(tmp_path, scratch):
    _, cand = fb.geo_partial(tmp_path / "geo")
    inv = inventory.inspect(cand, scratch)
    assert inv.missing_essential == []
    assert inv.chosen["matrix"] == "cell_feature_matrix/matrix.mtx.gz"
    assert inv.chosen["cells"] == "cells.csv.gz" and inv.chosen["transcripts"] == "transcripts.parquet"
    assert any(m.startswith("experiment.xenium") for m in inv.missing_optional)
    assert "cell boundaries" in inv.missing_optional and inv.n_cells == 25


def test_no_positions_is_an_essential_miss(tmp_path, scratch):
    _, cand = fb.geo_partial(tmp_path / "nopos", positions_ok=False)
    inv = inventory.inspect(cand, scratch)
    assert len(inv.missing_essential) == 1 and inv.missing_essential[0].startswith("cell positions")
    assert any("0-byte" in n and "cells.csv.gz" in n for n in inv.notes)


def test_no_matrix_is_an_essential_miss(tmp_path, scratch):
    folder = fb.xenium_full(tmp_path / "nomatrix")
    (folder / "cell_feature_matrix.h5").unlink()
    inv = inventory.inspect(fb.candidate(folder), scratch)
    assert inv.missing_essential and inv.missing_essential[0].startswith("count matrix")


def test_cells_of_another_cell_set_are_an_essential_miss(tmp_path, scratch):
    folder = fb.xenium_full(tmp_path / "mix")
    other = fb.xenium_reseg(tmp_path / "other")
    (folder / "cells.parquet").write_bytes((other / "cells.parquet").read_bytes())
    inv = inventory.inspect(fb.candidate(folder), scratch)
    # cells.zarr.zip holds the matrix's cells: positions come from there, and the note says why
    assert inv.missing_essential == [] and inv.chosen["cells"] == "cells.zarr.zip"
    assert any("only 0 of the matrix's 30 cells" in n for n in inv.notes)
    (folder / "cells.zarr.zip").unlink()
    inv = inventory.inspect(fb.candidate(folder), tmp_path / "scratch2")
    assert inv.missing_essential and "only 0 of the matrix's 30 cells" in inv.missing_essential[0]


def test_transcripts_chosen_by_cell_ids_not_by_name(tmp_path, scratch):
    inv = inventory.inspect(fb.candidate(fb.xenium_reseg(tmp_path / "reseg")), scratch)
    assert inv.chosen["transcripts"] == "reseg_transcripts.parquet"
    assert any("reseg_transcripts.parquet chosen" in n and "transcripts.parquet (0.0%" in n for n in inv.notes)


def test_transcripts_of_another_segmentation_alone_are_left_out(tmp_path, scratch):
    folder = fb.xenium_reseg(tmp_path / "reseg")
    (folder / "reseg_transcripts.parquet").unlink()
    inv = inventory.inspect(fb.candidate(folder), scratch)
    assert "transcripts" not in inv.chosen
    assert any(m.startswith("transcripts (cell IDs don't match") for m in inv.missing_optional)


def test_explorer_only_export(tmp_path, scratch):
    inv = inventory.inspect(fb.candidate(fb.xenium_explorer(tmp_path / "xe")), scratch)
    assert inv.missing_essential == []
    assert inv.chosen["matrix"] == "cell_feature_matrix.zarr.zip" and inv.chosen["cells"] == "cells.zarr.zip"
    assert inv.chosen["cell_boundaries"] == inv.chosen["nucleus_boundaries"] == "cells.zarr.zip"
    assert set(inv.var["feature_types"]) >= {"Gene Expression"}
    assert inv.n_cells == 20


def test_atera_bundle(tmp_path, scratch):
    inv = inventory.inspect(fb.candidate(fb.atera(tmp_path / "atera"), "atera"), scratch)
    assert inv.missing_essential == []
    assert inv.chosen == {"matrix": "cell_feature_matrix.zarr.zip", "cells": "cell_feature_matrix.zarr.zip",
                          "experiment": "experiment.spatial"}
    assert any("cells.zarr.zip" in m for m in inv.missing_optional) and inv.n_cells == 15


def test_bytes_transcripts_and_csv_boundaries(tmp_path, scratch):
    inv = inventory.inspect(fb.candidate(fb.xenium_bytes(tmp_path / "bytes")), scratch)
    assert inv.chosen["transcripts"] == "transcripts.parquet"          # bytes cell IDs decoded for the check
    assert inv.chosen["cell_boundaries"] == "cell_boundaries.csv.gz"


def test_gzipped_parquet_is_read_whatever_its_name(tmp_path, scratch):
    """A GEO .parquet.gz left with gz=False (name kept) is still sniffed and read."""
    folder = fb.xenium_full(tmp_path / "gz", labels=False)
    fb.gzip_file(folder / "cells.parquet", folder / "cells.parquet.gz")
    inv = inventory.inspect(fb.candidate(folder), scratch)
    assert inv.chosen["cells"] == "cells.parquet.gz" and inv.missing_essential == []


def test_describe_is_one_line(tmp_path, scratch):
    _, cand = fb.geo_partial(tmp_path / "geo")
    line = inventory.describe(inventory.inspect(cand, scratch))
    assert "\n" not in line and line.startswith("not available:")
