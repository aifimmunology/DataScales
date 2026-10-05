"""Golden-file tests: annizarr's own writers must reproduce a pinned on-disk layout.

Fixtures live in ``tests/golden/<case>.tar.gz`` (see ``tests/golden/README.md``), each a
self-snapshot built by ``tests/golden/generate.py`` from the CURRENT ``annizarr`` package. This
test extracts a tarball, re-runs ``annizarr.convert`` with the packed parameters, and asserts
the result matches the packed ``expected/`` store file-for-file — so a refactor that silently
changes chunking, codecs, encoding attrs, or sharding math fails loudly instead of drifting.
"""

from __future__ import annotations

import difflib
import filecmp
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

import annizarr
from _readable import assert_anndata_readable
from annizarr.config import apply_cli_overrides, load_config

GOLDEN_DIR = Path(__file__).parent / "golden"
TARBALLS = sorted(GOLDEN_DIR.glob("*.tar.gz"))


def _tree_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}


def _assert_stores_equal(expected_root: Path, actual_root: Path) -> None:
    expected_files = _tree_files(expected_root)
    actual_files = _tree_files(actual_root)
    assert set(actual_files) == set(expected_files), (
        f"file set differs.\n"
        f"only in expected: {sorted(set(expected_files) - set(actual_files))}\n"
        f"only in actual:   {sorted(set(actual_files) - set(expected_files))}"
    )
    for rel in sorted(expected_files):
        expected_path, actual_path = expected_files[rel], actual_files[rel]
        if filecmp.cmp(expected_path, actual_path, shallow=False):
            continue
        if rel.endswith("zarr.json"):
            e_text = expected_path.read_text().splitlines(keepends=True)
            a_text = actual_path.read_text().splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(e_text, a_text, fromfile=f"expected/{rel}", tofile=f"actual/{rel}"))
            pytest.fail(f"first differing file: {rel}\n{diff}")
        pytest.fail(
            f"first differing file (binary, {expected_path.stat().st_size}B vs {actual_path.stat().st_size}B): {rel}"
        )


def _build_cfg(case: dict[str, Any]) -> Any:
    # case.json's "backed" key is baked into the (un-regenerated) golden tarballs verbatim;
    # map it onto the renamed apply_cli_overrides(lazy=) kwarg here rather than touching them.
    return apply_cli_overrides(
        load_config(),
        x_storage=case["x_storage"],
        lazy=case.get("backed", False),
        cpus=case.get("cpus"),
        x_row_chunk=case.get("x_row_chunk"),
        x_col_chunk=case.get("x_col_chunk"),
        sparse_flat_chunk=case.get("sparse_flat_chunk"),
        x_shard_factor=case.get("x_shard_factor"),
        auto_shard=case.get("auto_shard", False),
        sort_by=case.get("sort_by"),
    )


@pytest.mark.parametrize("tarball", TARBALLS, ids=[p.name.removesuffix(".tar.gz") for p in TARBALLS])
def test_golden_writer_matches_old_converter(tarball: Path, tmp_path: Path) -> None:
    with tarfile.open(tarball) as tar:
        tar.extractall(tmp_path, filter="data")

    case = json.loads((tmp_path / "case.json").read_text())
    input_dir = tmp_path / "input"
    input_paths = [str(input_dir / name) for name in case["inputs"]]
    inputs = input_paths[0] if len(input_paths) == 1 else input_paths

    cfg = _build_cfg(case)
    actual_root = tmp_path / "actual"
    # the streamed backed-sort path mkdtemps a sibling of `output` before opening the store
    # (output_path.parent must pre-exist for it, same as the old converter); mirror how
    # generate.py's expected/ dir was pre-created before invoking the old CLI.
    actual_root.mkdir(parents=True, exist_ok=True)
    output = actual_root / case["output"]

    annizarr.convert(inputs, output=output, cfg=cfg)

    expected_root = tmp_path / "expected"
    _assert_stores_equal(expected_root, actual_root)
    assert_anndata_readable(output)
