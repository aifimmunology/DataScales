"""Minimal regression tests for the zarr_query_benchmarking CLI tool.

Covers the pieces that would silently produce WRONG benchmark numbers if a future
refactor broke them — not exhaustive coverage of the CLI surface:

* span reconstruction + multi-run read (celltype mode reads exactly the matched rows),
* the I/O-wall union (the concurrency-correct time split — the tool's core metric),
* format conversion (``native``/``dense``/``csr`` — the "fair across layouts" claim),
* selection (sequential slice; seeded random determinism),
* end-to-end ``main()`` on tiny dense/CSR stores: right rows selected, locality
  (``n_spans``) correct, scatter warning fires, JSON summary shape intact.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import zarr_query_benchmarking.__main__ as qbench
from zarr_query_benchmarking.__main__ import (
    TimingStore,
    _read_spans,
    _run_stats,
    _runs_from_sorted_indices,
    _to_final,
    main,
    select,
)


# ── pure logic: celltype row selection is read correctly ──────────────────────

def test_runs_from_sorted_indices():
    assert _runs_from_sorted_indices(np.array([], dtype=int)) == []
    assert _runs_from_sorted_indices(np.arange(5, 10)) == [(5, 10)]           # one block
    assert _runs_from_sorted_indices(np.array([0, 1, 2, 7, 8])) == [(0, 3), (7, 9)]
    assert _runs_from_sorted_indices(np.array([1, 3, 5])) == [(1, 2), (3, 4), (5, 6)]  # scattered


def test_run_stats_locality():
    _, n, avg = _run_stats(np.arange(0, 100))          # sorted store: one long run
    assert (n, avg) == (1, 100.0)
    _, n, avg = _run_stats(np.array([0, 2, 4, 6, 8]))  # scattered: many length-1 runs
    assert (n, avg) == (5, 1.0)
    _, n, avg = _run_stats(np.array([], dtype=int))     # empty selection
    assert (n, avg) == (0, 0.0)


def test_read_spans_single_multi_and_empty():
    dense = np.arange(20).reshape(10, 2)
    assert np.array_equal(_read_spans(dense, [(2, 5)]), dense[2:5])          # single run
    multi = _read_spans(dense, [(0, 2), (7, 9)])                             # concat multiple runs
    assert np.array_equal(multi, np.concatenate([dense[0:2], dense[7:9]]))
    csr = sp.csr_matrix(dense)                                              # sparse -> vstack
    out = _read_spans(csr, [(0, 2), (7, 9)])
    assert sp.issparse(out) and np.array_equal(out.toarray(), multi)
    assert _read_spans(dense, []).shape[0] == 0                             # empty -> no rows


# ── pure logic: format conversion (fair-across-layouts) ───────────────────────

def test_to_final_formats():
    dense = np.arange(6).reshape(2, 3).astype("float32")
    csr = sp.csr_matrix(dense)
    assert _to_final(csr, "native") is csr                                  # native = unchanged
    assert _to_final(dense, "native") is dense
    densified = _to_final(csr, "dense")
    assert isinstance(densified, np.ndarray) and np.array_equal(densified, dense)
    compressed = _to_final(dense, "csr")
    assert sp.issparse(compressed) and np.array_equal(compressed.toarray(), dense)


# ── pure logic: selection ─────────────────────────────────────────────────────

def test_select_sequential_and_seeded_random():
    assert select(100, 10, "sequential", 0) == slice(0, 10)
    r1 = select(100, 10, "random", 0)
    r2 = select(100, 10, "random", 0)
    assert np.array_equal(r1, r2)                       # deterministic for a fixed seed
    assert len(r1) == 10 and np.all(np.diff(r1) > 0)    # sorted + unique


# ── pure logic: I/O-wall = UNION of fetch intervals (not sum) ─────────────────

def test_io_wall_union_of_intervals():
    ts = TimingStore.__new__(TimingStore)               # bypass __init__ (needs a real store)
    ts._intervals = []
    assert ts.io_wall_s() == 0.0
    ts._intervals = [(0.0, 1.0), (2.0, 3.0)]            # disjoint -> sum of lengths
    assert ts.io_wall_s() == pytest.approx(2.0)
    ts._intervals = [(0.0, 2.0), (1.0, 3.0)]            # overlapping -> union 3.0, NOT sum 4.0
    assert ts.io_wall_s() == pytest.approx(3.0)
    ts._intervals = [(0.0, 5.0), (1.0, 2.0)]            # nested -> outer interval only
    assert ts.io_wall_s() == pytest.approx(5.0)


# ── end-to-end through main() ─────────────────────────────────────────────────

def _build_store(path, fmt: str, *, sorted_ct: bool = True) -> None:
    """Tiny anndata zarr store: 30 cells x 5 genes, obs `cell_type` in {A,B,C}.
    sorted_ct -> contiguous blocks (B is one run); else interleaved (B scattered)."""
    ad.settings.zarr_write_format = 3
    n, g = 30, 5
    ct = (np.array(["A"] * 12 + ["B"] * 10 + ["C"] * 8) if sorted_ct
          else np.array([["A", "B", "C"][i % 3] for i in range(n)]))
    dense = np.arange(n * g).reshape(n, g).astype("float32")
    X = sp.csr_matrix(dense) if fmt == "csr" else dense
    obs = pd.DataFrame({"cell_type": pd.Categorical(ct)}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    ad.AnnData(X=X, obs=obs, var=var).write_zarr(str(path))


def _run(monkeypatch, capsys, argv):
    """Run main() with the (subprocess) RSS probe stubbed out; return (summary, stderr)."""
    monkeypatch.setattr(qbench, "_measure_peak_rss", lambda args: 12345)
    main(argv + ["--json"])
    cap = capsys.readouterr()
    return json.loads(cap.out.strip().splitlines()[-1]), cap.err


@pytest.mark.parametrize("fmt", ["dense", "csr"])
def test_e2e_celltype_sorted(tmp_path, monkeypatch, capsys, fmt):
    store = tmp_path / f"{fmt}.zarr"
    _build_store(store, fmt, sorted_ct=True)
    d, err = _run(monkeypatch, capsys, [
        "--store", str(store), "--mode", "celltype",
        "--obs-column", "cell_type", "--obs-value", "B",
        "--native", "--repeats", "2", "--warmup", "1",
    ])
    assert d["selected"] == 10 and d["result_shape"][0] == 10   # exactly the 10 B cells
    assert d["n_spans"] == 1                                     # sorted -> single contiguous run
    assert "WARNING" not in err                                 # contiguous -> no scatter warning
    assert d["chunks_fetched"] >= 1                             # the TimingStore actually counted fetches
    assert "select_mode" not in d                               # removed knob stays gone
    for k in ("median_s", "io_wall_median_s", "cpu_wall_median_s", "convert_median_s"):
        assert k in d and d[k] >= 0.0                           # timing split intact


def test_e2e_celltype_scattered_warns(tmp_path, monkeypatch, capsys):
    store = tmp_path / "scattered.zarr"
    _build_store(store, "csr", sorted_ct=False)                 # B interleaved every 3rd row
    d, err = _run(monkeypatch, capsys, [
        "--store", str(store), "--mode", "celltype",
        "--obs-column", "cell_type", "--obs-value", "B",
        "--native", "--repeats", "1", "--warmup", "0",
    ])
    assert d["selected"] == 10
    assert d["n_spans"] > 1                                     # scattered -> many runs
    assert "WARNING" in err and "Sort the store" in err        # scatter warning fires


def test_e2e_sequential_dense(tmp_path, monkeypatch, capsys):
    store = tmp_path / "seq.zarr"
    _build_store(store, "dense", sorted_ct=True)
    d, _ = _run(monkeypatch, capsys, [
        "--store", str(store), "--axis", "row", "--count", "5",
        "--mode", "sequential", "--format", "dense",
    ])
    assert d["result_shape"] == [5, 5] and d["selected"] == 5
    assert d["n_spans"] is None                                # locality metric is celltype-only


def test_e2e_celltype_bad_value_exits(tmp_path, monkeypatch, capsys):
    store = tmp_path / "bad.zarr"
    _build_store(store, "csr", sorted_ct=True)
    monkeypatch.setattr(qbench, "_measure_peak_rss", lambda args: 0)
    with pytest.raises(SystemExit):                            # unmatched obs value -> guarded exit
        main([
            "--store", str(store), "--mode", "celltype",
            "--obs-column", "cell_type", "--obs-value", "NOPE", "--native",
        ])
