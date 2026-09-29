#!/usr/bin/env python
"""OLD (dask-based) vs NEW (thread-pool) writer engine benchmark.

AnniZarr unit 3.3: the writers were rewritten off dask onto chunk-aligned
thread/process pools (Phase 3a/3b). This compares the two CLI *engines* while
holding everything else (chunk math, codecs, dataset) identical, per
CLAUDE.md's "Benchmarking best practices" (isolate the variable, warm + N
repeats, median/min/max, peak RSS, pinned+recorded thread env, raw JSON).

    OLD: tools/convert-to-zarr  (pixi -e dev convert-to-zarr convert-h5ad ...)
    NEW: tools/annizarr         (pixi -e default annizarr convert ...)

Both envs pin zarr 3.3.0 / anndata 0.12.19 / numpy 2.4.x with identical
chunk/codec math (tests/test_golden_writers.py proves byte-identical output),
so the store engine (dask vs stdlib thread/process pool) is the only variable.

Each run is its OWN subprocess (the pixi CLI invocation) so peak RSS is
isolated: wall time via time.perf_counter() around subprocess.run, peak RSS
via the RUSAGE_CHILDREN delta (rolls up through the pixi -> python process
tree once each is wait()'d). 1 warm-up + 3 measured repeats per config;
output store removed between runs.

Usage
-----
    cd tools/annizarr
    pixi run -e default python benchmarking/writer_engine_bench.py \
        --json /path/to/raw.json
    # narrow to one case/engine while iterating:
    pixi run -e default python benchmarking/writer_engine_bench.py \
        --cases a --engines new --repeats 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".data"
DATASCALE_ROOT = HERE.parent.parent  # tools/annizarr/benchmarking/.. /.. -> tools
OLD_DIR = DATASCALE_ROOT / "convert-to-zarr"
NEW_DIR = DATASCALE_ROOT / "annizarr"

D1_H5AD = DATA_DIR / "D1_sparse_100k.h5ad"
D2_H5AD = DATA_DIR / "D2_dense_20k.h5ad"

# Pinned so N worker processes/threads don't each spawn N BLAS threads
# (CLAUDE.md silent-perf-killer #1). Passed explicitly to each subprocess's
# env (not just set here) so it is recorded and applies to every descendant.
PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


# ── dataset generation (fixed seed, skip if present) ──────────────────────────
def _gen_d1() -> None:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from anndata import AnnData

    rng = np.random.default_rng(42)
    n_obs, n_vars = 100_000, 2_000
    x = sp.random(n_obs, n_vars, density=0.10, format="csr", dtype=np.float32, random_state=42)
    x.data = np.abs(x.data) + 1.0
    cats = ["type_a", "type_b", "type_c", "type_d", "type_e"]
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(cats, n_obs), categories=cats)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    AnnData(X=x, obs=obs, var=var).write_h5ad(D1_H5AD)
    print(f"generated {D1_H5AD} ({x.nnz} nnz)", file=sys.stderr)


def _gen_d2() -> None:
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(7)
    n_obs, n_vars = 20_000, 2_000
    x = (rng.random((n_obs, n_vars), dtype=np.float32) * 5.0).astype(np.float32)
    cats = ["type_a", "type_b", "type_c"]
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(cats, n_obs), categories=cats)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    AnnData(X=x, obs=obs, var=var).write_h5ad(D2_H5AD)
    print(f"generated {D2_H5AD}", file=sys.stderr)


def ensure_datasets() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not D1_H5AD.exists():
        _gen_d1()
    if not D2_H5AD.exists():
        _gen_d2()


# ── measurement helpers ────────────────────────────────────────────────────────
def _rss_to_mb(raw: int) -> float:
    """ru_maxrss is bytes on macOS (darwin), kilobytes on Linux."""
    return raw / (1024**2) if sys.platform == "darwin" else raw / 1024


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**2)


def rmtree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


# ── engine command builders ────────────────────────────────────────────────────
class Engine:
    def __init__(self, name: str, cwd: Path, pixi_env: str):
        self.name = name
        self.cwd = cwd
        self.pixi_env = pixi_env

    def build_cmd(
        self,
        *,
        input_path: Path,
        output_path: Path,
        x_storage: str,
        backed: bool,
        cpus: int,
        row_chunk: int | None,
        col_chunk: int | None,
        shard_factor: int | None,
    ) -> list[str]:
        raise NotImplementedError

    def version_cmd(self) -> list[str]:
        raise NotImplementedError


class OldEngine(Engine):
    _STORAGE: ClassVar[dict[str, str]] = {"dense": "dense", "csr": "sparse-csr"}

    def __init__(self):
        super().__init__("old", OLD_DIR, "dev")

    def build_cmd(
        self, *, input_path, output_path, x_storage, backed, cpus, row_chunk, col_chunk, shard_factor
    ) -> list[str]:
        cmd = [
            "pixi",
            "run",
            "-e",
            self.pixi_env,
            "convert-to-zarr",
            "convert-h5ad",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--x-storage",
            self._STORAGE[x_storage],
            "--cpus",
            str(cpus),
            "--overwrite",
        ]
        if backed:
            cmd.append("--backed")
        if row_chunk is not None:
            cmd += ["--x-row-chunk", str(row_chunk)]
        if col_chunk is not None:
            cmd += ["--x-col-chunk", str(col_chunk)]
        if shard_factor is not None:
            cmd += ["--x-shard-factor", str(shard_factor)]
        return cmd

    def version_cmd(self) -> list[str]:
        return ["pixi", "run", "-e", self.pixi_env, "convert-to-zarr", "--help"]


class NewEngine(Engine):
    _STORAGE: ClassVar[dict[str, str]] = {"dense": "dense", "csr": "csr"}

    def __init__(self):
        super().__init__("new", NEW_DIR, "default")

    def build_cmd(
        self, *, input_path, output_path, x_storage, backed, cpus, row_chunk, col_chunk, shard_factor
    ) -> list[str]:
        cmd = [
            "pixi",
            "run",
            "-e",
            self.pixi_env,
            "annizarr",
            "convert",
            str(input_path),
            "-o",
            str(output_path),
            "--x-storage",
            self._STORAGE[x_storage],
            "--cpus",
            str(cpus),
            "--overwrite",
        ]
        if backed:
            cmd.append("--backed")
        if row_chunk is not None:
            cmd += ["--x-row-chunk", str(row_chunk)]
        if col_chunk is not None:
            cmd += ["--x-col-chunk", str(col_chunk)]
        if shard_factor is not None:
            cmd += ["--x-shard-factor", str(shard_factor)]
        return cmd

    def version_cmd(self) -> list[str]:
        return ["pixi", "run", "-e", self.pixi_env, "annizarr", "--version"]


ENGINES = {"old": OldEngine(), "new": NewEngine()}


# ── case matrix ─────────────────────────────────────────────────────────────
def build_cases(max_cpus: int) -> list[dict]:
    def cpus_of(*vals):
        return sorted({v for v in vals if v <= max_cpus})

    cases = [
        dict(
            case="a",
            dataset="D1",
            x_storage="dense",
            backed=False,
            cpus=cpus_of(1, 4, 8),
            row_chunk=None,
            col_chunk=None,
            shard_factor=None,
        ),
        dict(
            case="b",
            dataset="D1",
            x_storage="dense",
            backed=True,
            cpus=cpus_of(1, 4),
            row_chunk=None,
            col_chunk=None,
            shard_factor=None,
        ),
        dict(
            case="c",
            dataset="D1",
            x_storage="csr",
            backed=False,
            cpus=cpus_of(1, 4),
            row_chunk=None,
            col_chunk=None,
            shard_factor=None,
        ),
        dict(
            case="d",
            dataset="D2",
            x_storage="dense",
            backed=False,
            cpus=cpus_of(1, 4),
            row_chunk=None,
            col_chunk=None,
            shard_factor=None,
        ),
        dict(
            case="e",
            dataset="D1",
            x_storage="dense",
            backed=False,
            cpus=cpus_of(4),
            row_chunk=1024,
            col_chunk=1024,
            shard_factor=4,
        ),
    ]
    return cases


CASE_DESC = {
    "a": "D1 -> dense, in-memory",
    "b": "D1 -> dense, --backed (process pool)",
    "c": "D1 -> csr, in-memory",
    "d": "D2 -> dense, in-memory",
    "e": "D1 -> dense, in-memory, sharded (row/col 1024, shard x4)",
}


# ── run one (engine, config, rep) ──────────────────────────────────────────
# NOTE: resource.getrusage(RUSAGE_CHILDREN).ru_maxrss is a monotonic HIGH-WATER
# MARK across this process's whole lifetime, not a per-child value — it never
# decreases. Measuring it via before/after deltas in one long-lived orchestrator
# is wrong: a later, smaller run gets shadowed by an earlier, larger one and
# reads back 0. Fix: re-invoke THIS script as a fresh one-shot "measuring
# child" per run (mirrors convert_bench.py's --run pattern, one level deeper —
# the actual work happens in a pixi grandchild). Each measuring child is a
# brand-new process, so its own RUSAGE_CHILDREN starts at 0 and its post-run
# reading is exactly that run's peak.
def _measuring_child_main(payload: dict) -> None:
    engine = ENGINES[payload["engine"]]
    cmd = engine.build_cmd(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        x_storage=payload["x_storage"],
        backed=payload["backed"],
        cpus=payload["cpus"],
        row_chunk=payload["row_chunk"],
        col_chunk=payload["col_chunk"],
        shard_factor=payload["shard_factor"],
    )
    env = {**os.environ, **PINNED_ENV}
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=engine.cwd, env=env, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    peak_rss_mb = _rss_to_mb(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    out_mb = dir_size_mb(Path(payload["output_path"]))
    ok = proc.returncode == 0
    result = {
        "wall_s": round(wall_s, 3),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "out_mb": round(out_mb, 2),
        "returncode": proc.returncode,
        "ok": ok,
        "stderr_tail": None if ok else proc.stderr[-2000:],
    }
    print("RESULT_JSON:" + json.dumps(result))


def run_once(
    engine: Engine,
    *,
    input_path: Path,
    output_path: Path,
    x_storage: str,
    backed: bool,
    cpus: int,
    row_chunk,
    col_chunk,
    shard_factor,
) -> dict:
    rmtree_if_exists(output_path)
    payload = {
        "engine": engine.name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "x_storage": x_storage,
        "backed": backed,
        "cpus": cpus,
        "row_chunk": row_chunk,
        "col_chunk": col_chunk,
        "shard_factor": shard_factor,
    }
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--internal-measure", json.dumps(payload)],
        capture_output=True,
        text=True,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None)
    if line is None:
        return {
            "wall_s": None,
            "peak_rss_mb": None,
            "out_mb": dir_size_mb(output_path),
            "returncode": proc.returncode,
            "ok": False,
            "stderr_tail": (proc.stderr or "")[-2000:] + "\n[no RESULT_JSON line]",
        }
    return json.loads(line[len("RESULT_JSON:") :])


def summarize(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"median": round(median, 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


# ── provenance ──────────────────────────────────────────────────────────────
def provenance() -> dict:

    def ver(pkg: str, cwd: Path, pixi_env: str) -> str:
        try:
            out = subprocess.run(
                [
                    "pixi",
                    "run",
                    "-e",
                    pixi_env,
                    "python",
                    "-c",
                    f"import importlib.metadata as m; print(m.version('{pkg}'))",
                ],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip() or "MISSING"
        except Exception:
            return "MISSING"

    versions = {
        "old_env": {p: ver(p, OLD_DIR, "dev") for p in ("zarr", "anndata", "numpy", "numcodecs", "dask")},
        "new_env": {p: ver(p, NEW_DIR, "default") for p in ("zarr", "anndata", "numpy", "numcodecs")},
    }
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=DATASCALE_ROOT, text=True).strip()
    except Exception:
        git = "unknown"
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cpus_reported": os.cpu_count(),
        "python_version_this_process": sys.version,
        "git_short": git,
        "pinned_env": PINNED_ENV,
        "versions": versions,
    }


# ── CLI startup timing ──────────────────────────────────────────────────────
def cli_startup(repeats: int = 3) -> dict:
    results = {}
    for name, engine in ENGINES.items():
        cmd = engine.version_cmd()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            subprocess.run(cmd, cwd=engine.cwd, capture_output=True, text=True)
            times.append(time.perf_counter() - t0)
        results[name] = {"cmd": " ".join(cmd), **summarize(times), "raw_s": [round(t, 3) for t in times]}
    return results


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--json",
        default=str(
            HERE.parent.parent.parent
            / "benchmarking_results"
            / "writer_engine"
            / "results"
            / "writer_engine_bench.json"
        ),
        help="raw results JSON path",
    )
    ap.add_argument("--cases", nargs="+", default=list(CASE_DESC), choices=list(CASE_DESC))
    ap.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    ap.add_argument("--repeats", type=int, default=3, help="measured repeats (after 1 discarded warm-up)")
    ap.add_argument("--outdir", default=str(DATA_DIR / "out"), help="scratch dir for converted stores")
    ap.add_argument("--internal-measure", help=argparse.SUPPRESS)  # internal one-shot measuring child
    args = ap.parse_args()

    if args.internal_measure:  # measuring-child path: run one config, report, exit
        _measuring_child_main(json.loads(args.internal_measure))
        return

    ensure_datasets()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    max_cpus = os.cpu_count() or 1
    cases = [c for c in build_cases(max_cpus) if c["case"] in args.cases]

    prov = provenance()
    prov["max_cpus_used"] = max_cpus
    print("== provenance ==", file=sys.stderr)
    print(json.dumps(prov, indent=2), file=sys.stderr)

    all_runs = []
    config_rows = []  # one row per (engine, case, cpus) with summary stats

    for case in cases:
        dataset = D1_H5AD if case["dataset"] == "D1" else D2_H5AD
        for cpus in case["cpus"]:
            for engine_name in args.engines:
                engine = ENGINES[engine_name]
                out_path = outdir / f"{engine_name}_{case['case']}_cpu{cpus}.zarr"
                cfg_label = f"{engine_name}/{case['case']}/cpus={cpus}"
                print(f"-> {cfg_label}: {CASE_DESC[case['case']]} ({case['dataset']})", file=sys.stderr)

                # 1 discarded warm-up
                warm = run_once(
                    engine,
                    input_path=dataset,
                    output_path=out_path,
                    x_storage=case["x_storage"],
                    backed=case["backed"],
                    cpus=cpus,
                    row_chunk=case["row_chunk"],
                    col_chunk=case["col_chunk"],
                    shard_factor=case["shard_factor"],
                )
                if not warm["ok"]:
                    print(f"   WARMUP FAILED rc={warm['returncode']}\n{warm['stderr_tail']}", file=sys.stderr)

                reps = []
                for r in range(args.repeats):
                    res = run_once(
                        engine,
                        input_path=dataset,
                        output_path=out_path,
                        x_storage=case["x_storage"],
                        backed=case["backed"],
                        cpus=cpus,
                        row_chunk=case["row_chunk"],
                        col_chunk=case["col_chunk"],
                        shard_factor=case["shard_factor"],
                    )
                    reps.append(res)
                    all_runs.append(
                        {"engine": engine_name, "case": case["case"], "cpus": cpus, "rep": r, "warmup": False, **res}
                    )
                    tag = "OK" if res["ok"] else f"FAIL rc={res['returncode']}"
                    print(
                        f"   rep {r + 1}/{args.repeats}: {res['wall_s']}s "
                        f"peak_rss={res['peak_rss_mb']}MB out={res['out_mb']}MB [{tag}]",
                        file=sys.stderr,
                    )

                ok_reps = [r for r in reps if r["ok"]]
                wall_summary = summarize([r["wall_s"] for r in ok_reps]) if ok_reps else None
                rss_summary = summarize([r["peak_rss_mb"] for r in ok_reps]) if ok_reps else None
                out_mb = ok_reps[-1]["out_mb"] if ok_reps else None
                config_rows.append(
                    {
                        "engine": engine_name,
                        "case": case["case"],
                        "case_desc": CASE_DESC[case["case"]],
                        "dataset": case["dataset"],
                        "x_storage": case["x_storage"],
                        "backed": case["backed"],
                        "cpus": cpus,
                        "row_chunk": case["row_chunk"],
                        "col_chunk": case["col_chunk"],
                        "shard_factor": case["shard_factor"],
                        "n_ok": len(ok_reps),
                        "n_total": len(reps),
                        "wall_s": wall_summary,
                        "peak_rss_mb": rss_summary,
                        "out_mb": out_mb,
                    }
                )
                rmtree_if_exists(out_path)  # delete large output between configs

    startup = cli_startup(repeats=3)

    payload = {"provenance": prov, "cli_startup": startup, "configs": config_rows, "raw_runs": all_runs}
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nraw results -> {json_path}", file=sys.stderr)

    # summary table to stdout
    hdr = (
        f"{'engine':6s} {'case':5s} {'cpus':>4s} {'median_s':>9s} {'min_s':>7s} {'max_s':>7s}"
        f" {'rss_med_mb':>10s} {'out_mb':>8s}  desc"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in config_rows:
        w = row["wall_s"] or {"median": None, "min": None, "max": None}
        rss = row["peak_rss_mb"] or {"median": None}
        print(
            f"{row['engine']:6s} {row['case']:5s} {row['cpus']:>4d} "
            f"{w['median']!s:>9s} {w['min']!s:>7s} {w['max']!s:>7s} "
            f"{rss['median']!s:>10s} {row['out_mb']!s:>8s}  {row['case_desc']}"
        )

    print("\n== CLI startup (median of 3) ==")
    for name, res in startup.items():
        print(f"{name:6s} {res['median']}s  (min {res['min']}, max {res['max']})  [{res['cmd']}]")


if __name__ == "__main__":
    main()
