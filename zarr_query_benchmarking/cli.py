"""Command-line interface for the zarr query benchmark.

Run the same query against one or more stores and print a comparison table.

    python -m zarr_query_benchmarking \
        --store zarr_dbs/health_atlas_csr_1000.zarr \
        --store zarr_dbs/health_atlas_dense_1k_1k.zarr \
        --axis obs --count 1000 --mode contiguous --repeats 5

Use ``--inspect`` to just report the layout of each store without timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import BenchmarkResult, benchmark_request
from .query import QueryRequest, inspect_store


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zarr_query_benchmarking",
        description="Benchmark pulling data from single-cell AnnData zarr stores "
        "(dense / sparse-CSR / CSC) and converting to a dense result.",
    )
    p.add_argument(
        "--store", action="append", required=True, metavar="PATH",
        help="Path to a zarr store. Repeat to compare multiple setups.",
    )
    p.add_argument(
        "--axis", choices=["obs", "var"], default="obs",
        help="Select cells (obs/rows) or genes (var/cols). Default: obs.",
    )
    p.add_argument(
        "--count", type=int, default=1000,
        help="Number of rows/cols to pull. Default: 1000.",
    )
    p.add_argument(
        "--mode", choices=["contiguous", "random"], default="contiguous",
        help="Contiguous block or random sample along the axis. Default: contiguous.",
    )
    p.add_argument(
        "--offset", type=int, default=0,
        help="Start index for contiguous mode. Default: 0.",
    )
    p.add_argument(
        "--final-format", choices=["dense", "csr"], default="dense",
        help="Format the pulled data is converted to before timing ends. "
        "'dense' streams and discards (summary only); 'csr' returns a single "
        "compact matrix. Default: dense.",
    )
    p.add_argument(
        "--array", dest="array_path", default="X",
        help="Node within the store to query (e.g. 'X' or 'layers/counts'). Default: X.",
    )
    p.add_argument("--repeats", type=int, default=5, help="Timed runs. Default: 5.")
    p.add_argument(
        "--warmup", type=int, default=1,
        help="Untimed warmup runs before timing. Default: 1.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random mode. Default: 0.")
    p.add_argument(
        "--output", type=Path, default=None,
        help="Write full results as JSON to this path.",
    )
    p.add_argument("--json", action="store_true", help="Print results as JSON to stdout.")
    p.add_argument(
        "--inspect", action="store_true",
        help="Only report each store's layout; do not run the benchmark.",
    )
    return p


def _print_inspect(stores: list[str], array_path: str) -> int:
    rc = 0
    for s in stores:
        try:
            info = inspect_store(Path(s), array_path)
            chunks = "x".join(map(str, info.chunks)) if info.chunks else "-"
            nnz = f", nnz={info.nnz}" if info.nnz is not None else ""
            print(
                f"{s}\n  format={info.storage_format} shape={info.shape} "
                f"dtype={info.dtype} data_chunks={chunks}{nnz}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{s}\n  ERROR: {exc}", file=sys.stderr)
            rc = 1
    return rc


def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'store':<40} {'src':<5} {'out':<5} {'result':<14} {'MB':>8} "
        f"{'nnz':>10} {'bands':>6} {'min(s)':>10} {'med(s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        name = r.request.store.name
        if not r.ok:
            print(f"{name:<40} ERROR: {r.error}")
            continue
        shp = "x".join(map(str, r.result_shape))
        print(
            f"{name:<40} {r.info.storage_format:<5} {r.request.final_format:<5} "
            f"{shp:<14} {r.result_nbytes / 1e6:>8.2f} {r.result_nnz:>10} "
            f"{r.n_bands:>6} {r.min_s:>10.4f} {r.median_s:>10.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.inspect:
        return _print_inspect(args.store, args.array_path)

    results: list[BenchmarkResult] = []
    for s in args.store:
        req = QueryRequest(
            store=Path(s),
            axis=args.axis,
            count=args.count,
            mode=args.mode,
            final_format=args.final_format,
            array_path=args.array_path,
            offset=args.offset,
            seed=args.seed,
        )
        results.append(benchmark_request(req, repeats=args.repeats, warmup=args.warmup))

    payload = [r.as_dict() for r in results]
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"axis={args.axis} count={args.count} mode={args.mode} "
            f"final={args.final_format} repeats={args.repeats} warmup={args.warmup}\n"
        )
        _print_table(results)

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.output}", file=sys.stderr)

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
