# AnniZarr — final build spec

Supersedes `One_Datascale.md`. Folds `tools/convert-to-zarr` (creation + shared core),
`tools/zarrsmith` (store editing) and `tools/scizarr_IC` (Icechunk versioning) into one
package, **`annizarr`**, built in place at `tools/annizarr/` on branch `AnniZarr`.

Decisions were confirmed with Alex on 2026-09-28 unless marked *assumption*. The code
conventions in §3 apply to **all** code in the package, moved code included; Phase 1c is the
sweep that brings the existing implementation up to them.

---

## 1. Decisions (locked)

| Topic | Decision |
|---|---|
| Name | Brand **AnniZarr**; import name, PyPI name, CLI name all `annizarr`; short CLI alias `anz`. PyPI `annizarr` free as of 2026-09-28. *Assumption:* lowercase for import/CLI. |
| Where | Built in place at `tools/annizarr/`. The three old dirs stay untouched as reference until the merged suite and golden tests are green, then their code is deleted and each gets a stub README. `git filter-repo` to a standalone repo is Alex's step afterwards. |
| Scope of this workflow | Phases 1–5 in code + Phase 6 files. **Not** here: PyPI reservation, publishing, filter-repo, filing the anndata upstream issue. |
| Dask | Replaced by `ThreadPoolExecutor` and **deleted**. Golden stores generated from the old code first. |
| scanpy | Dropped. h5py reader for 10x HDF5 (v3 `matrix/features/*`, v2 `<genome>/genes,gene_names`). scanpy only in `dev` for the reference test. |
| icechunk | **Optional extra** `annizarr[icechunk]`, one `require_icechunk()` guard raising `StorageError` with the install hint (`ImportError` cannot share a base with `RuntimeError`, and the CLI must catch it as an `AnzError`). Plain zarr paths never import it. |
| GCS | `gs://` repos open/read/write via `icechunk.gcs_storage`. `ic copy` is local + `s3://` (boto3 via `[s3]`). No gs copy. |
| Compatibility | **Clean break.** No forwarders, no `scizarr_ic` shim; env vars `ANNIZARR_ORIGIN`, `ANNIZARR_HOME`; HEAD file `annizarr_head`. |
| CLI shape | Flat ops `convert`, `add-expr`, `rechunk`, `sort`, `append`; Icechunk commands under `ic`. Positional inputs, `-o/--output`, positional REPO for `ic`. Global `-v/-q`. |
| convert auto-detect | Format sniffed from HDF5 contents (h5ad vs 10x); `--from h5ad\|10x` overrides. **Several inputs = concat.** |
| Icechunk output | New stores are plain zarr unless `--ic`; remote output without `--ic` is an error. Existing repos as inputs or in-place targets are auto-detected. |
| Storage enum | *Assumption:* `x_storage` becomes `Literal["csr", "csc", "dense"]` everywhere (config, CLI `--x-storage`, `add-expr --format`). The old `sparse-csr`/`sparse-csc` spellings are not accepted (clean break). |
| Hard deps | `anndata>=0.12.10,<0.13`, `zarr>=3.3,<4`, `numpy`, `scipy`, `h5py`, `PyYAML`. |
| Extras | `icechunk` (icechunk + boto3), `dev`. No `s3`/`all` (Alex, 2026-09-29). |
| Versions | `requires-python >=3.12` — **corrected in Phase 4**: every zarr ≥3.2 release and every icechunk wheel require 3.12, so a 3.10 floor was never installable. CI 3.12–3.13. Lifted with evidence: `zarr>=3.3,<4` (3.4.0 keeps all 16 goldens byte-identical), `icechunk>=2.1.2,<3` (2.2.2), numpy unbounded (2.5.3). Keep `anndata<0.13` with reason + issue link. |
| Dev env pins | Phases 1–3 pinned `zarr 3.3.*`, `icechunk 2.1.*`, `numpy <2.5` so the move and the goldens were not confounded by a library bump; Phase 4 lifted them one at a time (see Versions). Envs: `default` (3.13, all extras), `core` (hard deps + pytest), `py312` (floor check). |
| Versioning | *Assumption:* static `version = "0.1.0"` while in the monorepo; `hatch-vcs` after the repo split. |
| Errors | One hierarchy rooted at `AnzError`; children `ConversionError`, `StorageError`, `ValidationError`, `RepoError`. CLI catches `AnzError` only. |
| Output of ops | *Assumption:* every op returns a frozen `OpResult(path, n_obs, n_vars, snapshot_id)`; warnings go through `logging`, not return values. |
| Mutation policy | Additive ops (`append`, `add-expr`) write in place; `convert`, `rechunk`, `sort` write a new store to `OUT.tmp-<uuid>`, verify, rename. One commit per op on an Icechunk repo, through `Repo`. |
| Autosharding (added 2026-09-29) | Writers partition concurrent writes from the **created array's** grid (`_layout.write_grid(arr)` = shards if sharded else chunks), never from config, so any sharding decision by zarr or anndata keeps writes aligned. `ChunkConfig.auto_shard` (CLI `--auto-shard/--no-auto-shard`) opts anndata-written elements and our 1-D sparse arrays into `shards="auto"`; **default `False`**: measured on a 100k × 2k CSR store, zarr's heuristic (2 chunks per shard) cut objects by 32% but slowed row reads 15–36% (`benchmarking_results/autoshard/`). Goldens are self-snapshots of annizarr's own output, 16 unsharded + 6 autoshard cases. |
| Memory | Every op has a streaming path and works on a store larger than RAM. The eager path is an optimisation used only below a documented threshold (`io.eager_max_bytes`, default 2 GiB, *assumption*). |

---

## 2. Package layout

anndata-style: implementation lives in `_`-prefixed modules and packages; public names are
re-exported from un-prefixed modules. "What is public" is exactly what is importable without an
underscore. Each private package is one ownership boundary and one agent work unit.

```
tools/annizarr/
  pyproject.toml            # hatchling; static version; extras; scripts; ruff/mypy/pytest/coverage config
  pixi.toml + pixi.lock     # contributor env: default = [dev,icechunk,s3]; core = no extras
  README.md  CHANGELOG.md  CITATION.cff  LICENSE
  src/annizarr/
    __init__.py             # _runtime.pin_blas() first; then the public surface (§6) with __all__
    __main__.py             # python -m annizarr → _cli.main
    errors.py               # PUBLIC  AnzError, ConversionError, StorageError, ValidationError, RepoError
    typing.py               # PUBLIC  XStorage, Backend, PathLike aliases
    config.py               # PUBLIC  re-exports from _config
    sources.py              # PUBLIC  re-exports from _sources (extension point)
    ic.py                   # PUBLIC  re-exports from _ic (Repo, DEFAULT_BRANCH)
    _version.py
    _config.py              # frozen slots dataclasses, load_config, apply_overrides, resolve_backend_cfg
    _runtime.py             # pin_blas, configure_runtime, run_parallel(mode=…), stage() timer → logging
    _layout.py              # PURE  DenseLayout/SparseLayout/BandPlan, dense_shards, band_plan, x_compressors, BATCH_BYTES
    _validation.py          # PURE  ValidationResult, validate_single_cell_anndata
    _zarr.py                # typed zarr accessors (as_array/get_group/...): the one place the Array|Group union is narrowed
    _sorting.py             # PURE  compute_sort → SortPlan;  I/O  stream_sorted_store(plan, …, commit_message)
    _sources/               # I/O read
      __init__.py           #   registry: open_source, register_source, detect_format
      _base.py  _detect.py  _h5ad.py  _tenx.py  _memory.py  _matrix.py
    _storage/               # I/O open
      __init__.py           #   open_output_store, open_store_rw, open_input_group
      _uri.py (PURE)  _backends.py  _open.py
    _writers/               # I/O write (anndata-zarr encoding)
      __init__.py
      _encoding.py  _dense.py  _sparse.py  _concat.py  _adata.py  _workers.py
    _ops/                   # source(s) → plan → writer → storage
      __init__.py
      _result.py            #   OpResult, AppendPlan
      _convert.py  _concat.py  _expr.py  _rechunk.py  _sort.py  _append.py
    _ic/                    # Icechunk versioning
      __init__.py
      _repo.py  _head.py  _copy.py
    _cli/
      __init__.py           #   main(), root parser, logging handler (-v/-q), AnzError handler
      _args.py  _convert.py  _edit.py  _ic.py
  tests/                    # §9
  benchmarking/convert_bench.py
```

### Pure core vs I/O edge

| Pure (plain-data in, plain-data out; tested without fixtures, hypothesis where numeric) | I/O edge (touches disk, network, or a session) |
|---|---|
| `_layout` (chunk/shard/codec/band math), `_validation`, `_config`, `_storage/_uri`, `_sorting.compute_sort`, `_sources/_detect` (given an opened handle), `_ops/_append.plan_append` (reads metadata only, returns `AppendPlan`) | `_sources/*` loaders, `_storage/_backends`, `_storage/_open`, `_writers/*`, `_sorting.stream_sorted_store`, `_ops/*` execution, `_ic/*` |

### Old → new mapping

| Old | New |
|---|---|
| `convert_to_zarr/errors.py`, `storage.StorageError`, `validation.ValidationError`, `scizarr_ic/errors.py` | `errors.py` |
| `convert_to_zarr/config.py` | `_config.py` (+ `IOConfig.eager_max_bytes`, `backed: bool \| None` = auto; `x_storage: XStorage`). `branch`/`message` are per-op keyword arguments, not config fields. |
| `convert_to_zarr/engine.py` | `_runtime.py`; worker fns → `_writers/_workers.py` |
| `convert_to_zarr/layout.py` | `_layout.py` (+ plan dataclasses, single `BATCH_BYTES`) |
| `convert_to_zarr/sources.py` | `_sources/{_h5ad,_matrix}.py` + registry |
| `convert_to_zarr/storage.py` + `scizarr_ic/storage.py` | `_storage/{_uri,_backends,_open}.py` |
| `convert_to_zarr/writers.py` | `_writers/{_encoding,_dense,_sparse,_concat,_adata,_workers}.py` |
| `convert_to_zarr/sorting.py` | `_sorting.py` |
| `convert_to_zarr/ops/{convert,concat}.py` | `_ops/{_convert,_concat}.py` |
| `zarrsmith/{expr,rechunk,sort,append}.py` | `_ops/{_expr,_rechunk,_sort,_append}.py` |
| `scizarr_ic/{repo,head,copy}.py` | `_ic/{_repo,_head,_copy}.py` |
| three `cli.py` | `_cli/` |

Private symbols zarrsmith imports from convert-to-zarr today (`_resolve_backend_cfg`, `_stage`,
`_run_parallel_threads`, `_store_name`, `_is_s3_url`, `_x_compressors`, `_dense_shards`,
`_stream_sorted_store`, plus `expr._introspect_gexp`/`_lognorm_band`) become ordinary
functions inside the private packages. Nothing outside `annizarr` needs them.

---

## 3. Code conventions (apply to all code, moved code included)

**Style**
- Public API (`__all__` members) gets numpydoc docstrings: summary line, Parameters, Returns,
  Raises where non-obvious. No Examples unless the call is genuinely non-obvious. Private
  functions and methods on internal classes get no docstring.
- One-line comments only for real constraints ("shard-aligned so writes are disjoint",
  "anndata <0.13 writes a None key here"), never narration. No module headers, author lines,
  or section dividers.
- `from __future__ import annotations` at the top of every module; type hints everywhere;
  heavy imports under `if TYPE_CHECKING:`; runtime imports of optional deps inside functions.
  `Literal[...]` for string enums, `numpy.typing.NDArray`, `str | os.PathLike[str]` for paths.
- `mypy --strict` on `src/` in CI and pre-commit. `# type: ignore[code]  # reason` only.
- Keyword-only after the data argument: `rechunk(store, *, output, array="X", …)`.
- Small pure functions; side effects at the edges (§2 table). Layout, validation and plan
  building return `@dataclass(frozen=True, slots=True)` objects; no ad-hoc dicts cross module
  boundaries.
- One exception hierarchy rooted at `AnzError`. CLI catches `AnzError` and prints the message;
  anything else is a bug and gets a traceback.
- No `print` in the library. `logging.getLogger(__name__)`, `NullHandler` on the package
  logger, quiet by default. Progress/stage timings at INFO, warnings at WARNING. The CLI
  installs the stderr handler: default INFO, `-q` WARNING, `-v` DEBUG.
- `_` prefix for private modules; public names re-exported from `__init__.py` and the
  un-prefixed facade modules.
- Ruff, `ruff format` (not black), line length **120** (scverse convention; fewer wrapped
  keyword-heavy signatures). Rules `E, F, W, I, UP, B, TID, RUF, D` with
  `pydocstyle.convention = "numpy"`; `D` ignored on `**/_*.py` and `tests/**`; `D100`/`D104`
  (module/package docstrings) off. Zero `# noqa` without a reason on the same line.

**Implementation rules**
- Never read a full matrix into memory unless it is below `io.eager_max_bytes`; every path
  through convert and edit ops streams. Enforced by one `-m slow` test on a synthetic
  50k × 30k sparse store (≈1 % density) run in a subprocess with a peak-RSS ceiling.
- Atomic outputs: new stores go to `OUT.tmp-<uuid>` beside the target, are verified
  (`ad.read_zarr` opens, shapes match), then renamed. A partial store never sits at the target.
- Encoding metadata is sacred: every element gets `encoding-type`/`encoding-version`; a shared
  test helper `assert_anndata_readable(path)` runs on every store the suite produces.
- Idempotent commands: `add-expr` on a store that already has the layer is an error unless
  `--overwrite` (then a replacement, never a duplicate). `append` whose obs names are **all**
  already present is an error ("already appended"); partial overlap stays a warning because
  barcodes legitimately collide across samples (*assumption*).

---

## 4. Seams (what makes future formats cheap)

**Sources (input).** `Source` = `adata` (eager or backed), `kind`, `backed`, `close()`.
Loaders and sniffers are registered by kind. `detect_format(path)`: `.h5ad` → h5ad;
`.h5`/`.hdf5` → h5py sniff (`matrix/barcodes` → 10x v3; genome groups with `barcodes`+`genes`
→ 10x v2; `obs`+`var` groups → h5ad; else an error naming `--from`); directory with
`zarr.json` → zarr store; directory with `repo` + `snapshots/`, or a remote URI → icechunk
repo. A new input format = one `_sources/_<kind>.py` with a loader, an optional sniffer, and
`register_source(...)`. Concat accepts any list of sources.

**Storage (location/backend).** Three seams: `open_output_store(path, cfg, *,
commit_message)`, `open_store_rw(path, cfg, *, commit_message)`, `open_input_group(path, *,
branch=None, snapshot_id=None)`. Each returns a `zarr.Group` plus `finalize()`. Icechunk
branches go through `Repo` (lazy import) so branch/HEAD/origin semantics live once.
`finalize()` = `repo.commit(message)` for icechunk; consolidate + atomic rename for plain zarr.

**Writers (format).** Take a `zarr.Group` and a matrix/AnnData plus a layout plan; know nothing
about paths or backends. All encoding attrs and the single `anndata._io.specs.write_elem`
import live in `_writers/_encoding.py`, verified against `.claude/vendor/anndata/src`. A future
zarr-to-other-format op is a new writer module and a `convert` target switch.

**Ops.** source(s) → validation → plan → writer → storage seam; 100–300 lines each.

**Repo.** API unchanged from scizarr-ic: `Repo(path, *, branch=None, origin=None)`, `create`,
`init`, `exists`, `copy`, `open_zarr("r"|"w", *, snapshot_id=None)`, `commit`, `discard`,
`log`, `tree`, `checkout(name, *, create=False)`, `cherrypick`, `branches`, `branch`, `resolved`.

---

## 5. CLI surface

```
annizarr [-v|-q] convert   INPUT [INPUT …] -o OUT [--from h5ad|10x] [--backed|--eager] [--x-storage csr|csc|dense]
                           [--cpus N] [--x-row-chunk] [--x-col-chunk] [--sparse-flat-chunk] [--x-shard-factor]
                           [--sort-by COL …] [--obs-columns COL …]       # --obs-columns only with ≥2 inputs
                           [--consolidate-metadata] [--overwrite] [--ic] [--branch B] [-m MSG] [--config FILE]
annizarr add-expr  STORE [--format csr|csc|dense] [--layer gexp] [--chunk-elems N] [--target-sum 1e4]
                   [--overwrite] [--branch B] [-m MSG] [--config FILE]
annizarr rechunk   STORE -o OUT [--array X] [chunk flags] [--cpus N] [--overwrite] [--consolidate-metadata]
                   [--ic] [--branch B] [-m MSG] [--config FILE]
annizarr sort      STORE -o OUT --by COL … [--cpus N] [--overwrite] [--consolidate-metadata]
                   [--ic] [--branch B] [-m MSG] [--config FILE]
annizarr append    STORE CELLS [--drop-derived] [--extend-layers] [-y] [--branch B] [-m MSG] [--config FILE]
annizarr ic init       SRC.zarr REPO [-m MSG]
annizarr ic log        REPO [-b BRANCH] [--oneline] [--origin URL]
annizarr ic tree       REPO [--origin URL]
annizarr ic checkout   REPO BRANCH [-b] [--origin URL]
annizarr ic cherrypick REPO SNAPSHOT [--origin URL]
annizarr ic copy       REPO DEST [--origin URL]
annizarr --version
anz …
```

`append` in the CLI calls `plan_append`, shows the loss plan, and prompts (or honours `-y` /
`--drop-derived`) before calling `append`; the library never prompts. Default commit message
`"annizarr <op> <key params>"`. Progress to stderr via logging; stdout stays clean.

---

## 6. Python API (public, `__all__`)

```python
import annizarr as az

az.convert(inputs, *, output, cfg=None, fmt=None) -> OpResult          # PathLike | AnnData | Sequence[PathLike]
az.add_expr(store, *, fmt="csc", layer="gexp", chunk_elems=1_000_000, target_sum=1e4,
            overwrite=False, cfg=None, branch=None, message=None) -> OpResult
az.rechunk(store, *, output, array="X", cfg=None, branch=None, message=None) -> OpResult
az.sort(store, *, output, by, cfg=None, branch=None, message=None) -> OpResult
az.plan_append(store, *, cells) -> AppendPlan                             # pure: metadata only
az.append(store, *, cells, drop_derived=False, extend_layers=False, cfg=None,
          branch=None, message=None) -> OpResult
az.OpResult, az.AppendPlan
az.Repo                                    # also annizarr.ic.Repo, annizarr.ic.DEFAULT_BRANCH
az.AppConfig, az.load_config               # full set in annizarr.config
az.AnzError, az.ConversionError, az.StorageError, az.ValidationError, az.RepoError
annizarr.sources: Source, Loader, Sniffer, open_source, register_source, detect_format
annizarr.typing:  XStorage, Backend, PathLike
```

`cfg=None` means `load_config()` defaults. Typed helpers (`convert_h5ad`, `convert_10x_h5`,
`convert_adata`, `concat`) live in `_ops` and are not public.

---

## 7. pyproject.toml (target)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "annizarr"
version = "0.1.0"                     # → hatch-vcs after the repo split
description = "AnnData zarr stores: convert, edit, and version with Icechunk"
readme = "README.md"
requires-python = ">=3.12"   # zarr>=3.3 and icechunk wheels need 3.12
license = "MIT"
authors = [{ name = "Alex Holly", email = "alex.holly@alleninstitute.org" }]
dependencies = [
  "anndata>=0.12.10,<0.13",   # <0.13: phantom None layer key breaks the layer writer (issue link TBD)
  "zarr>=3.3,<4",
  "numpy>=1.24",
  "scipy>=1.10",
  "h5py>=3.8",
  "PyYAML>=6",
]

[project.optional-dependencies]
icechunk = ["icechunk>=2.1.2,<3", "boto3>=1.28"]
dev      = ["pytest>=8", "pytest-cov", "hypothesis", "ruff", "mypy", "types-PyYAML",
            "moto[s3,server]>=5", "scanpy>=1.10", "build", "twine", "pre-commit"]

[project.scripts]
annizarr = "annizarr._cli:main"
anz      = "annizarr._cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/annizarr"]

[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "TID", "RUF", "D"]
ignore = ["D100", "D104", "D105", "D107"]
[tool.ruff.lint.pydocstyle]
convention = "numpy"
[tool.ruff.lint.per-file-ignores]
"src/annizarr/**/_*.py" = ["D"]
"src/annizarr/_*.py"    = ["D"]
"tests/**"              = ["D"]

[tool.mypy]
strict = true
files = ["src/annizarr"]
warn_unused_ignores = true

[tool.pytest.ini_options]
addopts = "-m 'not slow' --cov=annizarr --cov-fail-under=85"
markers = ["slow: large synthetic store, run with -m slow"]
[tool.coverage.run]
omit = ["src/annizarr/_cli/*", "src/annizarr/__main__.py"]
```

`pixi.toml` (separate): conda-forge; `python = ">=3.12,<3.14"` for dev; environments
`default` (extras dev+icechunk) and `core` (no extras).

---

## 8. Phases, work units, agents

Model policy: **Sonnet** for implementation and review; **Haiku** for mechanical units. Fable
writes prompts, reads reports, runs the suite, intervenes on failures. Every unit gets the
relevant §§ of this spec, the vendored-source protocol from `CLAUDE.md`, a disjoint
file-ownership list, and a report template (files touched, pytest tail, deviations). Every
reviewer checks §3 conventions as well as behaviour.

### Phase 1 — restructure

| Unit | Model | Owns | Depends |
|---|---|---|---|
| 1a.1 scaffold + move converter & editor | Sonnet | `tools/annizarr/{pyproject,pixi}.toml`, `src/annizarr/{__init__,__main__,errors,typing,config,sources}.py`, `_config,_runtime,_layout,_validation,_sorting`, `_sources/`, `_writers/`, `_ops/`, moved tests | — |
| 1a.2 move ic | Sonnet | `ic.py`, `_ic/`, `tests/test_ic_*.py`, env-var and HEAD-file renames | — (parallel) |
| 1b.1 storage + errors merge | Sonnet | `_storage/`, `errors.py`, `require_icechunk` | 1a |
| 1b.2 CLI | Sonnet | `_cli/`, `tests/test_cli.py` (subprocess `annizarr`/`anz`) | 1a (parallel with 1b.1, 1b.3) |
| 1b.3 sources registry + `convert()` dispatch + sniffing | Sonnet | `_sources/`, `_ops/_convert.py`, `tests/test_sources_detect.py` | 1a (parallel) |
| 1c conventions sweep | Sonnet ×2 (split: core+writers+ops / sources+storage+ic+cli) | logging instead of print, `from __future__`, full type hints + `mypy --strict` clean, kw-only signatures, `OpResult`, `plan_append`/`AppendPlan`, `frozen+slots` dataclasses, layout/validation plan objects, numpydoc on `__all__`, ruff config + format, remove double exception wrapping, one `BATCH_BYTES`, one `make_sparse_group`, one `run_parallel`, merged `conftest.py` with scoped zarr-v3 fixture and `assert_anndata_readable` | 1b |
| R1 review | Sonnet | read-only | 1c |

Acceptance: `pixi run pytest` green in `tools/annizarr`; `pixi run -e core python -c "import annizarr"`;
`ruff check`, `ruff format --check`, `mypy --strict` clean; coverage ≥ 85 %; old dirs untouched.
Dask and scanpy still present (temporary `_sources/_tenx.py` wraps scanpy; writers still dask).

### Phase 2 — drop scanpy

| Unit | Model | Owns |
|---|---|---|
| 2.1 h5py 10x reader | Sonnet | `_sources/_tenx.py`, `tests/test_sources_tenx.py` |

`var` columns match scanpy's (`gene_ids`, `feature_types`, `genome` for v3; `gene_ids` for v2);
on-disk CSC transposed to CSR (cells × genes). Reference test `pytest.importorskip("scanpy")`
on synthetic v2 and v3 files. The agent verifies scanpy's behaviour from the installed source in
the old `convert-to-zarr/.pixi` env, not memory.

### Phase 3 — drop dask

| Unit | Model | Owns | Depends |
|---|---|---|---|
| 3.0 golden generation | Haiku | `tests/golden/*.tar.gz` + `README.md`, produced with the **old** `tools/convert-to-zarr` env from fixed-seed 100 × 50 fixtures: dense eager, dense backed, sparse-as-dense, csr eager, csr backed, csc, concat csr, concat dense, sorted eager + backed, sharded dense | — |
| 3.1 writers rewrite | Sonnet | `_writers/{_dense,_sparse,_concat}.py`, `_runtime` progress at INFO, dask removed from deps | 3.0 |
| 3.2 hypothesis for layout | Sonnet | `tests/test_layout_props.py` (`dense_shards`, `band_plan`, `_uri`, `compute_sort` contiguity) | 1c |
| 3.3 benchmark | Sonnet | `benchmarking_results/writer_engine/` (README + JSON): old vs new, wall + peak RSS + size, one large dense (≈100k × 2k float32) and one sparse-as-dense, subprocess per run, cpus ∈ {1, 4, 8}, BLAS pinned and recorded | 3.1 |
| R3 review | Sonnet | focus: chunk alignment, no read-modify-write, no locks, memory = workers × band | 3.1 |

Writer design: in-memory dense → thread pool over whole-shard blocks from `DenseLayout`, one
row-band slice per task; in-memory CSR/CSC → thread pool over chunk-aligned flat segments (no
lock, no RMW); dense concat → output row-chunk-aligned bands gathering across input seams
(removes today's seam RMW); CSR concat → aligned flat segments spanning seams; backed inputs keep
the process pool (h5py is not thread-safe); backed dense stays serial. Acceptance: golden
tarballs match file-for-file; suite green; dask absent from `pyproject`, `pixi.lock`, imports.

### Phase 4 — deps & bounds

| Unit | Model | Owns |
|---|---|---|
| 4.1 extras, guards, CI | Sonnet | `pyproject.toml`, boto3 guard, `tests/test_import_guards.py`, moto test for s3 `copy_repo`, pre-commit (ruff, ruff-format, mypy), `.github/workflows/annizarr.yml` (jobs: `all` on 3.12/3.13 with `.[all,dev]`, `core` without extras, `build` + twine, anndata pre-release `continue-on-error`) |

Acceptance: `python -m build && twine check dist/*`; core env shows the friendly install-hint `StorageError`;
numpy upper bound removed if the suite passes on the resolved latest.

### Phase 5 — invariants: versioned, atomic, idempotent, memory-bounded

| Unit | Model | Owns | Depends |
|---|---|---|---|
| 5.1 storage through Repo | Sonnet | `_storage/_open.py`, `IOConfig.branch`, `_cli/_args.py` (`--branch`, `-m`) | 1 |
| 5.2 op wiring + atomic outputs | Sonnet | `_ops/*`, `_sorting.stream_sorted_store(commit_message)`, `OUT.tmp-<uuid>` + verify + rename, idempotency rules (incl. `append` "all obs names already present" → error; R1 finding 2), `open_input_group(path, *, branch=None, snapshot_id=None)` (R1 finding 5), `tests/test_versioned_edits.py`, `tests/test_atomic.py` | 5.1 |
| 5.3 streaming everywhere | Sonnet | streamed CSR→CSC for `convert --x-storage csc` from backed input (reuse the `_expr` bucket engine in `_writers/_sparse.py`), `io.eager_max_bytes` auto-select + CLI `--eager` (R1 finding 6), `tests/test_large_store.py` (`-m slow`, 50k × 30k, subprocess peak-RSS ceiling) | 3 |
| R5 review | Sonnet | read-only | 5.3 |

Acceptance: icechunk lifecycle (`convert --ic` → `add-expr` → `append` → `sort --ic`) shows one
commit per op with the right message on the chosen branch; a killed rechunk leaves no partial
output; re-running `add-expr` errors without `--overwrite`; slow test passes under the ceiling.

### Phase 6 — ship files (no publish)

| Unit | Model | Owns |
|---|---|---|
| 6.1 docs | Sonnet | `README.md` (pitch, install lines, 10-line quickstart, compatibility table), `CHANGELOG.md`, `CITATION.cff`, `LICENSE` |
| 6.2 monorepo wiring | Haiku | stub READMEs + code deletion in the three old dirs (**only after Phase 2 and 3 acceptance**), delete old workflows, add `publish.yml` (trusted publishing on `v*`, disabled until the split), root `README.md`, `CLAUDE.md` repo-map rows |

---

## 9. Testing

- Merged suites (≈92 tests) move with import and monkeypatch-string updates; warning
  assertions switch to `caplog`. One `conftest.py`: 100 × 50 synthetic AnnData fixtures,
  `zarr.config.set` per test (no session-wide autouse mutation), `assert_anndata_readable`.
- `@pytest.mark.parametrize` over `XStorage` × dtype (`float32`, `float64`, `int32`) × chunk
  layout (unsharded, sharded, tiny chunks) for convert, rechunk, add-expr.
- Golden-file tests: tarball equality for chunk bytes **and** a `zarr.json` snapshot per array
  (chunk grid, shards, codecs, fill value, attrs) so a refactor cannot silently change layout.
- `hypothesis` for `_layout`, `_uri`, `compute_sort`.
- New: `test_sources_detect.py`, `test_sources_tenx.py`, `test_golden_writers.py`,
  `test_layout_props.py`, `test_import_guards.py`, `test_versioned_edits.py`, `test_atomic.py`,
  `test_large_store.py` (`-m slow`), moto s3 copy test, CLI subprocess test.
- Full default suite under one minute; coverage gate 85 % on `src/`, CLI omitted.
- **Pruned 2026-09-29 at Alex's request:** 164 tests (one per behaviour/invariant; parametrised only along axes that take different code paths) and 9 golden cases (one per writer path + one autoshard); coverage unchanged at 93.3 %. Under a minute without coverage.

---

## 10. Doc-vs-code discrepancies fixed along the way

1. Temp + atomic swap for rechunk/sort/convert: claimed, not implemented → 5.2.
2. `sort` on icechunk commits as `"convert-to-zarr convert (sorted) → …"` → 5.2.
3. `convert_h5ad_to_zarr`/`convert_h5ads_to_zarr` double-wrap inner errors → 1c.
4. `gs://` documented but `copy` refuses it; `gcs://` a silent third scheme → keep open-only;
   `copy` error says "local and s3:// only" → 1b.1.
5. `Repo(path, branch=…)` does not persist HEAD (only `checkout` does) → documented, kept.
6. Old README claims (`--cpus`/`--backed`, 64 MB cap) → 6.1.

---

## 11. Budget and agent protocol

- Target: under $200 total (raised by Alex 2026-09-28); warn and pause when close.
- **Actuals (Sonnet subagent tokens):** surveys 0.25M · Phase 1 2.83M (1a 0.51M, 1b 0.68M, 1c 1.03M, R1+fix 0.57M) · Phase 2 0.17M · Phase 3 0.9M (goldens 0.19M, props 0.10M, writers ≈0.4M, R3+fix 0.25M, bench 0.15M) · Phase 4 0.31M · Phase 5 1.44M (5.1+5.2 0.33M, 5.3 0.42M, R5+fix 0.69M) · Phase 6 0.22M · autosharding follow-up 0.57M (unit 0.43M, review 0.14M). **Total ≈ 7.0M Sonnet tokens; all six phases + the autosharding follow-up complete 2026-09-29.**
- Estimate (Sonnet tokens): Phase 1 ≈ 2.3M (1c sweep adds ≈ 0.8M), Phase 2 ≈ 0.2M, Phase 3 ≈
  1.0M, Phase 4 ≈ 0.2M, Phase 5 ≈ 0.8M, Phase 6 ≈ 0.3M, reviews ≈ 0.7M → ≈ 5.5M. Fable turns
  capped at about three per phase.
- After every phase: one short message with cumulative subagent tokens and status. If the
  projection crosses ~$80, work pauses for a go/no-go.
- Every unit runs `pytest`, `ruff check`, `ruff format --check`, `mypy` itself and pastes the
  tails; Fable re-runs once per phase. A Sonnet reviewer reads each phase diff against §3 and the
  `CLAUDE.md` silent-performance-killers list before acceptance.
- Agents never commit. Fable commits one or two times per phase under Alex's git identity, no Claude attribution lines, commit messages that mark the phase/unit (Alex, 2026-09-28).

---

## 12. Alex's steps (outside this workflow)

Reserve `annizarr` on PyPI (0.0.1 placeholder); file the anndata phantom-None-layer-key issue
and paste the link into the pyproject comment; `git filter-repo` into the standalone repo and
switch to hatch-vcs; enable trusted publishing; update the Code Ocean capsule to `annizarr`
(`Repo` API unchanged, env vars renamed, `x_storage` spellings changed).

## 13. Out of scope

Dask/distributed engine; gs copy; MuData; several distributions; `push`/`pull` between repos;
tags; converting *from* zarr to other formats (writer seam prepared, not exercised);
pyright (mypy chosen).
