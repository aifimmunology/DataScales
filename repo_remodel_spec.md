# Repo Remodel Spec — Datascales Project

**Status:** executing on branch `Datascales_Project_Layout`. Tools relocated under `tools/`
(git-tracked renames, history preserved), per-tool pyprojects added, hub README/pyproject
trimmed, project folders scaffolded — **all 60 tests green**. Pending: `benchmarks/` folder
fate, CLAUDE.md path sweep, per-tool `pixi install`.

## Goal

Rebrand and restructure the current `DataScale` repo into a **single central repo (monorepo)**
that holds several **self-contained, installable tool packages** plus a set of **project folders**,
all served by **one Claude agent** at the root. This preserves the existing commit history (we keep
working in this repo — it *becomes* the hub) while giving each tool its own package boundary.

## Decisions locked (with rationale)

- **Monorepo of plain folders, NOT git submodules.** One continuous history (which we like), no
  submodule mechanics, and no cross-repo sync to maintain. A change to a tool is a normal commit,
  instantly visible everywhere — the "auto-update" problem never arises.
- **Each `tools/*` folder is a self-contained installable package** — its own `pyproject.toml`
  (+ pixi env), `src/`, `tests/`, `README`, and CLI. `pip install -e tools/<tool>` works. Logically
  separate, physically in one repo.
- **Each project folder = README + whatever scripts it needs** (NOT an installable package).
  Projects consume the tools; they are experiments/analyses, not libraries.
- **One agent at the root.** `.claude/` (skills + vendored library source) and `CLAUDE.md` stay at
  the repo root and see every tool and project. Tools do NOT each carry their own `vendor/`/skills —
  vendor is agent-reference, not a runtime dependency.
- **No `datascale-core` package yet (YAGNI).** Shared code (e.g. `storage.py`, `config.py`,
  `validation.py`) starts inside `<convert>`. If `<merge-sort>` genuinely needs
  `storage.open_output_store` (the icechunk/zarr switch), decide *then*: copy it, depend on
  `<convert>`, or extract a small shared package. Don't pre-build it.
- **`<merge-sort>` has a NEW purpose.** It operates **zarr ↔ zarr**: concatenate multiple zarr
  stores and **resort a large zarr**. This is distinct from the converter's h5ad-level small
  concat/sort. The two tools should be clearly separated in the README so no one grabs the wrong one.
- **Docs = READMEs only.** No published docs site / handbook. The "why" lives in the root README;
  findings live in `Benchmarks/README`.

## Naming (decided)

| Role | Package/module (kept for now) | Tool folder / brand |
|------|-------------------------------|---------------------|
| umbrella / hub repo | `datascales-project` | repo root |
| convert tool | `convert_to_zarr` (import) · `convert-to-zarr` (dist/CLI) | `tools/convert-to-zarr` |
| query benchmark | `zarr_query_bench` (`python -m …`) | `tools/zarr-query-bench` |
| merge/sort (deferred) | new (carved from `converter.py`) | `tools/<merge-sort>` — not started |

Package/module identifiers **renamed**: `datascale → convert_to_zarr` (dist/CLI `convert-to-zarr`)
and `zarr_query_benchmarking → zarr_query_bench`.
Env model: **one pixi env per tool** (each `tools/*` has its own `pyproject.toml`); the root is a
hub env for the project folders/notebooks.

## Target layout

```
Datascales-Project/               ← one repo, one history (this repo, remodeled)
├── README.md                     ← the "why" + index of tools & projects
├── CLAUDE.md                     ← hub agent brief, spans all tools + projects
├── .claude/                      ← the single agent: skills/ + vendor/ (one canonical copy)
├── tools/                        ← self-contained installable packages
│   ├── <convert>/     pyproject.toml · src/ · tests/ · README
│   ├── <query-bench>/ pyproject.toml · src/ · tests/ · README
│   └── <merge-sort>/  pyproject.toml · src/ · tests/ · README
├── Benchmarks/                   ← project: rapids/zarr findings (README + raw JSON/CSV + scripts)
├── IceChunk-MultiUser/           ← project: README + scripts
├── Rapids-User-Notebook/         ← project: standard-usage notebook + README
└── MegaZarr/                     ← project: multi-cohort giant-zarr build scripts + explainer README
```

## Tools vs. projects

- **Tools** (`tools/*`): reusable, installable, tested, versionable packages. Own env, own CLI.
- **Projects** (`Benchmarks/`, `IceChunk-MultiUser/`, `Rapids-User-Notebook/`, `MegaZarr/`):
  README + scripts. Pin/record which tool version produced a result (provenance rules still apply).

## Migration map (existing → destination)

| Current path | → Destination | Notes |
|--------------|---------------|-------|
| `datascale/` (converter, cli, config, storage, validation) | `tools/<convert>/src/` | git-tracked move, history follows |
| `example_config.toml` | `tools/<convert>/` | |
| zarr↔zarr concat + large-zarr resort logic in `converter.py` | `tools/<merge-sort>/src/` | **carved out** of converter.py — fresh code, no clean path history |
| `zarr_query_benchmarking/` | `tools/<query-bench>/src/` | git-tracked move |
| `zarr_dbs/` (fixture stores) | `tools/<query-bench>/` or `Benchmarks/` | decide — heavy; keep out of git if large |
| `tests/test_converter,_config,_validation` | `tools/<convert>/tests/` | |
| `tests/test_zarr_query_benchmarking` | `tools/<query-bench>/tests/` | |
| `tests/test_features` (icechunk RT + concat + sort) | split `<convert>` / `<merge-sort>` | |
| existing `benchmarks/` (e.g. `convert_bench.py`) | `Benchmarks/` | **macOS is case-insensitive** — `benchmarks/` and `Benchmarks/` are the same dir; fold, don't duplicate |
| `data/`, `cache/` | decide: leave / gitignore / drop | likely working data, not moved |
| root `pyproject.toml`, `pixi.lock`, `datascale.egg-info` | retire / split per tool | each tool owns its own env now |
| `.claude/`, `CLAUDE.md`, `README.md`, `LICENSE` | stay at root (hub) | README/CLAUDE repurposed to hub scope |

## Build plan (phases)

- **Phase 0 — Branch.** Commit the in-flight WIP (`datascale/cli.py`, `datascale/converter.py`,
  `tests/test_features.py`) on `Zarr_Usage_Features` first, then branch
  `Datascales_Project_Layout` off it so the remodel starts clean.
- **Phase 1 — Scaffold.** Create `tools/` + the four project folders (each with a README stub).
  Repurpose root `README.md` → hub landing; update `CLAUDE.md` → hub agent brief.
- **Phase 2 — Reorganize in place.** Move existing code into `tools/<convert>` and
  `tools/<query-bench>` (git mv, history follows); carve `<merge-sort>` out of `converter.py`;
  give each tool its own `pyproject.toml` + pixi env + tests + README.
- **Phase 3 — Cleanup.** Retire the root package/pyproject, resolve `data/`/`cache/`/`zarr_dbs/`,
  refresh `.gitignore`.

## Open items to resolve before building

1. **Final tool names** (see Naming table).
2. **WIP commit** on `Zarr_Usage_Features` before branching — confirm.
3. **Fate of `benchmarks/` (fold into `Benchmarks/`), `data/`, `cache/`, `zarr_dbs/`.**
4. **`<merge-sort>` ↔ `storage.py`** dependency: copy / depend-on-convert / extract-core — decide
   when the real shared surface is visible.

## Notes

- **Escape hatch (folders → own repo later).** Any `tools/*` folder can be promoted to a standalone
  repo *with full history* via `git subtree split -P tools/<tool> -b <branch>` when a concrete need
  appears (external consumer wants it alone, independent release tags, or different access control).
  Going folders-first keeps this option open; merging separate repos back is the harder direction —
  so we don't pay that cost until a trigger actually shows up.

- **Appendix — if we ever switch to submodules.** Not planned, kept for reference. Submodules pin a
  commit on purpose (no auto-follow). To automate pointer bumps: `.github/dependabot.yml` with
  `package-ecosystem: "gitsubmodule"` opens bump PRs on a schedule (simplest); a scheduled Action
  running `git submodule update --remote` is the alternative; `repository_dispatch` from each tool's
  CI gives near-instant bumps (needs a PAT). Local ergonomics: set `branch` per submodule in
  `.gitmodules` + `git submodule update --remote`, and `git config submodule.recurse true`.
