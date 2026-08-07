# MegaZarr Build

Assembling one large, sorted Zarr store from several single-cell cohorts — how to gather the
inputs, get the metadata to a standard schema, and merge everything into a store that supports
contiguous range reads.

The build spec for the current 5-cohort store lives at `~/megazarr/JOIN_SPEC2.md`.

## Folder layout

```
build_megazarr/
├── README.md                      ← this file: the general recipe
├── hise_datagrab.ipynb            ← pull cohort h5ads + metadata from HISE
└── metadata_cleanup/              ← getting obs to a standard, complete schema
    ├── metadata_cleanup.ipynb     ← the main cleanup: per-cohort obs → one cleaned parquet
    ├── altra-fix.ipynb            ← ALTRA kit→subject join, exploratory
    ├── altra_meta_fix.py          ← same join as a script
    └── schema.md                  ← CZI CELLxGENE schema, kept as an external reference
```

## General steps to reproduce

**1. Gather the h5ads.** One modality at a time (this build is scRNA only). Pull each cohort from
HISE with `hise_datagrab.ipynb`. Cohorts in other study spaces need their own credentials, so
expect to chase a few separately. Confirm up front that every file is the same modality, has raw
counts (not normalized), and shares a gene axis — mismatched gene sets force an intersection step
that is easy to avoid by asking for the right export.

**2. Build one cell × metadata table.** This is the step that takes real calendar time, because it
means bugging analysts and scientists for the *final* versions of labels and sample metadata.
Dump each cohort's obs to a parquet, then assemble them into a single table with one row per cell
and every metadata column you intend to keep. The goal is a table that is **complete** — no nulls
in anything that isn't null by design.

**3. Clean and standardize that table.** See `metadata_cleanup/`:

- Settle on a canonical column set. The AIFI scRNA download schema is the reference:
  <https://apps.allenimmunology.org/aifi/insights/dynamics-imm-health-age/downloads/scrna/>
- Normalize names across cohorts (dot vs underscore, singular vs plural) and drop per-cohort
  artifacts, derived QC, clustering and embedding columns.
- Join in missing sample/subject metadata from the authoritative CSV per cohort.
- **Regenerate the cell-label hierarchy from the deepest tier.** Only the finest labels (L3 here)
  are hand-curated; the coarser tiers in source obs drift out of sync with them. Rebuild L1/L2
  from L3 via a committed mapping file rather than carrying them forward.
- Filter out rows that don't belong (cells from another cohort that got included in an export).
- Harmonize dtypes so the per-cohort frames concatenate cleanly.

Write the result as one parquet. That file is the authority for both which cells are in the store
and what obs they carry.

**4. Stage each h5ad against the cleaned table.** Replace each cohort's obs with the canonical
columns and drop any rows not in the cleaned table, writing a **new** file rather than editing the
source. Do this before sorting — the source h5ads hold stale labels, so sorting on their own obs
gives the wrong order.

Work **one file end-to-end at a time** (stage → sort → delete the staged copy → next). Staged
copies are nearly as large as the sources, so materializing them all at once can easily exceed the
disk you have; sequentially, peak usage is one staged file. Start with the biggest.

**5. Sort each file individually.** Use the in-house convert tool's backed sort, which streams and stays
memory-bounded:

```bash
cd tools/convert-to-zarr          # the tool has its own pixi env
pixi run convert-to-zarr convert-h5ad --input <staged.h5ad> --output <sorted.zarr> \
  --backed --overwrite --sparse-flat-chunk <derived, see below> \
  --sort-by <your keys>
```


**6. Merge the sorted stores.** The classic approach is a streaming k-way merge: buffer a chunk
from each sorted input and repeatedly take the smallest next key. That works, but if you already
have a cleaned table with every cell's sort keys, the global order is *already known* — so do the
comparisons once, vectorized, in metadata-space, and let the merge be a run-length copy: walk the
sorted table and copy each contiguous same-source run in one slice, with a monotonic cursor per
input. Identical sequential I/O, minus ~one Python-level operation per cell.

Make the tie-break explicit by appending `(source_id, local_row)` to the sort keys. That guarantees
each input's rows appear in increasing local order, so a single cursor per input is provably
correct instead of relying on stable-sort luck. When the primary key is a cohort or batch ID, runs
tend to be enormous — here 30 M cells collapsed to 663 runs.

Then write obs and var, and consolidate metadata.

**7. Validate.** Row and gene counts, per-cohort counts, unique barcodes, rows non-decreasing in
the sort keys, label vocabularies the expected size, label hierarchy a strict tree, and a
spot-checked cell's counts matching its source file. Check nulls **after** the concat, not
per-cohort — a per-cohort pass can't see columns that only some cohorts have.

Note what those checks *can't* catch: they're all self-consistency against the table that produced
obs, so none of them detects X being misaligned with obs. Comparing the final obs to the cleaned
table is a tautology, since obs was written from it. The check that bites is to keep each sorted
intermediate's own obs, concatenate them in merge order, and assert equality with the final store's
barcodes — do that before deleting the intermediates.

Record provenance with the result: library versions, repo commit, host/CPU, shape + nnz + dtype,
chunk/codec layout, input file hashes, and per-step wall times.

## What we did for the current 5 cohorts

ALTRA, UP1, MM, BRI (SoundLife) and COVID — ~31 M raw cells of PBMC scRNA, raw `uint16` CSR.

The first build (`~/megazarr/Megazarr_h5ads/JOIN_SPEC.md`) harmonized obs itself: it renamed each
cohort's columns to a canonical set, derived MM's mito columns, intersected genes down to MM's
31,915, deduplicated overlapping barcodes with a cohort precedence rule, and sorted on four keys.
It produced a 30.07 M × 31,915 store.

Two problems showed up afterward. Several cohorts' L1/L2 labels disagreed with their own L3, so the
merged store's label hierarchy wasn't a tree — and `AIFI_L1` had been used as a sort key, so fixing
the labels invalidated part of the physical order. Separately, the COVID and MM exports both
contained BRI cells, which the first build absorbed as duplicate barcodes rather than filtering.

The cleanup in `metadata_cleanup/` addressed that outside the build: it standardized names,
regenerated L1/L2 from the curated L3, filtered the foreign cohort rows, and emitted one cleaned
parquet as the single source of truth for obs. The label vocabularies now match the published AIFI
schema (9 / 29 / 71) and the hierarchy is a strict tree.

The v2 build (`~/megazarr/JOIN_SPEC2.md`) takes obs verbatim from that parquet. Two steps from v1
disappear as a result: MM was re-exported with the full 33,538 genes so there is no gene
intersection, and the per-cohort row filters make the cohorts disjoint so there is no barcode
dedup. Duplicate cells now resolve to the SoundLife copy.
