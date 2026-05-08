#!/usr/bin/env python3
"""Simple cell-type query against a dense AnnData Zarr store.

Opens the store with zarr.open_group(), decodes the target cell-type column,
finds matching row indices, then pulls only the required row-chunks from X.

Usage:
    pixi run python dev/query_cell_type.py
"""

import numpy as np
import zarr

STORE_PATH = "zarr_dbs/health_atlas_dense_1k_1k.zarr"
CELLTYPE_COL = "AIFI_L3"   # which obs column to search (AIFI_L1, AIFI_L2, AIFI_L3)
TARGET_LABEL = "CLP cell"     # change to any label printed in "Available labels" below


# ── 1. Open store ─────────────────────────────────────────────────────────────

root = zarr.open_group(STORE_PATH, mode="r")

X = root["X"]
n_rows, n_cols = X.shape
row_chunk, col_chunk = X.chunks
print(f"X shape: {n_rows} cells × {n_cols} genes  |  chunk shape: {row_chunk}×{col_chunk}")


# ── 2. Decode categorical cell-type column ────────────────────────────────────
# Categorical columns are stored as two arrays:
#   categories  — the set of unique label strings
#   codes       — one integer per cell, indexing into categories

cat_group  = root["obs"][CELLTYPE_COL]
categories = cat_group["categories"][:]   # shape: (n_unique_labels,)  dtype: str
codes      = cat_group["codes"][:]        # shape: (n_cells,)          dtype: int

print(f"\nAvailable labels in '{CELLTYPE_COL}': {categories.tolist()}")


# ── 3. Find matching row indices ──────────────────────────────────────────────

label_to_code = {label: i for i, label in enumerate(categories.tolist())}

if TARGET_LABEL not in label_to_code:
    raise SystemExit(
        f"Label '{TARGET_LABEL}' not found. Choose from: {categories.tolist()}"
    )

target_code = label_to_code[TARGET_LABEL]
row_indices  = np.where(codes == target_code)[0]  # global row positions in X

print(f"\nFound {len(row_indices)} cells matching '{TARGET_LABEL}'")
print(f"Row indices (first 10): {row_indices[:10]}")


# ── 4. Work out which row-chunks are needed ───────────────────────────────────
# Each chunk covers rows [chunk_id * row_chunk : (chunk_id+1) * row_chunk].
# Group the matching row indices by their chunk id so we load each chunk once.

chunk_to_rows: dict[int, list[int]] = {}
for r in row_indices:
    chunk_id = int(r) // row_chunk
    chunk_to_rows.setdefault(chunk_id, []).append(int(r))

print(f"\nRow chunks needed: {sorted(chunk_to_rows.keys())}")


# ── 5. Load required chunks and extract matching rows ─────────────────────────
# zarr slices a contiguous block per chunk; inside that block the target rows
# are at local offset = global_row - chunk_start_row (fast array indexing, not scan).

result_blocks = []

for chunk_id, rows_in_chunk in sorted(chunk_to_rows.items()):
    r_start = chunk_id * row_chunk
    r_end   = min(r_start + row_chunk, n_rows)

    print(f"  Loading chunk {chunk_id}: rows [{r_start}:{r_end}]  "
          f"({len(rows_in_chunk)} matching rows inside)")

    # zarr fetches exactly this slice — only the chunks that overlap are read
    block = X[r_start:r_end, :]              # shape: (chunk_rows, n_cols)

    local_offsets = np.array(rows_in_chunk) - r_start
    result_blocks.append(block[local_offsets, :])  # direct row indexing, no scan

# Stack all matching rows into a single expression matrix
expression = np.vstack(result_blocks)
print(f"\nResult expression matrix: {expression.shape}  (cells × genes)")
print(f"First cell, first 10 gene values: {expression[0, :10]}")


# ── 6. Pull matching barcodes for reference ───────────────────────────────────

barcodes = root["obs"]["barcodes"][:]
matched_barcodes = barcodes[row_indices]
print(f"\nFirst 5 matched barcodes: {matched_barcodes[:5]}")
