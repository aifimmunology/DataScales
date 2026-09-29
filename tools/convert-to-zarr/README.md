# convert-to-zarr (merged)

Merged into [`tools/annizarr`](../annizarr/README.md) (2026-09).

Command mapping:
- `convert-h5ad --input X --output Y` → `annizarr convert X -o Y`
- `convert-10x-h5` → `annizarr convert X -o Y` (auto-detected)
- `concat-h5ads --inputs A B` → `annizarr convert A B -o Y`

Last standalone version: `git show 2dba863:tools/convert-to-zarr/`
