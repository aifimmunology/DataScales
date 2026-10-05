# convert-to-zarr (merged)

This tool was folded into **AnniZarr**, a bigger and more polished tool that now lives in its own
repository: https://github.com/A-Jolly-Holly/annizarr (`pip install annizarr` once it is published).

Command mapping:
- `convert-h5ad --input X --output Y` → `annizarr convert X -o Y`
- `convert-10x-h5` → `annizarr convert X -o Y` (auto-detected)
- `concat-h5ads --inputs A B` → `annizarr convert A B -o Y`

Last standalone version: `git show 2dba863:tools/convert-to-zarr/`
