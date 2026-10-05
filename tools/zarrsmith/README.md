# zarrsmith (merged)

This tool was folded into **AnniZarr**, a bigger and more polished tool that now lives in its own
repository: https://github.com/A-Jolly-Holly/annizarr (`pip install annizarr` once it is published).

Command mapping:
- `add-expr --store S` → `annizarr add-expr S`
- `rechunk --store S --output O` → `annizarr rechunk S -o O`
- `sort --store S --output O --by C` → `annizarr sort S -o O --by C`
- `append --store S --cells C` → `annizarr append S C`

Last standalone version: `git show 2dba863:tools/zarrsmith/`
