# zarrsmith

Edit existing AnnData Zarr v3 stores: derive expression layers, rechunk, physically sort, and
append cells. Builds on the [convert-to-zarr](../convert-to-zarr/README.md) package for its
shared core (readers, writers, chunk layout, config, Icechunk storage) — creating stores from
`.h5ad`/10x inputs lives there.

Design notes and the op-by-op spec are in [zarrsmith-buildspec.md](zarrsmith-buildspec.md).

## Install

Requires Python 3.10+. Run commands from this tool's directory (`tools/zarrsmith`) after
cloning; the local `convert-to-zarr` checkout is installed automatically as a path dependency.

```bash
cd tools/zarrsmith
pixi install
```

## Ops

```bash
# add a log-normalized expression layer (layers/gexp) for gene queries — csc, dense, or csr
pixi run zarrsmith add-expr --store path/to/store.zarr --format csc --chunk-elems 1000000

# rewrite a matrix with new chunking into a new store (everything else copied as-is)
pixi run zarrsmith rechunk --store in.zarr --output out.zarr --array X --x-row-chunk 2048 --x-col-chunk 512

# physically sort by obs column(s) into a new store (streamed, memory-bounded, CSR X)
pixi run zarrsmith sort --store in.zarr --output sorted.zarr --by AIFI_L1 batch_id

# append the cells of another zarr store, in place (strict var/obs/dtype match)
pixi run zarrsmith append --store store.zarr --cells new_cells.zarr [--drop-obsp] [--refresh-expr]
```

`add-expr` and `append` mutate the store in place (one commit with `--icechunk`); `rechunk` and
`sort` always write a new store. Appending refuses to silently invalidate things: obsp graphs
need `--drop-obsp`, an existing gexp layer needs `--refresh-expr` (re-derived after the append),
and a previously sorted store should be re-sorted. `sort` re-derives a lone `layers/gexp` on the
sorted output automatically.

Icechunk inputs are auto-detected; pass `--icechunk` to write outputs through an Icechunk
repository.

## Development

```bash
pixi run -e dev pytest tests/ -v
```

## License

MIT
