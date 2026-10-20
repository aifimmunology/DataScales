"""Build the run-2 cell-type subsets (all / tnk / b): 01_make_subsets.py <ds> ..."""
import sys

from scvi_pca import config, subsets


def main(ds_list):
    out = []
    for ds in ds_list:
        m = subsets.build(ds, config.out_root() / "subsets" / ds)
        out.append(m)
        print(f"[ok] {ds}: " + ", ".join(f"{s}={m[m.subset == s].n.sum():,}" for s in config.SUBSETS))
        for s in ("tnk", "b"):
            print(f"     {s}: {dict(m[m.subset == s].set_index('label').n)}")
    import pandas as pd
    dest = config.REPO / "results" / "subsets_manifest.parquet"
    pd.concat(out).to_parquet(dest, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
