"""Check a store before pointing the app at it: python -m server.validate <path or gs:// prefix> [--deep]
Errors mean the app cannot serve it; warnings mean a feature is missing."""

import sys
from concurrent.futures import ThreadPoolExecutor

from store_meta import latent_rep

from .config import MAX_BODY_BYTES

READ_THREADS = 16
SPARSE = ("csr_matrix", "csc_matrix")


def _column_problem(obs, n_obs: int, name: str) -> tuple[str | None, int | None]:
    try:
        col = obs[name]
        if col.attrs.get("encoding-type") == "categorical":
            return (None if col["codes"].shape == (n_obs,) else f"codes have {col['codes'].shape} rows, not {n_obs}"), \
                col["categories"].shape[0]
        shape = getattr(col, "shape", None)
        return (None if shape is None or shape == (n_obs,) else f"has shape {shape}, not ({n_obs},)"), None
    except Exception as e:
        return f"unreadable: {e}", None


def inspect_store(uri: str, deep: bool = False, max_body: int = MAX_BODY_BYTES) -> dict:
    import zarr

    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {"uri": uri}
    try:
        root = zarr.open_group(uri, mode="r", use_consolidated=False)
    except Exception as e:
        return {"ok": False, "errors": [f"cannot open {uri} as a zarr group: {e}"], "warnings": [], "info": info,
                "capabilities": {}}
    if root.attrs.get("encoding-type") != "anndata":
        warnings.append("root is not tagged encoding-type=anndata (a MuData root? point at mod/<name>)")

    n_obs = None
    obs = root.get("obs")
    if obs is None:
        errors.append("no obs group")
    else:
        idx = obs.attrs.get("_index", "_index")
        if idx not in obs:
            errors.append(f"obs has no index column '{idx}'")
        else:
            n_obs = obs[idx].shape[0]
            info["n_obs"] = n_obs
            cols = list(obs.attrs.get("column-order", []))
            with ThreadPoolExecutor(READ_THREADS) as pool:
                checked = list(pool.map(lambda c: _column_problem(obs, n_obs, c), cols))
            for c, (problem, _) in zip(cols, checked):
                if problem:
                    errors.append(f"obs/{c}: {problem}")
            info["categorical"] = {c: k for c, (_, k) in zip(cols, checked) if k is not None}
            if deep:
                import numpy as np

                names = obs[idx][:]
                if len(np.unique(names)) != len(names):
                    errors.append("obs index has duplicate barcodes")
                info["barcode_chars"] = int(max((len(str(b)) for b in names[:100_000]), default=0))

    umap = root.get("obsm/X_umap")
    if umap is None:
        errors.append("no obsm/X_umap: the viewer has nothing to draw")
    elif len(umap.shape) != 2 or umap.shape[1] != 2 or (n_obs is not None and umap.shape[0] != n_obs):
        errors.append(f"obsm/X_umap has shape {umap.shape}, expected ({n_obs}, 2)")

    latent = None
    try:
        latent = latent_rep(root)
    except Exception as e:
        warnings.append(f"could not read the recorded latent space: {e}")
    X = root.get("X")
    x_enc = X.attrs.get("encoding-type") if X is not None else None
    info.update(latent=latent, X=x_enc)
    can_reembed = latent is not None or x_enc in SPARSE
    if not can_reembed:
        warnings.append("no latent space (uns/neighbors/params/use_rep -> obsm) and no sparse X: Generate New UMAP is unavailable")
    elif latent is None and x_enc == "csc_matrix":
        warnings.append("X is CSC: row selection for re-embedding will be slow (CSR is strongly preferred)")
    can_genes = "layers/gexp" in root or x_enc in (*SPARSE, "array")
    if not can_genes:
        warnings.append("no layers/gexp or X: the Genes panel will be hidden")

    if n_obs:
        worst = n_obs * (len(str(n_obs)) + 2) + 1024  # digits + ", " per index
        info["worst_request_bytes"] = worst
        if worst > max_body:
            errors.append(f"selecting all {n_obs} cells makes a ~{worst >> 20} MB request, over the {max_body >> 20} MB body limit")

    info["views"] = len(root["umap_views"]) if "umap_views" in root else 0
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info,
            "capabilities": {"view": umap is not None, "label": obs is not None, "reembed": can_reembed,
                             "latent": latent is not None, "genes": can_genes}}


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    report = inspect_store(args[0], deep="--deep" in argv)
    for key in ("errors", "warnings"):
        for line in report[key]:
            print(f"{key[:-1].upper()}: {line}")
    print(f"info: {report['info']}\ncapabilities: {report['capabilities']}")
    print("OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
