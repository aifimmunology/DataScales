LABEL_ATTR = "datavis-label"
PARAMS_ATTR = "datavis-params"


def latent_rep(root) -> str | None:
    from anndata.io import read_elem

    params = root.get("uns/neighbors/params")
    if params is None or "use_rep" not in params:
        return None
    rep = str(read_elem(params["use_rep"]))
    return rep if rep in root["obsm"] else None
