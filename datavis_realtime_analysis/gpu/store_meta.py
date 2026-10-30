"""Store conventions shared by the backend and the GPU pipeline: the attributes a view's
root carries, and the latent space the store's own neighbor graph was built from."""

LABEL_ATTR = "datavis-label"
PARAMS_ATTR = "datavis-params"


def latent_rep(root) -> str | None:
    """The obsm key recorded in uns/neighbors/params/use_rep, if the store has it."""
    from anndata.io import read_elem

    params = root.get("uns/neighbors/params")
    if params is None or "use_rep" not in params:
        return None
    rep = str(read_elem(params["use_rep"]))
    return rep if rep in root["obsm"] else None
