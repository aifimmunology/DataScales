"""Ordered run-2 run ids for one dataset: 12_run_list.py <ds>. retrain on `all` is the setup run (02_train_full)."""
import sys

from scvi_pca import config

ORDER = ("b", "tnk", "all")


def ids(ds):
    rna = config.dataset(ds)["modality"] == "rna"
    strats = ["latent", "pca", "retrain"] if rna else ["latent", "retrain"]
    for sub in ORDER:
        for s in strats:
            if s == "retrain" and sub == "all":
                continue
            for rep in (1, 2):
                yield f"{ds}-{s}-{sub}-r{rep}"


if __name__ == "__main__":
    print("\n".join(ids(sys.argv[1])))
