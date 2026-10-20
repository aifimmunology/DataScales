"""latent (run 2): slice of our full-data scVI/PeakVI latent (models/<ds>/full_r1/latent.npy) to the subset rows."""
from .. import data

USES_RMM = True


def needs_cluster(ctx):
    return False


def load(ctx):
    ctx.data = data.load_own_latent(ctx.rid.ds, ctx.rows, ctx.params)


def run(ctx):
    import cupy as cp
    emb, ctx.data = ctx.data, None
    return cp.asarray(emb, dtype=cp.float32), "X_latent"
