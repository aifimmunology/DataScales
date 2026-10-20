"""Chunked JAX silhouette (euclidean) matching scib-metrics semantics; fixed shapes so it compiles per size bucket."""
import jax
import jax.numpy as jnp
import numpy as np

CHUNK = 2048


@jax.jit
def _chunk_scores(q, own, X, sq, onehot, counts):
    d2 = jnp.sum(q * q, 1)[:, None] + sq[None, :] - 2.0 * q @ X.T
    S = jnp.sqrt(jnp.maximum(d2, 0.0)) @ onehot
    own_oh = jax.nn.one_hot(own, onehot.shape[1], dtype=S.dtype)
    n_own = counts[own]
    a = jnp.sum(S * own_oh, 1) / jnp.maximum(n_own - 1, 1)
    mean_other = jnp.where((own_oh > 0) | (counts[None, :] == 0), jnp.inf, S / jnp.maximum(counts[None, :], 1))
    b = jnp.min(mean_other, 1)
    s = (b - a) / jnp.maximum(a, b)
    return jnp.where((n_own > 1) & jnp.isfinite(b), jnp.nan_to_num(s), 0.0)


def _bucket(n):
    return max(CHUNK, 1 << int(np.ceil(np.log2(n))))


def silhouette_samples(X, codes):
    """Per-cell silhouette of integer cluster codes; X (n, d) float32."""
    n, L = len(X), int(codes.max()) + 1
    m = _bucket(n)
    Xp = np.zeros((m, X.shape[1]), np.float32)
    Xp[:n] = X
    oh = np.zeros((m, L), np.float32)
    oh[np.arange(n), codes] = 1.0
    Xd, ohd = jnp.asarray(Xp), jnp.asarray(oh)
    sq, counts = jnp.sum(Xd * Xd, 1), jnp.asarray(np.bincount(codes, minlength=L).astype(np.float32))
    out = np.empty(n, np.float32)
    for s in range(0, n, CHUNK):
        q = np.zeros((CHUNK, X.shape[1]), np.float32)
        own = np.zeros(CHUNK, np.int32)
        e = min(s + CHUNK, n)
        q[: e - s], own[: e - s] = X[s:e], codes[s:e]
        out[s:e] = np.asarray(_chunk_scores(jnp.asarray(q), jnp.asarray(own), Xd, sq, ohd, counts))[: e - s]
    return out


def asw_label(X, labels):
    """scib silhouette_label: mean silhouette rescaled to [0, 1]."""
    return float((silhouette_samples(X, labels).mean() + 1) / 2)


def asw_batch(X, labels, batch):
    """scib silhouette_batch: per label 1-|silhouette| over batch codes, mean over labels with >1 batch."""
    scores = []
    for lab in np.unique(labels):
        m = labels == lab
        b = np.unique(batch[m], return_inverse=True)[1]
        if b.max() < 1 or m.sum() == b.max() + 1:
            continue
        scores.append(float((1 - np.abs(silhouette_samples(X[m], b))).mean()))
    return float(np.mean(scores))
