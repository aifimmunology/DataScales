"""TF-IDF + randomized GPU SVD for the ATAC uncorrected baseline (from run 1's pca_lsi strategy)."""
SVD_OVERSAMPLE, SVD_POWER_ITERS = 50, 15


def tfidf(X):
    """Binary csr (cupy) -> log1p(tf * idf * 1e4); tf = 1/peaks-in-cell, idf = n_cells/cells-with-peak (Signac method 1)."""
    import cupy as cp
    n = X.shape[0]
    per_cell = cp.diff(X.indptr).astype(cp.float32)
    n_cells_with = cp.bincount(X.indices, minlength=X.shape[1]).astype(cp.float32)
    idf = n / cp.maximum(n_cells_with, 1)
    tf = 1.0 / cp.maximum(per_cell, 1)
    X.data = cp.log1p(cp.repeat(tf, cp.diff(X.indptr).astype(cp.int64)) * idf[X.indices] * 1e4).astype(cp.float32)
    return X


def randomized_svd(A, k, seed=0, oversample=SVD_OVERSAMPLE, n_iter=SVD_POWER_ITERS):
    """Halko range finder with QR-stabilised power iterations; returns U*S (n, k) and S (k)."""
    import cupy as cp
    At = A.T.tocsr()
    Q = cp.linalg.qr(A @ cp.random.RandomState(seed).standard_normal((A.shape[1], k + oversample), dtype=cp.float32))[0]
    for _ in range(n_iter):
        Q = cp.linalg.qr(A @ cp.linalg.qr(At @ Q)[0])[0]
    _, S, Vyt = cp.linalg.svd(At @ Q, full_matrices=False)
    return ((Q @ Vyt.T) * S)[:, :k], S[:k]
