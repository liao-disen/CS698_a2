from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def pca(x: np.ndarray, n_components: int) -> np.ndarray:
    if n_components <= 0 or n_components >= x.shape[1]:
        return x
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:n_components].T
    return xc @ basis


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be >= 1")
    n = x.shape[0]
    if n < k:
        raise ValueError(f"need n >= k, got n={n}, k={k}")

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()

    assign = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, iters)):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assign = d2.argmin(axis=1)
        if np.array_equal(assign, new_assign):
            break
        assign = new_assign

        for c in range(k):
            idx = np.where(assign == c)[0]
            if len(idx) == 0:
                centers[c] = x[rng.integers(0, n)]
            else:
                centers[c] = x[idx].mean(axis=0)

    return assign, centers
