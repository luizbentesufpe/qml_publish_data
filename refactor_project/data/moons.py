from typing import Tuple

import numpy as np
from pytest import Config


def create_make_moons_dataset(
    n_samples: int = 400,
    noise: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera o dataset Two Moons via sklearn, normalizado para [0, 1]².

    Retorna:
        X : (n_samples, 2)  float32  — features x0, x1
        Y : (n_samples, 1)  float32  — labels 0 ou 1
    """
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # normaliza para [0, 1] — compatível com angle encoding (θ = π·x)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X = scaler.fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)

    # shuffle determinístico
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_make_moons_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten.

    percent_total=100 → 400 amostras (200 por classe)
    percent_total=60  → ~240 amostras (usado no SEARCH)
    """
    N_BASE = 400
    n = max(4, int(round(N_BASE * percent_total / 100.0)))
    # garante par para balanceamento
    if n % 2 != 0:
        n += 1
    return create_make_moons_dataset(n_samples=n, noise=0.15, seed=seed)
