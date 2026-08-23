from typing import Tuple

import numpy as np

#: Nomes das 9 features (grid 3x3 achatado) — usados nos gráficos
FEATURE_NAMES = [f"p{r}{c}" for r in range(3) for c in range(3)]


def create_stripes_dataset(
    n_samples_per_class: int = 100,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Padrão handcrafted "stripes" (listras horizontais vs. verticais em
    um grid 3x3 com ruído), no mesmo estilo de
    refactor_project.data.cross_circle.create_circle_cross_dataset.

    Retorna:
      X: (2*n_samples_per_class, 9) float32
      Y: (2*n_samples_per_class, 1) float32  (0=horizontal, 1=vertical)
    """
    horizontal_template = np.array(
        [[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32
    )
    vertical_template = np.array(
        [[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.float32
    )

    rng = np.random.default_rng(seed)

    X_horizontal = []
    X_vertical = []

    for _ in range(n_samples_per_class):
        noise = rng.normal(0, noise_std, (3, 3))
        horiz = np.clip(horizontal_template + noise, 0, 1)
        X_horizontal.append(horiz.reshape(-1))

        noise = rng.normal(0, noise_std, (3, 3))
        vert = np.clip(vertical_template + noise, 0, 1)
        X_vertical.append(vert.reshape(-1))

    X = np.vstack([np.asarray(X_horizontal), np.asarray(X_vertical)]).astype(np.float32)
    Y = (
        np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
        .astype(np.float32)
        .reshape(-1, 1)
    )

    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_stripes_pool_flatten(cfg, percent_total: int, seed: int):
    """
    Compatível com load_circle_cross_pool_flatten.

    percent_total=100 -> usa N_BASE_PER_CLASS por classe (1000 => 2000 total)
    percent_total=20  -> usa 20% de N_BASE_PER_CLASS por classe
    """
    N_BASE_PER_CLASS = 1000  # mesmo valor base usado em cross_circle.py

    frac = max(1, int(round(N_BASE_PER_CLASS * (percent_total / 100.0))))

    X, Y = create_stripes_dataset(
        n_samples_per_class=frac,
        noise_std=0.1,
        seed=int(seed),
    )
    return X, Y