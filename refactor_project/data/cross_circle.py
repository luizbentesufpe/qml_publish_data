from typing import Tuple

import numpy as np


def create_circle_cross_dataset(
    n_samples_per_class: int = 100,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      X: (2*n_samples_per_class, 9) float32
      Y: (2*n_samples_per_class, 1) float32  (0=círculo, 1=cruz)
    """
    circle_template = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    cross_template = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)

    rng = np.random.default_rng(seed)

    X_circles = []
    X_crosses = []

    for _ in range(n_samples_per_class):
        noise = rng.normal(0, noise_std, (3, 3))
        circle = np.clip(circle_template + noise, 0, 1)
        X_circles.append(circle.reshape(-1))

        noise = rng.normal(0, noise_std, (3, 3))
        cross = np.clip(cross_template + noise, 0, 1)
        X_crosses.append(cross.reshape(-1))

    X = np.vstack([np.asarray(X_circles), np.asarray(X_crosses)]).astype(np.float32)
    Y = (
        np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
        .astype(np.float32)
        .reshape(-1, 1)
    )

    # shuffle
    idx = rng.permutation(len(X))
    X = X[idx]
    Y = Y[idx]

    return X, Y


def load_circle_cross_pool_flatten(cfg, percent_total: int, seed: int):
    """
    Compatível com:
      X_full, Y_full = load_chestmnist_pool_flatten(cfg, percent_total=..., seed=...)

    percent_total aqui controla o tamanho do dataset final.
    Exemplo:
      percent_total=100 -> usa N_base por classe (ex: 100 por classe => 200 total)
      percent_total=20  -> usa 20% do N_base por classe
    """
    # Define um "tamanho base" por classe (se quiser maior, aumente aqui)
    N_BASE_PER_CLASS = 1000  # <- pode subir p/ 5000, 10000 etc (cuidado com custo)
    frac = max(1, int(round(N_BASE_PER_CLASS * (percent_total / 100.0))))

    # seed controla tudo
    X, Y = create_circle_cross_dataset(
        n_samples_per_class=frac,
        noise_std=0.1,
        seed=int(seed),
    )
    return X, Y
