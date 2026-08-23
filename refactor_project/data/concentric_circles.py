from typing import Tuple

import numpy as np

from refactor_project.config.config import Config

#: Nomes das 2 features — usados nos gráficos
FEATURE_NAMES = ["x0", "x1"]


def create_concentric_circles_dataset(
    n_samples: int = 400,
    noise: float = 0.08,
    factor: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera círculos concêntricos via sklearn.datasets.make_circles,
    normalizado para [0, 1]².

    Diferente de refactor_project.data.cross_circle (padrão de patch
    3x3 "círculo vs. cruz", 9 features handcrafted): aqui é o dataset
    clássico de fronteira radial — círculo interno (classe 1) dentro
    de um círculo externo (classe 0), NÃO linearmente separável no
    espaço original. Benchmark padrão para testar se o encoding
    aprendido consegue capturar uma fronteira não-linear em baixa
    dimensionalidade (2 features).

    Args:
        n_samples : total de amostras (dividido ~50/50 entre os 2 anéis)
        noise     : desvio padrão do ruído gaussiano radial
        factor    : razão entre o raio do círculo interno e externo
                    (0 < factor < 1; menor = anéis mais separados)

    Retorna:
        X : (n_samples, 2) float32
        Y : (n_samples, 1) float32  — labels 0.0 (externo) / 1.0 (interno)
    """
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
    )

    X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_concentric_circles_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten.

    percent_total=100 → 400 amostras (~200 por anel)
    percent_total=60  → ~240 amostras (usado no SEARCH)
    """
    N_BASE = 400
    n = max(4, int(round(N_BASE * percent_total / 100.0)))
    if n % 2 != 0:
        n += 1
    return create_concentric_circles_dataset(n_samples=n, noise=0.08, factor=0.5, seed=seed)