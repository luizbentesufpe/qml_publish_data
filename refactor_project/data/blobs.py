from typing import List, Tuple

import numpy as np

from refactor_project.config.config import Config

#: Contagens de features usadas nos testes de escalabilidade em alta
#: dimensionalidade (ver BLOBS_HIGHDIM_SCENARIOS em config/ablation/blobs.py)
HIGH_DIM_FEATURE_COUNTS = [4, 8, 16, 32]


def _feature_names(n_features: int) -> List[str]:
    """Nomes genéricos x0..x(n-1) — Blobs não tem semântica de domínio."""
    return [f"x{i}" for i in range(n_features)]


def create_blobs_dataset(
    n_samples: int = 400,
    n_features: int = 2,
    centers: int = 2,
    cluster_std: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera blobs gaussianos via sklearn.datasets.make_blobs, normalizado
    para [0, 1]^n_features.

    Caso mais "fácil" do framework: classes linearmente separáveis por
    construção — serve como piso/controle de sanidade (se o pipeline
    não conseguir ~100% AUC aqui, há bug em algum outro lugar).

    n_features é parametrizável para os testes de alta dimensionalidade
    (ver HIGH_DIM_FEATURE_COUNTS). Nota: com cluster_std fixo, a
    separação relativa entre classes tende a aumentar com d (distância
    euclidiana escala com sqrt(d) para ruído i.i.d.) — se for necessário
    manter a dificuldade constante entre dimensionalidades, escale
    cluster_std por sqrt(n_features) na chamada.

    Retorna:
        X : (n_samples, n_features) float32
        Y : (n_samples, 1)           float32  — labels 0.0 / 1.0
    """
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )

    X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_blobs_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    n_features: int = 2,
    cluster_std: float = 1.0,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten.

    percent_total=100 → 400 amostras (200 por classe)
    percent_total=60  → ~240 amostras (usado no SEARCH)

    n_features e cluster_std são normalmente lidos do cenário via
    overrides (ex.: BLOBS_HIGHDIM_SCENARIOS define "n_features": d),
    e repassados pelo runner (run_one_blobs.py).
    """
    N_BASE = 400
    n = max(4, int(round(N_BASE * percent_total / 100.0)))
    if n % 2 != 0:
        n += 1
    return create_blobs_dataset(
        n_samples=n,
        n_features=n_features,
        centers=2,
        cluster_std=cluster_std,
        seed=seed,
    )