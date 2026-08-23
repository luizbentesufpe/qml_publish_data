from pathlib import Path
from typing import List, Tuple

import numpy as np

from refactor_project.config.config import Config

#: Par de dígitos usado por padrão — 3 vs. 8 é o par visualmente mais
#: parecido em MNIST, benchmark clássico "difícil" para classificação
#: binária (equivalente ao papel do par versicolor/virginica no Iris).
DEFAULT_DIGIT_PAIR = (3, 8)

#: Nº de componentes PCA por padrão. 784 pixels brutos são inviáveis
#: para angle encoding em circuitos de poucos qubits; PCA concentra a
#: variância mais discriminativa entre os dois dígitos em poucas
#: dimensões.
DEFAULT_PCA_COMPONENTS = 8

#: Nº máximo de amostras por classe (MNIST tem ~7000/dígito; limitamos
#: por custo de PCA fit + custo de busca RL)
DEFAULT_MAX_PER_CLASS = 1000


def _feature_names(n_components: int) -> List[str]:
    return [f"pca_{i}" for i in range(n_components)]


def load_mnist_raw(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o MNIST completo (70000 amostras, 784 pixels) via
    sklearn.datasets.fetch_openml, com cache local em data_dir.

    Retorna:
        X : (70000, 784) float32  — pixels brutos em [0, 255]
        y : (70000,)      str     — dígitos '0'..'9'
    """
    from sklearn.datasets import fetch_openml

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print("[INFO] Carregando MNIST (fetch_openml, cache local)...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        data_home=data_dir,
        as_frame=False,
        cache=True,
    )
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(str)
    return X, y


def create_mnist_dataset(
    seed: int = 42,
    data_dir: str = "data",
    digit_pair: Tuple[int, int] = DEFAULT_DIGIT_PAIR,
    n_components: int = DEFAULT_PCA_COMPONENTS,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna o subconjunto MNIST binarizado (digit_pair), reduzido via
    PCA para n_components e normalizado para [0, 1]^n_components.

    Protocolo:
      1. Filtra apenas os 2 dígitos de digit_pair
      2. Undersampling balanceado até max_per_class por dígito
      3. PCA (fit no subconjunto já filtrado — evita leakage do resto
         do MNIST) reduz 784 -> n_components
      4. MinMaxScaler para [0,1]^n_components — compatível com
         angle encoding θ = π·x

    Retorna:
        X : (n, n_components) float32
        Y : (n, 1)             float32  — 0.0 = digit_pair[0], 1.0 = digit_pair[1]
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = load_mnist_raw(data_dir=data_dir)

    d0, d1 = digit_pair
    mask = np.isin(y_raw, [str(d0), str(d1)])
    X_sub, y_sub = X_raw[mask], y_raw[mask]

    rng = np.random.default_rng(seed)
    idx0 = np.where(y_sub == str(d0))[0]
    idx1 = np.where(y_sub == str(d1))[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n_per = min(max_per_class, len(idx0), len(idx1))
    idx_balanced = np.concatenate([idx0[:n_per], idx1[:n_per]])
    idx_balanced.sort()

    X_sub, y_sub = X_sub[idx_balanced], y_sub[idx_balanced]

    X_pca = PCA(n_components=n_components, random_state=seed).fit_transform(
        X_sub / 255.0
    )
    X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_pca).astype(np.float32)
    Y = (y_sub == str(d1)).astype(np.float32).reshape(-1, 1)

    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_mnist_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    data_dir: str = "data",
    digit_pair: Tuple[int, int] = DEFAULT_DIGIT_PAIR,
    n_components: int = DEFAULT_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten /
    load_breast_cancer_pool.

    percent_total=100 → todas as ~2*max_per_class amostras balanceadas
    percent_total=60  → ~60% (usado no SEARCH)

    Nota: PCA é sempre ajustado no subconjunto binário completo antes
    do split (mesma lógica de "normalização no dataset completo" dos
    demais datasets do framework, para evitar leakage entre splits).
    """
    X_full, Y_full = create_mnist_dataset(
        seed=seed,
        data_dir=data_dir,
        digit_pair=digit_pair,
        n_components=n_components,
    )

    n = max(4, int(round(len(X_full) * percent_total / 100.0)))
    if n % 2 != 0:
        n += 1
    n = min(n, len(X_full))

    rng = np.random.default_rng(seed + 9999)
    idx = rng.choice(len(X_full), size=n, replace=False)
    idx.sort()

    return X_full[idx], Y_full[idx]