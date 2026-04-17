#: URL canônica do UCI
from pathlib import Path
from typing import Tuple

import numpy as np

from refactor_project.config.config import Config

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00267/data_banknote_authentication.txt"
)

#: Nomes das 4 features — usados nos gráficos
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]


def _download_banknote(dest: Path) -> None:
    """Baixa o CSV do UCI se ainda não existir em *dest*."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Baixando Banknote Authentication de:\n  {_UCI_URL}")
    urllib.request.urlretrieve(_UCI_URL, str(dest))
    print(f"[INFO] Salvo em: {dest}")


def load_banknote_raw(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset Banknote Authentication.

    Tenta, em ordem:
      1. Arquivo local  data/data_banknote_authentication.txt
      2. Download do UCI (requer rede)

    Retorna:
        X : (1372, 4) float32  — features brutas (não normalizadas)
        y : (1372,)   int32    — labels 0 / 1
    """
    import pandas as pd

    local = Path(data_dir) / "data_banknote_authentication.txt"

    if not local.exists():
        try:
            _download_banknote(local)
        except Exception as e:
            raise FileNotFoundError(
                f"Não foi possível baixar o dataset Banknote.\n"
                f"Erro: {e}\n"
                f"Faça o download manual de:\n  {_UCI_URL}\n"
                f"e salve em:  {local.resolve()}"
            ) from e

    df = pd.read_csv(str(local), header=None)
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(np.int32)
    return X, y


def create_banknote_dataset(
    seed: int = 42,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna o dataset Banknote normalizado para [0, 1]⁴.

    Normalização MinMax é aplicada ao dataset completo antes
    de qualquer split — compatível com angle encoding θ = π·x.

    Retorna:
        X : (1372, 4) float32
        Y : (1372, 1) float32  — labels 0.0 / 1.0
    """
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = load_banknote_raw(data_dir=data_dir)

    # MinMax para [0, 1] — angle encoding espera features nesse intervalo
    X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)

    # shuffle determinístico para reproducibilidade
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_banknote_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten.

    percent_total=100 → todos os 1372 exemplos
    percent_total=60  → ~823 exemplos (usado no SEARCH)

    Nota: a normalização MinMax é sempre feita no dataset completo
    para evitar data leakage entre splits.
    """
    X_full, Y_full = create_banknote_dataset(seed=seed, data_dir=data_dir)

    n = max(4, int(round(len(X_full) * percent_total / 100.0)))
    # garante número par para balanceamento
    if n % 2 != 0:
        n += 1
    n = min(n, len(X_full))

    rng = np.random.default_rng(seed + 9999)  # seed diferente do shuffle global
    idx = rng.choice(len(X_full), size=n, replace=False)
    idx.sort()
    return X_full[idx], Y_full[idx]
