#: URL canônica do UCI — dataset Original (não Diagnostic)
from pathlib import Path
from typing import Tuple

import numpy as np

from refactor_project.config.config import Config

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/breast-cancer-wisconsin/wdbc.data"
)

#: Nomes das 9 features — usados nos gráficos
FEATURE_NAMES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry"
]


def _download_breast_cancer(dest: Path) -> None:
    """Baixa o CSV do UCI se ainda não existir em *dest*."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Baixando Breast Cancer Wisconsin de:\n  {_UCI_URL}")
    urllib.request.urlretrieve(_UCI_URL, str(dest))
    print(f"[INFO] Salvo em: {dest}")


def load_breast_cancer_raw(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset Breast Cancer Wisconsin (Original).
    
    Tenta, em ordem:
      1. Arquivo local  data/wdbc.data
      2. Download do UCI (requer rede)
    
    Retorna:
        X : (569, 9) float32  — features brutas (não normalizadas)
        y : (569,)   int32    — labels 0 (benign) / 1 (malignant)
    """
    import pandas as pd
    
    local = Path(data_dir) / "wdbc.data"
    if not local.exists():
        try:
            _download_breast_cancer(local)
        except Exception as e:
            raise FileNotFoundError(
                f"Não foi possível baixar o dataset Breast Cancer Wisconsin.\n"
                f"Erro: {e}\n"
                f"Faça o download manual de:\n  {_UCI_URL}\n"
                f"e salve em:  {local.resolve()}"
            ) from e
    
    # Carrega o arquivo CSV
    # Formato: ID, Diagnosis, 30 features (mas usamos apenas as 9 primeiras)
    df = pd.read_csv(str(local), header=None)
    
    # Extrai as 9 primeiras features (colunas 2 a 10)
    X = df.iloc[:, 2:11].values.astype(np.float32)
    
    # Coluna 1 contém diagnosis: 'M' (malignant=1) ou 'B' (benign=0)
    y = (df.iloc[:, 1].values == 'M').astype(np.int32)
    
    return X, y


def create_breast_cancer_dataset(
    seed: int = 42,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna o dataset Breast Cancer Wisconsin normalizado com StandardScaler.
    
    Normalização StandardScaler é aplicada ao dataset completo antes
    de qualquer split — compatível com pipelines de ML convencionais.
    
    Retorna:
        X : (569, 9) float32  — features normalizadas (μ=0, σ=1)
        Y : (569, 1) float32  — labels 0.0 / 1.0
    """
    from sklearn.preprocessing import StandardScaler
    
    X_raw, y_raw = load_breast_cancer_raw(data_dir=data_dir)
    
    # StandardScaler para μ=0, σ=1
    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)
    
    # shuffle determinístico para reproducibilidade
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    
    return X[idx], Y[idx]


def load_breast_cancer_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten.
    
    percent_total=100 → todos os 569 exemplos
    percent_total=60  → ~341 exemplos (usado no SEARCH)
    
    Nota: a normalização StandardScaler é sempre feita no dataset completo
    para evitar data leakage entre splits.
    
    Args:
        cfg : Config — configuração do projeto
        percent_total : int — percentual do dataset a usar (1-100)
        seed : int — seed para reproducibilidade
        data_dir : str — diretório contendo o arquivo wdbc.data
    
    Retorna:
        X : (n, 9) float32  — features normalizadas
        Y : (n, 1) float32  — labels
    """
    X_full, Y_full = create_breast_cancer_dataset(seed=seed, data_dir=data_dir)
    
    n = max(4, int(round(len(X_full) * percent_total / 100.0)))
    
    # garante número par para balanceamento
    if n % 2 != 0:
        n += 1
    n = min(n, len(X_full))
    
    rng = np.random.default_rng(seed + 9999)  # seed diferente do shuffle global
    idx = rng.choice(len(X_full), size=n, replace=False)
    idx.sort()
    
    return X_full[idx], Y_full[idx]