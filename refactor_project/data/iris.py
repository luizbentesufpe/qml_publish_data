#: URL canônica do UCI — dataset Iris
from pathlib import Path
from typing import Tuple
import numpy as np
from refactor_project.config.config import Config

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/iris/iris.data"
)

#: Nomes das 4 features — usados nos gráficos
FEATURE_NAMES = [
    "sepal_length", "sepal_width", "petal_length", "petal_width"
]

#: Nomes das 3 classes originais, na ordem em que aparecem no arquivo UCI
_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

#: Par de classes usado por padrão na binarização.
#: versicolor vs virginica é o par NÃO linearmente separável — é o
#: benchmark padrão usado na literatura de Quantum Machine Learning
#: (setosa vs. qualquer outra classe é trivial e não serve de comparação).
_DEFAULT_BINARY_CLASSES = ("Iris-versicolor", "Iris-virginica")


def _download_iris(dest: Path) -> None:
    """Baixa o CSV do UCI se ainda não existir em *dest*."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Baixando Iris de:\n  {_UCI_URL}")
    urllib.request.urlretrieve(_UCI_URL, str(dest))
    print(f"[INFO] Salvo em: {dest}")


def load_iris_raw(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset Iris (UCI, 150 exemplos, 3 classes).

    Tenta, em ordem:
      1. Arquivo local  data/iris.data
      2. Download do UCI (requer rede)

    Retorna:
        X : (150, 4) float32  — features brutas (não normalizadas)
        y : (150,)   object   — rótulos de classe como string
                                 ('Iris-setosa', 'Iris-versicolor',
                                  'Iris-virginica')
    """
    import pandas as pd

    local = Path(data_dir) / "iris.data"
    if not local.exists():
        try:
            _download_iris(local)
        except Exception as e:
            raise FileNotFoundError(
                f"Não foi possível baixar o dataset Iris.\n"
                f"Erro: {e}\n"
                f"Faça o download manual de:\n  {_UCI_URL}\n"
                f"e salve em:  {local.resolve()}"
            ) from e

    # Formato do iris.data: sepal_length, sepal_width, petal_length,
    # petal_width, class (string). Última linha pode vir vazia -> skip_blank_lines.
    df = pd.read_csv(
        str(local),
        header=None,
        names=FEATURE_NAMES + ["class"],
        skip_blank_lines=True,
    ).dropna()

    X = df.iloc[:, 0:4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(str)

    return X, y


def create_iris_dataset(
    seed: int = 42,
    data_dir: str = "data",
    binary: bool = True,
    binary_classes: Tuple[str, str] = _DEFAULT_BINARY_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna o dataset Iris normalizado com StandardScaler.

    Normalização StandardScaler é aplicada ao dataset completo antes
    de qualquer split — compatível com pipelines de ML convencionais.

    Args:
        seed : int — seed para o shuffle determinístico
        data_dir : str — diretório contendo/recebendo iris.data
        binary : bool — se True, filtra apenas duas classes (ver
            `binary_classes`) e binariza os rótulos, para comparação
            direta com benchmarks binários (ex.: literatura de QML).
            Se False, mantém as 3 classes originais como inteiros
            0/1/2 (não recomendado para comparação binária).
        binary_classes : tuple(str, str) — par de classes a manter
            quando binary=True. Default: ('Iris-versicolor',
            'Iris-virginica') — o par não linearmente separável,
            usado como benchmark padrão em QML. A primeira classe do
            par vira o rótulo 0.0 e a segunda o rótulo 1.0.

    Retorna:
        X : (n, 4) float32  — features normalizadas (μ=0, σ=1)
        Y : (n, 1) float32  — labels 0.0 / 1.0 (binary=True)
                               ou 0.0 / 1.0 / 2.0 (binary=False)
    """
    from sklearn.preprocessing import StandardScaler

    X_raw, y_raw = load_iris_raw(data_dir=data_dir)

    if binary:
        cls_a, cls_b = binary_classes
        mask = np.isin(y_raw, [cls_a, cls_b])
        X_raw = X_raw[mask]
        y_bin = (y_raw[mask] == cls_b).astype(np.float32)
        y_final = y_bin
    else:
        # mapeia as 3 classes para 0/1/2 na ordem de _CLASS_NAMES
        class_to_idx = {name: i for i, name in enumerate(_CLASS_NAMES)}
        y_final = np.array(
            [class_to_idx[c] for c in y_raw], dtype=np.float32
        )

    # StandardScaler para μ=0, σ=1 (sobre o subconjunto já filtrado,
    # igual à lógica de "sempre no dataset completo antes do split")
    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    Y = y_final.reshape(-1, 1)

    # shuffle determinístico para reproducibilidade
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    return X[idx], Y[idx]


def load_iris_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    data_dir: str = "data",
    binary: bool = True,
    binary_classes: Tuple[str, str] = _DEFAULT_BINARY_CLASSES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten /
    load_breast_cancer_pool.

    percent_total=100 → todos os exemplos (100 se binary=True, 150 se
                         binary=False)
    percent_total=60  → ~60% do total, arredondado para número par

    Nota: a normalização StandardScaler é sempre feita no dataset
    (já filtrado, se binary=True) para evitar data leakage entre splits.

    Args:
        cfg : Config — configuração do projeto
        percent_total : int — percentual do dataset a usar (1-100)
        seed : int — seed para reproducibilidade
        data_dir : str — diretório contendo o arquivo iris.data
        binary : bool — ver `create_iris_dataset`
        binary_classes : tuple(str, str) — ver `create_iris_dataset`

    Retorna:
        X : (n, 4) float32  — features normalizadas
        Y : (n, 1) float32  — labels
    """
    X_full, Y_full = create_iris_dataset(
        seed=seed,
        data_dir=data_dir,
        binary=binary,
        binary_classes=binary_classes,
    )

    n = max(4, int(round(len(X_full) * percent_total / 100.0)))

    # garante número par para balanceamento
    if n % 2 != 0:
        n += 1
    n = min(n, len(X_full))

    rng = np.random.default_rng(seed + 9999)  # seed diferente do shuffle global
    idx = rng.choice(len(X_full), size=n, replace=False)
    idx.sort()

    return X_full[idx], Y_full[idx]