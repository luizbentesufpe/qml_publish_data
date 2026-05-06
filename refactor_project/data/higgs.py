#: URL canônica do UCI
from pathlib import Path
from typing import Tuple

import numpy as np

from refactor_project.config.config import Config

_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

#: Nomes das 28 features — coluna 0 é o label
#: Cols 1-21: low-level (kinematic) features dos detectores
#: Cols 22-28: high-level features derivadas (massas invariantes)
FEATURE_NAMES = [
    # Low-level kinematic features (21)
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet1_pT",
    "jet1_eta",
    "jet1_phi",
    "jet1_b_tag",
    "jet2_pT",
    "jet2_eta",
    "jet2_phi",
    "jet2_b_tag",
    "jet3_pT",
    "jet3_eta",
    "jet3_phi",
    "jet3_b_tag",
    "jet4_pT",
    "jet4_eta",
    "jet4_phi",
    "jet4_b_tag",
    # High-level derived features (7)
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

#: Tamanho default do subset (compatível com nq <= 6 e custo de busca RL)
DEFAULT_SUBSET_SIZE = 10_000


def _download_higgs(dest: Path) -> None:
    """Baixa o HIGGS.csv.gz do UCI se ainda não existir em *dest*.

    O arquivo comprimido tem ~280 MB; descomprimido ~2.6 GB.
    Mantemos comprimido em disco — pandas lê .gz nativamente.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Baixando HIGGS Boson dataset de:\n  {_UCI_URL}")
    print("[INFO] Arquivo grande (~280 MB comprimido) — pode demorar alguns minutos...")
    urllib.request.urlretrieve(_UCI_URL, str(dest))
    print(f"[INFO] Salvo em: {dest}")


def load_higgs_raw(
    data_dir: str = "data",
    subset_size: int = DEFAULT_SUBSET_SIZE,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega um subset do dataset HIGGS Boson.

    Tenta, em ordem:
      1. Arquivo local  data/HIGGS.csv.gz
      2. Download do UCI (requer rede; ~280 MB)

    O dataset completo tem 11M amostras — para QAS com nq <= 6 isso é
    intratável. Usamos um subset balanceado (50/50) das primeiras
    ~2*subset_size linhas, o que mantém custo de I/O baixo via nrows.

    Args:
        data_dir: diretório onde o .csv.gz está/será salvo
        subset_size: número total de amostras retornadas (default 10k)
        balanced: se True, aplica undersampling 50/50 sobre as primeiras
                  ~2*subset_size linhas; se False, retorna os primeiros
                  subset_size exemplos sem rebalanceamento.

    Retorna:
        X : (subset_size, 28) float32  — features brutas (não normalizadas)
        y : (subset_size,)    int32    — labels 0 / 1

    Notas físicas:
        Label 1 = signal (Higgs boson production)
        Label 0 = background (top quark pair production)
        Features 0-20: cinemática direta dos detectores (low-level)
        Features 21-27: massas invariantes derivadas (high-level)
    """
    import pandas as pd

    local = Path(data_dir) / "HIGGS.csv.gz"
    if not local.exists():
        try:
            _download_higgs(local)
        except Exception as e:
            raise FileNotFoundError(
                f"Não foi possível baixar o dataset HIGGS.\n"
                f"Erro: {e}\n"
                f"Faça o download manual de:\n  {_UCI_URL}\n"
                f"e salve em:  {local.resolve()}"
            ) from e

    # Lê apenas as primeiras linhas necessárias — evita carregar 11M na RAM
    # Para subset balanceado, lemos ~2x o necessário e fazemos undersampling
    nrows_to_read = subset_size * 2 if balanced else subset_size

    print(f"[INFO] Lendo primeiras {nrows_to_read} linhas de HIGGS.csv.gz...")
    df = pd.read_csv(
        str(local),
        header=None,
        nrows=nrows_to_read,
        compression="gzip",
    )

    # Coluna 0 = label; colunas 1-28 = features
    y_full = df.iloc[:, 0].values.astype(np.int32)
    X_full = df.iloc[:, 1:].values.astype(np.float32)

    if balanced:
        # Undersampling 50/50 para classes balanceadas
        idx_pos = np.where(y_full == 1)[0]
        idx_neg = np.where(y_full == 0)[0]
        n_per_class = subset_size // 2
        if len(idx_pos) < n_per_class or len(idx_neg) < n_per_class:
            raise ValueError(
                f"Subset insuficiente para balanceamento: "
                f"pos={len(idx_pos)}, neg={len(idx_neg)}, "
                f"necessário={n_per_class} de cada. "
                f"Aumente nrows_to_read ou desabilite balanced=True."
            )
        idx_pos = idx_pos[:n_per_class]
        idx_neg = idx_neg[:n_per_class]
        idx_balanced = np.concatenate([idx_pos, idx_neg])
        idx_balanced.sort()
        X = X_full[idx_balanced]
        y = y_full[idx_balanced]
    else:
        X = X_full[:subset_size]
        y = y_full[:subset_size]

    return X, y


def create_higgs_dataset(
    seed: int = 42,
    data_dir: str = "data",
    subset_size: int = DEFAULT_SUBSET_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna o dataset HIGGS normalizado para [0, 1]^28.

    Normalização MinMax é aplicada ao subset completo antes
    de qualquer split — compatível com angle encoding θ = π·x.

    Importante: HIGGS tem features com escalas muito diferentes
    (pT em GeV, eta adimensional, phi em radianos, b_tag em [0,1]).
    MinMax garante que o angle encoding mapeie todas para [0, π]
    de forma uniforme, condição necessária para que αi seja
    interpretável como peso relativo entre features.

    Retorna:
        X : (subset_size, 28) float32
        Y : (subset_size, 1)  float32  — labels 0.0 / 1.0
    """
    from sklearn.preprocessing import MinMaxScaler

    X_raw, y_raw = load_higgs_raw(data_dir=data_dir, subset_size=subset_size)

    # MinMax para [0, 1] — angle encoding espera features nesse intervalo
    X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_raw).astype(np.float32)
    Y = y_raw.astype(np.float32).reshape(-1, 1)

    # shuffle determinístico para reproducibilidade
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


def load_higgs_pool(
    cfg: Config,
    percent_total: int,
    seed: int,
    data_dir: str = "data",
    subset_size: int = DEFAULT_SUBSET_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interface compatível com load_circle_cross_pool_flatten / load_banknote_pool.

    percent_total=100 → todos os subset_size exemplos (default 10k)
    percent_total=60  → ~6000 exemplos (usado no SEARCH)

    Nota: a normalização MinMax é sempre feita no subset completo
    para evitar data leakage entre splits — consistente com o protocolo
    dos demais datasets do framework.

    Args:
        cfg: configuração do experimento (mantida para compatibilidade)
        percent_total: percentual do subset retornado (1-100)
        seed: seed para shuffle reproduzível
        data_dir: diretório do .csv.gz
        subset_size: tamanho total do subset HIGGS (default 10k)

    Retorna:
        X : (n, 28) float32  — n = round(subset_size * percent_total / 100)
        Y : (n, 1)  float32
    """
    X_full, Y_full = create_higgs_dataset(seed=seed, data_dir=data_dir, subset_size=subset_size)
    n = max(4, int(round(len(X_full) * percent_total / 100.0)))
    # garante número par para balanceamento
    if n % 2 != 0:
        n += 1
    n = min(n, len(X_full))

    rng = np.random.default_rng(seed + 9999)  # seed diferente do shuffle global
    idx = rng.choice(len(X_full), size=n, replace=False)
    idx.sort()
    return X_full[idx], Y_full[idx]
