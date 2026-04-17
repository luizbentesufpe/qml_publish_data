import numpy as np
from sklearn.feature_selection import mutual_info_classif as _mi_classif


def select_feature_bank_mi(
    X_tr: np.ndarray,
    y_tr_bin: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Ordena features pelo Mutual Information com o label binário.
    Calculado UMA VEZ no __init__, nunca recomputado durante o RL.

    Retorna índices ordenados por MI decrescente (0 = mais informativo).
    """
    y = y_tr_bin.reshape(-1).astype(int)
    X = np.asarray(X_tr, dtype=np.float32)
    k = int(min(k, X.shape[1]))

    scores = _mi_classif(X, y, random_state=int(seed))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ordem decrescente: índice 0 = patch com maior MI com y
    order = np.argsort(-scores)
    return order[:k].astype(np.int64)  # NÃO ordena por índice — preserva ranking
