import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def compute_task_context(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Meta-features baseadas em Rivolli et al. (2022) e Pfahringer et al. (2000).
    
    Se len(X) > max_samples, subamostras estratificadamente antes de computar
    os landmarkers — evita custo O(n²) do 1-NN em datasets grandes.
    """
    # Subsample estratificado automático
    if len(X) > max_samples:
        rng = np.random.default_rng(int(seed))
        classes = np.unique(y)
        idx = np.concatenate([
            rng.choice(
                np.where(y == c)[0],
                int(max_samples * (y == c).mean()),
                replace=False,
            )
            for c in classes
        ])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    n, d = X.shape

    # 1. Balanço — entropia de classes (Brazdil 2003)
    counts = np.bincount(y.astype(int))
    class_entropy = float(scipy_entropy(counts / counts.sum()))

    # 2. Landmarker LR — AUC (Pfahringer 2000)
    try:
        lr = LogisticRegression(max_iter=300, random_state=0, C=1.0)
        lr.fit(X, y)
        separabilidade = float(roc_auc_score(y, lr.predict_proba(X)[:, 1]))
    except Exception:
        separabilidade = 0.5

    # 3. Landmarker 1-NN — AUC CV-5 (Pfahringer 2000)
    try:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn_auc = float(cross_val_score(
            knn, X, y, cv=5, scoring="roc_auc"
        ).mean())
    except Exception:
        knn_auc = 0.5

    # 4. PCA fraction — variância do 1º componente (Rivolli 2022)
    pca = PCA(n_components=min(d, n - 1)).fit(X)
    pca_fraction = float(pca.explained_variance_ratio_[0])

    # 5. PCA 90 — fração de componentes para 90% da variância (Rivolli 2022)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    pca_90 = float(n_90 / d)

    return {
        "class_entropy":  float(np.clip(class_entropy,  0.0, 1.0)),
        "separabilidade": float(np.clip(separabilidade, 0.0, 1.0)),
        "knn_auc":        float(np.clip(knn_auc,        0.0, 1.0)),
        "pca_fraction":   float(np.clip(pca_fraction,   0.0, 1.0)),
        "pca_90":         float(np.clip(pca_90,         0.0, 1.0)),
    }