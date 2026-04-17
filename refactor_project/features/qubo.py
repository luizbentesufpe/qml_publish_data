import numpy as np
from sklearn.feature_selection import mutual_info_classif as _mi_classif
from sklearn.metrics import mutual_info_score as _mi_score


def _compute_importance(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Vetor I ∈ R^n: MI entre cada feature x_i e o label y.
    I_i = I(x_i ; y)  — Eq. (3) do paper.
    """
    scores = _mi_classif(X, y, random_state=int(seed))
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _compute_redundancy(X: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """
    Matriz R ∈ R^{n×n}: MI entre cada par de features (x_i, x_j).
    R_ij = I(x_i ; x_j)  — Eq. (4) do paper.
    Diagonal forçada a 0 (feature não é redundante com ela mesma).

    Discretiza via quantis (B=20 bins) exatamente como o paper descreve
    na Seção 2.1 para estimar MI em dados contínuos.
    """
    N, n = X.shape
    R = np.zeros((n, n), dtype=np.float64)

    # Discretiza cada feature em B bins por quantis
    X_disc = np.zeros_like(X, dtype=np.int32)
    for i in range(n):
        quantiles = np.percentile(X[:, i], np.linspace(0, 100, n_bins + 1))
        quantiles = np.unique(quantiles)  # remove duplicatas
        X_disc[:, i] = np.digitize(X[:, i], quantiles[1:-1], right=False)

    # MI entre todos os pares (i < j), simétrico
    for i in range(n):
        for j in range(i + 1, n):
            mi = _mi_score(X_disc[:, i], X_disc[:, j])
            mi = float(max(0.0, mi))
            R[i, j] = mi
            R[j, i] = mi
    # Diagonal = 0 por definição (paper Seção 2.1: R_ii = 0)
    np.fill_diagonal(R, 0.0)
    return R


def _build_qubo_matrix(importance: np.ndarray, R: np.ndarray, alpha: float) -> np.ndarray:
    """
    Monta a matriz QUBO Q(α) — Eq. (14) do paper:
        Q_ij(α) = R_ij - α * (R_ij + δ_ij * I_i)

    onde δ_ij é o delta de Kronecker.
    """
    alpha = float(alpha)
    Q = R.copy()  # Q_ij = R_ij
    Q -= alpha * R  # Q_ij -= α * R_ij
    diag_correction = alpha * importance  # α * I_i para diagonal
    np.fill_diagonal(Q, np.diag(Q) - diag_correction)
    return Q


def _solve_qubo_simulated_annealing(
    Q: np.ndarray,
    seed: int = 0,
    n_restarts: int = 8,
    n_steps: int = 5_000,
) -> np.ndarray:
    """
    Resolve QUBO min x^T Q x  com x ∈ {0,1}^n via Simulated Annealing.

    Para P=49 patches esta abordagem é mais que suficiente classicamente.
    Usa múltiplos restarts para robustez.
    Retorna x* ∈ {0,1}^n.
    """
    n = Q.shape[0]
    rng = np.random.default_rng(int(seed))

    def _energy(x: np.ndarray) -> float:
        return float(x @ Q @ x)

    best_x = None
    best_e = float("inf")

    for restart in range(int(n_restarts)):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        e = _energy(x)
        T = 1.0  # temperatura inicial
        T_min = 1e-4
        decay = (T_min / T) ** (1.0 / max(1, n_steps))

        for step in range(int(n_steps)):
            # flip bit aleatório
            i = int(rng.integers(0, n))
            x[i] = 1.0 - x[i]
            e_new = _energy(x)
            delta = e_new - e
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                e = e_new  # aceita
            else:
                x[i] = 1.0 - x[i]  # reverte
            T *= decay

        if e < best_e:
            best_e = e
            best_x = x.copy()

    return (best_x > 0.5).astype(np.int32)


def select_feature_bank_qfs(
    X_tr: np.ndarray,
    y_tr_bin: np.ndarray,
    k: int,
    seed: int = 0,
    n_bins: int = 20,
    eps: float = 1e-8,
    mu: float = 1e-6,
    n_restarts: int = 8,
    n_steps: int = 5_000,
    max_bisect: int = 50,
) -> np.ndarray:
    """
    QUBO Feature Selection — implementação fiel ao paper (Mücke et al., 2023).

    Pipeline (Figura 1 do paper):
      1. Calcula I (importância via MI feature↔label)       — Eq. (3)
      2. Calcula R (redundância via MI entre pares)          — Eq. (4)
      3. Binary search em α para achar subset de tamanho k  — Algoritmo 1
      4. Resolve cada QUBO via Simulated Annealing           — Seção 3.1
      5. Aplica threshold ε para features com I_i ≈ 0       — Eq. (18)

    Parâmetros:
        X_tr      : (N, P) features já patchificadas e normalizadas
        y_tr_bin  : (N,) labels binários {0,1}
        k         : número exato de features a selecionar
        seed      : semente para reprodutibilidade
        n_bins    : bins de discretização para MI (paper usa B=20)
        eps       : threshold ε para features com importância ≈ 0 — Eq. (18)
        mu        : peso µ para penalizar features com I_i < ε
        n_restarts: restarts do SA por chamada QUBO
        n_steps   : passos do SA por restart
        max_bisect: iterações máximas da busca binária em α

    Retorna:
        índices das k features selecionadas (NÃO ordenados por índice — preserva ranking QFS)
    """
    y = y_tr_bin.reshape(-1).astype(int)
    X = np.asarray(X_tr, dtype=np.float64)
    n = X.shape[1]
    k = int(np.clip(k, 0, n))

    # ── Passo 1: importância e redundância ──────────────────────
    importance_vec = _compute_importance(X.astype(np.float32), y, seed=seed)
    R = _compute_redundancy(X, n_bins=int(n_bins))

    # ── Passo 2: normaliza importance_vec e R para [0,1] (estabilidade numérica) ──
    I_max = float(np.max(importance_vec)) if np.max(importance_vec) > 0 else 1.0
    R_max = float(np.max(R)) if np.max(R) > 0 else 1.0
    importance_vec = importance_vec / I_max
    R = R / R_max

    def _solve_for_alpha(alpha: float) -> np.ndarray:
        """Monta Q(α,ε,µ) — Eq. (18) — e resolve via SA."""
        Q = _build_qubo_matrix(importance_vec, R, alpha)
        # Aplica threshold ε: features com α*importance_vec_i < ε recebem peso µ > 0
        # para forçar exclusão (Eq. 18 do paper)
        for i in range(n):
            if float(alpha) * float(importance_vec[i]) < float(eps):
                Q[i, i] = float(mu)
        return _solve_qubo_simulated_annealing(
            Q, seed=seed, n_restarts=n_restarts, n_steps=n_steps
        )

    # ── Passo 3: binary search em α — Algoritmo 1 do paper ──────
    # Proposição 1 garante que ∃ α ∈ [0,1] tal que ||x*||_1 = k
    a, b = 0.0, 1.0
    x_star = _solve_for_alpha(0.5)

    for _ in range(int(max_bisect)):
        alpha = (a + b) / 2.0
        x_star = _solve_for_alpha(alpha)
        k_current = int(x_star.sum())
        if k_current == k:
            break
        elif k_current > k:
            b = alpha  # muitas features → diminui α (menos importância, mais redundância penaliza)
        else:
            a = alpha  # poucas features → aumenta α

    selected = np.where(x_star == 1)[0].astype(np.int64)

    # fallback: se a busca binária não convergiu exatamente para k,
    # completa/trunca pelo ranking de importância importance_vec
    if len(selected) != k:
        importance_order = np.argsort(-importance_vec)
        if len(selected) < k:
            extra = [i for i in importance_order if i not in set(selected.tolist())]
            selected = np.concatenate([selected, extra[: k - len(selected)]])
        else:
            # mantém as k com maior importância entre as selecionadas
            sel_importance = importance_vec[selected]
            top_idx = np.argsort(-sel_importance)[:k]
            selected = selected[top_idx]

    return selected.astype(np.int64)
