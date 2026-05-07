import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Subset

from refactor_project.config.config import Config
from refactor_project.environments.states_encode.state_encoder import sanitize_architecture
from refactor_project.model.arch_train import train_final_model_end2end
from refactor_project.util.util import Logger


def get_balanced_subset(dataset, percentage=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    y_bin = (labels.sum(axis=1) > 0).astype(np.int32)
    N = len(dataset)
    subset_size = max(int(N * (percentage / 100.0)), 2)

    idx_pos = np.where(y_bin == 1)[0]
    idx_neg = np.where(y_bin == 0)[0]

    k_pos = min(len(idx_pos), subset_size // 2)
    k_neg = min(len(idx_neg), subset_size - k_pos)

    sel_pos = rng.choice(idx_pos, k_pos, replace=False) if k_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, k_neg, replace=False) if k_neg > 0 else np.array([], dtype=int)

    sel = np.concatenate([sel_pos, sel_neg])
    if len(sel) == 0:
        sel = rng.choice(np.arange(N), subset_size, replace=False)

    rng.shuffle(sel)
    return Subset(dataset, sel)

def adapt_inner_train_subset_size(cfg: Config, n_train: int) -> Config:
    """
    Ajusta inner_train_subset_size ao tamanho real do dataset de treino.
    
    Regra: usa o mínimo entre o valor configurado e um teto baseado
    no dataset, garantindo sempre pelo menos batch_size × 8 amostras
    para não matar a diversidade por batch.
    """
    floor = int(cfg.batch_size) * 8          # mínimo absoluto
    ceiling = min(int(cfg.inner_train_subset_size), n_train)
    cfg.inner_train_subset_size = max(floor, ceiling)
    return cfg

def dataset_to_arrays(
    ds, batch_size: int, use_patch_bank: bool, compact: bool, patch_size: int, patch_stride: int
):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    X, Y = [], []
    for xb, yb in dl:
        B = xb.size(0)
        xflat = xb.view(B, -1).float()  # (B,784)
        if yb.ndim == 3:
            yb = yb.view(yb.size(0), -1)  # (B,14)
        y_bin = (yb.sum(dim=1) > 0).float().unsqueeze(1)  # (B,1)
        X.append(xflat)
        Y.append(y_bin)

    X = torch.cat(X, 0).cpu().numpy().astype(np.float32)
    Y = torch.cat(Y, 0).cpu().numpy().astype(np.float32)


def _balanced_subset_from_arrays(X: np.ndarray, Y: np.ndarray, percentage: int, seed: int = 0):
    """
    Build a balanced subset (pos/neg) directly from numpy arrays.
    This avoids any dataset-index mismatch and guarantees no holdout leakage
    when you pass X_train_all/Y_train_all.
    """
    rng = np.random.default_rng(int(seed))
    y = (Y.reshape(-1) > 0.5).astype(np.int32)
    N = int(len(y))
    subset_size = int(max(2, round(N * (float(percentage) / 100.0))))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    k_pos = int(min(len(idx_pos), subset_size // 2))
    k_neg = int(min(len(idx_neg), subset_size - k_pos))
    sel = np.concatenate([idx_pos[:k_pos], idx_neg[:k_neg]])
    if sel.size < 2:
        sel = rng.choice(np.arange(N), subset_size, replace=False)
    rng.shuffle(sel)
    return X[sel], Y[sel]


def _make_search_splits_from_train_all(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    percent_search: int,
    seed: int = 0,
    val_frac: float = 0.25,
):
    """
    Build RL-search splits from TRAIN_ALL only (holdout-safe).
    Returns:
    (XtrS, YtrS), (XvaS, YvaS)
    - Stratified by binary label.
    - Uses a small, balanced-ish subset size determined by percent_search.
    - Deterministic by seed.
    """
    # 1) take a balanced subset from TRAIN_ALL
    Xs, Ys = _balanced_subset_from_arrays(
        X_train_all, Y_train_all, percentage=int(percent_search), seed=int(seed)
    )
    y = (Ys.reshape(-1) > 0.5).astype(np.int32)

    # 2) stratified split subset -> (tr, val)
    # guard: if subset has only one class, fallback to simple split
    if len(np.unique(y)) < 2 or len(y) < 4:
        n_val = int(max(1, round(float(val_frac) * len(y))))
        XvaS, YvaS = Xs[:n_val], Ys[:n_val]
        XtrS, YtrS = Xs[n_val:], Ys[n_val:]
        if len(XtrS) < 2:
            XtrS, YtrS = Xs, Ys
            XvaS, YvaS = Xs[:1], Ys[:1]
        return (XtrS, YtrS), (XvaS, YvaS)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_frac), random_state=int(seed))
    tr_idx, va_idx = next(sss.split(Xs, y))
    XtrS, YtrS = Xs[tr_idx], Ys[tr_idx]
    XvaS, YvaS = Xs[va_idx], Ys[va_idx]
    return (XtrS, YtrS), (XvaS, YvaS)


def split_holdout(X, Y, frac=0.20, seed=0):
    """
    Publication-grade split:
    Hold out a final test set never used by RL search or threshold calibration.
    """
    rng = np.random.default_rng(int(seed))
    y = Y.reshape(-1).astype(int)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n = len(y)
    n_te = int(max(2, round(frac * n)))
    n_te_pos = int(min(len(idx_pos), n_te // 2))
    n_te_neg = int(min(len(idx_neg), n_te - n_te_pos))
    te_idx = np.concatenate([idx_pos[:n_te_pos], idx_neg[:n_te_neg]])
    rng.shuffle(te_idx)
    tr_idx = np.setdiff1d(np.arange(n), te_idx)
    return tr_idx, te_idx


def nested_cv_eval_fixed_arch(
    arch_mat,
    X,
    Y,
    cfg: Config,
    n_qubits: int,
    logger: Logger,
    seed=0,
    device: torch.device | str = "cpu",
):
    # FINAL phase: strict/clinical threshold constraints
    try:
        cfg.phase = "final"
    except Exception:
        pass
    """
    Publication protocol:
    Outer CV evaluates generalization on TRAIN ONLY (holdout untouched).
    Inner training calibrates thr* using fold-train only.
    """
    DEVICE = torch.device(device)
    arch_mat = sanitize_architecture(arch_mat, int(n_qubits))
    y = Y.reshape(-1).astype(int)
    outer = StratifiedKFold(
        n_splits=int(cfg.nested_cv_splits_outer), shuffle=True, random_state=int(seed)
    )
    aucs, sens, thrs = [], [], []
    for fo, (tr_idx, va_idx) in enumerate(outer.split(X, y), 1):
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xva, Yva = X[va_idx], Y[va_idx]
        # Train on Xtr, evaluate on Xva, thr* calibrated only on Xtr inside train_final_model_end2end
        a, s, t = train_final_model_end2end(
            arch_mat, n_qubits, Xtr, Ytr, Xva, Yva, cfg, logger, device=DEVICE
        )
        aucs.append(a)
        sens.append(s)
        thrs.append(t)
        logger.log_to_file("nested", f"[outer={fo}] thr*={t:.3f} AUC={a:.4f} SENS@thr*={s:.4f}")
    return {
        "aucs": aucs,
        "sens": sens,
        "thr_star": thrs,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sens_mean": float(np.mean(sens)),
        "sens_std": float(np.std(sens)),
    }
