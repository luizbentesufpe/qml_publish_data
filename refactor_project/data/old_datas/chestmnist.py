from medmnist import ChestMNIST
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from refactor_project.config.config import Config
from refactor_project.data.util import dataset_to_arrays, get_balanced_subset


def load_chestmnist_pool_flatten(cfg: Config, percent_total: int = 100, seed: int = 0):
    """
    CORRETO (para seu protocolo de holdout próprio):
    - Carrega train/val/test completos
    - Concatena tudo em um pool único
    - (Opcional) subamostra UMA vez no pool inteiro (estratificado)
    - Retorna X_all, Y_all
    """
    tf = T.Compose([T.ToTensor()])

    tr = ChestMNIST(split="train", download=True, transform=tf, size=28)
    va = ChestMNIST(split="val", download=True, transform=tf, size=28)
    te = ChestMNIST(split="test", download=True, transform=tf, size=28)

    Xtr, Ytr = dataset_to_arrays(
        tr,
        batch_size=int(cfg.batch_size),
        use_patch_bank=bool(cfg.use_patch_bank),
        compact=bool(cfg.patch_bank_compact_features),
        patch_size=int(cfg.patch_size),
        patch_stride=int(cfg.patch_stride),
    )
    Xva, Yva = dataset_to_arrays(
        va,
        batch_size=int(cfg.batch_size),
        use_patch_bank=bool(cfg.use_patch_bank),
        compact=bool(cfg.patch_bank_compact_features),
        patch_size=int(cfg.patch_size),
        patch_stride=int(cfg.patch_stride),
    )
    Xte, Yte = dataset_to_arrays(
        te,
        batch_size=int(cfg.batch_size),
        use_patch_bank=bool(cfg.use_patch_bank),
        compact=bool(cfg.patch_bank_compact_features),
        patch_size=int(cfg.patch_size),
        patch_stride=int(cfg.patch_stride),
    )

    X_all = np.concatenate([Xtr, Xva, Xte], axis=0)
    Y_all = np.concatenate([Ytr, Yva, Yte], axis=0)

    # Subamostra UMA vez no pool inteiro (opcional)
    percent_total = int(percent_total)
    if percent_total >= 100:
        return X_all, Y_all

    rng = np.random.default_rng(int(seed))
    y = (Y_all.reshape(-1) > 0.5).astype(np.int32)
    N = int(len(y))
    n_keep = int(max(10, round(N * (percent_total / 100.0))))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    k_pos = int(min(len(idx_pos), n_keep // 2))
    k_neg = int(min(len(idx_neg), n_keep - k_pos))
    sel = np.concatenate([idx_pos[:k_pos], idx_neg[:k_neg]])
    if sel.size < 10:
        sel = rng.choice(np.arange(N), size=n_keep, replace=False)
    rng.shuffle(sel)

    return X_all[sel], Y_all[sel]


def load_chestmnist_flatten(cfg: Config, percentage_each_split=3, seed=0):
    tf = T.Compose([T.ToTensor()])
    tr = ChestMNIST(split="train", download=True, transform=tf, size=28)
    va = ChestMNIST(split="val", download=True, transform=tf, size=28)
    te = ChestMNIST(split="test", download=True, transform=tf, size=28)

    tr = get_balanced_subset(tr, percentage_each_split, seed=seed)
    va = get_balanced_subset(va, percentage_each_split, seed=seed + 1)
    te = get_balanced_subset(te, percentage_each_split, seed=seed + 2)

    def to_arrays(ds):
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        X, Y = [], []
        for xb, yb in dl:
            B = xb.size(0)
            xflat = xb.view(B, -1).float()  # (B,784)
            if yb.ndim == 3:
                yb = yb.view(yb.size(0), -1)  # (B,14)
            y_bin = (yb.sum(dim=1) > 0).float().unsqueeze(1)  # (B,1)
            X.append(xflat)
            Y.append(y_bin)
        X = torch.cat(X, 0).cpu().numpy()
        Y = torch.cat(Y, 0).cpu().numpy()

        # ==========================
        # Patch-bank (publication fix):
        # If enabled + compact_features, convert pixels->patch means
        # so encoder indices match feature_bank domain (P patches).
        # ==========================
        X = X.astype(np.float32)
        return X, Y

    return to_arrays(tr), to_arrays(va), to_arrays(te)
