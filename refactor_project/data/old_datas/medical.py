import importlib

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T


def load_medmnist_pool_flatten(cfg, dataset_name: str, percent_total: int = 100, seed: int = 0):
    DatasetClass = getattr(importlib.import_module("medmnist"), dataset_name)
    tf = T.Compose([T.ToTensor()])

    Xs, Ys = [], []
    for split in ("train", "val", "test"):
        try:
            ds = DatasetClass(split=split, download=True, transform=tf, size=28)
        except Exception as e:
            print(f"[WARN] {dataset_name}/{split}: {e}")
            continue
        dl = DataLoader(
            ds,
            batch_size=int(batch_size=int(cfg.batch_size)),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        for xb, yb in dl:
            B = xb.size(0)
            xflat = xb.view(B, -1).float()
            if yb.ndim == 3:
                yb = yb.view(yb.size(0), -1)
            y_bin = (yb.float().sum(dim=-1) > 0).float().unsqueeze(1)
            Xs.append(xflat.cpu().numpy())
            Ys.append(y_bin.cpu().numpy())

    X_all = np.concatenate(Xs, axis=0).astype("float32")
    Y_all = np.concatenate(Ys, axis=0).astype("float32")

    if int(percent_total) >= 100:
        return X_all, Y_all

    rng = np.random.default_rng(int(seed))
    y = (Y_all.reshape(-1) > 0.5).astype("int32")
    n_keep = max(10, round(len(y) * percent_total / 100.0))
    ip = np.where(y == 1)[0]
    inn = np.where(y == 0)[0]

    rng.shuffle(ip)
    rng.shuffle(inn)

    kp = min(len(ip), n_keep // 2)
    kn = min(len(inn), n_keep - kp)

    sel = np.concatenate([ip[:kp], inn[:kn]])
    rng.shuffle(sel)

    return X_all[sel], Y_all[sel]
