from __future__ import annotations

import numpy as np
import torch

from refactor_project.config.config import Config
from refactor_project.rl.actions import OpType


def count_ops(arch_mat: torch.Tensor) -> dict[str, int]:
    ops = arch_mat[2, :].detach().cpu().numpy()
    return {
        "ENC": int(np.sum(ops == OpType.ENC.value)),
        "ROT": int(np.sum(ops == OpType.ROT.value)),
        "CNOT": int(np.sum(ops == OpType.CNOT.value)),
    }


def budget_excess(counts: dict[str, int], cfg: Config) -> dict[str, int]:
    return {
        "ENC": max(0, counts["ENC"] - int(cfg.ENC_budget)),
        "ROT": max(0, counts["ROT"] - int(cfg.ROT_budget)),
        "CNOT": max(0, counts["CNOT"] - int(cfg.CNOT_budget)),
    }


def would_exceed_budget(counts: dict[str, int], action, cfg: Config) -> bool:
    kind = action[0]
    if kind == "ENC":
        return counts["ENC"] >= int(cfg.ENC_budget)
    if kind == "ROT":
        return counts["ROT"] >= int(cfg.ROT_budget)
    if kind == "CNOT":
        return counts["CNOT"] >= int(cfg.CNOT_budget)
    return False


def dead_qubit_count(arch_mat: torch.Tensor, n_qubits: int) -> int:
    nq = int(n_qubits)
    if nq <= 0:
        return 0

    used = np.zeros(nq, dtype=bool)
    a = arch_mat.detach().cpu().numpy()

    for local in range(a.shape[1]):
        op = int(a[2, local])
        c = int(a[0, local])
        t = int(a[1, local])

        if op == OpType.CNOT.value:
            if 0 < c <= nq:
                used[c - 1] = True
            if 0 < t <= nq:
                used[t - 1] = True
        elif op in (OpType.ENC.value, OpType.ROT.value):
            if 0 < t <= nq:
                used[t - 1] = True

    return int((~used).sum())


def p95_p5(arr: np.ndarray) -> float:
    z = np.asarray(arr, dtype=float).reshape(-1)
    if z.size == 0:
        return float("nan")
    p5, p95 = np.percentile(z, [5, 95])
    return float(p95 - p5)


def safe_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return dict(
            min=np.nan,
            mean=np.nan,
            max=np.nan,
            p5=np.nan,
            p50=np.nan,
            p95=np.nan,
            range=np.nan,
        )
    mn = float(np.min(x))
    mx = float(np.max(x))
    return dict(
        min=mn,
        mean=float(np.mean(x)),
        max=mx,
        p5=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
        range=float(mx - mn),
    )


def collapse_health(
    x: np.ndarray, range_min: float = 0.06, std_min: float = 0.02
) -> dict[str, float]:
    """
    Simple collapse detector based on robust range (p95-p5) and std.
    Returns dict with flags and key stats.
    """
    z = np.asarray(x, dtype=float).reshape(-1)
    if z.size == 0:
        return {
            "collapsed": 1.0,
            "p95_p5": float("nan"),
            "std": float("nan"),
            "mean": float("nan"),
        }
    p5, p95 = np.percentile(z, [5, 95])
    pr = float(p95 - p5)
    sd = float(np.std(z))
    mu = float(np.mean(z))
    collapsed = 1.0 if (pr < float(range_min) or sd < float(std_min)) else 0.0
    return {
        "collapsed": float(collapsed),
        "p95_p5": float(pr),
        "std": float(sd),
        "mean": float(mu),
    }


def confusion_from_thr(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> tuple[int, int, int, int]:
    pred = (probs >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def rates_from_thr(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> tuple[float, float, float]:
    tp, fp, tn, fn = confusion_from_thr(y_true, probs, thr)
    n_pos = tp + fn
    n_neg = tn + fp
    sens = float(tp) / max(n_pos, 1)
    spec = float(tn) / max(n_neg, 1)
    fpr = float(fp) / max(n_neg, 1)
    return sens, spec, fpr


def ema_update(prev: float | None, x: float, alpha: float = 0.05) -> float:
    x = float(x)
    if prev is None or (not np.isfinite(prev)):
        return x
    return float((1.0 - float(alpha)) * float(prev) + float(alpha) * x)
