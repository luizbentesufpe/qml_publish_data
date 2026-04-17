from typing import Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def _youden01(sens: float, spec: float) -> float:
    """Map sensitivity and specificity into [0,1] using Youden's J statistic.
    Youden's J statistic is defined as J = sensitivity + specificity - 1, which ranges from -1 to 1.
    We then transform this to the [0,1] range by applying the formula: (J + 1) / 2.
    This metric captures the balance between sensitivity and specificity,
    with a value of 1 indicating perfect sensitivity and specificity, and a value of 0 indicating
    that the test performs no better than random chance.
    variables:
        sens: Sensitivity (true positive rate) of the model, expected to be in the range [0,1].
        spec: Specificity (true negative rate) of the model, expected to be in the range [0,1].
    returns:
        A float value in the range [0,1] that represents the Youden's J
        statistic transformed to the [0,1] range, where higher values indicate better performance.
    """
    y = float(np.clip(float(sens) + float(spec) - 1.0, -1.0, 1.0))  # [-1,1]
    return float(0.5 * (y + 1.0))  # [0,1]


def _bacc(sens: float, spec: float) -> float:
    """Map sensitivity and specificity into [0,1] using balanced accuracy.
    Balanced accuracy is defined as the average of sensitivity and specificity: (sensitivity + specificity) / 2.
    This metric gives equal weight to sensitivity and specificity, making it useful for imbalanced datasets.
    variables:
        sens: Sensitivity (true positive rate) of the model, expected to be in the range [0,1].
        spec: Specificity (true negative rate) of the model, expected to be in the range [0,1].
    returns:
        A float value in the range [0,1] that represents the balanced accuracy, where higher values indicate better performance.
    """
    return float(0.5 * (float(sens) + float(spec)))


def _sep01_from_logit_margin(logit_p95_p5: float, scale: float = 0.25) -> float:
    """Map the logit margin between the 95th and 5th percentiles of the predicted probabilities into [0,1].
    The logit margin is calculated as the difference between the logit of the 95th
    percentile and the logit of the 5th percentile of the predicted probabilities.
    This function applies a hyperbolic tangent transformation to the logit margin, scaled by a factor, to map it into the [0,1] range.
    variables:
        - logit_p95_p5: The logit margin between the 95th and 5th percentiles
            of the predicted probabilities. This value can range from negative
            to positive infinity, where higher values indicate better separation
            between the positive and negative classes.
        - scale: A scaling factor that controls the sensitivity
            of the transformation. A smaller scale will
            make the function more sensitive to changes
            in the logit margin, while a larger scale
            will make it less sensitive.
            The default value is 0.25, which was chosen empirically
            to provide a good range of values for typical logit margins o
            bserved in practice.
    returns:
    A float value in the range [0,1] that represents the separation between the positive"""
    m = float(logit_p95_p5)
    if not np.isfinite(m):
        return 0.0
    m = max(0.0, m)
    s = max(float(scale), 1e-6)
    return float(np.tanh(m / s))  # 0..1


def _proxy_score_search(
    auc: float,
    sens: float,
    spec: float,
    logit_margin: float | None,
    phase: str,
    cfg: Any,
) -> tuple[float, dict]:
    """Search for the best proxy score based on the specified phase and configuration.
    This function evaluates different proxy scores (such as AUC, Youden's J statistic, balanced accuracy, and logit margin separation)
    based on the current phase of training or evaluation and the configuration settings.
    It returns the best proxy score along with a dictionary of all computed scores for analysis.
    variables:
        - auc: The area under the ROC curve (AUC) for the model's predictions, expected to be in the range [0,1].
        - sens: Sensitivity (true positive rate) of the model, expected to be in the range [0,1].
        - spec: Specificity (true negative rate) of the model, expected to be in the range [0,1].
        - logit_margin: The logit margin between the 95th and 5th percentiles of the predicted probabilities. This value can range from negative to positive infinity.
        - phase: A string indicating the current phase of training or evaluation (e.g., "train", "val", "test").
        - cfg: A configuration object that contains settings for which proxy score to use during each phase."""

    auc01 = float(np.clip(float(auc), 0.0, 1.0))
    sens01 = float(np.clip(float(sens), 0.0, 1.0))
    spec01 = float(np.clip(float(spec), 0.0, 1.0))
    youden01 = _youden01(sens01, spec01)

    phase_l = str(phase).lower()
    is_final = phase_l.startswith("final")
    sep_scale = cfg.final_sep_scale if is_final else cfg.search_sep_scale

    # 1) Separability based on logit margin (between 95th and 5th percentiles of predicted probabilities)
    # without gates
    sep01_raw = _sep01_from_logit_margin(
        float(logit_margin) if logit_margin is not None else float("nan"), scale=sep_scale
    )

    # 2) weights (phase-aware): during training, we might want to prioritize a
    # different metric than during final evaluation.
    w_auc = float(cfg.search_w_auc) if not is_final else float(cfg.final_w_auc)
    w_y = float(cfg.search_w_youden) if not is_final else float(cfg.final_w_youden)
    w_s = float(cfg.search_w_sep) if not is_final else float(cfg.final_w_sep)
    w_sum = max(1e-9, (w_auc + w_y + w_s))

    # 3) AUC-gate (sigmoid, search only)
    auc_floor = cfg.final_auc_floor if is_final else cfg.search_auc_floor
    tau_gate = cfg.final_auc_gate_tau if is_final else cfg.search_auc_gate_tau

    if not is_final:
        z = (auc01 - auc_floor) / max(tau_gate, 1e-6)
        auc_gate = float(1.0 / (1.0 + np.exp(-z)))

    else:
        auc_gate = 1.0

    # 4) gate applied ONLY in sep01
    sep01_gated = float(sep01_raw * auc_gate)

    # 5) Base score (without double gate)
    base = (w_auc * auc01 + w_y * youden01 + w_s * sep01_gated) / w_sum  # [0,1]

    # 6) Penalty to minimal separitily
    sep_floor = cfg.final_sep_floor if is_final else cfg.search_sep_floor
    lam_sep_floor = cfg.final_sep_floor_lam if is_final else cfg.search_sep_floor_lam

    pen_sep = 0.0
    if (not is_final) and (sep01_gated < sep_floor):
        pen_sep = lam_sep_floor * float(sep_floor - sep01_gated)

    score = float(np.clip(base - pen_sep, 0.0, 1.0))

    # Debug dict values
    dbg = {
        "auc01": float(auc01),
        "youden01": float(youden01),
        "sep01_raw": float(sep01_raw),
        "sep01_gated": float(sep01_gated),
        "base": float(base),
        "pen_sep_floor": float(pen_sep),
        "w": (w_auc, w_y, w_s),
        "auc_floor": float(auc_floor),
        "auc_gate": float(auc_gate),
        "auc_gate_tau": float(tau_gate),
        "sep_scale": float(sep_scale),
    }

    return score, dbg


@torch.no_grad()
def eval_metrics_final(model, X, Y, thr: float, device: torch.device | str):
    DEVICE = torch.device(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    logits = model(X_t)
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    yt = (Y_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
    n_pos = int(yt.sum())
    n_neg = int(len(yt) - n_pos)
    if (n_pos == 0) or (n_neg == 0):
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(yt, probs))
            if not np.isfinite(auc):
                auc = 0.5
        except Exception:
            auc = 0.5
    thr = float(thr)
    yp = (probs >= thr).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    sens = float(tp) / float(max(n_pos, 1))
    return float(auc), float(sens)


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    pos_weight: Optional[torch.Tensor] = None,
):
    logits = logits.view(-1)
    targets = targets.view(-1).float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    w = torch.where(
        targets > 0.5,
        torch.tensor(alpha, device=logits.device),
        torch.tensor(1.0 - alpha, device=logits.device),
    )
    loss = w * (1.0 - pt).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
