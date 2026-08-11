import math

import numpy as np
import torch
from torch import nn

from refactor_project.config.config import Config
from refactor_project.features.features import num_patches
from refactor_project.model.util.auto_config import compute_auto_config


def make_cfg_for_qubits(
    base_cfg: Config,
    n_qubits: int,
    n_train: int | None = None,
    in_dim: int | None = None,
) -> Config:
    """Given a base configuration and a desired number of qubits, create a new configuration that is adjusted for the specified number of qubits.
    This function modifies the feature bank size and schedule based on the number of qubits, ensuring
    that the feature bank is appropriately sized for the input dimensionality and the model's capacity.
    various heuristics are applied to determine the starting size of the feature bank and its schedule of reduction during training,
    vars:
        base_cfg: The original configuration object that contains the default settings for the model and training.
        n_qubits: The desired number of qubits to be used in the quantum circuit, which will influence the size of the feature bank and its schedule.
    returns:
        A new Config object that has been adjusted based on the number of qubits, with updated feature bank
    """
    cfg = Config(**base_cfg.__dict__)

    nq = int(n_qubits)
    cfg.n_qubits = nq
    cfg.start_qubits = int(np.clip(nq, int(cfg.min_qubits), int(cfg.max_qubits)))

    # Adjust feature bank size and schedule based on the number of qubits and whether patch-based features are used.
    if bool(cfg.use_patch_bank):
        P = num_patches(28, 28, int(cfg.patch_size), int(cfg.patch_stride))
        cfg.feature_bank_size = int(P)
        cfg.feature_bank_min_size = int(min(cfg.feature_bank_min_size, P))
        cfg.feature_bank_schedule = (
            tuple(int(min(int(k), P)) for k in cfg.feature_bank_schedule)
            if len(cfg.feature_bank_schedule)
            else (P,)
        )
    else:
        # Heuristic for feature bank size: start with a size that is at least 4 times the number of qubits,
        # but not more than 784 (the total number of pixels in a 28x28 image).
        bank_start = int(min(784, max(32, 2 * (2**cfg.n_qubits))))
        bank_min = int(max(32, bank_start // 4))
        cfg.feature_bank_size = int(max(cfg.feature_bank_size, bank_start))
        cfg.feature_bank_min_size = int(min(cfg.feature_bank_min_size, bank_min))
        cfg.feature_bank_schedule = (
            cfg.feature_bank_size,
            int(max(cfg.feature_bank_min_size, round(0.75 * cfg.feature_bank_size))),
            int(max(cfg.feature_bank_min_size, round(0.50 * cfg.feature_bank_size))),
            cfg.feature_bank_min_size,
        )
        auto = compute_auto_config(
            in_dim=int(in_dim),
            n_train=int(n_train),
            batch_size=int(cfg.batch_size),
            enc_budget=int(cfg.ENC_budget) if cfg.ENC_budget > 0 else None,
            rot_budget=int(cfg.ROT_budget) if cfg.ROT_budget > 0 else None,
            cnot_budget=int(cfg.CNOT_budget) if cfg.CNOT_budget > 0 else None,
        )
        cfg.L_max = auto.L_max
        cfg.budget_penalty = auto.budget_penalty
        cfg.inner_train_subset_size = auto.inner_train_subset_size

    return cfg


def compute_pos_weight(Y_tr: torch.Tensor, device: str) -> torch.Tensor:
    y = Y_tr.detach().cpu().numpy().reshape(-1)
    pos = float(y.sum())
    neg = float(y.shape[0] - pos)
    w = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    if not np.isfinite(w):
        w = 1.0
    device = torch.device(device)
    return torch.tensor([w], device=device)


def init_head_bias_with_prevalence(model: nn.Module, Y_tr_np, *, force_p: float | None = None):
    if force_p is not None:
        p = float(force_p)
    else:
        p = float(Y_tr_np.mean())

    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    b = torch.tensor(
        [np.log(p / (1 - p))], dtype=torch.float32, device=next(model.parameters()).device
    )
    with torch.no_grad():
        model.head.bias.copy_(b)


def clamp_and_clip_head_(head: torch.nn.Module, cfg) -> dict:
    if head is None:
        return {}

    w = getattr(head, "weight", None)
    b = getattr(head, "bias", None)

    # Clamps (valores seguros por default; tune via cfg)
    w_abs_max = cfg.head_weight_abs_max
    b_abs_max = cfg.head_bias_abs_max

    # Clips por norma (se quiser)
    w_norm_max = cfg.head_weight_norm_max
    b_norm_max = cfg.head_bias_norm_max
    dbg = {}
    with torch.no_grad():
        if w is not None:
            dbg["w_norm_before"] = float(w.data.norm().item())
            dbg["w_absmax_before"] = float(w.data.abs().max().item())

            # clamp por valor
            if math.isfinite(w_abs_max) and w_abs_max > 0:
                w.data.clamp_(-w_abs_max, w_abs_max)

            # clip por norma
            if math.isfinite(w_norm_max) and w_norm_max > 0:
                wn = float(w.data.norm().item())
                if wn > w_norm_max:
                    w.data.mul_(w_norm_max / (wn + 1e-12))

            dbg["w_norm_after"] = float(w.data.norm().item())
            dbg["w_absmax_after"] = float(w.data.abs().max().item())

        if b is not None:
            dbg["b_norm_before"] = float(b.data.norm().item())
            dbg["b_absmax_before"] = float(b.data.abs().max().item())

            if math.isfinite(b_abs_max) and b_abs_max > 0:
                b.data.clamp_(-b_abs_max, b_abs_max)

            if math.isfinite(b_norm_max) and b_norm_max > 0:
                bn = float(b.data.norm().item())
                if bn > b_norm_max:
                    b.data.mul_(b_norm_max / (bn + 1e-12))

            dbg["b_norm_after"] = float(b.data.norm().item())
            dbg["b_absmax_after"] = float(b.data.abs().max().item())

    return dbg
