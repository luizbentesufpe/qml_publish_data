#!/usr/bin/env python3
"""
benchmark_all_datasets.py — Handcrafted benchmark para todos os datasets.

Datasets suportados:
  - cross_circle, make_moons, banknote, breast_cancer  : 3 baselines (B0, B1, B2)
  - higgs                                              : 4 baselines (B0, B1, B2, B3)
                                                         + diagnóstico físico HL vs LL

O Higgs adiciona um controle contrastivo (B3, low-level) e métricas
α_HL/α_LL para circuitos treináveis, testando a hipótese de que
encoding paramétrico recupera a hierarquia física conhecida do dataset
(features high-level mais discriminativas que low-level).

Uso:
    python benchmark_all_datasets.py --seeds 0 1 2
    python benchmark_all_datasets.py --datasets higgs --subset_size 10000
    python benchmark_all_datasets.py --datasets cross_circle banknote higgs
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn.functional as F

from refactor_project.config.config import Config
from refactor_project.data.banknote import create_banknote_dataset
from refactor_project.data.breast_cancer import create_breast_cancer_dataset
from refactor_project.data.cross_circle import create_circle_cross_dataset
from refactor_project.data.higgs import create_higgs_dataset
from refactor_project.data.moons import create_make_moons_dataset
from refactor_project.environments.states_encode.state_encoder import sanitize_architecture
from refactor_project.environments.states_encode.threshold import find_threshold
from refactor_project.model.model import CQV_End2End
from refactor_project.rl.actions import OpType

# ════════════════════════════════════════════════════════════════════
# GRUPOS FÍSICOS (apenas Higgs por enquanto)
# ════════════════════════════════════════════════════════════════════

#: Mapeia dataset_key → grupos de features. None se não há hierarquia
#: física aplicável. Adicione novos datasets aqui se tiverem grupos
#: semânticos (e.g., "demographic" vs "clinical" em datasets médicos).
_PHYSICS_GROUPS: Dict[str, Optional[Dict[str, List[int]]]] = {
    "cross_circle": None,
    "make_moons": None,
    "banknote": None,
    "breast_cancer": None,
    "higgs": {
        "low": list(range(0, 21)),    # cinemática direta dos detectores
        "high": list(range(21, 28)),  # massas invariantes derivadas
    },
}


# ════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ════════════════════════════════════════════════════════════════════


def make_arch_mat(ops, n_qubits=4, L_max=20):
    arch = np.zeros((5, L_max), dtype=np.int64)
    for col, op in enumerate(ops):
        if col >= L_max:
            break
        kind = op["type"]
        if kind == "ENC":
            arch[1, col] = op["qubit"] + 1
            arch[2, col] = OpType.ENC.value
            arch[3, col] = op.get("axis", 2)
            arch[4, col] = op["feature"] + 1
        elif kind == "ROT":
            arch[1, col] = op["qubit"] + 1
            arch[2, col] = OpType.ROT.value
            arch[3, col] = op.get("axis", 2)
        elif kind == "CNOT":
            arch[0, col] = op["ctrl"] + 1
            arch[1, col] = op["qubit"] + 1
            arch[2, col] = OpType.CNOT.value
    mat = torch.tensor(arch, dtype=torch.int64)
    return sanitize_architecture(mat, n_qubits)


def count_ops(arch_mat):
    ops = arch_mat[2, :].numpy()
    nonzero = np.where(ops > 0)[0]
    return {
        "n_enc": int(np.sum(ops == OpType.ENC.value)),
        "n_rot": int(np.sum(ops == OpType.ROT.value)),
        "n_cnot": int(np.sum(ops == OpType.CNOT.value)),
        "n_total": int(np.sum(ops > 0)),
        "depth": int(nonzero[-1] + 1) if len(nonzero) > 0 else 0,
    }


def get_used_features(arch_mat) -> List[int]:
    """Retorna lista ordenada de feature indices ativos no circuito."""
    ops = arch_mat[2, :].numpy()
    feats = arch_mat[4, :].numpy()
    used = []
    for col in range(arch_mat.shape[1]):
        if ops[col] == OpType.ENC.value and feats[col] > 0:
            used.append(int(feats[col] - 1))
    return sorted(set(used))


def alpha_group_split(
    alpha_vals: Optional[List[float]],
    used_features: List[int],
    groups: Optional[Dict[str, List[int]]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcula mean(α[low]) e mean(α[high]) restrito às features efetivamente
    usadas no circuito. Retorna (low_mean, high_mean, ratio_high_over_low).

    Crítico: ignora features fora do circuito (que ficam em α_init=0.5
    por nunca receberem gradiente), evitando contaminação da média.

    Retorna (None, None, None) se groups for None (dataset sem hierarquia).
    """
    if groups is None or alpha_vals is None:
        return None, None, None
    a = np.asarray(alpha_vals, dtype=np.float32)
    used_set = set(used_features)
    low_used = [i for i in groups["low"] if i in used_set and i < len(a)]
    high_used = [i for i in groups["high"] if i in used_set and i < len(a)]
    low_mean = float(a[low_used].mean()) if low_used else None
    high_mean = float(a[high_used].mean()) if high_used else None
    ratio = (
        (high_mean / low_mean) if (low_mean is not None and low_mean > 1e-6 and high_mean is not None) else None
    )
    return low_mean, high_mean, ratio


def train_and_eval(
    arch_mat,
    X_tr,
    Y_tr,
    X_va,
    Y_va,
    n_qubits,
    cfg,
    freeze_alpha=False,
    n_epochs=200,
    lr=0.02,
    device="cpu",
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CQV_End2End(
        arch_mat=arch_mat,
        n_qubits=n_qubits,
        enc_lambda=float(cfg.enc_lambda),
        diff_method="backprop",
        input_dim=int(X_tr.shape[1]),
        enc_affine_mode=str(cfg.enc_affine_mode),
        enc_alpha_init=float(cfg.enc_alpha_init),
        enc_beta_init=float(cfg.enc_beta_init),
        enc_beta_max=float(cfg.enc_beta_max),
    ).to(device)

    if freeze_alpha:
        model.enc_alpha_raw.requires_grad_(False)
        model.enc_beta_raw.requires_grad_(False)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(Y_tr.reshape(-1), dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, Ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs_tr = torch.sigmoid(model(Xtr_t).squeeze(-1)).cpu().numpy()
        probs_va = torch.sigmoid(model(Xva_t).squeeze(-1)).cpu().numpy()

    y_tr_int = (Y_tr.reshape(-1) > 0.5).astype(int)
    y_va_int = (Y_va.reshape(-1) > 0.5).astype(int)

    thr = find_threshold(y_tr_int, probs_tr, mode="soft")
    auc = float(roc_auc_score(y_va_int, probs_va)) if len(np.unique(y_va_int)) > 1 else 0.5

    alpha_vals = None
    if hasattr(model, "enc_alpha_raw"):
        alpha_vals = F.softplus(model.enc_alpha_raw).detach().cpu().numpy().tolist()

    return {"auc": round(auc, 4), "thr_star": round(float(thr), 4), "alpha": alpha_vals}


def run_benchmark(
    name: str,
    baselines: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    cfg: Config,
    args,
    physics_groups: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Treina e avalia (frozen vs trainable) cada baseline em `baselines`.

    Se `physics_groups` for não-None (e.g., {"low": [...], "high": [...]}),
    coleta também estatísticas α_low / α_high / ratio para circuitos
    treináveis — usado em Higgs para diagnóstico físico HL vs LL.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    results = {}
    has_physics = physics_groups is not None

    for circuit_name, arch_mat in baselines.items():
        ops = count_ops(arch_mat)
        used_feats = get_used_features(arch_mat)

        print(f"\n{'─' * 50}\n[{circuit_name}]")
        print(
            f"  ENC={ops['n_enc']} ROT={ops['n_rot']} CNOT={ops['n_cnot']} "
            f"total={ops['n_total']} depth={ops['depth']}"
        )

        if has_physics:
            low_used = [i for i in used_feats if i in physics_groups["low"]]
            high_used = [i for i in used_feats if i in physics_groups["high"]]
            print(f"  Used: {used_feats}  |  low={low_used}  high={high_used}")
        else:
            print(f"  Used features: {used_feats}")

        frozen_aucs, train_aucs = [], []
        low_alphas, high_alphas, ratios = [], [], []

        for seed in args.seeds:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            y_int = (Y.reshape(-1) > 0.5).astype(int)
            tr_idx, va_idx = next(sss.split(X, y_int))
            X_tr, Y_tr = X[tr_idx], Y[tr_idx]
            X_va, Y_va = X[va_idx], Y[va_idx]

            r_fr = train_and_eval(
                arch_mat, X_tr, Y_tr, X_va, Y_va,
                n_qubits=args.n_qubits, cfg=cfg, freeze_alpha=True,
                n_epochs=args.n_epochs, device="cpu", seed=seed,
            )
            r_tr = train_and_eval(
                arch_mat, X_tr, Y_tr, X_va, Y_va,
                n_qubits=args.n_qubits, cfg=cfg, freeze_alpha=False,
                n_epochs=args.n_epochs, device="cpu", seed=seed,
            )

            frozen_aucs.append(r_fr["auc"])
            train_aucs.append(r_tr["auc"])

            # Diagnóstico físico (apenas datasets com physics_groups)
            phys_str = ""
            if has_physics:
                ll_m, hl_m, ratio = alpha_group_split(
                    r_tr["alpha"], used_feats, physics_groups
                )
                if ll_m is not None:
                    low_alphas.append(ll_m)
                if hl_m is not None:
                    high_alphas.append(hl_m)
                if ratio is not None:
                    ratios.append(ratio)

                if hl_m is not None and ll_m is not None:
                    phys_str = f"  α_HL={hl_m:.3f} α_LL={ll_m:.3f} ratio={ratio:.2f}"
                elif hl_m is not None:
                    phys_str = f"  α_HL={hl_m:.3f} (HL-only)"
                elif ll_m is not None:
                    phys_str = f"  α_LL={ll_m:.3f} (LL-only)"
            else:
                # Fallback: log α médio global como antes
                if r_tr["alpha"]:
                    a = np.array(r_tr["alpha"])
                    phys_str = f"  α={a.mean():.3f}±{a.std():.3f}"

            print(
                f"  seed={seed}  frozen={r_fr['auc']:.4f}  "
                f"trainable={r_tr['auc']:.4f}  "
                f"Δ={r_tr['auc'] - r_fr['auc']:+.4f}{phys_str}"
            )

        f_mean = np.mean(frozen_aucs)
        t_mean = np.mean(train_aucs)
        print(
            f"  → frozen  {f_mean:.4f}±{np.std(frozen_aucs):.4f}  "
            f"train {t_mean:.4f}±{np.std(train_aucs):.4f}  "
            f"Δ={t_mean - f_mean:+.4f}"
        )

        if has_physics and ratios:
            print(
                f"  → α_LL={np.mean(low_alphas):.3f}±{np.std(low_alphas):.3f}  "
                f"α_HL={np.mean(high_alphas):.3f}±{np.std(high_alphas):.3f}  "
                f"ratio HL/LL={np.mean(ratios):.2f}±{np.std(ratios):.2f}"
            )

        result_entry = {
            "circuit": ops,
            "frozen_auc": round(f_mean, 4),
            "frozen_std": round(float(np.std(frozen_aucs)), 4),
            "trainable_auc": round(t_mean, 4),
            "trainable_std": round(float(np.std(train_aucs)), 4),
            "delta_auc": round(t_mean - f_mean, 4),
            "per_seed": [
                {"seed": s, "frozen": f, "trainable": t}
                for s, f, t in zip(args.seeds, frozen_aucs, train_aucs)
            ],
        }

        # Campos físicos só para datasets com hierarquia
        if has_physics:
            result_entry["used_features"] = used_feats
            result_entry["low_used"] = [i for i in used_feats if i in physics_groups["low"]]
            result_entry["high_used"] = [i for i in used_feats if i in physics_groups["high"]]
            result_entry["alpha_physics"] = (
                {
                    "low_mean": round(float(np.mean(low_alphas)), 4) if low_alphas else None,
                    "low_std": round(float(np.std(low_alphas)), 4) if low_alphas else None,
                    "high_mean": round(float(np.mean(high_alphas)), 4) if high_alphas else None,
                    "high_std": round(float(np.std(high_alphas)), 4) if high_alphas else None,
                    "ratio_high_over_low": round(float(np.mean(ratios)), 4) if ratios else None,
                    "ratio_std": round(float(np.std(ratios)), 4) if ratios else None,
                }
                if (low_alphas or high_alphas)
                else None
            )

        results[circuit_name] = result_entry

    return results


# ════════════════════════════════════════════════════════════════════
# DEFINIÇÃO DOS CIRCUITOS POR DATASET
# ════════════════════════════════════════════════════════════════════


def build_cross_circle(n_qubits=4, L_max=20):
    baselines = {}

    # B0 — pixel centro (f4) + canto (f0), 1 CNOT
    baselines["B0_minimal"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 4},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 0},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    # B1 — 4 bordas (f1,f3,f5,f7), 2 CNOTs
    baselines["B1_structural"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 1},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 3},
            {"type": "ENC", "qubit": 2, "axis": 2, "feature": 5},
            {"type": "ENC", "qubit": 3, "axis": 2, "feature": 7},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "ROT", "qubit": 2, "axis": 2},
            {"type": "ROT", "qubit": 3, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "CNOT", "qubit": 3, "ctrl": 2},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "ROT", "qubit": 2, "axis": 3},
            {"type": "ROT", "qubit": 3, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    # B2 — HW-efficient, 9 features round-robin, 3 CNOTs
    hw_ops = []
    for q in range(n_qubits):
        hw_ops.append({"type": "ENC", "qubit": q, "axis": 2, "feature": q % 9})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 1}, {"type": "ROT", "qubit": q, "axis": 2}]
    for q in range(n_qubits - 1):
        hw_ops.append({"type": "CNOT", "qubit": q + 1, "ctrl": q})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 2}, {"type": "ROT", "qubit": q, "axis": 3}]
    baselines["B2_hw_efficient"] = make_arch_mat(hw_ops, n_qubits=n_qubits, L_max=L_max)

    return baselines


def build_make_moons(n_qubits=4, L_max=20):
    baselines = {}

    baselines["B0_minimal"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 1},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    baselines["B1_bidirectional"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 1},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "CNOT", "qubit": 0, "ctrl": 1},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    baselines["B2_reupload"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 1},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "ENC", "qubit": 0, "axis": 1, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 1, "feature": 1},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    return baselines


def build_banknote(n_qubits=4, L_max=20):
    baselines = {}

    baselines["B0_var_curt"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 2},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    baselines["B1_all_features"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 0},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 1},
            {"type": "ENC", "qubit": 2, "axis": 2, "feature": 2},
            {"type": "ENC", "qubit": 3, "axis": 2, "feature": 3},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "ROT", "qubit": 2, "axis": 2},
            {"type": "ROT", "qubit": 3, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "CNOT", "qubit": 3, "ctrl": 2},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "ROT", "qubit": 2, "axis": 3},
            {"type": "ROT", "qubit": 3, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    hw_ops = []
    for q in range(n_qubits):
        hw_ops.append({"type": "ENC", "qubit": q, "axis": 2, "feature": q % 4})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 1}, {"type": "ROT", "qubit": q, "axis": 2}]
    for q in range(n_qubits - 1):
        hw_ops.append({"type": "CNOT", "qubit": q + 1, "ctrl": q})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 2}, {"type": "ROT", "qubit": q, "axis": 3}]
    baselines["B2_hw_efficient"] = make_arch_mat(hw_ops, n_qubits=n_qubits, L_max=L_max)

    return baselines


def build_breast_cancer(n_qubits=4, L_max=20):
    baselines = {}

    baselines["B0_concavity"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 6},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 7},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    baselines["B1_clinical_subset"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 6},
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 7},
            {"type": "ENC", "qubit": 2, "axis": 2, "feature": 2},
            {"type": "ENC", "qubit": 3, "axis": 2, "feature": 0},
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "ROT", "qubit": 2, "axis": 2},
            {"type": "ROT", "qubit": 3, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "CNOT", "qubit": 3, "ctrl": 2},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "ROT", "qubit": 2, "axis": 3},
            {"type": "ROT", "qubit": 3, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    hw_ops = []
    for q in range(n_qubits):
        hw_ops.append({"type": "ENC", "qubit": q, "axis": 2, "feature": q % 9})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 1}, {"type": "ROT", "qubit": q, "axis": 2}]
    for q in range(n_qubits - 1):
        hw_ops.append({"type": "CNOT", "qubit": q + 1, "ctrl": q})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 2}, {"type": "ROT", "qubit": q, "axis": 3}]
    baselines["B2_hw_efficient"] = make_arch_mat(hw_ops, n_qubits=n_qubits, L_max=L_max)

    return baselines


def build_higgs(n_qubits=4, L_max=20):
    """
    Quatro baselines handcrafted para HIGGS (d=28).

    B0_minimal_HL:    2 features high-level dominantes (m_wwbb, m_bb), 1 CNOT
                      — referência mínima
    B1_high_level:    4 features high-level (massas invariantes), 2 CNOTs
                      — testa hipótese "HL é suficiente"
    B2_hw_efficient:  round-robin sobre 28 features, chain CNOT, 3 CNOTs
                      — paradigma HW-efficient padrão
    B3_low_level:     4 features low-level (cinemática direta), 2 CNOTs
                      — controle contrastivo a B1, mesma topologia
                      Hipótese: AUC(B3) < AUC(B1) frozen → confirma
                      hierarquia física conhecida (Baldi et al. 2014).
    """
    baselines = {}

    # B0 — m_wwbb (f27) + m_bb (f25), 1 CNOT
    baselines["B0_minimal_HL"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 27},  # m_wwbb
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 25},  # m_bb
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    # B1 — 4 features high-level, 2 CNOTs
    baselines["B1_high_level_subset"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 27},  # m_wwbb
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 25},  # m_bb
            {"type": "ENC", "qubit": 2, "axis": 2, "feature": 22},  # m_jjj
            {"type": "ENC", "qubit": 3, "axis": 2, "feature": 26},  # m_wbb
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "ROT", "qubit": 2, "axis": 2},
            {"type": "ROT", "qubit": 3, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "CNOT", "qubit": 3, "ctrl": 2},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "ROT", "qubit": 2, "axis": 3},
            {"type": "ROT", "qubit": 3, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    # B2 — HW-efficient, round-robin sobre 28 features (mas nq=4 → só
    # as 4 primeiras LL ficam codificadas). Baseline naïve da literatura.
    hw_ops = []
    for q in range(n_qubits):
        hw_ops.append({"type": "ENC", "qubit": q, "axis": 2, "feature": q % 28})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 1}, {"type": "ROT", "qubit": q, "axis": 2}]
    for q in range(n_qubits - 1):
        hw_ops.append({"type": "CNOT", "qubit": q + 1, "ctrl": q})
    for q in range(n_qubits):
        hw_ops += [{"type": "ROT", "qubit": q, "axis": 2}, {"type": "ROT", "qubit": q, "axis": 3}]
    baselines["B2_hw_efficient"] = make_arch_mat(hw_ops, n_qubits=n_qubits, L_max=L_max)

    # B3 — controle contrastivo: 4 features low-level, mesma topologia que B1
    # Diferença frente a B1 isola o efeito da escolha de features (HL vs LL).
    baselines["B3_low_level_subset"] = make_arch_mat(
        [
            {"type": "ENC", "qubit": 0, "axis": 2, "feature": 3},   # missing_E_mag
            {"type": "ENC", "qubit": 1, "axis": 2, "feature": 5},   # jet1_pT
            {"type": "ENC", "qubit": 2, "axis": 2, "feature": 0},   # lepton_pT
            {"type": "ENC", "qubit": 3, "axis": 2, "feature": 9},   # jet2_pT
            {"type": "ROT", "qubit": 0, "axis": 2},
            {"type": "ROT", "qubit": 1, "axis": 2},
            {"type": "ROT", "qubit": 2, "axis": 2},
            {"type": "ROT", "qubit": 3, "axis": 2},
            {"type": "CNOT", "qubit": 1, "ctrl": 0},
            {"type": "CNOT", "qubit": 3, "ctrl": 2},
            {"type": "ROT", "qubit": 0, "axis": 3},
            {"type": "ROT", "qubit": 1, "axis": 3},
            {"type": "ROT", "qubit": 2, "axis": 3},
            {"type": "ROT", "qubit": 3, "axis": 3},
        ],
        n_qubits=n_qubits, L_max=L_max,
    )

    return baselines


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["cross_circle", "make_moons", "banknote", "breast_cancer", "higgs"],
        choices=["cross_circle", "make_moons", "banknote", "breast_cancer", "higgs"],
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10_000,
        help="Tamanho do subset balanceado HIGGS (default 10k). Ignorado por outros datasets.",
    )
    args = parser.parse_args()

    cfg = Config()
    all_results = {}

    dataset_loaders = {
        "cross_circle": lambda: create_circle_cross_dataset(
            n_samples_per_class=200, noise_std=0.1, seed=42
        ),
        "make_moons": lambda: create_make_moons_dataset(n_samples=400, noise=0.15, seed=42),
        "banknote": lambda: create_banknote_dataset(seed=42),
        "breast_cancer": lambda: create_breast_cancer_dataset(seed=42, data_dir=args.data_dir),
        "higgs": lambda: create_higgs_dataset(
            seed=42, data_dir=args.data_dir, subset_size=args.subset_size
        ),
    }

    dataset_builders = {
        "cross_circle": lambda: build_cross_circle(args.n_qubits),
        "make_moons": lambda: build_make_moons(args.n_qubits),
        "banknote": lambda: build_banknote(args.n_qubits),
        "breast_cancer": lambda: build_breast_cancer(args.n_qubits),
        "higgs": lambda: build_higgs(args.n_qubits),
    }

    dataset_labels = {
        "cross_circle": "Cross/Circle",
        "make_moons": "Make Moons",
        "banknote": "Banknote",
        "breast_cancer": "BC Wisconsin",
        "higgs": "HIGGS Boson",
    }

    # Pré-download serial do Higgs (evita race conditions se rodado em paralelo)
    if "higgs" in args.datasets:
        print(f"[INFO] Pré-carregando HIGGS subset (size={args.subset_size})...")

    for ds_key in args.datasets:
        X, Y = dataset_loaders[ds_key]()
        baselines = dataset_builders[ds_key]()
        label = dataset_labels[ds_key]
        physics_groups = _PHYSICS_GROUPS.get(ds_key)

        info_extra = ""
        if ds_key == "higgs":
            info_extra = f"  subset_size={args.subset_size}"
        print(
            f"\n[DATA] {label}: X={X.shape}  pos={int(Y.sum())}  "
            f"neg={int((1 - Y).sum())}  balance={float(Y.mean()):.3f}{info_extra}"
        )

        all_results[label] = run_benchmark(
            label, baselines, X, Y, cfg, args, physics_groups=physics_groups
        )

    # ── Tabela final consolidada ──────────────────────────────────────
    print(f"\n\n{'=' * 96}")
    print("  TABELA CONSOLIDADA — todos os datasets")
    print(f"{'=' * 96}")
    print(
        f"{'Dataset':<14} {'Circuito':<24} {'CNOTs':>5} {'Gates':>5} "
        f"{'Frozen':>8} {'Train':>8} {'Δ':>7} "
        f"{'α_LL':>7} {'α_HL':>7} {'HL/LL':>6}"
    )
    print("─" * 96)

    for ds_label, ds_results in all_results.items():
        for circuit_name, r in ds_results.items():
            c = r["circuit"]
            ap = r.get("alpha_physics") or {}
            ll = ap.get("low_mean")
            hl = ap.get("high_mean")
            ratio = ap.get("ratio_high_over_low")
            ll_str = f"{ll:.3f}" if ll is not None else "   —   "
            hl_str = f"{hl:.3f}" if hl is not None else "   —   "
            ratio_str = f"{ratio:.2f}" if ratio is not None else "  —  "
            print(
                f"{ds_label:<14} {circuit_name:<24} {c['n_cnot']:>5} "
                f"{c['n_total']:>5} {r['frozen_auc']:>8.4f} "
                f"{r['trainable_auc']:>8.4f} {r['delta_auc']:>+7.4f} "
                f"{ll_str:>7} {hl_str:>7} {ratio_str:>6}"
            )
        print()

    # ── Diagnóstico físico HIGGS: B1 (HL) vs B3 (LL) ──────────────────
    if "HIGGS Boson" in all_results:
        higgs_r = all_results["HIGGS Boson"]
        if "B1_high_level_subset" in higgs_r and "B3_low_level_subset" in higgs_r:
            b1 = higgs_r["B1_high_level_subset"]
            b3 = higgs_r["B3_low_level_subset"]
            gap_frozen = b1["frozen_auc"] - b3["frozen_auc"]
            gap_train = b1["trainable_auc"] - b3["trainable_auc"]
            print(f"\n{'=' * 96}")
            print("  DIAGNÓSTICO FÍSICO HIGGS: B1 (high-level) vs B3 (low-level)")
            print(f"{'=' * 96}")
            print(
                f"  AUC frozen:     B1={b1['frozen_auc']:.4f}  "
                f"B3={b3['frozen_auc']:.4f}  gap={gap_frozen:+.4f}"
            )
            print(
                f"  AUC trainable:  B1={b1['trainable_auc']:.4f}  "
                f"B3={b3['trainable_auc']:.4f}  gap={gap_train:+.4f}"
            )
            verdict = (
                "CONFIRMADA (frozen)" if gap_frozen > 0
                else "NÃO confirmada (frozen) — investigar"
            )
            print(f"  → Hipótese física HL > LL: {verdict}")

    # ── Salva JSON consolidado ────────────────────────────────────────
    out = Path("benchmark_all_datasets_results.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n[OK] {out}")
    print("[DONE]")


if __name__ == "__main__":
    main()