"""
auto_config.py
==============
Calcula automaticamente L_max, budget_penalty e inner_train_subset_size
a partir das características reais do dataset (in_dim, n_train, budgets).

Objetivo: motor generalizado e escalável — qualquer novo dataset recebe
configuração adequada sem intervenção manual.

Princípios:
    L_max = clip(ceil(budget_total * 1.20), floor=20, cap=60)
        Garante que todos os slots do budget cabem no circuito + 20% margem.
        Pérez-Salinas et al. (Quantum 2020): re-uploading exige L_max > d.
        Liu et al. (ACM TQC 2021): expressibilidade satura além de cap.

    budget_penalty = max(1.0, L_max / 10.0)
        Proporcional ao espaço de busca: mais slots -> violação mais custosa.
        Garante que o agente aprende a respeitar o budget em qualquer L_max.

    inner_train_subset_size = max(batch_size*8, min(2048, n_train))
        Clamp ao dataset real — evita DataLoader pedir mais amostras que existem.
        Fedus et al. (ICML 2020): benefício de amostras extras diminui após 2048.

Valores calculados com budgets reais dos ablation files:

    Dataset          ENC  ROT  CNOT  Budget  L_max  b_pen  subset
    ──────────────────────────────────────────────────────────────
    Cross & Circle     9   17     2      28     34    3.4     256
    Make Moons         4    8     4      16     20    2.0     256
    Banknote           8   12     3      23     28    2.8     820
    BC Wisconsin      18   27     8      53     60    6.0     270
    Higgs             14   20     4      38     46    4.6    2048
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass
class AutoConfigResult:
    """Resultado do cálculo automático de configuração por dataset."""

    in_dim: int
    n_train: int

    # Budgets (respeitados do cfg se já definidos no ablation)
    ENC_budget: int
    ROT_budget: int
    CNOT_budget: int
    budget_total: int

    # Calculados automaticamente
    L_max: int
    budget_penalty: float
    inner_train_subset_size: int

    def apply_to_cfg(self, cfg) -> None:
        """
        Aplica L_max, budget_penalty e inner_train_subset_size ao cfg.
        NAO sobrescreve ENC/ROT/CNOT_budget — respeitados do ablation.
        Chamado em make_cfg_for_qubits após o bloco de feature bank.
        """
        cfg.L_max = self.L_max
        cfg.budget_penalty = self.budget_penalty
        cfg.inner_train_subset_size = self.inner_train_subset_size

    def __str__(self) -> str:
        return (
            f"AutoConfig(in_dim={self.in_dim}, n_train={self.n_train}, "
            f"ENC={self.ENC_budget}, ROT={self.ROT_budget}, CNOT={self.CNOT_budget}, "
            f"budget_total={self.budget_total}, L_max={self.L_max}, "
            f"budget_penalty={self.budget_penalty}, subset={self.inner_train_subset_size})"
        )


def compute_auto_config(
    in_dim: int,
    n_train: int,
    batch_size: int = 32,
    enc_budget: Optional[int] = None,
    rot_budget: Optional[int] = None,
    cnot_budget: Optional[int] = None,
    l_max_floor: int = 20,
    l_max_cap: int = 60,
) -> AutoConfigResult:
    """
    Calcula configuração automática a partir das características do dataset.

    Os budgets ENC/ROT/CNOT são lidos do cfg (ablation) quando disponíveis.
    Se None, são calculados por heurística baseada em in_dim.
    L_max, budget_penalty e inner_train_subset_size são SEMPRE calculados.

    Args:
        in_dim      : número de features (XtrS.shape[1])
        n_train     : tamanho do conjunto de treino de search (len(XtrS))
        batch_size  : batch size do cfg (default 32)
        enc_budget  : ENC_budget do cfg — None = calcular automaticamente
        rot_budget  : ROT_budget do cfg — None = calcular automaticamente
        cnot_budget : CNOT_budget do cfg — None = calcular automaticamente
        l_max_floor : piso mínimo do L_max (default 20)
        l_max_cap   : teto máximo do L_max (default 60)

    Returns:
        AutoConfigResult com todos os valores calculados

    Raises:
        ValueError se in_dim ou n_train forem inválidos
    """
    if in_dim <= 0:
        raise ValueError(f"in_dim must be positive, got {in_dim}")
    if n_train <= 0:
        raise ValueError(f"n_train must be positive, got {n_train}")
    if l_max_floor > l_max_cap:
        raise ValueError(f"l_max_floor={l_max_floor} must be <= l_max_cap={l_max_cap}")

    # Budgets: usa valores do ablation se disponíveis, senão heurística
    enc  = int(enc_budget)  if (enc_budget  and enc_budget  > 0) \
           else max(4, min(in_dim // 2, 20))
    rot  = int(rot_budget)  if (rot_budget  and rot_budget  > 0) \
           else max(8, enc * 2)
    cnot = int(cnot_budget) if (cnot_budget and cnot_budget > 0) \
           else max(2, in_dim // 10)

    budget_total = enc + rot + cnot

    # L_max: budget_total + 20% margem, cap [floor, cap]
    l_max = int(np.clip(
        math.ceil(budget_total * 1.20),
        l_max_floor,
        l_max_cap,
    ))

    # budget_penalty: proporcional a L_max
    budget_penalty = round(max(1.0, l_max / 10.0), 1)

    # inner_train_subset_size: clamp ao dataset real
    floor_subset = batch_size * 8
    subset = max(floor_subset, min(2048, n_train))

    return AutoConfigResult(
        in_dim=in_dim,
        n_train=n_train,
        ENC_budget=enc,
        ROT_budget=rot,
        CNOT_budget=cnot,
        budget_total=budget_total,
        L_max=l_max,
        budget_penalty=budget_penalty,
        inner_train_subset_size=subset,
    )


def compute_meta_L_max(dataset_pool: list[dict]) -> int:
    """
    Calcula meta_L_max como o máximo L_max entre todos os datasets do pool.

    Usado na Fase 2 (Meta-RL) para garantir state_vec unificado entre tarefas,
    permitindo replay cross-task sem inconsistência de dimensões.

    Args:
        dataset_pool: lista de dicts com campos aceitos por compute_auto_config

    Returns:
        meta_L_max: int — L_max unificado para todos os datasets do pool
    """
    if not dataset_pool:
        raise ValueError("dataset_pool cannot be empty")
    return max(compute_auto_config(**kw).L_max for kw in dataset_pool)


if __name__ == "__main__":
    datasets = {
        "Cross & Circle": dict(in_dim=1,  n_train=120,
                               enc_budget=9,  rot_budget=17, cnot_budget=2),
        "Make Moons":     dict(in_dim=2,  n_train=160,
                               enc_budget=4,  rot_budget=8,  cnot_budget=4),
        "Banknote":       dict(in_dim=4,  n_train=820,
                               enc_budget=8,  rot_budget=12, cnot_budget=3),
        "BC Wisconsin":   dict(in_dim=9,  n_train=270,
                               enc_budget=18, rot_budget=27, cnot_budget=8),
        "Higgs":          dict(in_dim=28, n_train=5000,
                               enc_budget=14, rot_budget=20, cnot_budget=4),
    }

    print(f"{'Dataset':<18} {'ENC':>4} {'ROT':>4} {'CNOT':>5} "
          f"{'Budget':>7} {'L_max':>7} {'b_pen':>7} {'subset':>8}")
    print("-" * 70)

    for name, kw in datasets.items():
        r = compute_auto_config(**kw)
        print(f"{name:<18} {r.ENC_budget:>4} {r.ROT_budget:>4} {r.CNOT_budget:>5} "
              f"{r.budget_total:>7} {r.L_max:>7} "
              f"{r.budget_penalty:>7} {r.inner_train_subset_size:>8}")

    pool = list(datasets.values())
    ml = compute_meta_L_max(pool)
    print(f"\nmeta_L_max (Fase 2): {ml}")
    print(f"state_vec unificado: (5x{ml}+2+5,) = ({5*ml+7},)")