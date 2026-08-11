import math


def compute_adaptive_l_max(
    in_dim: int,
    factor: float = 1.5,
    floor: int = 20,
    cap: int = 40,
) -> int:
    """
    Calcula L_max adaptativo: clip(ceil(in_dim * factor), floor, cap).

    Base: in_dim (número de features do dataset), não ENC_budget.
    Motivação: Pérez-Salinas et al. (Quantum 2020) — re-uploading exige
    L_max > d para cobertura completa de features no circuito.
    Liu et al. (ACM TQC 2021) — expressibilidade VQC satura com depth,
    cap de 40 evita overhead desnecessário e risco de barren plateau.

    Args:
        in_dim  : número de features do dataset (XtrS.shape[1])
        factor  : fator multiplicativo — ablação: {1.0, 1.5, 2.0}
        floor   : piso mínimo (default 20 — preserva comportamento atual)
        cap     : teto máximo (default 40 — limite prático NISQ)

    Comportamento por dataset com factor=1.5, floor=20, cap=40:
        CC    (in_dim=1)  → clip(ceil(1*1.5)=2,   20, 40) = 20 (sem mudança)
        MM    (in_dim=2)  → clip(ceil(2*1.5)=3,   20, 40) = 20 (sem mudança)
        BN    (in_dim=4)  → clip(ceil(4*1.5)=6,   20, 40) = 20 (sem mudança)
        BC    (in_dim=9)  → clip(ceil(9*1.5)=14,  20, 40) = 20 (sem mudança)
        Higgs (in_dim=28) → clip(ceil(28*1.5)=42, 20, 40) = 40 (mudança real)

    Ablação α∈{1.0, 1.5, 2.0} no Higgs:
        α=1.0 → L_max=28  (abaixo do budget total=38 — agente truncado)
        α=1.5 → L_max=40  (acomoda budget completo com margem)
        α=2.0 → L_max=40  (cap domina — igual ao α=1.5)
    Resultado esperado: α=1.5 >= α=1.0, α=2.0 ≈ α=1.5 com custo igual.
    Isso valida empiricamente o fator 1.5 como escolha ótima.
    """
    if in_dim <= 0:
        raise ValueError(f"in_dim must be positive, got {in_dim}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if floor > cap:
        raise ValueError(f"floor={floor} must be <= cap={cap}")

    raw = math.ceil(in_dim * factor)
    return int(max(int(floor), min(raw, int(cap))))
 
 
