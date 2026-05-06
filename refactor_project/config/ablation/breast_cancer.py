from typing import Any, Dict

_BC_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Breast Cancer: 9 features
    "feature_bank_size": 9,
    "feature_bank_min_size": 9,
    "feature_bank_schedule": (9,),
    # Budgets: 9 features / fronteira não-linear real
    "CNOT_budget": 8,
    "ENC_budget": 18,
    "ROT_budget": 27,
}

BREAST_CANCER_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "bc_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bn_s0 / cc_s0 / mm_s0 — referência para ablation. "
            "Dataset: 569 amostras, 357 malignant (62.7%), 212 benign (37.3%)."
        ),
        "overrides": {
            **_BC_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,     # sobrescreve 0.5 do _BC_BASE
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────
    {
        "name": "bc_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado moderado para diagnóstico de câncer. "
            "α convergido reflete importância das 9 features: "
            "radius, texture, perimeter, area, smoothness, "
            "compactness, concavity, concave_points, symmetry."
        ),
        "overrides": {
            **_BC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "bc_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Elimina colapso all-positive — relevante em Breast Cancer (62.7/37.3): "
            "desbalanceamento moderado aumenta risco de collapse para malignant. "
            "Crítico para diagnóstico médico: evita falsos positivos."
        ),
        "overrides": {
            **_BC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "bc_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot esperado — mesmo padrão de bn_s3 / cc_s3 / mm_s3. "
            "CNOT_budget=8 reflete fronteira não-linear real com 9 features. "
            "Restrição de complexidade garante executabilidade em QHW."
        ),
        "overrides": {
            **_BC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "bc_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Breast Cancer. "
            "Dataset 62.7/37.3: focal loss reduz custo de hard negatives (benign), "
            "focando em exemplos difíceis de classificar — crítico em diagnóstico médico. "
            "Diferente de bn_s4 (55/45) / cc_s4 / mm_s4 (50/50)."
        ),
        "overrides": {
            **_BC_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]