from typing import Any, Dict

_BN_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Banknote: 4 features
    "feature_bank_size": 4,
    "feature_bank_min_size": 4,
    "feature_bank_schedule": (4,),
    # Budgets: 4 features / fronteira não-linear real
    "CNOT_budget": 3,
    "ENC_budget": 8,
    "ROT_budget": 12,
}

BANKNOTE_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "bn_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a cc_s0 / mm_s0 — referência para ablation."
        ),
        "overrides": {
            **_BN_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,     # sobrescreve 0.5 do _BN_BASE
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────
    {
        "name": "bn_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado menor que Moons (AUC baseline ~0.998) mas "
            "α convergido mais rico: 4 features com importâncias distintas "
            "(variance, skewness, curtosis, entropy)."
        ),
        "overrides": {
            **_BN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "bn_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Elimina colapso all-positive — relevante em Banknote (~55/45): "
            "leve desbalanceamento aumenta risco de collapse para classe majoritária."
        ),
        "overrides": {
            **_BN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "bn_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot esperado — mesmo padrão de cc_s3 / mm_s3. "
            "CNOT_budget=3 reflete fronteira não-linear real com 4 features."
        ),
        "overrides": {
            **_BN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "bn_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Banknote. "
            "Dataset ~55/45: ganho marginal esperado sobre S3, "
            "diferente de cc_s4 / mm_s4 que são 50/50."
        ),
        "overrides": {
            **_BN_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]