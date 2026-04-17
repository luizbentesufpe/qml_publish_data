_MM_BASE = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Make Moons: 2 features
    "feature_bank_size": 2,
    "feature_bank_min_size": 2,
    "feature_bank_schedule": (2,),
    # Budgets: fronteira não-linear precisa de mais entanglement que Cross/Circle
    "CNOT_budget": 4,
    "ENC_budget": 4,
    "ROT_budget": 8,
}

MAKE_MOONS_SCENARIOS = [
    # ── S0: baseline mínimo ──────────────────────────────────────────
    {
        "name": "mm_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente ao cc_s0 — referência sem nenhum componente proposto."
        ),
        "overrides": {
            **_MM_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,  # sobrescreve 0.5 do _MM_BASE — replica B0
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",  # sem efeito em thr_mode=hard — mantido por consistência
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────────
    {
        "name": "mm_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado maior que Cross/Circle: fronteira não-linear "
            "exige feature weighting adaptativo."
        ),
        "overrides": {
            **_MM_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────────
    {
        "name": "mm_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Elimina colapso all-positive — mais crítico em Moons que em "
            "Cross/Circle porque as classes têm overlap real na região de noise."
        ),
        "overrides": {
            **_MM_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ─────────────────────────────────────────────
    {
        "name": "mm_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot esperado — mesmo padrão do cc_s3. "
            "CNOT_budget=4 permite mais entanglement que Cross/Circle (budget=2)."
        ),
        "overrides": {
            **_MM_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ──────────────────────────────────────────────
    {
        "name": "mm_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Make Moons. "
            "Dataset balanceado (50/50) → focal neutro, confirma padrão cc_s4."
        ),
        "overrides": {
            **_MM_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]
