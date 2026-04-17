_CC_BASE = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
}

CROSS_CIRCLE_SCENARIOS = [
    # ── S0: mínimo absoluto ───────────────────────────────────────────
    {
        "name": "cc_s0_baseline_minimum",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Referência: o que o RL consegue SEM nenhum dos nossos componentes."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,     # sobrescreve 0.5 do _CC_BASE
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",    # sem efeito em thr_mode=hard — mantido por consistência
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────────
    {
        "name": "cc_s1_plus_enc_parametric",
        "semantics_flag": "strong",
        "notes": "S0 + encoding parametrizado (α/β treináveis end-to-end).",
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ────────────────────────────────────
    {
        "name": "cc_s2_plus_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Elimina collapse all-positive — "
            "J = sens + spec - 1 penaliza soluções degeneradas."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────────
    {
        "name": "cc_s3_plus_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot: reward → -0.34, budget penalty → 0.0."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────────
    {
        "name": "cc_s4_plus_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Cross/Circle. "
            "Neutro em dados balanceados (50/50)."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]