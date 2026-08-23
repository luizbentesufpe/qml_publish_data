from typing import Any, Dict

_CC_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Círculos Concêntricos: 2 features
    "feature_bank_size": 2,
    "feature_bank_min_size": 2,
    "feature_bank_schedule": (2,),
    # Budgets: 2 features / fronteira radial (não-linear no espaço
    # original — precisa de mapeamento não-trivial p/ ser separável)
    "CNOT_budget": 2,
    "ENC_budget": 4,
    "ROT_budget": 8,
}

CONCENTRIC_CIRCLES_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "cc2_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bc_s0 / iris_s0 / bl_s0 — referência para ablation. "
            "Dataset: círculos concêntricos (sklearn.make_circles), 400 "
            "amostras, ~50/50, fronteira RADIAL — NÃO linearmente "
            "separável no espaço original (XOR-like em 2D), diferente "
            "de Blobs (linear) e mais difícil que Iris/Breast Cancer "
            "(fronteiras suaves). Testa se o encoding aprendido consegue "
            "recuperar uma fronteira genuinamente não-linear com apenas "
            "2 features."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────
    {
        "name": "cc2_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Com apenas 2 features simétricas (x0, x1 têm papel "
            "equivalente na fronteira radial x0²+x1²=r²), α convergido "
            "deve ficar próximo entre as duas — assimetria forte aqui "
            "seria um sinal de overfitting ao ruído da seed, não de "
            "importância real de feature."
        ),
        "overrides": {
            **_CC_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "cc2_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Dataset ~50/50 (make_circles gera anéis balanceados por "
            "padrão) — ganho esperado pequeno, mantido por consistência "
            "do protocolo de ablation entre datasets."
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
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "cc2_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "CNOT_budget=2 (vs. 1 no Blobs 2D linear) — a fronteira "
            "radial exige entrelaçamento mínimo para separar os dois "
            "anéis, diferente do caso linearmente separável."
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
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "cc2_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Círculos "
            "Concêntricos. Dataset balanceado (~50/50): ganho de focal "
            "loss deve ser pequeno, útil como controle isolando o "
            "efeito de fronteira não-linear pura (sem desbalanceamento "
            "de classes como fator de confusão) em comparação com "
            "Breast Cancer (62.7/37.3) e Iris (50/50, mas fronteira "
            "menos extrema)."
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