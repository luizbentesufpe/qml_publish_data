from typing import Any, Dict

_IRIS_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Iris (binarizado versicolor vs virginica): 4 features
    "feature_bank_size": 4,
    "feature_bank_min_size": 4,
    "feature_bank_schedule": (4,),
    # Budgets: 4 features / fronteira não-linear (par não linearmente
    # separável) — mantém a mesma razão CNOT=(n-1), ENC=2n, ROT=3n
    # usada no Breast Cancer (n=9 → 8/18/27), aqui com n=4.
    "CNOT_budget": 3,
    "ENC_budget": 8,
    "ROT_budget": 12,
}

IRIS_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "iris_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bc_s0 / bn_s0 / cc_s0 / mm_s0 — referência para ablation. "
            "Dataset: 100 amostras (versicolor vs virginica), 50/50 (50.0%) — "
            "balanceado, mas não linearmente separável (diferente de "
            "setosa vs. resto, que é trivial). Par padrão em benchmarks de QML."
        ),
        "overrides": {
            **_IRIS_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,     # sobrescreve 0.5 do _IRIS_BASE
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────
    {
        "name": "iris_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado moderado — problema de baixa dimensionalidade "
            "(4 features), mas fronteira genuinamente não-linear. "
            "α convergido reflete importância das 4 features: "
            "sepal_length, sepal_width, petal_length, petal_width "
            "(petal_length/petal_width tendem a dominar na separação "
            "versicolor/virginica)."
        ),
        "overrides": {
            **_IRIS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "iris_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Elimina colapso all-positive/all-negative — menos crítico que "
            "em Breast Cancer (50/50 vs. 62.7/37.3), mas mantido para "
            "consistência do protocolo de ablation entre datasets."
        ),
        "overrides": {
            **_IRIS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "iris_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot esperado — mesmo padrão de bc_s3 / bn_s3 / cc_s3 / mm_s3. "
            "CNOT_budget=3 reflete a baixa dimensionalidade (4 features) "
            "mantendo restrição de complexidade compatível com QHW."
        ),
        "overrides": {
            **_IRIS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "iris_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Iris. "
            "Dataset 50/50: ganho de focal loss deve ser pequeno "
            "(sem desbalanceamento de classes a compensar), útil como "
            "controle para isolar o efeito de focal loss quando o "
            "desbalanceamento NÃO é o fator dominante — em contraste "
            "com bc_s4 (62.7/37.3), bn_s4 (55/45) e cc_s4/mm_s4 (50/50)."
        ),
        "overrides": {
            **_IRIS_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]