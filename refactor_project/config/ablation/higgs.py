from typing import Any, Dict

_HIGGS_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Higgs: 28 features (21 low-level kinematic + 7 high-level invariant masses)
    "feature_bank_size": 28,
    "feature_bank_min_size": 14,
    "feature_bank_schedule": (28,),
    # Budgets: d=28 features / fronteira não-linear genuína (LR ~0.68 AUC)
    # BENC=14 (metade) força seleção sem permitir cobertura completa trivial.
    # CNOT=4 reflete maior complexidade que Banknote (LR satura em ~0.99).
    "CNOT_budget": 4,
    "ENC_budget": 14,
    "ROT_budget": 20,
}

HIGGS_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "higgs_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a cc_s0 / mm_s0 / bn_s0 — referência para ablation. "
            "Em Higgs, baseline esperado é especialmente fraco: 28 features "
            "com escalas heterogêneas (pT em GeV, eta adimensional, b_tag em [0,1]) "
            "tornam encoding fixo θ=π·x particularmente subótimo."
        ),
        "overrides": {
            **_HIGGS_BASE,
            "freeze_enc_params": True,
            "enc_alpha_init": 1.0,     # sobrescreve 0.5 do _HIGGS_BASE
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S1: +encoding parametrizado ──────────────────────────────
    {
        "name": "higgs_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado MÁXIMO entre todos os datasets: "
            "d=28 com escalas heterogêneas é o regime onde α paramétrico "
            "tem maior alavancagem. "
            "Hipótese física testável: α deve amplificar features high-level "
            "(idx 21-27: m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb — "
            "massas invariantes derivadas) sobre low-level cinemáticas (idx 0-20). "
            "Validação esperada: ρ(α, SHAP) significativo (p < 0.05)."
        ),
        "overrides": {
            **_HIGGS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "higgs_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Subset balanceado 50/50 por undersampling — risco de collapse "
            "all-positive baixo, similar a CC/MM. "
            "Ganho esperado modesto sobre S1, diferente de bn_s2 (~55/45)."
        ),
        "overrides": {
            **_HIGGS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "higgs_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Sweet-spot esperado — mesmo padrão de cc_s3 / mm_s3 / bn_s3. "
            "CNOT_budget=4 vs 3 do Banknote: reflete que fronteira HIGGS "
            "é genuinamente não-linear (LR atinge apenas ~0.68 AUC vs ~0.99 BN). "
            "Espera-se que agente sature CNOT_budget (entanglement-bound) "
            "e atenue features low-level redundantes via α."
        ),
        "overrides": {
            **_HIGGS_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "higgs_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Higgs. "
            "Subset balanceado 50/50: ganho marginal esperado sobre S3, "
            "similar a cc_s4 / mm_s4 (também 50/50). "
            "Diferente de bn_s4 (~55/45) onde focal pode dar ganho residual."
        ),
        "overrides": {
            **_HIGGS_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]