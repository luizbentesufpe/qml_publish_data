from typing import Any, Dict, List


def _blobs_base(n_features: int) -> Dict[str, Any]:
    """
    Overrides base parametrizados por n_features — reaproveitado tanto
    pelos cenários S0-S4 (2D) quanto pelos cenários de alta
    dimensionalidade (BLOBS_HIGHDIM_SCENARIOS).

    Budgets seguem a mesma razão usada em Breast Cancer/Iris:
    CNOT=(n-1), ENC=2n, ROT=3n.
    """
    return {
        "use_patch_bank": False,
        "patch_bank_compact_features": False,
        "feature_bank_update": "none",
        "feature_bank_decay_enabled": False,
        "enc_affine_mode": "per_feature",
        "enc_alpha_init": 0.5,
        "enc_beta_init": 0.0,
        "enc_beta_max": 1.0,
        "feature_bank_size": n_features,
        "feature_bank_min_size": n_features,
        "feature_bank_schedule": (n_features,),
        "CNOT_budget": max(1, n_features - 1),
        "ENC_budget": 2 * n_features,
        "ROT_budget": 3 * n_features,
        # repassado pelo runner (run_one_blobs.py) para load_blobs_pool
        "n_features": n_features,
        "cluster_std": 1.0,
    }


_BL_BASE: Dict[str, Any] = _blobs_base(2)

# ══════════════════════════════════════════════════════════════════════
# S0→S4 — baseline 2D (mesmo protocolo de ablation dos demais datasets)
# ══════════════════════════════════════════════════════════════════════

BLOBS_SCENARIOS: List[Dict[str, Any]] = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "bl_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bc_s0 / iris_s0 / bn_s0 — referência para ablation. "
            "Dataset: blobs gaussianos 2D, 2 centros, cluster_std=1.0 — "
            "classes linearmente separáveis por construção. Caso mais "
            "'fácil' do framework: serve de piso/controle de sanidade."
        ),
        "overrides": {
            **_BL_BASE,
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
        "name": "bl_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado mínimo — com apenas 2 features linearmente "
            "separáveis, α convergido deve ficar próximo de 1.0 para "
            "ambas (sem feature dominante a descobrir)."
        ),
        "overrides": {
            **_BL_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "bl_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Balanceado 50/50 por construção do make_blobs — ganho "
            "esperado pequeno, mantido por consistência do protocolo."
        ),
        "overrides": {
            **_BL_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "bl_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "CNOT_budget=1 reflete a baixa dimensionalidade (2 features) "
            "e a fronteira linear — circuito mínimo deve bastar."
        ),
        "overrides": {
            **_BL_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "bl_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Blobs 2D. "
            "Dataset balanceado e trivialmente separável: ganho de "
            "focal loss deve ser desprezível. Serve como controle "
            "'fácil' no cross-dataset comparado a Iris/Breast Cancer "
            "(fronteiras não-lineares) e às Círculos Concêntricos "
            "(fronteira radial, XOR-like)."
        ),
        "overrides": {
            **_BL_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]

# ══════════════════════════════════════════════════════════════════════
# Testes de escalabilidade em alta dimensionalidade
# ══════════════════════════════════════════════════════════════════════
#
# Mesma config "full pipeline" do S4 (enc parametrizado + hard budget +
# thr soft + focal), variando apenas n_features. Mede como custo
# (CNOT/ENC/ROT) e desempenho (AUC/sens) escalam com a dimensionalidade
# — relevante para comparação com a literatura de QML, que tipicamente
# reporta esse tipo de curva de escalabilidade.

HIGH_DIM_FEATURE_COUNTS: List[int] = [4, 8, 16, 32]


def _make_highdim_scenario(n_features: int) -> Dict[str, Any]:
    base = _blobs_base(n_features)
    return {
        "name": f"bl_hd_d{n_features}",
        "semantics_flag": "strong",
        "notes": (
            f"Teste de escalabilidade em alta dimensionalidade: blobs "
            f"gaussianos com {n_features} features, 2 centros, "
            f"cluster_std=1.0. Config idêntica ao S4 (full pipeline) do "
            f"baseline 2D — único fator variado é a dimensionalidade. "
            f"CNOT_budget={base['CNOT_budget']}, ENC_budget={base['ENC_budget']}, "
            f"ROT_budget={base['ROT_budget']} (mesma razão do 2D, escalada "
            f"linearmente com n_features). "
            f"Caveat: com cluster_std fixo, a separabilidade relativa entre "
            f"classes tende a AUMENTAR com d (distância euclidiana escala "
            f"~sqrt(d) para ruído i.i.d.) — resultados de AUC/sens entre "
            f"diferentes d não são diretamente comparáveis sem normalizar "
            f"cluster_std por sqrt(n_features)."
        ),
        "overrides": {
            **base,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    }


BLOBS_HIGHDIM_SCENARIOS: List[Dict[str, Any]] = [
    _make_highdim_scenario(d) for d in HIGH_DIM_FEATURE_COUNTS
]

#: Conjunto completo (S0-S4 2D + varredura de alta dimensionalidade),
#: pronto para ser consumido diretamente por main_blobs.py
ALL_BLOBS_SCENARIOS: List[Dict[str, Any]] = BLOBS_SCENARIOS + BLOBS_HIGHDIM_SCENARIOS