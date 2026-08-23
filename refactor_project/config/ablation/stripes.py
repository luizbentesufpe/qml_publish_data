from typing import Any, Dict

_ST_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # Stripes: grid 3x3 achatado -> 9 features (mesmo shape de cross_circle)
    "feature_bank_size": 9,
    "feature_bank_min_size": 9,
    "feature_bank_schedule": (9,),
    # Budgets: idênticos a cross_circle (mesma dimensionalidade e
    # natureza "structural pattern" do problema)
    "CNOT_budget": 8,
    "ENC_budget": 18,
    "ROT_budget": 27,
}

STRIPES_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "st_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bc_s0 / iris_s0 / bl_s0 / cc2_s0 — referência "
            "para ablation. Dataset: padrão handcrafted 'stripes' "
            "(listras horizontais vs. verticais, grid 3x3 + ruído "
            "gaussiano), mesmo estilo do cross_circle (círculo vs. "
            "cruz). Balanceado 50/50 por construção. As 9 features "
            "correspondem às células p00..p22 do grid; a estrutura "
            "espacial (linhas vs. colunas) é o padrão discriminativo, "
            "diferente de cross_circle (padrão centro-preenchido vs. "
            "vazio)."
        ),
        "overrides": {
            **_ST_BASE,
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
        "name": "st_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "α convergido deve refletir a estrutura de linhas/colunas: "
            "para o template horizontal (linhas 0 e 2 preenchidas, "
            "linha 1 vazia) vs. vertical (colunas 0 e 2 preenchidas, "
            "coluna 1 vazia), espera-se que TODAS as 9 células "
            "contribuam de forma relativamente uniforme (o sinal "
            "discriminativo está distribuído no grid inteiro, não "
            "concentrado em poucas células como em cross_circle)."
        ),
        "overrides": {
            **_ST_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "st_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Dataset 50/50 por construção — ganho esperado pequeno, "
            "mantido por consistência do protocolo entre datasets."
        ),
        "overrides": {
            **_ST_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "st_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "Mesmo padrão de bc_s3 / iris_s3 / cc2_s3. "
            "CNOT_budget=8 (idêntico a cross_circle) reflete a mesma "
            "dimensionalidade (9 features) e natureza estrutural do "
            "problema — comparação direta entre os dois padrões "
            "handcrafted do framework fica isolada do efeito de budget."
        ),
        "overrides": {
            **_ST_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "st_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final Stripes. "
            "Dataset 50/50: ganho de focal loss deve ser pequeno, "
            "servindo como controle 'balanceado' emparelhado com "
            "cross_circle (também 50/50, também handcrafted 3x3), "
            "isolando diferenças que vêm da estrutura do padrão "
            "(linhas/colunas vs. centro-preenchido) e não do "
            "desbalanceamento de classes."
        ),
        "overrides": {
            **_ST_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]