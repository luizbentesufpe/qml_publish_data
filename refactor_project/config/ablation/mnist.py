from typing import Any, Dict

#: Nº de componentes PCA usado nos cenários — deve bater com
#: DEFAULT_PCA_COMPONENTS em data/mnist.py
_N_COMPONENTS = 8

_MN_BASE: Dict[str, Any] = {
    "use_patch_bank": False,
    "patch_bank_compact_features": False,
    "feature_bank_update": "none",
    "feature_bank_decay_enabled": False,
    "enc_affine_mode": "per_feature",
    "enc_alpha_init": 0.5,
    "enc_beta_init": 0.0,
    "enc_beta_max": 1.0,
    # MNIST (3 vs 8, reduzido via PCA): 8 features (componentes principais)
    "feature_bank_size": _N_COMPONENTS,
    "feature_bank_min_size": _N_COMPONENTS,
    "feature_bank_schedule": (_N_COMPONENTS,),
    # Budgets: 8 features / imagens reais, fronteira complexa e não
    # conhecida a priori (diferente de Blobs/Círculos, que têm
    # fronteira geométrica definida por construção)
    "CNOT_budget": 7,
    "ENC_budget": 16,
    "ROT_budget": 24,
}

MNIST_SCENARIOS = [
    # ── S0: baseline ─────────────────────────────────────────────
    {
        "name": "mn_s0_baseline",
        "semantics_flag": "strong",
        "notes": (
            "Baseline mínimo: enc fixo (α=1.0 congelado), BCE, "
            "sem hard budget, thr hard. "
            "Equivalente a bc_s0 / iris_s0 / st_s0 — referência para "
            "ablation. Dataset: MNIST binarizado (dígitos 3 vs. 8 — par "
            "visualmente mais parecido, benchmark difícil clássico), "
            "reduzido de 784 pixels para 8 componentes PCA (necessário "
            "para angle encoding em poucos qubits). Balanceado 50/50 "
            "por construção (undersampling). Único dataset do "
            "framework com imagens reais e features derivadas "
            "(componentes PCA) em vez de pixels ou features de domínio "
            "diretas — testa generalização do pipeline para "
            "representações aprendidas por outro método (PCA) como "
            "input do encoding quântico."
        ),
        "overrides": {
            **_MN_BASE,
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
        "name": "mn_s1_enc_param",
        "semantics_flag": "strong",
        "notes": (
            "S0 + encoding parametrizado (α/β treináveis end-to-end). "
            "Ganho esperado relevante: componentes PCA são ordenados "
            "por variância explicada, não por poder discriminativo "
            "entre 3 e 8 — α convergido pode revelar que componentes "
            "de variância mais baixa (ex.: pca_4, pca_5) são mais "
            "úteis para a classificação do que os primeiros (pca_0, "
            "pca_1), que capturam variação geral de escrita à mão "
            "não necessariamente ligada ao dígito específico."
        ),
        "overrides": {
            **_MN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "hard",
            "thr_policy": "youden",
        },
    },
    # ── S2: +threshold soft Youden ───────────────────────────────
    {
        "name": "mn_s2_thr_soft_youden",
        "semantics_flag": "strong",
        "notes": (
            "S1 + threshold soft Youden J. "
            "Dataset 50/50 por undersampling — ganho esperado pequeno, "
            "mantido por consistência do protocolo entre datasets."
        ),
        "overrides": {
            **_MN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": False,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S3: +hard budget ──────────────────────────────────────────
    {
        "name": "mn_s3_hard_budget",
        "semantics_flag": "strong",
        "notes": (
            "S2 + hard budget blocking. "
            "CNOT_budget=7 reflete a dimensionalidade intermediária "
            "(8 features PCA) — dataset real com fronteira de "
            "decisão desconhecida a priori (diferente de Blobs/Círculos, "
            "geometricamente definidos por construção), então o budget "
            "aqui é mais uma restrição de custo/executabilidade em QHW "
            "do que uma hipótese sobre a complexidade mínima necessária."
        ),
        "overrides": {
            **_MN_BASE,
            "freeze_enc_params": False,
            "use_focal": False,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
    # ── S4: +focal loss ───────────────────────────────────────────
    {
        "name": "mn_s4_focal",
        "semantics_flag": "strong",
        "notes": (
            "S3 + focal loss. "
            "FULL: todos os componentes — referência final MNIST (3 vs 8). "
            "Dataset 50/50: ganho de focal loss deve ser pequeno "
            "quanto a desbalanceamento, mas o par 3/8 tem overlap "
            "visual real (ambíguo mesmo para humanos em letra ruim) — "
            "focal loss pode ainda ajudar focando em exemplos "
            "genuinamente difíceis de separar após a redução PCA, "
            "diferente dos datasets sintéticos (Blobs/Círculos/Stripes) "
            "onde a dificuldade é inteiramente controlada pela geração."
        ),
        "overrides": {
            **_MN_BASE,
            "freeze_enc_params": False,
            "use_focal": True,
            "hard_block_budget": True,
            "thr_mode": "soft",
            "thr_policy": "youden",
        },
    },
]