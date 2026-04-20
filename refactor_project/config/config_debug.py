from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ConfigDebug:
    # ── Gate budgets ──────────────────────────────────────────────────────────
    ENC_budget: int = 3  # original: 9
    ROT_budget: int = 4  # original: 17
    CNOT_budget: int = 1  # original: 2
    allow_nop: bool = True
    L_max: int = 5  # original: 20
    enc_lambda: float = float(np.pi)

    # ── Data fractions ────────────────────────────────────────────────────────
    val_frac_search: float = 0.40
    percent_search: int = 10  # original: 60  — usa só 10% dos dados na busca
    percent_eval: int = 20  # original: 100 — usa só 20% na avaliação
    holdout_frac: float = 0.20

    # ── Cross-validation ──────────────────────────────────────────────────────
    nested_cv_splits_outer: int = 2  # original: 3
    nested_cv_splits_inner: int = 2  # original: 3
    qubit_grace_steps: int = 3  # number of steps before applying dead qubit penalty
    noise_p: float = 0.01  # probability of random noise in actions (for robustness)
    dead_qubit_penalty_terminal: float = (
        0.20  # penalty applied at episode end if dead qubits are detected
    )
    # ── Confidence intervals ──────────────────────────────────────────────────
    ci_method: str = "bootstrap"
    bootstrap_B: int = 50  # original: 1000
    ci_alpha: float = 0.05

    # ── Threshold lambdas — SEARCH ────────────────────────────────────────────
    search_thr_lam_spec: float = 1.5
    search_thr_lam_fpr: float = 2.0
    search_thr_lam_sens: float = 1.5

    # ── Threshold lambdas — FINAL ─────────────────────────────────────────────
    final_thr_lam_spec: float = 2.0
    final_thr_lam_fpr: float = 2.0
    final_thr_lam_sens: float = 2.0

    # ── Final phase training ──────────────────────────────────────────────────
    final_terminal_diff_method: str = "adjoint"
    final_inner_train_batches_head: int = 2  # original: 5
    final_inner_train_batches_vqc: int = 4  # original: 32
    final_inner_epochs_classif: int = 2  # original: 15
    final_head_epochs: int = 1

    # ── Threshold policy multipliers ──────────────────────────────────────────
    thr_policy_sens_mult: float = 2.0
    thr_policy_spec_mult: float = 1.0
    thr_policy_fpr_mult: float = 1.0
    thr_policy_youden_spec_mult: float = 1.5
    thr_policy_youden_sens_mult: float = 1.5

    # ── Search phase training ─────────────────────────────────────────────────
    search_inner_train_batches_vqc: int = 128  # original: 128
    search_inner_train_batches_head: int = 16  # original: 16
    search_terminal_diff_method: str = "adjoint"
    search_inner_epochs_classif: int = 2  # original: 6
    search_allow_bias_learn: bool = True

    # ── Threshold bounds ──────────────────────────────────────────────────────
    thr_min: float = 0.05
    thr_max: float = 0.95

    # ── Optimiser — VQC ───────────────────────────────────────────────────────
    wd_vqc: float = 1e-4
    lr_vqc: float = 0.01

    # ── Collapse detection ────────────────────────────────────────────────────
    collapse_saturation_abslogit_p95_thr: float = 12.0
    collapse_lr_scale: float = 2.0
    collapse_boost_epochs: int = 2  # original: 5
    collapse_range_min: float = 0.08
    collapse_std_min: float = 0.02
    collapse_boost_batches_mult: int = 2  # original: 3
    collapse_streak_max: int = 5
    collapse_streak_trigger: int = 3
    collapse_retry: bool = True
    collapse_max_retry: int = 2  # original: 3
    collapse_logit_margin_min: float = 0.05
    collapse_prob_std_weak: float = 0.01
    collapse_auc_eps: float = 0.01
    collapse_fail_streak_k: int = 2
    collapse_log_every: int = 5  # original: 25
    collapse_logit_gate_mult: float = 2.0
    collapse_prob_std_min_for_healthlog: float = 0.005
    saturation_fallback_thr: float = 0.95

    # ── Weight decay per phase ────────────────────────────────────────────────
    search_wd_enc: float = 0.0
    search_wd_theta: float = 0.0
    final_wd_enc: float = 0.0
    final_wd_theta: float = 0.0

    # ── Gradient clipping ─────────────────────────────────────────────────────
    clip_head: float = 5.0
    clip_body: float = 2.0
    clip_all: float = 2.0

    # ── Qubits ───────────────────────────────────────────────────────────────
    min_qubits: int = 4
    n_qubits = 4
    max_qubits: int = 6  # original: 10
    start_qubits: int = 4
    qubit_change_cooldown: int = 1
    qubit_penalty: float = 0.05

    # ── Learning rates ────────────────────────────────────────────────────────
    lr_head: float = 3e-2
    lr_enc: float = 2e-3
    lr_theta: float = 5e-3
    wd_head: float = 0.0

    # ── Head training ─────────────────────────────────────────────────────────
    search_head_epochs: int = 2  # original: 8
    search_head_only: bool = False
    search_head_bias_clamp: float = 2.0
    search_allow_bias_learning: bool = True

    # ── Budget control ────────────────────────────────────────────────────────
    hard_block_budget: bool = True
    budget_penalty: float = 0.25

    # ── Feature bank ─────────────────────────────────────────────────────────
    feature_bank_size: int = 9
    feature_bank_update: str = "none"
    feature_bank_min_size: int = 3
    feature_bank_decay_enabled: bool = True
    feature_bank_decay_every: int = 10  # original: 50
    feature_bank_rescore_every: int = 5  # original: 25
    feature_bank_schedule: Tuple[int, ...] = (16, 12, 8, 4)  # original: (128, 96, 64, 32)

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_per_feature: bool = True
    normalize_inputs: bool = True
    norm_eps: float = 1e-6
    norm_clip: float = 0.0
    norm_tanh: bool = False

    # ── Batch / data sizes ────────────────────────────────────────────────────
    inner_train_subset_size: int = 2048  # original: 2048
    batch_size: int = 32  # original: 32
    patch_size: int = 4
    patch_stride: int = 4
    cost_measure_samples: int = 4  # original: 16
    recenter_cap: int = 128  # original: 1024
    val_loss_cap: int = 64  # original: 512

    # ── Head layer clamp ──────────────────────────────────────────────────────
    head_weight_abs_max: float = 2.0
    head_bias_abs_max: float = 2.0
    head_weight_norm_max: float = 1.0
    head_bias_norm_max: float = 2.0
    log_head_clamp: bool = True
    dead_head_norm_thr: float = 0.02

    # ── Separation scales ─────────────────────────────────────────────────────
    final_sep_scale: float = 0.50
    search_sep_scale: float = 0.25

    # ── Composite metric weights — FINAL ─────────────────────────────────────
    final_w_auc: float = 0.55
    final_w_youden: float = 0.20
    final_w_sep: float = 0.25
    final_auc_floor: float = 0.60
    final_auc_gate_tau: float = 0.03
    final_sep_floor: float = 0.15
    final_sep_floor_lam: float = 0.50

    # ── Composite metric weights — SEARCH ────────────────────────────────────
    search_w_auc: float = 0.70
    search_w_youden: float = 0.20
    search_w_sep: float = 0.25
    search_auc_floor: float = 0.50
    search_auc_gate_tau: float = 0.03
    search_sep_floor: float = 0.15
    search_sep_floor_lam: float = 0.50

    # ── AUC floor streak ──────────────────────────────────────────────────────
    search_auc_floor_streak: int = 3

    # ── VQC initialisation ────────────────────────────────────────────────────
    vqc_init_max_attempts: int = 3  # original: 5
    vqc_theta_init_std: float = 0.1
    use_batched_qnode: bool = True

    # ── Main training loop ────────────────────────────────────────────────────
    episodes: int = 5  # original: 400

    # ── Reward curriculum ─────────────────────────────────────────────────────
    curriculum_T1: int = 2  # original: 40
    curriculum_T2: int = 4  # original: 100
    curriculum_w_metric_early: float = 0.4
    curriculum_w_metric_mid: float = 0.5
    curriculum_w_metric_late: float = 1.0

    # ── Terminal reward ───────────────────────────────────────────────────────
    terminal_w_bacc: float = 0.0
    terminal_spec_floor: float = 0.10
    terminal_collapse_penalty: float = 0.05
    terminal_reward_K: float = 5.0
    terminal_K_start: float = 1.0
    terminal_K_end: float = 5.0
    terminal_curriculum_K_T: int = 4  # original: 120
    terminal_K_max: float = 10.0
    terminal_adaptive_K: bool = True
    terminal_K_warmup: int = 2  # original: 20
    terminal_agg_floor: float = 0.30
    terminal_K_abs: float = 0.5
    terminal_use_delta_proxy: bool = True
    terminal_proxy_ema: float = 0.10
    terminal_target_std: float = 0.20
    terminal_clip: float = 4.0
    calib_prev_override: float | None = None
    recenter_after_warmup: bool = False

    # ── Instability penalties ─────────────────────────────────────────────────
    lambda_thr: float = 0.75
    lambda_var: float = 0.50
    lambda_thr_std: float = 0.60
    std_proxy_threshold: float = 0.05
    thr_std_threshold: float = 0.05
    thr_degenerate_lo: float = 0.05
    thr_degenerate_hi: float = 0.95

    # ── Architecture repeat penalty ───────────────────────────────────────────
    repeat_arch_penalty: float = 0.20
    repeat_arch_window: int = 10  # original: 50

    # ── Exploration boost on collapse ─────────────────────────────────────────
    eps_boost_on_collapse: float = 0.05
    eps_decay_mult: float = 1.0
    eps_min_when_collapsing: float = 0.30

    # ── Reliability gate ──────────────────────────────────────────────────────
    reliable_score_min: float = 0.75
    use_stability_weighted_gate: bool = True

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop_patience: int = 5  # original: 80
    early_stop_min_eps: int = 3  # original: 250
    early_stop_log: int = 2  # original: 10

    # ── Encoder ───────────────────────────────────────────────────────────────
    freeze_enc_params: bool = False
    enc_affine_mode: str = "per_feature"
    enc_alpha_init: float = 0.5
    enc_beta_init: float = 0.0
    enc_beta_max: float = 1.0

    # ── Threshold calibration ─────────────────────────────────────────────────
    thr_calib_seed: int = 12345
    thr_calib_frac: float = 0.20
    thr_lam_spec: float = 2.0
    thr_lam_fpr: float = 2.0
    thr_lam_sens: float = 1.0
    thr_init: float = 0.5
    search_calib_cap: int = 32  # original: 256
    final_calib_cap: int = 64  # original: 1024
    calib_seed_delta: int = 99991

    # ── Threshold constraints — SEARCH ───────────────────────────────────────
    search_sens_target: float = 0.42
    search_thr_spec_min: float = 0.65
    search_thr_fpr_max: float = 0.30

    # ── Threshold constraints — FINAL ────────────────────────────────────────
    final_sens_target: float = 0.85
    final_thr_spec_min: float = 0.80
    final_thr_fpr_max: float = 0.10

    # ── Threshold selection policy ────────────────────────────────────────────
    thr_policy: str = "youden"
    thr_mode: str = "soft"

    # ── Patch bank ────────────────────────────────────────────────────────────
    use_patch_bank: bool = False
    patch_bank_compact_features: bool = False

    # ── Execution phase ───────────────────────────────────────────────────────
    phase: str = "search"

    # ── Logit scale ───────────────────────────────────────────────────────────
    search_logit_scale_trainable: bool = True
    search_logit_scale: float = 2.0
    search_logit_scale_max: float = 8.0
    search_logit_scale_min: float = 2.0
    final_logit_clamp: float = 30.0

    # ── Metric shaping and circuit depth ─────────────────────────────────────
    metric_shaping_scale: float = 0.20
    depth_ref_pctl: float = 95.0
    depth_ref_buf: int = 32  # original: 512
    depth_ref_default: float = 10.0
    depth_ref_min: float = 8.0
    dead_qubit_penalty: float = 0.01
    spec_floor_shaping: float = 0.10
    spec_collapse_penalty: float = 0.05

    # ── Repeated actions within episode ──────────────────────────────────────
    recent_actions_maxlen: int = 64  # original: 512
    repeat_penalty: float = 0.01

    # ── NOP penalty ───────────────────────────────────────────────────────────
    min_steps_before_nop: int = 3  # original: 5
    nop_penalty: float = 0.01

    # ── Depth proxy ───────────────────────────────────────────────────────────
    depth_use_proxy_when_skip: bool = True
    proxy_depth_per_op: float = 1.0
    proxy_cnot_per_cnot: int = 1

    # ── Search phase extras ───────────────────────────────────────────────────
    search_inner_train_batches_vqc_override: int = -1
    search_use_pos_weight: bool = False
    search_recenter_logits: bool = False
    search_recenter_before_thr: bool = False
    search_recenter_mu_min: float = 0.25
    search_recenter_mu_clamp: float = 0.5
    search_recenter_mu_damp: float = 0.25

    # ── Multi-seed proxy ──────────────────────────────────────────────────────
    proxy_seed_delta: int = 1337
    proxy_n_seeds: int = 2  # original: 3
    proxy_aggregation: str = "mean"
    proxy_quantile: float = 0.20
    proxy_two_seeds: bool = True

    # ── Logging and diagnostics ───────────────────────────────────────────────
    log_thr_stability: bool = True
    log_loss_per_iter: bool = True
    loss_log_every: int = 2  # original: 10
    loss_ema_alpha: float = 0.05
    log_loss_per_epoch: bool = True

    # ── Grid sizes ────────────────────────────────────────────────────────────
    search_grid_size: int = 11  # original: 51
    grid_size: int = 21  # original: 201

    # ── Reward weights ────────────────────────────────────────────────────────
    alpha_auc: float = 1.0
    beta_sens: float = 0.8
    depth_penalty: float = 0.01
    cnot_penalty: float = 0.012
    rot_penalty: float = 0.002

    # ── Replay buffer ─────────────────────────────────────────────────────────
    replay_capacity: int = 256  # original: 16384
    n_steps: int = 5  # original: 15

    # ── Epsilon-greedy exploration ────────────────────────────────────────────
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_steps: int = 200  # original: 20000
    target_sync_steps: int = 32  # original: 512

    # ── Final phase ───────────────────────────────────────────────────────────
    final_epochs: int = 2  # original: 16
    final_lr_vqc: float = 0.03
    inner_train_batches_head: int = 2  # original: 16

    # ── Sensitivity target ────────────────────────────────────────────────────
    sens_target: float = 0.85

    # ── Focal loss ────────────────────────────────────────────────────────────
    use_focal: bool = True
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0

    def gamma(self) -> float:
        return 0.99
