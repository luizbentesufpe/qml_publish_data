from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Config:
    ENC_budget: int = 9
    ROT_budget: int = 17
    CNOT_budget: int = 2
    allow_nop: bool = True
    L_max: int = 20
    enc_lambda: float = float(np.pi)
    val_frac_search: float = 0.40
    percent_search: int = 60
    percent_eval: int = 100
    holdout_frac: float = 0.20  # final untouched test holdout
    nested_cv_splits_outer: int = 3
    nested_cv_splits_inner: int = 3
    ci_method: str = "bootstrap"  # "t" | "bootstrap"
    bootstrap_B: int = 1000
    ci_alpha: float = 0.05
    search_thr_lam_spec: float = 1.5
    search_thr_lam_fpr: float = 2.0
    search_thr_lam_sens: float = 1.5
    final_thr_lam_spec: float = 2.0
    final_thr_lam_fpr: float = 2.0
    final_thr_lam_sens: float = 2.0
    final_terminal_diff_method: str = "adjoint"
    final_inner_train_batches_head: int = 5
    final_inner_train_batches_vqc: int = 32
    final_inner_epochs_classif: int = 15
    final_head_epochs: int = 1
    thr_policy_sens_mult: float = 2.0  # sens policy makes sens penalty heavier
    thr_policy_spec_mult: float = 1.0  # sens policy may soften spec a bit
    thr_policy_fpr_mult: float = 1.0
    thr_policy_youden_spec_mult: float = 1.5
    thr_policy_youden_sens_mult: float = 1.5
    search_inner_train_batches_vqc: int = 128
    search_inner_train_batches_head: int = 16
    thr_min: float = 0.05
    thr_max: float = 0.95
    wd_vqc: float = 1e-4
    collapse_saturation_abslogit_p95_thr: float = 12.0
    search_terminal_diff_method: str = "adjoint"
    search_inner_epochs_classif: int = 6
    search_allow_bias_learn: bool = True
    lr_vqc: float = 0.01
    collapse_lr_scale: float = 2.0
    collapse_boost_epochs: int = 5
    collapse_range_min: float = 0.08
    collapse_std_min: float = 0.02
    collapse_boost_batches_mult: int = 3
    # Phase-aware defaults (SEARCH: less regularization to avoid collapse)
    search_wd_enc: float = 0.0
    search_wd_theta: float = 0.0
    final_wd_enc: float = 0.0
    final_wd_theta: float = 0.0

    # Gradient clipping (Priority #1: allow head to move)
    clip_head: float = 5.0
    clip_body: float = 2.0
    clip_all: float = 2.0
    # variable qubits
    min_qubits: int = 4
    n_qubits = 4
    lr_head: float = 3e-2
    search_head_epochs: int = 8
    max_qubits: int = 10
    start_qubits: int = 4
    qubit_change_cooldown: int = 1
    qubit_penalty: float = 0.05
    hard_block_budget: bool = True
    budget_penalty: float = 0.25
    feature_bank_size: int = 9
    feature_bank_update: str = "none"  # "saliency" | "chi2" | "random" | "none"
    feature_bank_min_size: int = 3

    norm_per_feature: bool = True
    inner_train_subset_size: int = 2048
    thr_init: float = 0.5
    batch_size: int = 32
    patch_size: int = 4
    patch_stride: int = 4
    cost_measure_samples: int = 16
    # ── Head layer clamp / clip ────────────────────────────────────────────────
    head_weight_abs_max: float = 2.0  # absolute max value for head weights
    head_bias_abs_max: float = 2.0  # absolute max value for head bias
    head_weight_norm_max: float = 1.0  # max norm for head weights
    head_bias_norm_max: float = 2.0  # max norm for head bias
    log_head_clamp: bool = True  # log when clamp is applied

    # ── Separation scales per phase ───────────────────────────────────────────
    final_sep_scale: float = 0.50  # separation weight in final phase
    search_sep_scale: float = 0.25  # separation weight in search phase

    # ── Composite metric weights and floors — FINAL phase ─────────────────────
    final_w_auc: float = 0.55  # AUC weight in final score
    final_w_youden: float = 0.20  # Youden index weight in final score
    final_w_sep: float = 0.25  # separation weight in final score
    final_auc_floor: float = 0.60  # minimum acceptable AUC in final phase
    final_auc_gate_tau: float = 0.03  # tolerance margin for final AUC gate
    final_sep_floor: float = 0.15  # minimum acceptable separation in final phase
    final_sep_floor_lam: float = 0.50  # penalty when separation drops below floor

    # ── Composite metric weights and floors — SEARCH phase (RL proxy) ─────────
    search_w_auc: float = 0.70  # AUC weight in search score
    search_w_youden: float = 0.20  # Youden index weight in search score
    search_w_sep: float = 0.25  # separation weight in search score
    search_auc_floor: float = 0.50  # minimum acceptable AUC in search phase
    search_auc_gate_tau: float = 0.03  # tolerance margin for search AUC gate
    search_sep_floor: float = 0.15  # minimum acceptable separation in search phase
    search_sep_floor_lam: float = 0.50  # penalty when separation drops below floor

    # ── Variational Quantum Circuit (VQC) initialisation ──────────────────────
    vqc_init_max_attempts: int = 5  # max initialisation attempts for VQC

    # ── Prediction collapse control ───────────────────────────────────────────
    collapse_streak_max: int = 5  # max consecutive collapses before intervention

    # ── Main training loop ────────────────────────────────────────────────────
    episodes: int = 400  # total number of training episodes

    # ── Reward curriculum (ramp over episodes) ────────────────────────────────
    curriculum_T1: int = 40  # end of early curriculum phase
    curriculum_T2: int = 100  # end of mid curriculum phase
    curriculum_w_metric_early: float = 0.4  # metric weight in early phase
    curriculum_w_metric_mid: float = 0.5  # metric weight in mid phase
    curriculum_w_metric_late: float = 1.0  # metric weight in late phase

    feature_bank_decay_enabled: bool = True  # enables feature bank decay

    # ── Terminal reward ───────────────────────────────────────────────────────
    terminal_w_bacc: float = 0.0  # balanced-accuracy weight (off by default)
    terminal_spec_floor: float = 0.10  # min specificity; below this counts as collapse
    terminal_collapse_penalty: float = 0.05  # penalty applied when collapse is detected
    collapse_log_every: int = 25  # collapse health log frequency (in episodes)

    terminal_reward_K: float = 5.0  # base K factor for terminal reward
    terminal_K_start: float = 1.0  # K value at episode 0
    terminal_K_end: float = 5.0  # K value at end of ramp
    terminal_curriculum_K_T: int = 120  # K ramp duration in episodes
    terminal_K_max: float = 10.0  # hard cap for K
    terminal_adaptive_K: bool = True  # enables adaptive K adjustment
    terminal_K_warmup: int = 20  # warmup episodes before adapting K
    terminal_agg_floor: float = 0.30  # proxy_score below this zeroes the absolute signal
    terminal_K_abs: float = 0.5  # weight of the absolute signal term in reward
    calib_prev_override: float | None = None  # overrides calibration prevalence (None = auto)
    recenter_after_warmup: bool = False  # recenters reward distribution after warmup
    terminal_target_std: float = 0.20  # target std for normalised reward
    terminal_clip: float = 4.0  # clip value for normalised terminal reward

    lambda_thr: float = 0.75  # weight of threshold term in score
    thr_degenerate_lo: float = 0.05  # threshold below this is degenerate (too low)
    thr_degenerate_hi: float = 0.95  # threshold above this is degenerate (too high)

    # ── Delta-proxy mode for terminal reward ──────────────────────────────────
    terminal_use_delta_proxy: bool = True  # use proxy delta instead of absolute value
    terminal_proxy_ema: float = 0.10  # exponential smoothing factor for proxy
    lambda_var: float = 0.50  # instability penalty across seeds
    std_proxy_threshold: float = 0.05  # max acceptable proxy std across seeds
    thr_std_threshold: float = 0.05  # max acceptable threshold std across seeds
    lambda_thr_std: float = 0.60  # weight of threshold instability penalty

    # ── Architecture repeat penalty ───────────────────────────────────────────
    repeat_arch_penalty: float = 0.20  # penalty for reusing a previously seen architecture
    repeat_arch_window: int = 50  # max architecture hashes kept in memory

    # ── Exploration boost on repeated collapse ────────────────────────────────
    eps_boost_on_collapse: float = 0.05  # epsilon increment when collapse is detected
    eps_decay_mult: float = 1.0  # multiplier for epsilon decay rate after boost
    eps_min_when_collapsing: float = 0.30  # minimum epsilon during collapse period
    collapse_streak_trigger: int = 3  # collapse streak length that triggers boost
    reliable_score_min: float = 0.75  # min score to consider an episode reliable
    use_stability_weighted_gate: bool = True  # weights the gate by stability degree

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop_patience: int = 80  # episodes without improvement before stopping
    early_stop_min_eps: int = 250  # minimum episodes before early stop is allowed

    # ── Encoder / input normalisation ─────────────────────────────────────────
    freeze_enc_params: bool = False  # freezes encoder parameters during training
    thr_calib_seed: int = 12345  # seed for threshold calibration
    thr_calib_frac: float = 0.20  # fraction of data used for calibration
    search_head_bias_clamp: float = 2.0  # head bias clamp during search
    thr_lam_spec: float = 2.0  # specificity weight in threshold optimisation
    thr_lam_fpr: float = 2.0  # false-positive rate weight in threshold optimisation
    thr_lam_sens: float = 1.0  # sensitivity weight in threshold optimisation

    # ── Threshold constraints — SEARCH phase (relaxed) ────────────────────────
    search_sens_target: float = 0.42  # sensitivity target in search (relaxed criterion)
    search_thr_spec_min: float = 0.65  # minimum specificity in search
    search_thr_fpr_max: float = 0.30  # maximum false-positive rate in search

    # ── Threshold constraints — FINAL phase (strict) ──────────────────────────
    final_sens_target: float = 0.85  # sensitivity target in final phase
    final_thr_spec_min: float = 0.80  # minimum specificity in final phase
    final_thr_fpr_max: float = 0.10  # maximum false-positive rate in final phase

    # ── Threshold selection policy ────────────────────────────────────────────
    thr_policy: str = "youden"  # criterion: "youden" | "sens" | "f2"
    thr_mode: str = "soft"  # application mode: "soft" (smoothed) | "hard"

    # ── Calibration capacity per phase ────────────────────────────────────────
    search_calib_cap: int = 256  # max calibration samples in search
    final_calib_cap: int = 1024  # max calibration samples in final phase

    # ── Feature bank and normalisation ───────────────────────────────────────
    feature_bank_schedule: Tuple[int, ...] = (128, 96, 64, 32)  # bank sizes along training
    use_patch_bank: bool = False  # enables patch bank
    patch_bank_compact_features: bool = False  # compacts features in patch bank
    normalize_inputs: bool = True  # normalises inputs before encoder
    norm_eps: float = 1e-6  # numerical epsilon for normalisation
    norm_clip: float = 0.0  # post-normalisation clip (0 = disabled)
    norm_tanh: bool = False  # applies tanh after normalisation to bound output

    # ── Encoder affine mode ───────────────────────────────────────────────────
    enc_affine_mode: str = "per_feature"  # affine transform granularity: "per_feature" | "shared"
    enc_alpha_init: float = 0.5  # initial value for encoder alpha parameter
    enc_beta_init: float = 0.0  # initial value for encoder beta parameter
    enc_beta_max: float = 1.0  # maximum value for beta

    # ── VQC / quantum circuit ─────────────────────────────────────────────────
    use_batched_qnode: bool = True  # uses qnode in batch mode (more efficient)
    vqc_theta_init_std: float = 0.1  # std for VQC angle initialisation

    # ── Execution phase ───────────────────────────────────────────────────────
    phase: str = "search"  # current phase: "search" | "final"

    # ── Logit scale (output head) ─────────────────────────────────────────────
    search_logit_scale_trainable: bool = True  # makes logit scale trainable during search
    search_logit_scale: float = 2.0  # initial logit scale in search
    search_logit_scale_max: float = 8.0  # maximum logit scale
    search_logit_scale_min: float = 2.0  # minimum logit scale

    # ── Metric shaping and circuit depth ──────────────────────────────────────
    metric_shaping_scale: float = 0.20  # scale of the metric shaping term
    depth_ref_pctl: float = 95.0  # percentile used to compute reference depth
    depth_ref_buf: int = 512  # buffer size for estimating reference depth
    depth_ref_default: float = 10.0  # default reference depth (no history)
    depth_ref_min: float = 8.0  # minimum reference depth
    dead_qubit_penalty: float = 0.05  # penalty for qubits with no operations
    qubit_grace_steps:  int   = 3  # number of steps before applying dead qubit penalty
    noise_p: float = 0.01  # probability of random noise in actions (for robustness)
    dead_qubit_penalty_terminal: float = 0.20 # penalty applied at episode end if dead qubits are detected
    spec_floor_shaping: float = 0.10  # specificity floor in shaping
    spec_collapse_penalty: float = 0.05  # extra penalty when specificity collapses

    # ── Repeated actions within episode ──────────────────────────────────────
    recent_actions_maxlen: int = 512  # max length of recent actions history
    repeat_penalty: float = 0.01  # penalty for repeating the same action in an episode

    # ── NOP (empty action) penalty ────────────────────────────────────────────
    min_steps_before_nop: int = 5  # minimum steps before NOP is allowed
    nop_penalty: float = 0.01  # penalty for emitting NOP

    # ── Depth proxy when step is skipped ─────────────────────────────────────
    depth_use_proxy_when_skip: bool = True  # use depth proxy when skipping a step
    proxy_depth_per_op: float = 1.0  # depth factor per operation in proxy
    proxy_cnot_per_cnot: int = 1  # CNOT count factor in proxy

    # ── Search phase configuration ────────────────────────────────────────────
    search_head_only: bool = False  # train head only (frozen encoder)
    search_inner_train_batches_vqc_override: int = -1  # overrides VQC batch count (-1 = auto)
    calib_seed_delta: int = 99991  # delta applied to calibration seed
    search_use_pos_weight: bool = False  # use positive class weight in search

    # ── Multi-seed proxy ──────────────────────────────────────────────────────
    proxy_seed_delta: int = 1337  # base delta for generating proxy seeds
    proxy_n_seeds: int = 3  # number of seeds used in proxy
    proxy_aggregation: str = "mean"  # aggregation across seeds: "mean" | "min" | "quantile"
    proxy_quantile: float = 0.20  # quantile used when aggregation="quantile"

    # ── Logging and diagnostics ───────────────────────────────────────────────
    log_thr_stability: bool = True  # logs threshold stability across seeds

    # ── Collapse retry ────────────────────────────────────────────────────────
    collapse_retry: bool = True  # retry after collapse is detected
    collapse_max_retry: int = 3  # max retry attempts after collapse
    collapse_logit_margin_min: float = 0.05  # min logit margin to avoid collapse classification
    collapse_prob_std_weak: float = 0.01  # min probability std (weak threshold)
    collapse_auc_eps: float = 0.01  # epsilon for collapse detection via AUC
    collapse_fail_streak_k: int = 2  # failure streak length that activates collapse retry
    search_auc_floor_streak: int = 3  # consecutive episodes below floor before intervention

    # ── Head diagnostics ─────────────────────────────────────────────────────
    dead_head_norm_thr: float = 0.02  # norm below this value indicates a "dead" head

    # ── Per-iteration loss logging ────────────────────────────────────────────
    log_loss_per_iter: bool = True  # enables loss logging every iteration
    loss_log_every: int = 10  # log frequency (in iterations)
    loss_ema_alpha: float = 0.05  # exponential smoothing factor for loss log

    # ── Head clamp ────────────────────────────────────────────────────────────────
    proxy_two_seeds: bool = True

    wd_head: float = 0.0  # weight decay for head parameters
    lr_enc: float = 2e-3  # learning rate for encoder parameters
    lr_theta: float = 5e-3  # learning rate for VQC parameters
    search_allow_bias_learning: bool = True  # allows bias to be updated during search phase
    search_recenter_logits: bool = False  # recenters logits during search phase
    search_recenter_before_thr: bool = False
    search_recenter_mu_min: float = 0.25
    search_recenter_mu_clamp: float = 0.5
    search_recenter_mu_damp: float = 0.25
    recenter_cap: int = 1024
    val_loss_cap: int = 512
    collapse_logit_gate_mult: float = 2.0
    collapse_prob_std_min_for_healthlog: float = 0.005
    saturation_fallback_thr: float = 0.95
    final_logit_clamp: float = 30.0  # absolute clamp for final logits to prevent overflow

    # ── Logging ───────────────────────────────────────────────────────────────────
    log_loss_per_epoch: bool = True  # já tínhamos, confirmar presente

    search_grid_size: int = 51
    grid_size: int = 201
    # reward weights
    alpha_auc: float = 1.0  # 0.6
    beta_sens: float = 0.8  # 0.6
    depth_penalty: float = 0.01
    cnot_penalty: float = 0.012
    rot_penalty: float = 0.002
    replay_capacity: int = 16384
    n_steps: int = 15
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_steps: int = 20000
    target_sync_steps: int = 512
    early_stop_log: int = 10
    feature_bank_decay_every: int = 50
    feature_bank_rescore_every: int = 25

    final_epochs: int = 16
    final_lr_vqc: float = 0.03
    inner_train_batches_head: int = 16

    sens_target: float = 0.85

    use_focal: bool = True
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0

    def gamma(self) -> float:
        return 0.99
