from pathlib import Path

import torch
import torch as _torch

from refactor_project.data.cross_circle import load_circle_cross_pool_flatten
from refactor_project.data.util import (
    _make_search_splits_from_train_all,
    nested_cv_eval_fixed_arch,
    split_holdout,
)
from refactor_project.environments.states_encode.state_encoder import sanitize_architecture
from refactor_project.model.arch_search import run_arch_search_end2end
from refactor_project.model.arch_train import train_final_model_end2end
from refactor_project.model.math.cost import measure_cost_from_arch
from refactor_project.model.util.util import make_cfg_for_qubits
from refactor_project.util.util import Logger, dump_run_metadata, set_seeds


def _run_one_seed_cross_circle(
    seed: int,
    nq: int,
    sc_name: str,
    sc_tag: str,
    cfg_base,
    sc_dir: "Path",
    PERCENT_SEARCH: int,
    PERCENT_EVAL: int,
) -> dict:
    _torch.set_num_threads(1)
    DEVICE = "cpu"  # forçado: PennyLane não é multi-GPU-safe em subprocessos
    set_seeds(seed)  # crítico: dentro do worker

    # Logger isolado por (seed × nq) — sem colisão de paths
    nq_dir = sc_dir / f"nq{nq}" / f"seed{seed}"
    nq_dir.mkdir(parents=True, exist_ok=True)
    nq_logger = Logger(nq_dir / "logs")

    noise_dir = nq_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    noise_logger = Logger(noise_dir / "logs")

    # ── Dados ────────────────────────────────────────────────────────────
    X_full, Y_full = load_circle_cross_pool_flatten(
        cfg_base, percent_total=int(PERCENT_EVAL), seed=int(seed)
    )
    tr_idx, ho_idx = split_holdout(X_full, Y_full, frac=float(cfg_base.holdout_frac), seed=seed)
    X_train_all, Y_train_all = X_full[tr_idx], Y_full[tr_idx]
    X_holdout, Y_holdout = X_full[ho_idx], Y_full[ho_idx]

    (XtrS, YtrS), (XvaS, YvaS) = _make_search_splits_from_train_all(
        X_train_all,
        Y_train_all,
        percent_search=int(cfg_base.percent_search),
        seed=int(seed),
        val_frac=float(cfg_base.val_frac_search),
    )

    in_dim = int(XtrS.shape[1])
    # ── Config por nq ────────────────────────────────────────────────────
    cfg_nq = make_cfg_for_qubits(cfg_base, int(nq), n_train=len(XtrS), in_dim=in_dim)
    cfg_nq.use_patch_bank = False
    cfg_nq.feature_bank_update = "none"
    cfg_nq.feature_bank_size = in_dim
    cfg_nq.feature_bank_min_size = in_dim
    cfg_nq.feature_bank_schedule = (in_dim,)
    cfg_nq.start_qubits = int(nq)
    cfg_nq.min_qubits = max(4, int(nq) - 2)
    cfg_nq.max_qubits = min(10, int(nq) + 2)
    for attr in ["feature_bank_size", "n_features", "in_dim", "d_in", "input_dim", "n_inputs"]:
        if hasattr(cfg_nq, attr):
            setattr(cfg_nq, attr, in_dim)

    dump_run_metadata(
        nq_logger,
        cfg_nq,
        extra={
            "scenario": sc_name,
            "seed": seed,
            "n_qubits": int(nq),
            "percent_search": PERCENT_SEARCH,
            "percent_eval": PERCENT_EVAL,
        },
    )
    # ── RL Search ────────────────────────────────────────────────────────
    arch_mat, best_nq, best_proxy = run_arch_search_end2end(
        XtrS, YtrS, XvaS, YvaS, cfg=cfg_nq, logger=nq_logger, seed=seed, device=DEVICE
    )
    arch_mat = sanitize_architecture(arch_mat, int(best_nq))

    # ── Nested CV ────────────────────────────────────────────────────────
    nested = nested_cv_eval_fixed_arch(
        arch_mat,
        X_train_all,
        Y_train_all,
        cfg=cfg_nq,
        n_qubits=int(best_nq),
        logger=nq_logger,
        seed=seed,
        device=DEVICE,
    )

    # ── Final holdout ─────────────────────────────────────────────────────
    auc_ho, sens_ho, thr_ho = train_final_model_end2end(
        arch_mat,
        int(best_nq),
        X_train_all,
        Y_train_all,
        X_holdout,
        Y_holdout,
        cfg_nq,
        nq_logger,
        device=DEVICE,
        noise=False,
    )


    # ── Final holdout ─────────────────────────────────────────────────────
    auc_with_noise, sens_with_noise, thr_with_noise = train_final_model_end2end(
        arch_mat,
        int(best_nq),
        X_train_all,
        Y_train_all,
        X_holdout,
        Y_holdout,
        cfg_nq,
        noise_logger,
        device=DEVICE,
        noise=True,
    )

    # ── Lê α/β do logger isolado por seed (sem colisão) ──────────────────
    _alpha_final = None
    _beta_final = None
    try:
        _pt_path = nq_dir / "logs" / "enc_params_final.pt"
        if _pt_path.exists():
            _enc = torch.load(str(_pt_path), weights_only=True, map_location="cpu")
            _alpha_final = _enc.get("alpha", None)
            _beta_final = _enc.get("beta", None)
            if _alpha_final is not None:
                _alpha_final = _alpha_final.tolist()
                _beta_final = _beta_final.tolist()
    except Exception as _e:
        nq_logger.log_to_file("enc_params", f"[WARN] could not load enc_params: {_e}")

    # ── Cost ──────────────────────────────────────────────────────────────
    X_ref = X_train_all[: max(64, int(cfg_nq.cost_measure_samples))]
    cost_obj = measure_cost_from_arch(arch_mat, int(best_nq), cfg_nq, X_ref, seed=seed)
    cost = float(cost_obj["cost"])
    perf = 0.5 * (nested["auc_mean"] + nested["sens_mean"])

    return {
        "seed": int(seed),
        "nq": int(nq),
        "best_nq": int(best_nq),
        "best_proxy_rl": (None if best_proxy is None else float(best_proxy)),
        "arch_mat": arch_mat.cpu().numpy().tolist(),
        "nested_cv": nested,
        "holdout": {"thr_star": float(thr_ho), "auc": float(auc_ho), "sens@thr*": float(sens_ho)},
        "holdout_noisy": {           # ← adiciona
            "thr_star":  float(thr_with_noise),
            "auc":       float(auc_with_noise),
            "sens@thr*": float(sens_with_noise),
            "noise_p":   float(getattr(cfg_nq, "noise_p", 0.01)),
        },
        "perf": float(perf),
        "cost": float(cost),
        "measured_cost": cost_obj,
        "budgets": {
            "ENC": int(cfg_nq.ENC_budget),
            "ROT": int(cfg_nq.ROT_budget),
            "CNOT": int(cfg_nq.CNOT_budget),
        },
        "enc_params": {
            "alpha": _alpha_final,
            "beta": _beta_final,
            "mode": str(getattr(cfg_nq, "enc_affine_mode", "per_feature")),
        },
    }
