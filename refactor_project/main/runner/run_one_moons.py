"""
refactor_project/main/runner/run_one_make_moons.py
==================================================
Worker isolado por (seed × nq) para o dataset Make Moons.
Chamado em paralelo por main_make_moons.py via joblib.Parallel.

Não importa nada de main_make_moons.py — auto-contido por design,
para ser serializável pelo backend loky.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from refactor_project.data.moons import load_make_moons_pool
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


def _run_one_seed_make_moons(
    seed: int,
    nq: int,
    sc_name: str,
    sc_tag: str,
    cfg_base,
    sc_dir: Path,
    PERCENT_SEARCH: int,
    PERCENT_EVAL: int,
) -> Dict[str, Any]:
    """
    Executa um run completo (search → nested CV → holdout) para
    uma combinação (seed, nq) do dataset Make Moons.

    Retorna um dict com todos os resultados — sem side-effects
    além de escrever logs e enc_params_final.pt no diretório
    nq_dir isolado por (nq, seed).

    Parâmetros
    ----------
    seed          : semente aleatória para este worker
    nq            : número de qubits solicitado
    sc_name       : nome do cenário (para logging)
    sc_tag        : tag curto do cenário (para nomes de arquivo)
    cfg_base      : Config com overrides do cenário já aplicados
    sc_dir        : diretório raiz do cenário
    PERCENT_SEARCH: percentual de dados para o stage SEARCH
    PERCENT_EVAL  : percentual de dados para o stage FINAL
    """
    # DEVICE forçado para CPU: PennyLane não é multi-GPU-safe em subprocessos
    DEVICE = "cpu"
    set_seeds(seed)  # crítico: deve ser a primeira chamada dentro do worker

    # Logger isolado por (nq × seed) — sem colisão de paths entre workers
    nq_dir = sc_dir / f"nq{nq}" / f"seed{seed}"
    nq_dir.mkdir(parents=True, exist_ok=True)
    nq_logger = Logger(nq_dir / "logs")

    # ── Dados completos (PERCENT_EVAL) → holdout ──────────────────────────
    X_full, Y_full = load_make_moons_pool(
        cfg_base,
        percent_total=int(PERCENT_EVAL),
        seed=int(seed),
    )
    tr_idx, ho_idx = split_holdout(
        X_full,
        Y_full,
        frac=float(cfg_base.holdout_frac),
        seed=seed,
    )
    X_train_all, Y_train_all = X_full[tr_idx], Y_full[tr_idx]
    X_holdout, Y_holdout = X_full[ho_idx], Y_full[ho_idx]

    # ── Dados reduzidos (PERCENT_SEARCH) — sem tocar no holdout ──────────
    (XtrS, YtrS), (XvaS, YvaS) = _make_search_splits_from_train_all(
        X_train_all,
        Y_train_all,
        percent_search=int(PERCENT_SEARCH),
        seed=int(seed),
        val_frac=float(getattr(cfg_base, "val_frac_search", 0.40)),
    )
    in_dim = int(XtrS.shape[1])  # 2 para Make Moons

    # ── Config por nq ─────────────────────────────────────────────────────
    cfg_nq = make_cfg_for_qubits(cfg_base, int(nq))

    # Make Moons: 2 features — desativa feature bank dinâmico
    cfg_nq.use_patch_bank = False
    cfg_nq.feature_bank_update = "none"
    cfg_nq.feature_bank_size = in_dim
    cfg_nq.feature_bank_min_size = in_dim
    cfg_nq.feature_bank_schedule = (in_dim,)
    cfg_nq.feature_bank_decay_enabled = False

    cfg_nq.start_qubits = int(nq)
    cfg_nq.min_qubits = max(4, int(nq) - 2)
    cfg_nq.max_qubits = min(10, int(nq) + 2)

    for attr in [
        "feature_bank_size",
        "n_features",
        "in_dim",
        "d_in",
        "input_dim",
        "n_inputs",
    ]:
        if hasattr(cfg_nq, attr):
            setattr(cfg_nq, attr, in_dim)

    dump_run_metadata(
        nq_logger,
        cfg_nq,
        extra={
            "scenario": sc_name,
            "seed": seed,
            "n_qubits": int(nq),
            "dataset": "make_moons",
            "in_dim": in_dim,
            "percent_search": PERCENT_SEARCH,
            "percent_eval": PERCENT_EVAL,
        },
    )

    # ── RL Search ─────────────────────────────────────────────────────────
    arch_mat, best_nq, best_proxy = run_arch_search_end2end(
        XtrS,
        YtrS,
        XvaS,
        YvaS,
        cfg=cfg_nq,
        logger=nq_logger,
        seed=seed,
        device=DEVICE,
    )
    arch_mat = sanitize_architecture(arch_mat, int(best_nq))

    # ── Nested CV em TRAIN_ALL ────────────────────────────────────────────
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
    )

    # ── Lê α/β do logger isolado por seed (sem colisão entre workers) ─────
    _alpha_final: Optional[List[float]] = None
    _beta_final: Optional[List[float]] = None
    try:
        _pt_path = nq_dir / "logs" / "enc_params_final.pt"
        if _pt_path.exists():
            _enc = torch.load(
                str(_pt_path),
                map_location="cpu",
                weights_only=False,  # suporta numpy arrays no .pt
            )
            _a = _enc.get("alpha", None)
            _b = _enc.get("beta", None)
            if _a is not None:
                _alpha_final = np.asarray(_a).flatten().tolist()
                _beta_final = np.asarray(_b).flatten().tolist()
    except Exception as _e:
        nq_logger.log_to_file(
            "enc_params",
            f"[WARN] could not load enc_params: {_e}",
        )

    # ── Custo (tape-based) ────────────────────────────────────────────────
    X_ref = X_train_all[: max(64, int(cfg_nq.cost_measure_samples))]
    cost_obj = measure_cost_from_arch(
        arch_mat,
        int(best_nq),
        cfg_nq,
        X_ref,
        seed=seed,
    )
    cost = float(cost_obj["cost"])
    perf = 0.5 * (nested["auc_mean"] + nested["sens_mean"])

    return {
        "seed": int(seed),
        "nq": int(nq),
        "best_nq": int(best_nq),
        "best_proxy_rl": (None if best_proxy is None else float(best_proxy)),
        "arch_mat": arch_mat.cpu().numpy().tolist(),
        "nested_cv": nested,
        "holdout": {
            "thr_star": float(thr_ho),
            "auc": float(auc_ho),
            "sens@thr*": float(sens_ho),
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
