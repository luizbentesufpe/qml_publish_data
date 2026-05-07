from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from refactor_project.data.higgs import (
    FEATURE_NAMES as _HIGGS_FEATURE_NAMES,
)
from refactor_project.data.higgs import (
    load_higgs_pool,
)
from refactor_project.data.task_context import compute_task_context
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

#: Nomes semânticos das 28 features físicas — importados de higgs.py
#: (21 low-level kinematic + 7 high-level invariant masses)
#: Single source of truth para evitar drift entre worker e loader.
FEATURE_NAMES: List[str] = list(_HIGGS_FEATURE_NAMES)

#: Tamanho default do subset (compatível com nq <= 6 e custo de busca RL).
#: Sobrescreve via parâmetro `subset_size` do worker.
DEFAULT_SUBSET_SIZE: int = 10_000


def _run_one_seed_higgs(
    seed: int,
    nq: int,
    sc_name: str,
    sc_tag: str,
    cfg_base,
    sc_dir: Path,
    PERCENT_SEARCH: int,
    PERCENT_EVAL: int,
    data_dir: str = "data",
    subset_size: int = DEFAULT_SUBSET_SIZE,
) -> Dict[str, Any]:
    """
    Executa um run completo (search → nested CV → holdout) para
    uma combinação (seed, nq) do dataset HIGGS Boson.

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
    data_dir      : diretório onde HIGGS.csv.gz está (ou será baixado)
    subset_size   : tamanho do subset balanceado do HIGGS (default 10k).
                    Dataset original tem 11M amostras — intratável para QAS.

    Notas
    -----
    HIGGS é um regime estruturalmente distinto dos demais datasets:
      - d=28 (vs 4-9 dos outros) → state space do RL é maior
      - Features com escalas heterogêneas (pT em GeV, eta adimensional,
        b_tag em [0,1]) → encoding paramétrico tem alavancagem máxima
      - Fronteira genuinamente não-linear (LR atinge ~0.68 AUC)
      - 21 low-level + 7 high-level features → α deve recuperar
        hierarquia física conhecida (high-level mais discriminativas)
    """
    # if torch.cuda.is_available():
    #     n_gpus = torch.cuda.device_count()
    #     gpu_id = (seed * len(str(nq)) + nq) % n_gpus
    #     DEVICE = f"cuda:{gpu_id}"
    #     torch.cuda.set_device(gpu_id)
    # else:
    #     DEVICE = "cpu"
    DEVICE = "cpu"  # força CPU para evitar OOMs e interferência entre workers
    set_seeds(seed)  # crítico: deve ser a primeira chamada dentro do worker

    # Logger isolado por (nq × seed) — sem colisão de paths entre workers
    nq_dir = sc_dir / f"nq{nq}" / f"seed{seed}"
    nq_dir.mkdir(parents=True, exist_ok=True)
    nq_logger = Logger(nq_dir / "logs")

    noise_dir = nq_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    noise_logger = Logger(noise_dir / "logs")

    # ── Dados completos (PERCENT_EVAL) → holdout ──────────────────────────
    X_full, Y_full = load_higgs_pool(
        cfg_base,
        percent_total=int(PERCENT_EVAL),
        seed=int(seed),
        data_dir=data_dir,
        subset_size=int(subset_size),
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
    in_dim = int(XtrS.shape[1])  # 28 para Higgs

    # Sanity check: protege contra mismatch entre dataset e FEATURE_NAMES
    if in_dim != len(FEATURE_NAMES):
        raise ValueError(
            f"Higgs in_dim mismatch: got {in_dim}, expected "
            f"{len(FEATURE_NAMES)} (FEATURE_NAMES). "
            f"Verifique higgs.py / load_higgs_pool."
        )
    task_ctx = compute_task_context(XtrS, YtrS.reshape(-1).astype(int))
    # ── Config por nq ─────────────────────────────────────────────────────
    cfg_nq = make_cfg_for_qubits(cfg_base, int(nq), n_train=len(XtrS))
    cfg_nq.task_context = [
        task_ctx["class_entropy"],
        task_ctx["separabilidade"],
        task_ctx["knn_auc"],
        task_ctx["pca_fraction"],
        task_ctx["pca_90"],
    ]

    # Higgs: d=28 — feature bank dinâmico DESATIVADO por consistência com
    # os demais datasets do paper (CC/MM/BN/BCW). O agente seleciona
    # features via ENC(q,a,i), não via redução do bank.
    # NOTA: feature_bank_size mantido em 28 (= in_dim) para que o action
    # space ENC(q,a,i) tenha i ∈ [0, 27] cobrindo todas as features.
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
            "dataset": "higgs_boson",
            "in_dim": in_dim,
            "feature_names": FEATURE_NAMES,
            "percent_search": PERCENT_SEARCH,
            "percent_eval": PERCENT_EVAL,
            "subset_size": int(subset_size),
            # Metadados específicos do Higgs — facilita análise post-hoc
            # da hipótese "α amplifica high-level sobre low-level"
            "low_level_indices": list(range(0, 21)),
            "high_level_indices": list(range(21, 28)),
            "task_context": task_ctx, 
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
        noise=False,
    )

    # ── Final holdout (noisy) ─────────────────────────────────────────────
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
        "holdout_noisy": {
            "thr_star": float(thr_with_noise),
            "auc": float(auc_with_noise),
            "sens@thr*": float(sens_with_noise),
            "noise_p": float(getattr(cfg_nq, "noise_p", 0.01)),
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
            "feature_names": FEATURE_NAMES,
            # Hint para análise post-hoc da hipótese física
            "low_level_indices": list(range(0, 21)),
            "high_level_indices": list(range(21, 28)),
        },
        "task_context": task_ctx, 
    }