"""
main_higgs.py
=============
Ablation S0→S4 no dataset HIGGS Boson (UCI).
Mesmo padrão estrutural do main_banknote.py refatorado:

  - Scenarios importados de refactor_project.config.ablation.higgs
  - Worker por seed em run_one_higgs._run_one_seed_higgs
  - Paralelismo via joblib.Parallel + loky
  - Flag DEBUG para rodar com ConfigDebug (episódios mínimos)
  - Output em publication_out_higgs/  (normal)
              ou debug_out_higgs/      (DEBUG=True)

Dataset:
  UCI — HIGGS Boson (28 features físicas, 11M amostras → subset 10k)
  https://archive.ics.uci.edu/ml/datasets/HIGGS

  Estrutura física conhecida:
    - Cols 0-20: low-level kinematic features (cinemática direta)
    - Cols 21-27: high-level invariant masses (derivadas)
  Hipótese testável: α deve amplificar high-level sobre low-level.

Uso:
    python main_higgs.py            # publicação (10k samples, 5 seeds)
    python main_higgs.py --debug    # debug rápido (10k samples, 1 seed)
"""

# ── stdlib ────────────────────────────────────────────────────────────
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── terceiros ─────────────────────────────────────────────────────────
import matplotlib

from refactor_project.config.ablation.higgs import HIGGS_SCENARIOS

matplotlib.use("Agg")
import numpy as np
from sklearn.calibration import Parallel, delayed
from torch import multiprocessing

# ── projeto — config ──────────────────────────────────────────────────
from refactor_project.config.config import Config
from refactor_project.config.config_debug import ConfigDebug
from refactor_project.config.util import apply_overrides, scenario_tag

# ── dataset — validação inicial ───────────────────────────────────────
from refactor_project.data.higgs import create_higgs_dataset

# ── projeto — runner isolado por seed ────────────────────────────────
from refactor_project.main.runner.run_one_higgs import (
    DEFAULT_SUBSET_SIZE,
    FEATURE_NAMES,
    _run_one_seed_higgs,
)

# ── projeto — utilitários ─────────────────────────────────────────────
from refactor_project.model.math.ci import mean_ci_bootstrap, mean_ci_t
from refactor_project.model.math.cost import pareto_front
from refactor_project.util.util import (
    Logger,
    _pearson,
    _plot_alpha_bar_higgs,
    _spearman,
    dump_run_metadata,
)

#: Índices das duas famílias de features (consistente com higgs.py)
_LOW_LEVEL_IDX = list(range(0, 21))
_HIGH_LEVEL_IDX = list(range(21, 28))


def _plot_alpha_lowlevel_vs_highlevel(
    alpha_per_seed: List[np.ndarray],
    out_path: str,
    title: str,
) -> None:
    """
    Barplot agregado low-level (idx 0-20) vs high-level (idx 21-27).

    Plot diagnóstico específico do Higgs: testa a hipótese física de que
    α aprendido amplifica features high-level (massas invariantes
    derivadas) sobre low-level (cinemática direta dos detectores).

    Se mean(α[high]) >> mean(α[low]) com baixa variância entre seeds,
    isso constitui validação não-supervisionada da hierarquia física
    conhecida do dataset HIGGS.
    """
    import matplotlib.pyplot as plt

    A = np.stack(alpha_per_seed, axis=0)  # (n_seeds, 28)
    if A.shape[1] != 28:
        # Se feature_bank reduziu, fail-soft — apenas avisa e retorna.
        print(
            f"[WARN] alpha_lowlevel_vs_highlevel: esperado d=28, "
            f"got d={A.shape[1]} — pulando plot agregado."
        )
        return

    ll_mean = A[:, _LOW_LEVEL_IDX].mean(axis=1)   # (n_seeds,)
    hl_mean = A[:, _HIGH_LEVEL_IDX].mean(axis=1)  # (n_seeds,)

    fig, ax = plt.subplots(figsize=(6, 4))
    means = [ll_mean.mean(), hl_mean.mean()]
    stds = [ll_mean.std(), hl_mean.std()]
    labels = ["low-level\n(idx 0-20)", "high-level\n(idx 21-27)"]
    colors = ["#7f8fa6", "#c0392b"]

    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label=r"$\alpha_{\mathrm{init}}=0.5$")
    ax.set_ylabel(r"mean $\alpha$ (across features)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    # Annotate ratio HL/LL (resultado-chave para o paper)
    if ll_mean.mean() > 1e-6:
        ratio = hl_mean.mean() / ll_mean.mean()
        ax.text(
            0.98, 0.98,
            f"ratio HL/LL = {ratio:.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def _run_one_scenario_higgs(
    sc: Dict[str, Any],
    cfg0,
    root_out: Path,
    SEEDS: List[int],
    QUBITS_LIST: List[int],
    PERCENT_SEARCH: int,
    PERCENT_EVAL: int,
    data_dir: str,
    subset_size: int,
    inner_n_jobs: int,
) -> Dict[str, Any]:
    """
    Executa UM cenário completo (todas as seeds × qubits) e retorna o
    dict de resumo que antes era feito inline no loop `for sc in
    scenarios:` de main_higgs(). Extraído para permitir que o PRÓPRIO
    main_higgs() rode múltiplos cenários em paralelo (outer Parallel),
    cada um com seu próprio orçamento de núcleos para o paralelismo
    interno (seed × nq).

    Roda dentro de um worker do outer Parallel — por isso `inner_n_jobs`
    é passado explicitamente em vez de recalculado via
    multiprocessing.cpu_count() aqui dentro (isso evitaria estourar o
    orçamento total quando vários cenários rodam ao mesmo tempo).
    """
    sc_name = str(sc["name"])
    sc_over = dict(sc.get("overrides", {}))
    sc_tag_ = scenario_tag(sc_name)
    sc_sem = str(sc.get("semantics_flag", "strong"))
    sc_note = str(sc.get("notes", ""))

    print(f"\n{'─' * 68}")
    print(f"  Cenário: {sc_name}  [{sc_tag_}]  (inner_n_jobs={inner_n_jobs})")
    print(f"{'─' * 68}")

    sc_dir = root_out / sc_tag_
    (sc_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = Logger(sc_dir / "logs")

    cfg_base = apply_overrides(cfg0, sc_over)

    dump_run_metadata(
        logger,
        cfg_base,
        extra={
            "scenario": sc_name,
            "scenario_tag": sc_tag_,
            "semantics_flag": sc_sem,
            "notes": sc_note,
            "overrides": sc_over,
            "dataset": "higgs_boson",
            "feature_names": FEATURE_NAMES,
            "percent_search": PERCENT_SEARCH,
            "percent_eval": PERCENT_EVAL,
            "seeds": SEEDS,
            "qubits_list": QUBITS_LIST,
            "subset_size": int(subset_size),
            "low_level_indices": _LOW_LEVEL_IDX,
            "high_level_indices": _HIGH_LEVEL_IDX,
        },
    )

    corr_csv = sc_dir / "rl_proxy_vs_final.csv"
    if not corr_csv.exists():
        with open(corr_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "scenario_tag",
                    "seed",
                    "nq_requested",
                    "best_nq",
                    "best_proxy_rl",
                    "final_auc_nested",
                ]
            )

    # ── Paralelismo interno: n_jobs limitado pelo orçamento do outer ──
    N_JOBS = min(
        len(SEEDS) * len(QUBITS_LIST),
        int(inner_n_jobs),
    )
    print(
        f"[parallel][{sc_tag_}] {len(SEEDS)} seeds × {len(QUBITS_LIST)} "
        f"qubits → n_jobs={N_JOBS} (cap={inner_n_jobs})"
    )

    raw_results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(_run_one_seed_higgs)(
            seed=seed,
            nq=nq,
            sc_name=sc_name,
            sc_tag=sc_tag_,
            cfg_base=cfg_base,
            sc_dir=sc_dir,
            PERCENT_SEARCH=PERCENT_SEARCH,
            PERCENT_EVAL=PERCENT_EVAL,
            data_dir=data_dir,
            subset_size=int(subset_size),
        )
        for seed in SEEDS
        for nq in QUBITS_LIST
    )

    # ── Agrega resultados dos workers (idêntico ao original) ──────────
    by_seed: Dict[int, Dict[str, Any]] = defaultdict(dict)
    all_pareto: List[Dict[str, Any]] = []

    for res in raw_results:
        s, nq_res = res["seed"], res["nq"]
        by_seed[s][str(nq_res)] = res
        all_pareto.append(
            {
                "seed": s,
                "n_qubits": nq_res,
                "cost": res["cost"],
                "perf": res["perf"],
            }
        )

    scenario_runs: List[Dict[str, Any]] = []

    for seed in SEEDS:
        rlqcv = by_seed[seed]
        pareto_points = [p for p in all_pareto if p["seed"] == seed]

        try:
            with open(corr_csv, "a", newline="") as f:
                w = csv.writer(f)
                for nq_str, obj in rlqcv.items():
                    w.writerow(
                        [
                            str(sc_tag_),
                            int(seed),
                            int(nq_str),
                            obj["best_nq"],
                            (
                                ""
                                if obj["best_proxy_rl"] is None
                                else float(obj["best_proxy_rl"])
                            ),
                            float(obj["nested_cv"]["auc_mean"]),
                        ]
                    )
        except Exception as e:
            logger.log_to_file("corr", f"[WARN] corr csv: {e}")

        best: Optional[Dict[str, Any]] = max(
            rlqcv.values(),
            key=lambda x: x["perf"],
            default=None,
        )
        if best is not None:
            best = {"seed": int(seed), "n_qubits": best["nq"], **best}

        scenario_runs.append(
            {
                "scenario": sc_name,
                "seed": int(seed),
                "percent_search": int(PERCENT_SEARCH),
                "percent_eval": int(PERCENT_EVAL),
                "holdout_frac": float(cfg_base.holdout_frac),
                "rlqcv": {
                    k: {
                        "best_n_qubits": v["best_nq"],
                        "best_proxy_rl": v["best_proxy_rl"],
                        "arch_mat": v["arch_mat"],
                        "nested_cv": v["nested_cv"],
                        "holdout": v["holdout"],
                        "perf": v["perf"],
                        "cost": v["cost"],
                        "budgets": v["budgets"],
                        "measured_cost": v["measured_cost"],
                        "enc_params": v["enc_params"],
                    }
                    for k, v in rlqcv.items()
                },
                "best_rlqcv": best,
                "pareto": pareto_points,
            }
        )

    # ── Barplots α por cenário (idêntico ao original) ─────────────────
    try:
        alpha_per_seed: List[Optional[np.ndarray]] = []
        for r in scenario_runs:
            best_r = r.get("best_rlqcv", {})
            enc = (best_r or {}).get("enc_params", {})
            a = enc.get("alpha", None)
            if a is not None:
                alpha_per_seed.append(np.asarray(a, dtype=np.float32))

        if alpha_per_seed:
            bar_path = sc_dir / f"alpha_bar_{sc_tag_}.png"
            try:
                _plot_alpha_bar_higgs(
                    alpha_per_seed=alpha_per_seed,
                    out_path=str(bar_path),
                    title=(
                        f"α convergido — {sc_name}\n"
                        f"(média ± std  ·  {len(alpha_per_seed)} seeds  ·  d=28)"
                    ),
                )
                print(f"[OK] Barplot α por feature: {bar_path}")
            except Exception as e:
                print(f"[WARN] Barplot α por feature falhou: {e}")

            hl_path = sc_dir / f"alpha_lowlevel_vs_highlevel_{sc_tag_}.png"
            _plot_alpha_lowlevel_vs_highlevel(
                alpha_per_seed=alpha_per_seed,
                out_path=str(hl_path),
                title=(
                    f"α: low-level vs high-level — {sc_name}\n"
                    f"({len(alpha_per_seed)} seeds  ·  hipótese: HL > LL)"
                ),
            )
            print(f"[OK] Barplot α LL-vs-HL: {hl_path}")
        else:
            print(f"[WARN] Sem α disponíveis para barplot: {sc_tag_}")
    except Exception as e:
        print(f"[WARN] Barplots α falharam para {sc_tag_}: {e}")

    # ── Métricas agregadas (idêntico ao original) ──────────────────────
    best_per_seed = [r["best_rlqcv"] for r in scenario_runs]
    perf_list = [b["perf"] for b in best_per_seed if b is not None]

    if str(cfg_base.ci_method).lower() == "bootstrap":
        perf_mean, perf_lo, perf_hi = mean_ci_bootstrap(
            perf_list,
            B=int(cfg_base.bootstrap_B),
            alpha=float(cfg_base.ci_alpha),
            seed=0,
        )
    else:
        perf_mean, perf_lo, perf_hi = mean_ci_t(
            perf_list,
            alpha=float(cfg_base.ci_alpha),
        )

    pf = pareto_front([p for r in scenario_runs for p in r.get("pareto", [])])

    hl_ll_stats: Optional[Dict[str, Any]] = None
    try:
        alphas_valid = [
            np.asarray(b["enc_params"]["alpha"], dtype=np.float32)
            for b in best_per_seed
            if b is not None
            and b.get("enc_params", {}).get("alpha") is not None
        ]
        alphas_valid = [a for a in alphas_valid if a.size == 28]
        if alphas_valid:
            A = np.stack(alphas_valid, axis=0)
            ll = A[:, _LOW_LEVEL_IDX].mean(axis=1)
            hl = A[:, _HIGH_LEVEL_IDX].mean(axis=1)
            ratio_per_seed = hl / np.clip(ll, 1e-6, None)
            hl_ll_stats = {
                "n_seeds": int(len(alphas_valid)),
                "low_level_mean_alpha": {
                    "mean": float(ll.mean()),
                    "std": float(ll.std()),
                },
                "high_level_mean_alpha": {
                    "mean": float(hl.mean()),
                    "std": float(hl.std()),
                },
                "ratio_HL_over_LL": {
                    "mean": float(ratio_per_seed.mean()),
                    "std": float(ratio_per_seed.std()),
                    "per_seed": [float(x) for x in ratio_per_seed.tolist()],
                },
            }
            logger.log_to_file(
                "alpha_physics",
                f"[HL/LL] scenario={sc_tag_} "
                f"LL={ll.mean():.3f}±{ll.std():.3f} "
                f"HL={hl.mean():.3f}±{hl.std():.3f} "
                f"ratio={ratio_per_seed.mean():.2f}±{ratio_per_seed.std():.2f}",
            )
    except Exception as e:
        logger.log_to_file("alpha_physics", f"[WARN] HL/LL stats: {e}")

    scenario_result: Dict[str, Any] = {
        "scenario": {
            "name": sc_name,
            "tag": sc_tag_,
            "overrides": sc_over,
            "semantics_flag": sc_sem,
            "notes": sc_note,
            "dataset": "higgs_boson",
            "feature_names": FEATURE_NAMES,
            "low_level_indices": _LOW_LEVEL_IDX,
            "high_level_indices": _HIGH_LEVEL_IDX,
        },
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": SEEDS,
            "percent_search": int(PERCENT_SEARCH),
            "percent_eval": int(PERCENT_EVAL),
            "holdout_frac": float(cfg_base.holdout_frac),
            "qubits_list": QUBITS_LIST,
            "subset_size": int(subset_size),
            "ci_method": str(cfg_base.ci_method),
            "ci_alpha": float(cfg_base.ci_alpha),
            "bootstrap_B": int(cfg_base.bootstrap_B),
        },
        "best_rlqcv_per_seed": best_per_seed,
        "rlqcv_perf_ci95": {
            "mean": float(perf_mean),
            "ci95": [float(perf_lo), float(perf_hi)],
            "per_seed": [float(x) for x in perf_list],
        },
        "alpha_physics": hl_ll_stats,
        "baselines": {},
        "pareto_front": pf,
        "runs": scenario_runs,
    }

    try:
        rows = []
        with open(corr_csv, "r") as f:
            for row in csv.DictReader(f):
                try:
                    bp = float(row["best_proxy_rl"])
                    fa = float(row["final_auc_nested"])
                    if np.isfinite(bp) and np.isfinite(fa):
                        rows.append((bp, fa))
                except Exception:
                    pass
        if len(rows) >= 2:
            xs = [a for a, _ in rows]
            ys = [b for _, b in rows]
            pr = _pearson(xs, ys)
            sr = _spearman(xs, ys)
            logger.log_to_file(
                "corr",
                f"[CORR] scenario={sc_tag_} n={len(rows)} "
                f"pearson={pr:.3f} spearman={sr:.3f} "
                f"(rl_proxy vs final_auc)",
            )
            scenario_result["corr_rlproxy_vs_finalauc"] = {
                "n": int(len(rows)),
                "pearson": float(pr),
                "spearman": float(sr),
            }
    except Exception as e:
        logger.log_to_file("corr", f"[WARN] correlação falhou: {e}")

    out_path = sc_dir / f"results_{sc_tag_}.json"
    out_path.write_text(
        json.dumps(scenario_result, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Saved scenario: {out_path}")

    return {
        "scenario": {
            "name": sc_name,
            "tag": sc_tag_,
            "overrides": sc_over,
            "semantics_flag": sc_sem,
            "notes": sc_note,
        },
        "summary": {
            "rlqcv_perf_ci95": scenario_result["rlqcv_perf_ci95"],
            "alpha_physics": scenario_result["alpha_physics"],
            "baselines": scenario_result["baselines"],
            "pareto_front": scenario_result["pareto_front"],
        },
        "results_file": str(out_path),
    }


def main_higgs(
    DEBUG: bool = False,
    data_dir: str = "data",
    subset_size: int = DEFAULT_SUBSET_SIZE,
    scenario_filter: Optional[List[str]] = None,
    total_core_budget: Optional[int] = None,
    outer_n_jobs: int = 2,
) -> None:
    """
    Ablation S0→S4 no dataset HIGGS Boson — com paralelismo em DOIS
    NÍVEIS: cenários entre si (outer) e (seed, nq) dentro de cada
    cenário (inner). Isso substitui o loop sequencial `for sc in
    scenarios:` por um Parallel externo, evitando deixar núcleos
    ociosos quando o cluster tem mais cores do que um único cenário
    usa sozinho (ex.: S0-S4 usam 10 cores cada; com 28 disponíveis,
    rodar 1 cenário por vez desperdiça 18 cores o tempo todo).

    Orçamento de núcleos (a parte que "exige cuidado" para não
    estourar o total):

        TOTAL_CORE_BUDGET = total_core_budget or cpu_count()
        INNER_N_JOBS_CAP  = max(1, TOTAL_CORE_BUDGET // outer_n_jobs)

        outer_n_jobs cenários rodam ao mesmo tempo, cada um limitado a
        INNER_N_JOBS_CAP núcleos para seu paralelismo interno de
        (seed, nq). Isso garante:

            outer_n_jobs × INNER_N_JOBS_CAP <= TOTAL_CORE_BUDGET

    Parâmetros novos
    ----------------
    scenario_filter   : se fornecido, roda só os cenários cujo "name"
                         está nessa lista (permite rodar em rounds
                         manuais, ex.: 2 cenários grandes + 1 pequeno
                         por vez, sem exceder o orçamento de núcleos).
    total_core_budget : teto de núcleos a usar no total. Default:
                         multiprocessing.cpu_count() (todos os núcleos
                         disponíveis — CUIDADO em máquina compartilhada,
                         prefira passar um valor explícito menor que o
                         total se outros processos também usam a CPU).
    outer_n_jobs      : quantos cenários rodam simultaneamente.
                         Default 2 — ajuste conforme RAM disponível
                         (mais cenários simultâneos = mais processos
                         com replay buffer + dados na memória ao
                         mesmo tempo).

    Exemplos
    --------
        # Tudo, orçamento total de 28 núcleos, 2 cenários por vez
        # (S0-S4: 10 cores cada -> 2×10=20/28 em uso; cenários menores
        # de L_max com 3 seeds usam só 6 cores cada)
        main_higgs(total_core_budget=28, outer_n_jobs=2)

        # Rodar só os 2 cenários novos de L_max, 26 núcleos, 2 por vez
        main_higgs(
            scenario_filter=["higgs_s4_lmax_factor_1p5", "higgs_s4_lmax_factor_2p0"],
            total_core_budget=26,
            outer_n_jobs=2,
        )
    """
    print(f"\n{'=' * 68}")
    print("  ABLATION — HIGGS Boson (UCI, 28 features físicas)")
    mode_str = "DEBUG" if DEBUG else "PUBLICAÇÃO"
    print(f"  Modo: {mode_str}")
    print(f"  Subset size: {subset_size}")
    print(f"{'=' * 68}\n")

    if DEBUG:
        cfg0 = ConfigDebug()
        root_out = Path("debug_out_higgs")
        SEEDS = [0]
    else:
        cfg0 = Config()
        root_out = Path("publication_out_higgs")
        SEEDS = [0, 1, 2, 3, 4]

    root_out.mkdir(parents=True, exist_ok=True)

    QUBITS_LIST = [4, 6]
    PERCENT_SEARCH = int(cfg0.percent_search)
    PERCENT_EVAL = int(cfg0.percent_eval)

    # ── valida dataset antes de iniciar (serial, evita race condition) ──
    print("[INFO] Verificando dataset HIGGS (pré-download serial)...")
    X_chk, Y_chk = create_higgs_dataset(
        seed=0,
        data_dir=data_dir,
        subset_size=subset_size,
    )
    n_pos = int((Y_chk.flatten() > 0.5).sum())
    n_neg = int(len(Y_chk) - n_pos)
    print(
        f"[INFO] OK: {len(X_chk)} amostras | "
        f"pos={n_pos} neg={n_neg} | "
        f"balance={n_pos / len(Y_chk):.2%} | "
        f"features={X_chk.shape[1]}"
    )
    if X_chk.shape[1] != 28:
        raise RuntimeError(
            f"HIGGS: esperado d=28, got d={X_chk.shape[1]}. "
            f"Verifique higgs.py."
        )

    scenarios = HIGGS_SCENARIOS
    if scenario_filter:
        scenarios = [sc for sc in scenarios if sc["name"] in scenario_filter]
        if not scenarios:
            raise ValueError(
                f"scenario_filter={scenario_filter} não bateu com nenhum "
                f"nome em HIGGS_SCENARIOS."
            )

    # ── orçamento de núcleos: outer (cenários) × inner (seed×nq) ────────
    TOTAL_CORE_BUDGET = int(total_core_budget) if total_core_budget else int(multiprocessing.cpu_count())
    OUTER_N_JOBS = max(1, min(int(outer_n_jobs), len(scenarios)))
    INNER_N_JOBS_CAP = max(1, TOTAL_CORE_BUDGET // OUTER_N_JOBS)

    print(
        f"[budget] total_cores={TOTAL_CORE_BUDGET}  outer_n_jobs={OUTER_N_JOBS}  "
        f"inner_n_jobs_cap={INNER_N_JOBS_CAP}  "
        f"(uso máximo teórico: {OUTER_N_JOBS * INNER_N_JOBS_CAP}/{TOTAL_CORE_BUDGET} cores)"
    )
    print(f"[scenarios] rodando {len(scenarios)} cenário(s): {[sc['name'] for sc in scenarios]}")

    # ── outer Parallel: cenários simultâneos ─────────────────────────────
    all_scenarios_results: List[Dict[str, Any]] = []
    for sc in scenarios:
        result = _run_one_scenario_higgs(
            sc=sc,
            cfg0=cfg0,
            root_out=root_out,
            SEEDS=SEEDS,
            QUBITS_LIST=QUBITS_LIST,
            PERCENT_SEARCH=PERCENT_SEARCH,
            PERCENT_EVAL=PERCENT_EVAL,
            data_dir=data_dir,
            subset_size=int(subset_size),
            inner_n_jobs=INNER_N_JOBS_CAP,
        )
        all_scenarios_results.append(result)

    # ── Índice agregado ───────────────────────────────────────────────
    agg = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "higgs_boson",
            "subset_size": int(subset_size),
            "n_scenarios": len(all_scenarios_results),
            "core_budget": {
                "total": TOTAL_CORE_BUDGET,
                "outer_n_jobs": OUTER_N_JOBS,
                "inner_n_jobs_cap": INNER_N_JOBS_CAP,
            },
        },
        "scenarios": all_scenarios_results,
    }
    agg_path = root_out / "ALL_SCENARIOS_INDEX.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[OK] Saved aggregate index: {agg_path}")