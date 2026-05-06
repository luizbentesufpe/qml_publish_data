"""
main_make_moons.py  (refatorado)
================================
Ablation S0→S4 no dataset Make Moons (sklearn).
Mesmo padrão estrutural do main_cross_circle.py refatorado:

  - Scenarios importados de refactor_project.config.build_ablations
  - Worker por seed em run_one_make_moons._run_one_seed_make_moons
  - Paralelismo via joblib.Parallel + loky
  - Flag DEBUG para rodar com ConfigDebug (episódios mínimos)
  - Output em publication_out_make_moons/  (normal)
              ou debug_out_make_moons/      (DEBUG=True)

Dataset:
  sklearn.datasets.make_moons — 2 features, fronteira não-linear
  400 amostras, noise=0.15, normalizado para [0,1]²

Uso:
    python main_make_moons.py            # publicação
    python main_make_moons.py --debug    # debug rápido
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

from refactor_project.config.ablation.make_moon import MAKE_MOONS_SCENARIOS
from refactor_project.main.runner.run_one_moons import _run_one_seed_make_moons

matplotlib.use("Agg")
import numpy as np
from sklearn.calibration import Parallel, delayed
from torch import multiprocessing

# ── projeto — config ──────────────────────────────────────────────────
from refactor_project.config.config import Config
from refactor_project.config.config_debug import ConfigDebug
from refactor_project.config.util import apply_overrides, scenario_tag

# ── projeto — utilitários ─────────────────────────────────────────────
from refactor_project.model.math.ci import mean_ci_bootstrap, mean_ci_t
from refactor_project.model.math.cost import pareto_front
from refactor_project.util.util import (
    Logger,
    _pearson,
    _plot_alpha_bar_moons,
    _spearman,
    dump_run_metadata,
)

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def main_make_moons(DEBUG: bool = False) -> None:
    """
    Ablation S0→S4 no dataset Make Moons.

    Protocolo idêntico ao main_cross_circle():
      - 5 seeds × 5 cenários  (1 seed em DEBUG)
      - 4 qubits fixos
      - 80/20 holdout estratificado
      - 400 episódios por run  (1 em DEBUG)
      - Paralelismo via joblib.Parallel (n_jobs = n_seeds × n_qubits)
    """
    print(f"\n{'=' * 68}")
    print("  ABLATION — Make Moons (Two Moons, 2 features)")
    mode_str = "DEBUG" if DEBUG else "PUBLICAÇÃO"
    print(f"  Modo: {mode_str}")






    # ── config base ───────────────────────────────────────────────────
    if DEBUG:
        cfg0 = ConfigDebug()
        root_out = Path("debug_out_make_moons")
        SEEDS = [0]
    else:
        cfg0 = Config()
        root_out = Path("publication_out_make_moons")
        SEEDS = [0, 1, 2, 3, 4]

    root_out.mkdir(parents=True, exist_ok=True)

    QUBITS_LIST = [4, 6]
    PERCENT_SEARCH = int(cfg0.percent_search)
    PERCENT_EVAL = int(cfg0.percent_eval)

    scenarios = MAKE_MOONS_SCENARIOS
    all_scenarios_results: List[Dict[str, Any]] = []

    # ── loop de cenários ─────────────────────────────────────────────
    for sc in scenarios:
        sc_name = str(sc["name"])
        sc_over = dict(sc.get("overrides", {}))
        sc_tag_ = scenario_tag(sc_name)
        sc_sem = str(sc.get("semantics_flag", "strong"))
        sc_note = str(sc.get("notes", ""))

        print(f"\n{'─' * 68}")
        print(f"  Cenário: {sc_name}  [{sc_tag_}]")
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
                "dataset": "make_moons",
                "percent_search": PERCENT_SEARCH,
                "percent_eval": PERCENT_EVAL,
                "seeds": SEEDS,
                "qubits_list": QUBITS_LIST,
            },
        )

        # CSV correlação proxy vs AUC final
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

        # ── Paralelismo: n_jobs = seeds × qubits ─────────────────────
        N_JOBS = min(
            len(SEEDS) * len(QUBITS_LIST),
            multiprocessing.cpu_count(),
        )
        print(f"[parallel] {len(SEEDS)} seeds × {len(QUBITS_LIST)} qubits → n_jobs={N_JOBS}")

        raw_results = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=10,
        )(
            delayed(_run_one_seed_make_moons)(
                seed=seed,
                nq=nq,
                sc_name=sc_name,
                sc_tag=sc_tag_,
                cfg_base=cfg_base,
                sc_dir=sc_dir,
                PERCENT_SEARCH=PERCENT_SEARCH,
                PERCENT_EVAL=PERCENT_EVAL,
            )
            for seed in SEEDS
            for nq in QUBITS_LIST
        )

        # ── Agrega resultados dos workers ─────────────────────────────
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

            # log correlação proxy vs AUC para este seed
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

            # melhor nq por perf
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

        # ── Barplot α por cenário ─────────────────────────────────────
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
                _plot_alpha_bar_moons(
                    alpha_per_seed=alpha_per_seed,
                    out_path=str(bar_path),
                    title=(
                        f"α convergido — {sc_name}\n(média ± std  ·  {len(alpha_per_seed)} seeds)"
                    ),
                )
                print(f"[OK] Barplot α: {bar_path}")
            else:
                print(f"[WARN] Sem α disponíveis para barplot: {sc_tag_}")
        except Exception as e:
            print(f"[WARN] Barplot α falhou para {sc_tag_}: {e}")

        # ── Métricas agregadas ────────────────────────────────────────
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

        scenario_result: Dict[str, Any] = {
            "scenario": {
                "name": sc_name,
                "tag": sc_tag_,
                "overrides": sc_over,
                "semantics_flag": sc_sem,
                "notes": sc_note,
                "dataset": "make_moons",
            },
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seeds": SEEDS,
                "percent_search": int(PERCENT_SEARCH),
                "percent_eval": int(PERCENT_EVAL),
                "holdout_frac": float(cfg_base.holdout_frac),
                "qubits_list": QUBITS_LIST,
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
            "baselines": {},
            "pareto_front": pf,
            "runs": scenario_runs,
        }

        # ── Correlação proxy vs AUC final ─────────────────────────────
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

        # ── Salva JSON do cenário ─────────────────────────────────────
        out_path = sc_dir / f"results_{sc_tag_}.json"
        out_path.write_text(
            json.dumps(scenario_result, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] Saved scenario: {out_path}")

        all_scenarios_results.append(
            {
                "scenario": {
                    "name": sc_name,
                    "tag": sc_tag_,
                    "overrides": sc_over,
                    "semantics_flag": sc_sem,
                    "notes": sc_note,
                },
                "summary": {
                    "rlqcv_perf_ci95": scenario_result["rlqcv_perf_ci95"],
                    "baselines": scenario_result["baselines"],
                    "pareto_front": scenario_result["pareto_front"],
                },
                "results_file": str(out_path),
            }
        )
    
    #Ending scenarios loop
    # ── Índice agregado ───────────────────────────────────────────────
    agg = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "make_moons",
            "n_scenarios": len(all_scenarios_results),
        },
        "scenarios": all_scenarios_results,
    }
    agg_path = root_out / "ALL_SCENARIOS_INDEX.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[OK] Saved aggregate index: {agg_path}")
