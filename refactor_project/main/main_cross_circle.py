from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path

from sklearn.calibration import Parallel, delayed
from torch import multiprocessing

from refactor_project.config.ablation.cross_circle import CROSS_CIRCLE_SCENARIOS
from refactor_project.config.config import Config
from refactor_project.config.config_debug import ConfigDebug
from refactor_project.config.util import apply_overrides, scenario_tag
from refactor_project.main.runner.run_one_cross_circle import _run_one_seed_cross_circle
from refactor_project.model.math.ci import mean_ci_bootstrap, mean_ci_t
from refactor_project.model.math.cost import pareto_front
from refactor_project.util.save_image_article import _plot_alpha_heatmap_3x3
from refactor_project.util.util import Logger, _pearson, _spearman, dump_run_metadata


def main_cross_circle(DEBUG=False):
    print("[HB] entered main_cross_circle()")
    # Base config
    if DEBUG:
        cfg0 = ConfigDebug()
        root_out = Path("debug_out_cross_circle")
        root_out.mkdir(parents=True, exist_ok=True)
    else:
        cfg0 = Config()
        root_out = Path("publication_out_cross_circle")
        root_out.mkdir(parents=True, exist_ok=True)

    SEEDS = [0, 1, 2, 3, 4]
    # SEEDS = [0]
    QUBITS_LIST = [4]
    nq = QUBITS_LIST[0]

    PERCENT_SEARCH = int(cfg0.percent_search)
    PERCENT_EVAL = int(cfg0.percent_eval)

    scenarios = CROSS_CIRCLE_SCENARIOS

    all_scenarios_results = []

    for sc in scenarios:
        sc_name = str(sc["name"])
        sc_over = dict(sc.get("overrides", {}))
        sc_tag = scenario_tag(sc_name)
        sc_sem = str(sc.get("semantics_flag", "strong"))
        sc_note = str(sc.get("notes", ""))

        sc_dir = root_out / sc_tag
        (sc_dir / "logs").mkdir(parents=True, exist_ok=True)

        logger = Logger(sc_dir / "logs")

        cfg_base = apply_overrides(cfg0, sc_over)

        # Save scenario meta
        dump_run_metadata(
            logger,
            cfg_base,
            extra={
                "scenario": sc_name,
                "scenario_tag": sc_tag,
                "semantics_flag": sc_sem,
                "notes": sc_note,
                "overrides": sc_over,
                "percent_search": PERCENT_SEARCH,
                "percent_eval": PERCENT_EVAL,
                "seeds": SEEDS,
                "qubits_list": QUBITS_LIST,
            },
        )

        scenario_runs = []
        corr_csv = sc_dir / "rl_proxy_vs_final.csv"
        if not corr_csv.exists():
            with open(corr_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "scenario_tag",
                        "seed",
                        "nq_requested",
                        "best_nq",
                        "best_proxy_rl",
                        "final_auc_nested",
                    ]
                )

            N_JOBS = min(len(SEEDS) * len(QUBITS_LIST), multiprocessing.cpu_count())
            print(f"[parallel] {len(SEEDS)} seeds × {len(QUBITS_LIST)} qubits → n_jobs={N_JOBS}")

            raw_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
                delayed(_run_one_seed_cross_circle)(
                    seed=seed,
                    nq=nq,
                    sc_name=sc_name,
                    sc_tag=sc_tag,
                    cfg_base=cfg_base,
                    sc_dir=sc_dir,
                    PERCENT_SEARCH=PERCENT_SEARCH,
                    PERCENT_EVAL=PERCENT_EVAL,
                )
                for seed in SEEDS
            )

            by_seed = defaultdict(dict)
            all_pareto = []
            for res in raw_results:
                s, nq = res["seed"], res["nq"]
                by_seed[s][str(nq)] = res
                all_pareto.append(
                    {"seed": s, "n_qubits": nq, "cost": res["cost"], "perf": res["perf"]}
                )

            for seed in SEEDS:
                rlqcv = by_seed[seed]
                pareto_points = [p for p in all_pareto if p["seed"] == seed]
                try:
                    with open(corr_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        for nq_str, obj in rlqcv.items():
                            w.writerow(
                                [
                                    str(sc_tag),
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

                best = max(rlqcv.values(), key=lambda x: x["perf"], default=None)
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

            best = None
            for nq, obj in rlqcv.items():
                if best is None or obj["perf"] > best["perf"]:
                    best = {"seed": int(seed), "n_qubits": int(nq), **obj}

            scenario_runs.append(
                {
                    "scenario": sc_name,
                    "seed": int(seed),
                    "percent_search": int(PERCENT_SEARCH),
                    "percent_eval": int(PERCENT_EVAL),
                    "holdout_frac": float(cfg_base.holdout_frac),
                    # "baselines": baselines,
                    "rlqcv": rlqcv,
                    "best_rlqcv": best,
                    "pareto": pareto_points,
                }
            )
        #
        try:
            import numpy as np

            alpha_per_seed = []
            for r in scenario_runs:
                best_r = r.get("best_rlqcv", {})
                enc = best_r.get("enc_params", {}) if best_r else {}
                a = enc.get("alpha", None)
                if a is not None:
                    alpha_per_seed.append(np.asarray(a, dtype=np.float32))

            if len(alpha_per_seed) > 0:
                heatmap_path = sc_dir / f"alpha_heatmap_{sc_tag}.png"
                _plot_alpha_heatmap_3x3(
                    alpha_per_seed=alpha_per_seed,
                    out_path=str(heatmap_path),
                    title=f"α finais — {sc_name}\n(média ± std entre {len(alpha_per_seed)} seeds)",
                    grid_shape=(3, 3),  # Cross/Circle tem 9 features = 3×3
                )
                print(f"[OK] Heatmap α: {heatmap_path}")
            else:
                print(f"[WARN] Sem α disponíveis para heatmap do cenário {sc_tag}")
        except Exception as e:
            print(f"[WARN] Heatmap falhou para {sc_tag}: {e}")
        # ─────────────────────────────────────────────────────────────────

        # Aggregate per-scenario
        best_per_seed = [r["best_rlqcv"] for r in scenario_runs]
        perf_list = [b["perf"] for b in best_per_seed if b is not None]

        if str(cfg_base.ci_method).lower() == "bootstrap":
            perf_mean, perf_lo, perf_hi = mean_ci_bootstrap(
                perf_list, B=int(cfg_base.bootstrap_B), alpha=float(cfg_base.ci_alpha), seed=0
            )
        else:
            perf_mean, perf_lo, perf_hi = mean_ci_t(perf_list, alpha=float(cfg_base.ci_alpha))

        base_aggr = {}
        all_points = [p for r in scenario_runs for p in r["pareto"]]
        pf = pareto_front(all_points)

        scenario_result = {
            "scenario": {
                "name": sc_name,
                "tag": sc_tag,
                "overrides": sc_over,
                "semantics_flag": sc_sem,
                "notes": sc_note,
            },
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seeds": SEEDS,
                "percent_search": int(PERCENT_SEARCH),
                "percent_eval": int(PERCENT_EVAL),
                "holdout_frac": float(cfg_base.holdout_frac),
                "qubits_list": QUBITS_LIST,
                "metric": "0.5*(AUC_mean + SENS_mean) for RLQCV (nested CV), baselines use SENS@thr*",
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
            "baselines": base_aggr,
            "pareto_front": pf,
            "runs": scenario_runs,
        }
        try:
            rows = []
            with open(corr_csv, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        bp = float(row["best_proxy_rl"])
                        fa = float(row["final_auc_nested"])
                        if np.isfinite(bp) and np.isfinite(fa):
                            rows.append((bp, fa))
                    except Exception:
                        pass
            if len(rows) >= 2:
                xs = [a for a, b in rows]
                ys = [b for a, b in rows]
                pr = _pearson(xs, ys)
                sr = _spearman(xs, ys)
                logger.log_to_file(
                    "corr",
                    f"[CORR] scenario={sc_tag} n={len(rows)} pearson={pr:.3f} spearman={sr:.3f} (rl_proxy vs final_auc)",
                )
                scenario_result["corr_rlproxy_vs_finalauc"] = {
                    "n": int(len(rows)),
                    "pearson": float(pr),
                    "spearman": float(sr),
                }
        except Exception as e:
            logger.log_to_file("corr", f"[WARN] correlation computation failed: {e}")

        out_path = sc_dir / f"results_{sc_tag}.json"
        out_path.write_text(json.dumps(scenario_result, indent=2), encoding="utf-8")
        print(f"[OK] Saved scenario: {out_path}")

        all_scenarios_results.append(
            {
                "scenario": {
                    "name": sc_name,
                    "tag": sc_tag,
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
    agg = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_scenarios": len(all_scenarios_results),
        },
        "scenarios": all_scenarios_results,
    }
    agg_path = root_out / "ALL_SCENARIOS_INDEX.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[OK] Saved aggregate index: {agg_path}")
