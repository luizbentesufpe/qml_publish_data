"""
extract_classical_baselines.py
================================
Extrai ou calcula baselines clássicos (LR, RF, SVM) para todos os
datasets — Cross/Circle, Make Moons, Banknote, BC Wisconsin, HIGGS Boson.

Tenta primeiro extrair dos JSONs existentes (campo 'baselines').
Se não encontrar, calcula diretamente com kfold_baselines_calibrated
usando os mesmos dados dos experimentos.

Para o HIGGS, extrai também o diagnóstico físico α_HL/α_LL (ratio
high-level / low-level) do campo 'alpha_physics' nos JSONs do cenário
S1 — se disponível.

Uso:
    python extract_classical_baselines.py
    python extract_classical_baselines.py --subset_size 10000

Outputs:
    - Terminal: tabela comparativa baselines vs RL-QAS + LaTeX
    - classical_baselines_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from refactor_project.data.banknote import create_banknote_dataset
from refactor_project.data.breast_cancer import create_breast_cancer_dataset
from refactor_project.data.cross_circle import create_circle_cross_dataset
from refactor_project.data.higgs import create_higgs_dataset
from refactor_project.data.moons import create_make_moons_dataset

#: Tamanho do subset HIGGS — mesmo valor usado no main_higgs.py para
#: garantir que os baselines clássicos sejam calculados no mesmo conjunto.
_HIGGS_SUBSET_SIZE_DEFAULT = 10_000

# ════════════════════════════════════════════════════════════════════
# 1.  EXTRAÇÃO DOS JSONs EXISTENTES
# ════════════════════════════════════════════════════════════════════


def extract_baselines_from_json(base_dir: str, scenarios: List[str]) -> Optional[Dict]:
    """
    Tenta extrair baselines do campo 'baselines' nos JSONs existentes.
    Retorna None se não encontrar ou se o campo estiver vazio.
    """
    base = Path(base_dir)
    for sc in scenarios:
        found = list((base / sc).rglob("results_*.json"))
        if not found:
            continue
        data = json.loads(found[0].read_text())
        bl = data.get("baselines", {})
        if bl and not bl.get("error"):
            return bl
    return None


def extract_alpha_physics_from_json(
    base_dir: str,
    s1_scenario: str,
) -> Optional[Dict[str, Any]]:
    """
    Extrai o diagnóstico físico α_HL / α_LL do cenário S1 do HIGGS.

    Procura o campo 'alpha_physics' no JSON do cenário S1, que é
    populado por main_higgs.py com mean/std de α por grupo (LL e HL)
    e o ratio HL/LL. Retorna None se o campo não existir (e.g., se
    o experimento ainda não foi rodado ou foi rodado sem main_higgs.py).
    """
    base = Path(base_dir)
    found = list((base / s1_scenario).rglob("results_*.json"))
    if not found:
        return None
    data = json.loads(found[0].read_text())
    return data.get("alpha_physics")


# ════════════════════════════════════════════════════════════════════
# 2.  CÁLCULO DIRETO — mesmo protocolo dos experimentos
# ════════════════════════════════════════════════════════════════════


def compute_baselines(
    X: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Calcula AUC (média ± std) via StratifiedKFold para:
      - Logistic Regression  (com StandardScaler)
      - Random Forest        (sem scaler — tree-based)
      - SVM (RBF kernel)     (com StandardScaler)

    Protocolo idêntico ao kfold_baselines_calibrated do projeto.
    """
    y_flat = Y.reshape(-1).astype(float)

    classifiers = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        C=1.0,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=seed,
                        C=1.0,
                    ),
                ),
            ]
        ),
    }

    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_int = (y_flat > 0.5).astype(int)

    for name, pipeline in classifiers.items():
        aucs = []
        for tr_idx, va_idx in skf.split(X, y_int):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y_int[tr_idx], y_int[va_idx]
            try:
                pipeline.fit(X_tr, y_tr)
                prob = pipeline.predict_proba(X_va)[:, 1]
                auc = roc_auc_score(y_va, prob)
                aucs.append(float(auc))
            except Exception as e:
                print(f"    [WARN] {name} fold falhou: {e}")

        if aucs:
            results[name] = {
                "auc_mean": round(float(np.mean(aucs)), 4),
                "auc_std": round(float(np.std(aucs)), 4),
                "n_folds": len(aucs),
            }
        else:
            results[name] = {"auc_mean": None, "auc_std": None, "n_folds": 0}

    return results


# ════════════════════════════════════════════════════════════════════
# 3.  RESULTADOS RL-QAS — extraídos dos JSONs
# ════════════════════════════════════════════════════════════════════


def extract_rlqas_results(base_dir: str, scenarios: List[str]) -> Dict[str, Any]:
    """
    Extrai AUC holdout médio por cenário dos JSONs existentes.
    Retorna dict {scenario_name: {auc_mean, auc_std, seeds_ok}}.

    Compatível com o schema de JSON produzido por main_banknote.py,
    main_higgs.py e demais mains — todos usam o mesmo campo
    'runs[*].best_rlqcv.holdout.auc'.
    """
    base = Path(base_dir)
    results = {}

    for sc in scenarios:
        found = list((base / sc).rglob("results_*.json"))
        if not found:
            results[sc] = None
            continue

        data = json.loads(found[0].read_text())
        auc_list = []

        for run in data.get("runs", []):
            best = run.get("best_rlqcv") or {}
            ho = best.get("holdout", {})
            auc = ho.get("auc")
            if auc is not None:
                auc_list.append(float(auc))

        if auc_list:
            results[sc] = {
                "auc_mean": round(float(np.mean(auc_list)), 4),
                "auc_std": round(float(np.std(auc_list)), 4),
                "seeds_ok": sum(1 for a in auc_list if a >= 0.99),
                "n_seeds": len(auc_list),
            }
        else:
            results[sc] = None

    return results


# ════════════════════════════════════════════════════════════════════
# 4.  CONFIGURAÇÃO DOS DATASETS
# ════════════════════════════════════════════════════════════════════

#: Chaves S0 e S3 de cada dataset — usadas para identificar o cenário
#: correto dentro da lista de cenários de cada dataset.
S0_KEYS = [
    "cc_s0_baseline_minimum",
    "mm_s0_baseline",
    "bn_s0_baseline",
    "bc_s0_baseline",
    "higgs_s0_baseline",
]
S3_KEYS = [
    "cc_s3_plus_hard_budget",
    "mm_s3_hard_budget",
    "bn_s3_hard_budget",
    "bc_s3_hard_budget",
    "higgs_s3_hard_budget",
]
#: Chave S1 do HIGGS — usada para extrair alpha_physics
_HIGGS_S1_KEY = "higgs_s1_enc_param"


def build_datasets_config(
    higgs_subset_size: int = _HIGGS_SUBSET_SIZE_DEFAULT,
    data_dir: str = "data",
) -> List[Dict[str, Any]]:
    """
    Retorna a configuração de todos os datasets, incluindo HIGGS.
    O `higgs_subset_size` é injetado no loader do HIGGS para garantir
    que o mesmo subset usado no treino RL seja usado para os baselines.
    """
    return [
        {
            "name": "Cross/Circle",
            "key": "cross_circle",
            "base_dir": "publication_out_cross_circle",
            "scenarios": [
                "cc_s0_baseline_minimum",
                "cc_s1_plus_enc_parametric",
                "cc_s2_plus_thr_soft_youden",
                "cc_s3_plus_hard_budget",
                "cc_s4_plus_focal",
            ],
            "load_fn": lambda: create_circle_cross_dataset(
                n_samples_per_class=200, noise_std=0.1, seed=42
            ),
            "has_physics": False,
        },
        {
            "name": "Make Moons",
            "key": "make_moons",
            "base_dir": "publication_out_make_moons",
            "scenarios": [
                "mm_s0_baseline",
                "mm_s1_enc_param",
                "mm_s2_thr_soft_youden",
                "mm_s3_hard_budget",
                "mm_s4_focal",
            ],
            "load_fn": lambda: create_make_moons_dataset(n_samples=400, noise=0.15, seed=42),
            "has_physics": False,
        },
        {
            "name": "Banknote",
            "key": "banknote",
            "base_dir": "publication_out_banknote",
            "scenarios": [
                "bn_s0_baseline",
                "bn_s1_enc_param",
                "bn_s2_thr_soft_youden",
                "bn_s3_hard_budget",
                "bn_s4_focal",
            ],
            "load_fn": lambda: create_banknote_dataset(seed=42),
            "has_physics": False,
        },
        {
            "name": "BC Wisconsin",
            "key": "breast_cancer",
            "base_dir": "publication_out_breast_cancer",
            "scenarios": [
                "bc_s0_baseline",
                "bc_s1_enc_param",
                "bc_s2_thr_soft_youden",
                "bc_s3_hard_budget",
                "bc_s4_focal",
            ],
            "load_fn": lambda: create_breast_cancer_dataset(seed=42, data_dir=data_dir),
            "has_physics": False,
        },
        {
            "name": "HIGGS Boson",
            "key": "higgs",
            "base_dir": "publication_out_higgs",
            "scenarios": [
                "higgs_s0_baseline",
                "higgs_s1_enc_param",
                "higgs_s2_thr_soft_youden",
                "higgs_s3_hard_budget",
                "higgs_s4_focal",
            ],
            # Lambda captura higgs_subset_size por closure — correto porque
            # build_datasets_config é chamado após parse dos args.
            "load_fn": (
                lambda ss, dd: lambda: create_higgs_dataset(seed=42, data_dir=dd, subset_size=ss)
            )(higgs_subset_size, data_dir),
            "has_physics": True,
            "s1_scenario": _HIGGS_S1_KEY,
        },
    ]


# ════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Extrai baselines clássicos e RL-QAS de todos os datasets."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=_HIGGS_SUBSET_SIZE_DEFAULT,
        help=f"Tamanho do subset HIGGS (default: {_HIGGS_SUBSET_SIZE_DEFAULT}).",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    DATASETS = build_datasets_config(
        higgs_subset_size=args.subset_size,
        data_dir=args.data_dir,
    )

    all_results: Dict[str, Any] = {}

    for ds in DATASETS:
        name = ds["name"]
        base_dir = ds["base_dir"]
        scenarios = ds["scenarios"]
        has_physics = ds.get("has_physics", False)

        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        # ── tenta extrair baselines dos JSONs ─────────────────────────
        bl = extract_baselines_from_json(base_dir, scenarios)
        if bl:
            print("  [JSON] Baselines encontrados nos JSONs existentes")
            baselines = bl
        else:
            print("  [CALC] Calculando baselines diretamente...")
            X, Y = ds["load_fn"]()
            n_pos = int((Y.reshape(-1) > 0.5).sum())
            print(
                f"  [DATA] X={X.shape}  pos={n_pos}  "
                f"neg={len(Y) - n_pos}  balance={n_pos / len(Y):.3f}"
            )
            baselines = compute_baselines(X, Y)
            for clf_name, res in baselines.items():
                if res["auc_mean"] is not None:
                    print(
                        f"    {clf_name:<25}: AUC = {res['auc_mean']:.4f} ± {res['auc_std']:.4f}"
                    )

        # ── extrai resultados RL-QAS ──────────────────────────────────
        print("  [JSON] Extraindo resultados RL-QAS...")
        rl_results = extract_rlqas_results(base_dir, scenarios)

        s0_res = next(
            (rl_results[sc] for sc in scenarios if sc in S0_KEYS and rl_results.get(sc)),
            None,
        )
        s3_res = next(
            (rl_results[sc] for sc in scenarios if sc in S3_KEYS and rl_results.get(sc)),
            None,
        )

        if s0_res:
            print(
                f"    {'RL-QAS S0':<25}: "
                f"AUC = {s0_res['auc_mean']:.4f} ± {s0_res['auc_std']:.4f}"
                f"  ({s0_res['seeds_ok']}/{s0_res['n_seeds']} seeds ≥0.99)"
            )
        if s3_res:
            print(
                f"    {'RL-QAS S3 (ours)':<25}: "
                f"AUC = {s3_res['auc_mean']:.4f} ± {s3_res['auc_std']:.4f}"
                f"  ({s3_res['seeds_ok']}/{s3_res['n_seeds']} seeds ≥0.99)"
            )

        # ── diagnóstico físico (apenas HIGGS) ─────────────────────────
        alpha_physics = None
        if has_physics:
            s1_scenario = ds.get("s1_scenario", "")
            alpha_physics = extract_alpha_physics_from_json(base_dir, s1_scenario)
            if alpha_physics:
                ll = alpha_physics.get("low_level_mean_alpha", {})
                hl = alpha_physics.get("high_level_mean_alpha", {})
                ratio = alpha_physics.get("ratio_HL_over_LL", {})
                print(
                    f"  [PHYS] α_LL = {ll.get('mean', '?'):.3f}±{ll.get('std', '?'):.3f}  "
                    f"α_HL = {hl.get('mean', '?'):.3f}±{hl.get('std', '?'):.3f}  "
                    f"ratio HL/LL = {ratio.get('mean', '?'):.2f}±{ratio.get('std', '?'):.2f}"
                )
            else:
                print(
                    "  [PHYS] alpha_physics não encontrado no JSON "
                    "(rode main_higgs.py para populá-lo)"
                )

        all_results[name] = {
            "baselines": baselines,
            "rl_s0": s0_res,
            "rl_s3": s3_res,
            "rl_all": rl_results,
            "alpha_physics": alpha_physics,
        }

    # ── tabela de texto consolidada ───────────────────────────────────
    print(f"\n\n{'=' * 78}")
    print("  TABELA CONSOLIDADA — baselines vs RL-QAS")
    print(f"{'=' * 78}")

    ds_names = list(all_results.keys())
    clf_names = []
    for ds_res in all_results.values():
        for k in ds_res["baselines"]:
            if k not in clf_names:
                clf_names.append(k)

    col_w = 14
    header = f"{'Method':<28}" + "".join(f"{n:>{col_w}}" for n in ds_names)
    print(header)
    print("─" * (28 + col_w * len(ds_names)))

    def fmt_cell(res):
        if res is None:
            return "?"
        m = res.get("auc_mean")
        s = res.get("auc_std")
        if m is None:
            return "?"
        return f"{m:.3f}±{s:.3f}" if s else f"{m:.3f}"

    for clf in clf_names:
        row = f"{clf:<28}"
        for ds_name in ds_names:
            res = all_results[ds_name]["baselines"].get(clf)
            row += f"{fmt_cell(res):>{col_w}}"
        print(row)

    print("─" * (28 + col_w * len(ds_names)))

    row = f"{'RL-QAS S0 (fixed enc.)':<28}"
    for ds_name in ds_names:
        row += f"{fmt_cell(all_results[ds_name]['rl_s0']):>{col_w}}"
    print(row)

    row = f"{'RL-QAS S3 (ours)':<28}"
    for ds_name in ds_names:
        row += f"{fmt_cell(all_results[ds_name]['rl_s3']):>{col_w}}"
    print(row)

    # ── diagnóstico físico HIGGS (se disponível) ──────────────────────
    higgs_phys = all_results.get("HIGGS Boson", {}).get("alpha_physics")
    if higgs_phys:
        ratio_data = higgs_phys.get("ratio_HL_over_LL", {})
        ratio_mean = ratio_data.get("mean")
        ratio_std = ratio_data.get("std")
        if ratio_mean is not None:
            print("\n  HIGGS — diagnóstico físico S1:")
            print(
                f"    ratio HL/LL = {ratio_mean:.2f}±{ratio_std:.2f}  "
                f"({'HL > LL: hipótese confirmada' if ratio_mean > 1.0 else 'HL ≤ LL: investigar'})"
            )

    # ── tabela LaTeX ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TABELA LaTeX — baselines vs RL-QAS")
    print(f"{'=' * 60}\n")

    def fmt_auc_latex(res):
        if res is None:
            return r"\textcolor{red}{?}"
        m = res.get("auc_mean")
        s = res.get("auc_std")
        if m is None:
            return r"\textcolor{red}{?}"
        return f"${m:.3f}{{\\pm}}{s:.3f}$" if s else f"${m:.3f}$"

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Classical ML baselines vs.\ RL-QAS on holdout AUC")
    print(r"         (mean$\pm$std, 5-fold CV for classical methods,")
    print(r"         5 seeds for RL-QAS). HIGGS uses a balanced 10k subset.}")
    print(r"\label{tab:baselines}")
    print(r"\begin{tabular}{l" + "c" * len(ds_names) + "}")
    print(r"\toprule")
    header_latex = "Method & " + " & ".join(f"\\textbf{{{n}}}" for n in ds_names) + r" \\"
    print(header_latex)
    print(r"\midrule")

    for clf in clf_names:
        row = clf.replace("&", r"\&")
        for ds_name in ds_names:
            res = all_results[ds_name]["baselines"].get(clf)
            row += " & " + fmt_auc_latex(res)
        row += r" \\"
        print(row)

    print(r"\midrule")

    row = r"RL-QAS S0 (fixed enc.)"
    for ds_name in ds_names:
        row += " & " + fmt_auc_latex(all_results[ds_name]["rl_s0"])
    row += r" \\"
    print(row)

    row = r"RL-QAS S3 \textbf{(ours)}"
    for ds_name in ds_names:
        row += " & " + fmt_auc_latex(all_results[ds_name]["rl_s3"])
    row += r" \\"
    print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── salva JSON ────────────────────────────────────────────────────
    Path("classical_baselines_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print("\n[OK] classical_baselines_results.json")
    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
