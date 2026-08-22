"""
extract_classical_baselines.py
================================
Extrai ou calcula baselines clássicos (LR, RF, SVM) para os
três datasets — Cross/Circle, Make Moons, Banknote.

Tenta primeiro extrair dos JSONs existentes (campo 'baselines').
Se não encontrar, calcula diretamente com kfold_baselines_calibrated
usando os mesmos dados dos experimentos.

Uso:
    python extract_classical_baselines.py

Outputs:
    - Terminal: tabela comparativa baselines vs RL-QAS
    - classical_baselines_results.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from refactor_project.data.banknote import create_banknote_dataset
from refactor_project.data.cross_circle import create_circle_cross_dataset
from refactor_project.data.moons import create_make_moons_dataset

# ════════════════════════════════════════════════════════════════════
# 1.  EXTRAÇÃO DOS JSONs EXISTENTES
# ════════════════════════════════════════════════════════════════════

def extract_baselines_from_json(base_dir: str, scenarios: list) -> dict | None:
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
        bl   = data.get("baselines", {})
        if bl and not bl.get("error"):
            return bl
    return None


# ════════════════════════════════════════════════════════════════════
# 2.  CÁLCULO DIRETO — mesmo protocolo dos experimentos
# ════════════════════════════════════════════════════════════════════

def compute_baselines(
    X       : np.ndarray,
    Y       : np.ndarray,
    n_splits: int = 5,
    seed    : int = 42,
) -> dict:
    """
    Calcula AUC (média ± std) via StratifiedKFold para:
      - Logistic Regression  (com StandardScaler)
      - Random Forest        (sem scaler — tree-based)
      - SVM (RBF kernel)     (com StandardScaler)

    Protocolo idêntico ao kfold_baselines_calibrated do projeto.
    """
    y_flat = Y.reshape(-1).astype(float)

    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, random_state=seed, C=1.0,
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100, random_state=seed,
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", probability=True,
                random_state=seed, C=1.0,
            )),
        ]),
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
                auc  = roc_auc_score(y_va, prob)
                aucs.append(float(auc))
            except Exception as e:
                print(f"    [WARN] {name} fold falhou: {e}")

        if aucs:
            results[name] = {
                "auc_mean": round(float(np.mean(aucs)), 4),
                "auc_std" : round(float(np.std(aucs)),  4),
                "n_folds" : len(aucs),
            }
        else:
            results[name] = {"auc_mean": None, "auc_std": None, "n_folds": 0}

    return results


# ════════════════════════════════════════════════════════════════════
# 3.  RESULTADOS RL-QAS — extraídos dos JSONs
# ════════════════════════════════════════════════════════════════════

def extract_rlqas_results(base_dir: str, scenarios: list) -> dict:
    """
    Extrai AUC holdout médio por cenário dos JSONs existentes.
    Retorna dict {scenario_name: {auc_mean, auc_std, seeds_ok}}.
    """
    base    = Path(base_dir)
    results = {}

    for sc in scenarios:
        found = list((base / sc).rglob("results_*.json"))
        if not found:
            results[sc] = None
            continue

        data     = json.loads(found[0].read_text())
        auc_list = []

        for run in data.get("runs", []):
            best = run.get("best_rlqcv") or {}
            ho   = best.get("holdout", {})
            auc  = ho.get("auc")
            if auc is not None:
                auc_list.append(float(auc))

        if auc_list:
            results[sc] = {
                "auc_mean" : round(float(np.mean(auc_list)), 4),
                "auc_std"  : round(float(np.std(auc_list)),  4),
                "seeds_ok" : sum(1 for a in auc_list if a >= 0.99),
                "n_seeds"  : len(auc_list),
            }
        else:
            results[sc] = None

    return results


# ════════════════════════════════════════════════════════════════════
# 4.  CONFIGURAÇÃO DOS DATASETS
# ════════════════════════════════════════════════════════════════════

DATASETS = [
    {
        "name"      : "Cross/Circle",
        "base_dir"  : "publication_out_ablation_cross_only_4_qubits",
        "scenarios" : [
            "cc_s0_baseline_minimum",
            "cc_s1_plus_enc_parametric",
            "cc_s2_plus_thr_soft_youden",
            "cc_s3_plus_hard_budget",
            "cc_s4_plus_focal",
        ],
        "s0_label"  : "S0 (fixed enc)",
        "s3_label"  : "S3 (ours)",
        "load_fn"   : lambda: create_circle_cross_dataset(
                          n_samples_per_class=200, noise_std=0.1, seed=42
                      ),
    },
    {
        "name"      : "Make Moons",
        "base_dir"  : "publication_out_make_moons",
        "scenarios" : [
            "mm_s0_baseline",
            "mm_s1_enc_param",
            "mm_s2_thr_soft_youden",
            "mm_s3_hard_budget",
            "mm_s4_focal",
        ],
        "s0_label"  : "S0 (fixed enc)",
        "s3_label"  : "S3 (ours)",
        "load_fn"   : lambda: create_make_moons_dataset(
                          n_samples=400, noise=0.15, seed=42
                      ),
    },
    {
        "name"      : "Banknote",
        "base_dir"  : "publication_out_banknote",
        "scenarios" : [
            "bn_s0_baseline",
            "bn_s1_enc_param",
            "bn_s2_thr_soft_youden",
            "bn_s3_hard_budget",
            "bn_s4_focal",
        ],
        "s0_label"  : "S0 (fixed enc)",
        "s3_label"  : "S3 (ours)",
        "load_fn"   : lambda: create_banknote_dataset(seed=42),
    },
]

# nomes curtos dos cenários S0 e S3 para a tabela
S0_KEYS = ["cc_s0_baseline_minimum", "mm_s0_baseline", "bn_s0_baseline"]
S3_KEYS = ["cc_s3_plus_hard_budget", "mm_s3_hard_budget", "bn_s3_hard_budget"]


# ════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    all_results = {}

    for ds in DATASETS:
        name      = ds["name"]
        base_dir  = ds["base_dir"]
        scenarios = ds["scenarios"]

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # ── tenta extrair baselines dos JSONs ─────────────────────────
        bl = extract_baselines_from_json(base_dir, scenarios)
        if bl:
            print(f"  [JSON] Baselines encontrados nos JSONs existentes")
            baselines = bl
        else:
            print(f"  [CALC] Calculando baselines diretamente...")
            X, Y = ds["load_fn"]()
            print(f"  [DATA] X={X.shape}  Y={Y.shape}")
            baselines = compute_baselines(X, Y)
            for clf_name, res in baselines.items():
                print(
                    f"    {clf_name:<25}: "
                    f"AUC = {res['auc_mean']:.4f} ± {res['auc_std']:.4f}"
                )

        # ── extrai resultados RL-QAS ──────────────────────────────────
        print(f"  [JSON] Extraindo resultados RL-QAS...")
        rl_results = extract_rlqas_results(base_dir, scenarios)

        # S0 e S3
        s0_res = next(
            (rl_results[sc] for sc in scenarios if sc in S0_KEYS and rl_results.get(sc)),
            None
        )
        s3_res = next(
            (rl_results[sc] for sc in scenarios if sc in S3_KEYS and rl_results.get(sc)),
            None
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

        all_results[name] = {
            "baselines" : baselines,
            "rl_s0"     : s0_res,
            "rl_s3"     : s3_res,
            "rl_all"    : rl_results,
        }

    # ── tabela LaTeX ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TABELA LaTeX — baselines vs RL-QAS")
    print(f"{'='*60}\n")

    # coleta nomes de classifiers presentes
    clf_names = []
    for ds_res in all_results.values():
        for k in ds_res["baselines"]:
            if k not in clf_names:
                clf_names.append(k)

    def fmt_auc(res):
        if res is None:
            return r"\textcolor{red}{?}"
        m = res.get("auc_mean")
        s = res.get("auc_std")
        if m is None:
            return r"\textcolor{red}{?}"
        return f"${m:.3f}${{\pm}}{s:.3f}$" if s else f"${m:.3f}$"

    ds_names = list(all_results.keys())

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Classical ML baselines vs.\ RL-QAS on holdout AUC")
    print(r"         (mean$\pm$std, 5-fold CV for classical methods,")
    print(r"         5 seeds for RL-QAS).}")
    print(r"\label{tab:baselines}")
    print(r"\begin{tabular}{l" + "c" * len(ds_names) + "}")
    print(r"\toprule")
    header = "Method & " + " & ".join(
        f"\\textbf{{{n}}}" for n in ds_names
    ) + r" \\"
    print(header)
    print(r"\midrule")

    # baselines clássicos
    for clf in clf_names:
        row = clf
        for ds_name in ds_names:
            res = all_results[ds_name]["baselines"].get(clf)
            row += " & " + fmt_auc(res)
        row += r" \\"
        print(row)

    print(r"\midrule")

    # RL-QAS S0
    row = "RL-QAS S0 (fixed enc.)"
    for ds_name in ds_names:
        row += " & " + fmt_auc(all_results[ds_name]["rl_s0"])
    row += r" \\"
    print(row)

    # RL-QAS S3
    row = r"RL-QAS S3 \textbf{(ours)}"
    for ds_name in ds_names:
        row += " & " + fmt_auc(all_results[ds_name]["rl_s3"])
    row += r" \\"
    print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── salva JSON ────────────────────────────────────────────────────
    Path("classical_baselines_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print(f"\n[OK] classical_baselines_results.json")
    print("\n[DONE]\n")


if __name__ == "__main__":
    main()