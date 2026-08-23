"""
classical_baselines.py
======================
Executa baselines clássicos em todos os datasets do framework RL-QAS
com o mesmo protocolo de avaliação: nested CV (3×3) + holdout 20%.

Baselines:
    - Logistic Regression (LR)
    - SVM RBF (SVC)
    - Random Forest (RF)
    - XGBoost (XGB)
    - k-Nearest Neighbors (KNN)

Protocolo idêntico ao VQC:
    - Holdout 20% separado antes de qualquer treino
    - Nested CV 3×3 no train_all para AUC/F1/Acc nested
    - Avaliação final no holdout com threshold Youden

Saída:
    - classical_baselines_results.csv
    - classical_baselines_table.tex  (LaTeX pronto para o paper)
    - classical_baselines_summary.json

Uso:
    cd refactor_project
    python3 classical_baselines.py
    python3 classical_baselines.py --dataset higgs
    python3 classical_baselines.py --seeds 0 1 2 3 4
"""

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost não instalado — pulando XGB baseline.")

# Resolve root automaticamente: results/ → refactor_project/refactor_project/
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = str(_PROJECT_ROOT / "data")
_OUT_DIR = _SCRIPT_DIR

sys.path.insert(0, str(_PROJECT_ROOT))


# ── Loaders dos datasets ───────────────────────────────────────────────────
def load_cross_circle(data_dir="data", seed=42):
    from refactor_project.data.cross_circle import create_circle_cross_dataset
    X, Y = create_circle_cross_dataset(seed=seed)
    y = Y.flatten().astype(int)
    feature_names = [f"p{r}{c}" for r in range(3) for c in range(3)]
    return X, y, feature_names

def load_moons(data_dir="data", seed=42):
    from refactor_project.data.moons import create_make_moons_dataset
    X, Y = create_make_moons_dataset(seed=seed)
    y = Y.flatten().astype(int)
    return X, y, ["x0", "x1"]

def load_blobs(data_dir="data", seed=42):
    from refactor_project.data.blobs import create_blobs_dataset
    X, Y = create_blobs_dataset(seed=seed)
    y = Y.flatten().astype(int)
    return X, y, ["x0", "x1"]

def load_concentric_circles(data_dir="data", seed=42):
    from refactor_project.data.concentric_circles import create_concentric_circles_dataset
    X, Y = create_concentric_circles_dataset(seed=seed)
    y = Y.flatten().astype(int)
    return X, y, ["x0", "x1"]

def load_stripes(data_dir="data", seed=42):
    from refactor_project.data.stripes import create_stripes_dataset
    X, Y = create_stripes_dataset(seed=seed)
    y = Y.flatten().astype(int)
    feature_names = [f"p{r}{c}" for r in range(3) for c in range(3)]
    return X, y, feature_names

def load_banknote(data_dir="data", seed=42):
    from refactor_project.data.banknote import FEATURE_NAMES, create_banknote_dataset
    X, Y = create_banknote_dataset(seed=seed, data_dir=data_dir)
    y = Y.flatten().astype(int)
    return X, y, list(FEATURE_NAMES)

def load_breast_cancer(data_dir="data", seed=42):
    from refactor_project.data.breast_cancer import FEATURE_NAMES, create_breast_cancer_dataset
    X, Y = create_breast_cancer_dataset(seed=seed, data_dir=data_dir)
    y = Y.flatten().astype(int)
    return X, y, list(FEATURE_NAMES)

def load_iris(data_dir="data", seed=42):
    from refactor_project.data.iris import FEATURE_NAMES, create_iris_dataset
    X, Y = create_iris_dataset(seed=seed, data_dir=data_dir)
    y = Y.flatten().astype(int)
    return X, y, list(FEATURE_NAMES)

def load_higgs(data_dir="data", seed=42):
    from refactor_project.data.higgs import FEATURE_NAMES, create_higgs_dataset
    X, Y = create_higgs_dataset(seed=seed, data_dir=data_dir, subset_size=10000)
    y = Y.flatten().astype(int)
    return X, y, list(FEATURE_NAMES)

def load_mnist(data_dir="data", seed=42):
    from refactor_project.data.mnist import DEFAULT_PCA_COMPONENTS, create_mnist_dataset
    X, Y = create_mnist_dataset(seed=seed, data_dir=data_dir)
    y = Y.flatten().astype(int)
    feature_names = [f"pca_{i}" for i in range(DEFAULT_PCA_COMPONENTS)]
    return X, y, feature_names


# ── Configuração dos datasets ──────────────────────────────────────────────
DATASETS = {
    "cross_circle":       {"loader": load_cross_circle,       "label": "Cross/Circle",    "holdout_frac": 0.20},
    "moons":              {"loader": load_moons,              "label": "Make Moons",      "holdout_frac": 0.20},
    "blobs":              {"loader": load_blobs,              "label": "Blobs",           "holdout_frac": 0.20},
    "concentric_circles": {"loader": load_concentric_circles, "label": "Conc. Circles",   "holdout_frac": 0.20},
    "stripes":            {"loader": load_stripes,            "label": "Stripes",         "holdout_frac": 0.20},
    "banknote":           {"loader": load_banknote,           "label": "Banknote",        "holdout_frac": 0.20},
    "breast_cancer":      {"loader": load_breast_cancer,      "label": "BC Wisconsin",    "holdout_frac": 0.20},
    "iris":               {"loader": load_iris,               "label": "Iris",            "holdout_frac": 0.20},
    "higgs":              {"loader": load_higgs,              "label": "Higgs Boson",     "holdout_frac": 0.20},
    "mnist":              {"loader": load_mnist,              "label": "MNIST (3 vs 8)",  "holdout_frac": 0.20},
}


# ── Definição dos baselines ────────────────────────────────────────────────
def build_classifiers():
    clfs = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight="balanced")),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42,
                        class_weight="balanced")),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }
    if HAS_XGB:
        clfs["XGB"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="auc",
            verbosity=0, use_label_encoder=False,
        )
    return clfs


def youden_threshold(y_true, y_prob):
    """Encontra threshold ótimo pelo índice J de Youden (max TPR + TNR - 1)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def evaluate_at_threshold(y_true, y_prob, thr):
    """Calcula métricas no threshold dado."""
    y_pred = (y_prob >= thr).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"auc": auc, "f1": f1, "accuracy": acc,
            "sensitivity": sens, "specificity": spec, "threshold": thr}


def nested_cv_eval(clf, X, y, outer_splits=3, inner_splits=3, seed=42):
    """
    Nested CV 3×3 idêntico ao protocolo VQC.
    Outer: avaliação; Inner: calibração do threshold Youden.
    """
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    aucs, f1s, accs, senss, specs = [], [], [], [], []

    for tr_idx, va_idx in outer.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Inner: calibra threshold no fold de treino
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                random_state=seed + 1)
        thrs = []
        for itr_idx, ica_idx in inner.split(X_tr, y_tr):
            Xi_tr, yi_tr = X_tr[itr_idx], y_tr[itr_idx]
            Xi_ca, yi_ca = X_tr[ica_idx], y_tr[ica_idx]
            try:
                clf_clone = clone_clf(clf)
                clf_clone.fit(Xi_tr, yi_tr)
                prob = clf_clone.predict_proba(Xi_ca)[:, 1]
                thrs.append(youden_threshold(yi_ca, prob))
            except Exception:
                thrs.append(0.5)

        thr_star = float(np.mean(thrs))

        # Treina no fold completo, avalia no fold de validação
        try:
            clf_clone = clone_clf(clf)
            clf_clone.fit(X_tr, y_tr)
            prob_va = clf_clone.predict_proba(X_va)[:, 1]
            m = evaluate_at_threshold(y_va, prob_va, thr_star)
            aucs.append(m["auc"])
            f1s.append(m["f1"])
            accs.append(m["accuracy"])
            senss.append(m["sensitivity"])
            specs.append(m["specificity"])
        except Exception as e:
            print(f"    [WARN] fold falhou: {e}")

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "sens_mean": float(np.mean(senss)),
        "spec_mean": float(np.mean(specs)),
    }


def clone_clf(clf):
    """Clona estimador sklearn preservando parâmetros."""
    from sklearn.base import clone
    return clone(clf)


def run_dataset(dataset_key: str, seeds: list,
                data_dir: str = "data") -> list:
    """Roda todos os baselines em um dataset para múltiplas seeds."""
    cfg = DATASETS[dataset_key]
    label = cfg["label"]
    holdout_frac = cfg["holdout_frac"]

    print(f"\n{'='*65}")
    print(f"  Dataset: {label}")
    print(f"{'='*65}")

    # Carrega dataset
    try:
        X, y, feature_names = cfg["loader"](data_dir=data_dir, seed=42)
    except Exception as e:
        print(f"  [ERRO] Não foi possível carregar {dataset_key}: {e}")
        return []

    print(f"  X={X.shape}  y={np.bincount(y)}")

    clfs = build_classifiers()
    results = []

    for clf_name, clf in clfs.items():
        seed_results = []
        print(f"\n  [{clf_name}]")

        for seed in seeds:
            # Holdout split idêntico ao VQC
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=holdout_frac, random_state=seed
            )
            tr_idx, ho_idx = next(sss.split(X, y))
            X_train_all, y_train_all = X[tr_idx], y[tr_idx]
            X_holdout, y_holdout = X[ho_idx], y[ho_idx]

            # Nested CV no train_all
            try:
                nested = nested_cv_eval(
                    clf, X_train_all, y_train_all, seed=seed
                )
            except Exception as e:
                print(f"    seed={seed} ERRO nested CV: {e}")
                continue

            # Treino final + holdout
            try:
                clf_final = clone_clf(clf)
                clf_final.fit(X_train_all, y_train_all)
                prob_ho = clf_final.predict_proba(X_holdout)[:, 1]
                thr_ho = youden_threshold(y_holdout, prob_ho)
                holdout = evaluate_at_threshold(y_holdout, prob_ho, thr_ho)
            except Exception as e:
                print(f"    seed={seed} ERRO holdout: {e}")
                continue

            seed_results.append({
                "seed": seed,
                "nested": nested,
                "holdout": holdout,
            })

            print(f"    seed={seed}  nested_AUC={nested['auc_mean']:.4f}  "
                  f"holdout_AUC={holdout['auc']:.4f}  "
                  f"F1={holdout['f1']:.4f}  Acc={holdout['accuracy']:.4f}")

        if not seed_results:
            continue

        # Agrega por seeds
        def agg(key_path):
            vals = []
            for sr in seed_results:
                obj = sr
                for k in key_path:
                    obj = obj[k]
                vals.append(float(obj))
            return float(np.mean(vals)), float(np.std(vals))

        ho_auc_mu, ho_auc_sd = agg(["holdout", "auc"])
        ho_f1_mu, ho_f1_sd = agg(["holdout", "f1"])
        ho_acc_mu, ho_acc_sd = agg(["holdout", "accuracy"])
        ho_sens_mu, _ = agg(["holdout", "sensitivity"])
        ho_spec_mu, _ = agg(["holdout", "specificity"])
        ne_auc_mu, ne_auc_sd = agg(["nested", "auc_mean"])

        print(f"  → {clf_name}: AUC={ho_auc_mu:.4f}±{ho_auc_sd:.4f}  "
              f"F1={ho_f1_mu:.4f}±{ho_f1_sd:.4f}  "
              f"Acc={ho_acc_mu:.4f}±{ho_acc_sd:.4f}")

        results.append({
            "dataset": dataset_key,
            "dataset_label": label,
            "classifier": clf_name,
            "n_seeds": len(seed_results),
            "nested_auc_mean": round(ne_auc_mu, 4),
            "nested_auc_std": round(ne_auc_sd, 4),
            "holdout_auc_mean": round(ho_auc_mu, 4),
            "holdout_auc_std": round(ho_auc_sd, 4),
            "holdout_f1_mean": round(ho_f1_mu, 4),
            "holdout_f1_std": round(ho_f1_sd, 4),
            "holdout_acc_mean": round(ho_acc_mu, 4),
            "holdout_acc_std": round(ho_acc_sd, 4),
            "holdout_sens_mean": round(ho_sens_mu, 4),
            "holdout_spec_mean": round(ho_spec_mu, 4),
        })

    return results


def write_csv(results: list, out_path: pathlib.Path) -> None:
    fields = [
        "dataset_label", "classifier",
        "holdout_auc_mean", "holdout_auc_std",
        "holdout_f1_mean", "holdout_f1_std",
        "holdout_acc_mean", "holdout_acc_std",
        "holdout_sens_mean", "holdout_spec_mean",
        "nested_auc_mean", "nested_auc_std",
        "n_seeds",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n[OK] CSV: {out_path}")


def write_latex(results: list, out_path: pathlib.Path) -> None:
    """Tabela LaTeX pronta para o paper — AUC, F1, Acc por dataset e método."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Baselines clássicos — holdout 20\% (média$\pm$std, 5 seeds, "
        r"threshold Youden). Protocolo idêntico ao VQC.}",
        r"\label{tab:classical_baselines}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Método & AUC & F1 & Acc \\",
        r"\midrule",
    ]

    current_ds = None
    for r in results:
        ds = r["dataset_label"]
        if ds != current_ds:
            if current_ds is not None:
                lines.append(r"\midrule")
            current_ds = ds

        auc = f"${r['holdout_auc_mean']:.3f}{{\\pm}}{r['holdout_auc_std']:.3f}$"
        f1 = f"${r['holdout_f1_mean']:.3f}{{\\pm}}{r['holdout_f1_std']:.3f}$"
        acc = f"${r['holdout_acc_mean']:.3f}{{\\pm}}{r['holdout_acc_std']:.3f}$"
        lines.append(f"{ds} & {r['classifier']} & {auc} & {f1} & {acc} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines))
    print(f"[OK] LaTeX: {out_path}")


def print_summary(results: list) -> None:
    """Imprime tabela resumo no terminal."""
    print(f"\n{'='*80}")
    print(f"{'Dataset':<18} {'Método':<10} {'AUC':>12} {'F1':>8} {'Acc':>8} "
          f"{'Sens':>7} {'Spec':>7}")
    print("-" * 80)
    current_ds = None
    for r in results:
        if r["dataset_label"] != current_ds:
            if current_ds is not None:
                print()
            current_ds = r["dataset_label"]
        auc_str = f"{r['holdout_auc_mean']:.4f}±{r['holdout_auc_std']:.4f}"
        f1_str = f"{r['holdout_f1_mean']:.4f}"
        acc_str = f"{r['holdout_acc_mean']:.4f}"
        sens_str = f"{r['holdout_sens_mean']:.4f}"
        spec_str = f"{r['holdout_spec_mean']:.4f}"
        print(f"{current_ds:<18} {r['classifier']:<10} {auc_str:>12} "
              f"{f1_str:>8} {acc_str:>8} {sens_str:>7} {spec_str:>7}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Baselines clássicos para todos os datasets do RL-QAS"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()),
        help="Rodar apenas um dataset (default: todos)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Seeds a usar (default: 0 1 2 3 4)"
    )
    parser.add_argument(
        "--data-dir", default=_DATA_DIR,
        help="Diretório com os dados (default: data)"
    )
    parser.add_argument(
        "--out", default=str(_OUT_DIR),
        help="Diretório de saída (default: diretório atual)"
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    print(f"\n[INFO] Datasets: {datasets_to_run}")
    print(f"[INFO] Seeds: {args.seeds}")
    print(f"[INFO] Baselines: LR, SVM_RBF, RF, KNN" +
          (" + XGB" if HAS_XGB else ""))

    all_results = []
    for ds_key in datasets_to_run:
        results = run_dataset(ds_key, args.seeds, data_dir=args.data_dir)
        all_results.extend(results)

    if not all_results:
        print("[ERRO] Nenhum resultado gerado.")
        sys.exit(1)

    print_summary(all_results)
    write_csv(all_results, out_dir / "classical_baselines_results.csv")
    write_latex(all_results, out_dir / "classical_baselines_table.tex")

    json_path = out_dir / "classical_baselines_summary.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"[OK] JSON: {json_path}")

    print(f"\n[DONE] {len(all_results)} resultados em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()