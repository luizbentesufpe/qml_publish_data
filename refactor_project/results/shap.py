"""
compute_shap_correlation.py
===========================
Calcula SHAP values de um XGBoost treinado no mesmo dataset
e correlaciona com os α convergidos do VQC (RL-QAS).

Hipótese testada:
    ρ(|SHAP_mean|, α_VQC) > 0  com p < 0.05
    → o VQC descobriu importância de features sem supervisão explícita

Uso:
    python3 compute_shap_correlation.py --dataset higgs
    python3 compute_shap_correlation.py --dataset banknote
    python3 compute_shap_correlation.py --dataset breast_cancer
    python3 compute_shap_correlation.py --dataset iris
    python3 compute_shap_correlation.py --all

Requisitos:
    pip install shap xgboost scikit-learn --break-system-packages

Saída:
    - shap_correlation_<dataset>.png  (scatter α vs |SHAP|)
    - shap_summary_<dataset>.png      (SHAP summary bar plot)
    - shap_results_<dataset>.json     (métricas numéricas)
"""

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Imports opcionais (instala se necessário) ──────────────────────────────
try:
    import shap
except ImportError:
    print("[ERRO] shap não instalado. Execute:")
    print("  pip install shap --break-system-packages")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("[ERRO] xgboost não instalado. Execute:")
    print("  pip install xgboost --break-system-packages")
    sys.exit(1)

from scipy.stats import spearmanr, pearsonr

# Resolve root automaticamente: results/ → refactor_project/refactor_project/
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_OUT_DIR = _SCRIPT_DIR / "shap"
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ── Loaders dos datasets ───────────────────────────────────────────────────
def load_higgs(data_dir: str = "data", subset_size: int = 10000):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from refactor_project.data.higgs import create_higgs_dataset, FEATURE_NAMES
    X, Y = create_higgs_dataset(seed=42, data_dir=data_dir,
                                subset_size=subset_size)
    y = Y.flatten().astype(int)
    return X, y, list(FEATURE_NAMES)


def load_banknote(data_dir: str = "data"):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        from refactor_project.data.banknote import create_banknote_dataset, FEATURE_NAMES
        X, Y = create_banknote_dataset(seed=42, data_dir=data_dir)
        return X, Y.flatten().astype(int), list(FEATURE_NAMES)
    except Exception:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("banknote-authentication", version=1, as_frame=False)
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        feature_names = ["variance_wavelet", "skewness_wavelet",
                         "curtosis_wavelet", "entropy_image"]
        return X, y, feature_names


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import MinMaxScaler
    data = load_breast_cancer()
    X = MinMaxScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(int)
    return X, y, list(data.feature_names)


def load_iris():
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    data = load_iris()
    X_raw = data.data.astype(np.float32)
    y_raw = data.target.astype(int)
    # Binarização: Setosa (0) vs resto
    y = (y_raw != 0).astype(int)
    # PCA d=2 (97.8% variância)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_raw).astype(np.float32)
    X = MinMaxScaler().fit_transform(X_pca).astype(np.float32)
    feature_names = ["PC1_size", "PC2_shape_ratio"]
    print(f"[INFO] Iris PCA: PC1+PC2 explicam "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% da variância")
    return X, y, feature_names


DATASET_LOADERS = {
    "higgs": load_higgs,
    "banknote": load_banknote,
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
}

DATASET_LABELS = {
    "higgs": "Higgs Boson (d=28)",
    "banknote": "Banknote Authentication (d=4)",
    "breast_cancer": "BC Wisconsin (d=30)",
    "iris": "Iris — PCA (d=2)",
}

# Índices HL/LL para o Higgs
HIGGS_LOW_LEVEL = list(range(0, 21))
HIGGS_HIGH_LEVEL = list(range(21, 28))


def load_alpha_from_results(dataset: str, scenario: str = "s4",
                             root: pathlib.Path = pathlib.Path(".")) -> dict:
    """
    Carrega α médio (entre seeds) do JSON de resultado do RL-QAS.
    Procura nos diretórios de publicação e debug.
    """
    search_dirs = [
        root / f"publication_out_{dataset}",
        root / f"debug_out_{dataset}",
        root / f"publication_out_breat_cancer",  # typo no nome original
        root / f"debug_out_breat_cancer",
    ]

    alpha_per_seed = []
    found_path = None

    for d in search_dirs:
        if not d.exists():
            continue
        # Busca JSONs que contenham o scenario
        for j in sorted(d.rglob("results_*.json")):
            if scenario not in j.stem:
                continue
            try:
                data = json.loads(j.read_text())
                seeds = data.get("best_rlqcv_per_seed", [])
                for s in seeds:
                    alpha = s.get("enc_params", {}).get("alpha")
                    if alpha:
                        alpha_per_seed.append(np.array(alpha, dtype=np.float32))
                if alpha_per_seed:
                    found_path = j
                    break
            except Exception:
                continue
        if alpha_per_seed:
            break

    if not alpha_per_seed:
        return {}

    alpha_mean = np.mean(alpha_per_seed, axis=0)
    alpha_std = np.std(alpha_per_seed, axis=0)

    print(f"[INFO] α carregado de: {found_path}")
    print(f"[INFO] {len(alpha_per_seed)} seeds, d={len(alpha_mean)}")

    return {
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "n_seeds": len(alpha_per_seed),
        "source": str(found_path),
    }


def run_shap_analysis(dataset: str, scenario: str = "s4",
                      root: pathlib.Path = pathlib.Path("."),
                      out_dir: pathlib.Path = pathlib.Path("."),
                      seed: int = 42) -> dict:
    """
    Pipeline completo:
    1. Carrega dataset
    2. Treina XGBoost
    3. Calcula SHAP
    4. Carrega α do VQC
    5. Correlaciona
    6. Salva plots e JSON
    """
    print(f"\n{'='*60}")
    print(f"  SHAP Analysis: {dataset.upper()}")
    print(f"{'='*60}")

    # ── 1. Carrega dataset ─────────────────────────────────────────────
    try:
        loader = DATASET_LOADERS[dataset]
        if dataset == "higgs":
            X, y, feature_names = loader(data_dir=str(root / "data"))
        else:
            X, y, feature_names = loader()
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar {dataset}: {e}")
        return {}

    print(f"[INFO] Dataset: X={X.shape}, y={np.bincount(y)}")

    # ── 2. Treina XGBoost ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="auc",
        verbosity=0,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)
    auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"[INFO] XGBoost AUC (holdout 20%): {auc_xgb:.4f}")

    # ── 3. Calcula SHAP ────────────────────────────────────────────────
    print("[INFO] Calculando SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # SHAP mean absolute por feature
    shap_abs_mean = np.abs(shap_values).mean(axis=0)  # (n_features,)
    shap_abs_mean_norm = shap_abs_mean / (shap_abs_mean.sum() + 1e-10)

    # ── 4. Carrega α do VQC ────────────────────────────────────────────
    alpha_data = load_alpha_from_results(dataset, scenario, root)
    if not alpha_data:
        print(f"[WARN] α não encontrado para {dataset}/{scenario}")
        print("       Verifique se a campanha foi concluída.")
        alpha_mean = None
    else:
        alpha_mean = alpha_data["alpha_mean"]
        alpha_std = alpha_data["alpha_std"]

    # ── 5. Correlação SHAP vs α ────────────────────────────────────────
    results = {
        "dataset": dataset,
        "scenario": scenario,
        "xgb_auc": float(auc_xgb),
        "feature_names": feature_names,
        "shap_abs_mean": shap_abs_mean.tolist(),
        "shap_abs_mean_norm": shap_abs_mean_norm.tolist(),
    }

    if alpha_mean is not None and len(alpha_mean) == len(shap_abs_mean):
        # Normaliza α para comparação
        alpha_norm = alpha_mean / (alpha_mean.sum() + 1e-10)

        rho, p_rho = spearmanr(shap_abs_mean_norm, alpha_norm)
        r, p_r = pearsonr(shap_abs_mean_norm, alpha_norm)

        print(f"\n[RESULTADO] Correlação SHAP vs α:")
        print(f"  Spearman ρ = {rho:.3f}  (p = {p_rho:.4f})")
        print(f"  Pearson  r = {r:.3f}  (p = {p_r:.4f})")
        print(f"  {'✓ SIGNIFICATIVO (p<0.05)' if p_rho < 0.05 else '✗ não significativo'}")

        results.update({
            "alpha_mean": alpha_mean.tolist(),
            "alpha_std": alpha_std.tolist() if alpha_mean is not None else None,
            "alpha_norm": alpha_norm.tolist(),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
            "significant": bool(p_rho < 0.05),
        })

        # Higgs específico: HL vs LL
        if dataset == "higgs" and len(alpha_mean) == 28:
            ll_alpha = alpha_mean[HIGGS_LOW_LEVEL].mean()
            hl_alpha = alpha_mean[HIGGS_HIGH_LEVEL].mean()
            ll_shap = shap_abs_mean[HIGGS_LOW_LEVEL].mean()
            hl_shap = shap_abs_mean[HIGGS_HIGH_LEVEL].mean()
            ratio_alpha = hl_alpha / (ll_alpha + 1e-10)
            ratio_shap = hl_shap / (ll_shap + 1e-10)

            print(f"\n[HIGGS HL/LL]")
            print(f"  α:    LL={ll_alpha:.3f}  HL={hl_alpha:.3f}  ratio={ratio_alpha:.2f}")
            print(f"  SHAP: LL={ll_shap:.4f}  HL={hl_shap:.4f}  ratio={ratio_shap:.2f}")
            print(f"  {'✓ SHAP confirma HL>LL' if hl_shap > ll_shap else '✗ SHAP não confirma'}")

            results.update({
                "higgs_ll_alpha": float(ll_alpha),
                "higgs_hl_alpha": float(hl_alpha),
                "higgs_ratio_alpha": float(ratio_alpha),
                "higgs_ll_shap": float(ll_shap),
                "higgs_hl_shap": float(hl_shap),
                "higgs_ratio_shap": float(ratio_shap),
                "higgs_shap_confirms_hl_ll": bool(hl_shap > ll_shap),
            })

    # ── 6. Plots ───────────────────────────────────────────────────────
    label = DATASET_LABELS.get(dataset, dataset)

    # Plot 1: SHAP summary bar
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
    colors = ["#c0392b" if i in HIGGS_HIGH_LEVEL else "#7f8fa6"
              for i in range(len(feature_names))] if dataset == "higgs" \
        else ["#2980b9"] * len(feature_names)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, shap_abs_mean, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("|SHAP| médio")
    ax.set_title(f"SHAP Feature Importance — {label}\n(XGBoost, AUC={auc_xgb:.3f})")
    if dataset == "higgs":
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#c0392b", alpha=0.85, label="high-level (invariant masses)"),
            Patch(color="#7f8fa6", alpha=0.85, label="low-level (kinematic)"),
        ], fontsize=8)
    fig.tight_layout()
    shap_bar_path = out_dir / f"shap_summary_{dataset}.png"
    fig.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] SHAP summary: {shap_bar_path}")

    # Plot 2: Scatter α vs |SHAP| (se α disponível)
    if alpha_mean is not None and len(alpha_mean) == len(shap_abs_mean):
        alpha_norm = alpha_mean / (alpha_mean.sum() + 1e-10)

        fig, ax = plt.subplots(figsize=(6, 5))
        colors_scatter = []
        for i in range(len(feature_names)):
            if dataset == "higgs":
                colors_scatter.append("#c0392b" if i in HIGGS_HIGH_LEVEL
                                      else "#7f8fa6")
            else:
                colors_scatter.append("#2980b9")

        ax.scatter(shap_abs_mean_norm, alpha_norm,
                   c=colors_scatter, alpha=0.75, s=60, edgecolors="white", lw=0.5)

        # Linha de tendência
        z = np.polyfit(shap_abs_mean_norm, alpha_norm, 1)
        p = np.poly1d(z)
        x_line = np.linspace(shap_abs_mean_norm.min(), shap_abs_mean_norm.max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.4, lw=1.5)

        # Anotações (top 5 features por SHAP)
        top5 = np.argsort(shap_abs_mean_norm)[-5:]
        for i in top5:
            ax.annotate(feature_names[i],
                        (shap_abs_mean_norm[i], alpha_norm[i]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

        rho = results.get("spearman_rho", 0)
        p_rho = results.get("spearman_p", 1)
        ax.set_xlabel("|SHAP| normalizado (XGBoost)", fontsize=10)
        ax.set_ylabel("α normalizado (VQC)", fontsize=10)
        ax.set_title(f"SHAP vs α — {label}\n"
                     f"ρ={rho:.3f}  p={p_rho:.4f}  "
                     f"{'(p<0.05 ✓)' if p_rho < 0.05 else '(n.s.)'}", fontsize=10)

        if dataset == "higgs":
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color="#c0392b", alpha=0.75, label="high-level"),
                Patch(color="#7f8fa6", alpha=0.75, label="low-level"),
            ], fontsize=8)

        fig.tight_layout()
        scatter_path = out_dir / f"shap_correlation_{dataset}.png"
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Scatter α vs SHAP: {scatter_path}")

    # Salva JSON
    json_path = out_dir / f"shap_results_{dataset}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] JSON: {json_path}")

    return results


def get_scenario_tags(dataset: str, root: pathlib.Path) -> list:
    """Detecta automaticamente os cenários disponíveis para um dataset."""
    # Mapeamento de out_dirs por dataset (typos incluídos do projeto original)
    _OUT_DIRS_MAP = {
        "breast_cancer": [f"publication_out_breast_cancer",
                          f"debug_out_breast_cancer",
                          f"publication_out_breat_cancer",
                          f"debug_out_breat_cancer"],
    }
    out_dirs = _OUT_DIRS_MAP.get(dataset, [
        f"publication_out_{dataset}",
        f"debug_out_{dataset}",
    ])
    tags = []
    for d_name in out_dirs:
        d = root / d_name
        if not d.exists():
            continue
        for j in sorted(d.rglob("results_*.json")):
            tag = j.stem.replace("results_", "")
            if tag not in tags:
                tags.append(tag)
    return tags


def plot_rho_evolution(scenario_results: list, dataset: str,
                       out_dir: pathlib.Path) -> None:
    """
    Gráfico de evolução de ρ (Spearman) ao longo da ablation S0→S4.
    Mostra como a correlação SHAP vs α evolui com cada componente adicionado.
    """
    # Filtra apenas cenários S0→S4 principais (sem lmax, sem debug)
    s_order = ["s0", "s1", "s2", "s3", "s4"]
    filtered = []
    for tag in s_order:
        for r in scenario_results:
            sc_tag = r.get("scenario_tag", r.get("scenario", ""))
            # Pega o primeiro que contém o tag e não tem "lmax"
            if tag in sc_tag.lower() and "lmax" not in sc_tag.lower():
                filtered.append(r)
                break

    if len(filtered) < 2:
        print(f"[WARN] Poucos cenários para plotar evolução de ρ ({len(filtered)})")
        return

    labels = [r.get("scenario_tag", "?").split("_", 2)[-1]
              if "_" in r.get("scenario_tag", "") else r.get("scenario_tag", "?")
              for r in filtered]
    rhos = [r.get("spearman_rho") for r in filtered]
    ps = [r.get("spearman_p") for r in filtered]

    # Remove cenários sem α (S0 com freeze_enc=True terá α constante)
    valid = [(i, l, rho, p) for i, (l, rho, p) in enumerate(zip(labels, rhos, ps))
             if rho is not None]

    if len(valid) < 2:
        print("[WARN] Dados insuficientes para o gráfico de evolução.")
        return

    idx, labels_v, rhos_v, ps_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in ps_v]
    bars = ax.bar(range(len(labels_v)), rhos_v, color=colors, alpha=0.85,
                  edgecolor="white", width=0.6)

    # Linha de significância p=0.05
    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.axhline(0.3, color="#27ae60", lw=1, ls="--", alpha=0.6,
               label="ρ=0.3 (efeito moderado)")

    # Anotações p-value
    for i, (rho, p) in enumerate(zip(rhos_v, ps_v)):
        if rho is not None:
            sig_str = f"p={p:.3f}" + (" *" if p < 0.05 else "")
            ax.text(i, rho + 0.01 if rho >= 0 else rho - 0.04,
                    sig_str, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels_v)))
    ax.set_xticklabels(labels_v, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ (SHAP vs α)", fontsize=10)
    ax.set_title(f"Evolução da Correlação SHAP vs α — {dataset.upper()}\n"
                 f"Ablation S0→S4  |  vermelho = p<0.05 significativo", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 1.0)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", alpha=0.85, label="p < 0.05 (significativo)"),
        Patch(color="#95a5a6", alpha=0.85, label="p ≥ 0.05 (não significativo)"),
    ], fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = out_dir / f"shap_rho_evolution_{dataset}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Evolução ρ: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SHAP vs α correlation analysis")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                        help="Dataset a analisar")
    parser.add_argument("--all", action="store_true",
                        help="Roda todos os datasets (cenário s4)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Roda S0→S4 para o dataset especificado")
    parser.add_argument("--scenario", default="s4",
                        help="Tag do cenário (default: s4)")
    parser.add_argument("--root", default=str(_PROJECT_ROOT),
                        help="Diretório raiz do projeto")
    parser.add_argument("--out", default=str(_OUT_DIR),
                        help="Diretório de saída")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Modo: todos os datasets, cenário único ─────────────────────────
    if args.all:
        datasets = list(DATASET_LOADERS.keys())
        all_results = []
        for ds in datasets:
            r = run_shap_analysis(ds, args.scenario, root, out_dir)
            if r:
                all_results.append(r)

        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("RESUMO — Correlação SHAP vs α")
            print(f"{'='*60}")
            print(f"{'Dataset':<20} {'ρ':>7} {'p':>8} {'Sig':>5} {'XGB AUC':>9}")
            print("-" * 55)
            for r in all_results:
                rho = r.get("spearman_rho")
                p = r.get("spearman_p")
                sig = "✓" if r.get("significant") else "✗"
                auc = r.get("xgb_auc", 0)
                rho_str = f"{rho:.3f}" if rho is not None else "  N/A"
                p_str = f"{p:.4f}" if p is not None else "    N/A"
                print(f"{r['dataset']:<20} {rho_str:>7} {p_str:>8} {sig:>5} {auc:>9.4f}")

        print(f"\n[DONE] Resultados em: {out_dir.resolve()}")
        return

    # ── Modo: todos os cenários S0→S4 para um dataset ─────────────────
    if args.all_scenarios:
        if not args.dataset:
            print("[ERRO] --all-scenarios requer --dataset")
            sys.exit(1)

        ds = args.dataset
        print(f"\n[INFO] Rodando S0→S4 para dataset: {ds}")

        # Detecta cenários disponíveis
        scenario_tags = get_scenario_tags(ds, root)
        if not scenario_tags:
            print(f"[ERRO] Nenhum cenário encontrado para {ds} em {root}")
            sys.exit(1)

        print(f"[INFO] Cenários encontrados: {scenario_tags}")

        # Filtra S0→S4 (exclui lmax e outros)
        main_scenarios = [t for t in scenario_tags
                          if any(f"s{i}" in t for i in range(5))
                          and "lmax" not in t]

        if not main_scenarios:
            main_scenarios = scenario_tags  # fallback: todos

        print(f"[INFO] Rodando: {main_scenarios}")

        scenario_results = []
        for sc_tag in main_scenarios:
            # Extrai o sufixo do cenário (ex: "higgs_s1_enc_param" → "s1")
            sc_key = None
            for i in range(5):
                if f"s{i}" in sc_tag:
                    sc_key = sc_tag
                    break
            if sc_key is None:
                continue

            r = run_shap_analysis(ds, sc_key, root, out_dir)
            if r:
                r["scenario_tag"] = sc_tag
                scenario_results.append(r)

        # Gráfico de evolução
        if scenario_results:
            plot_rho_evolution(scenario_results, ds, out_dir)

            # Tabela resumo
            print(f"\n{'='*70}")
            print(f"EVOLUÇÃO ρ — {ds.upper()} — S0→S4")
            print(f"{'='*70}")
            print(f"{'Cenário':<35} {'ρ':>7} {'p':>8} {'Sig':>5}")
            print("-" * 60)
            for r in scenario_results:
                sc = r.get("scenario_tag", r.get("scenario", "?"))[:34]
                rho = r.get("spearman_rho")
                p = r.get("spearman_p")
                sig = "✓" if r.get("significant") else "✗"
                rho_str = f"{rho:.3f}" if rho is not None else "  N/A"
                p_str = f"{p:.4f}" if p is not None else "    N/A"
                print(f"{sc:<35} {rho_str:>7} {p_str:>8} {sig:>5}")

        print(f"\n[DONE] Resultados em: {out_dir.resolve()}")
        return

    # ── Modo: dataset + cenário único ─────────────────────────────────
    if not args.dataset:
        parser.print_help()
        sys.exit(1)

    r = run_shap_analysis(args.dataset, args.scenario, root, out_dir)
    print(f"\n[DONE] Resultados em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()