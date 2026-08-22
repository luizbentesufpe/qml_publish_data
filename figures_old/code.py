"""
extract_spearman_and_feature_usage.py
======================================
Extrai dos JSONs existentes:

  1. Spearman ρ — correlação proxy-to-final por dataset e cenário
     (best_proxy_rl vs nested_cv.auc_mean)

  2. Feature usage frequency — frequência de uso de cada feature
     nos ENC gates dos arch_mat descobertos pelo agente

Uso:
    python extract_spearman_and_feature_usage.py

Outputs:
    - Terminal: tabelas de Spearman e feature usage
    - spearman_results.json
    - feature_usage_results.json
    - feature_usage_<dataset>.png  (barplots por cenário)
"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

# ── OpType values — ajuste se necessário ────────────────────────────
ENC_VAL = 2
ROT_VAL = 1
CNOT_VAL = 3

REPORT_PATH = "reports/figures"

# ── Configuração dos datasets ────────────────────────────────────────
DATASETS = [
    {
        "name": "Cross/Circle",
        "base_dir": "publication_out_cross_circle",
        "scenarios": [
            "cc_s0_baseline_minimum",
            "cc_s1_plus_enc_parametric",
            "cc_s2_plus_thr_soft_youden",
            "cc_s3_plus_hard_budget",
            "cc_s4_plus_focal",
        ],
        "n_features": 9,
        "feature_names": [f"f{i}" for i in range(9)],
        "key_features": [4, 0],  # esperados: central e canto
    },
    {
        "name": "Make Moons",
        "base_dir": "publication_out_make_moons",
        "scenarios": [
            "mm_s0_baseline",
            "mm_s1_enc_param",
            "mm_s2_thr_soft_youden",
            "mm_s3_hard_budget",
            "mm_s4_focal",
        ],
        "n_features": 2,
        "feature_names": ["x0", "x1"],
        "key_features": [0, 1],
    },
    {
        "name": "Banknote",
        "base_dir": "publication_out_banknote",
        "scenarios": [
            "bn_s0_baseline",
            "bn_s1_enc_param",
            "bn_s2_thr_soft_youden",
            "bn_s3_hard_budget",
            "bn_s4_focal",
        ],
        "n_features": 4,
        "feature_names": ["variance", "skewness", "curtosis", "entropy"],
        "key_features": [0, 2],  # esperados: variance e curtosis
    },
]

# ════════════════════════════════════════════════════════════════════
# 1.  SPEARMAN CORRELATION
# ════════════════════════════════════════════════════════════════════


def compute_spearman(base_dir: str, scenarios: list) -> dict:
    """
    Calcula Spearman ρ entre best_proxy_rl e nested_cv.auc_mean
    agregando todos os cenários e seeds do dataset.
    """
    base = Path(base_dir)
    all_proxy = []
    all_final = []
    per_scenario = {}

    for sc in scenarios:
        # tenta dois padrões de nome de arquivo
        candidates = [
            base / sc / f"results_{sc}.json",
            base / sc / f"results_{sc.split('_', 1)[-1]}.json",
        ]
        f = next((c for c in candidates if c.exists()), None)
        if f is None:
            # procura qualquer results_*.json no diretório do cenário
            found = list((base / sc).rglob("results_*.json"))
            f = found[0] if found else None
        if f is None:
            print(f"  [SKIP] {sc} — JSON não encontrado")
            continue

        data = json.loads(f.read_text())
        sc_proxy, sc_final = [], []

        for run in data.get("runs", []):
            best = run.get("best_rlqcv") or {}
            proxy = best.get("best_proxy_rl")
            final = (best.get("nested_cv") or {}).get("auc_mean")
            if proxy is not None and final is not None:
                try:
                    p, fi = float(proxy), float(final)
                    if np.isfinite(p) and np.isfinite(fi):
                        sc_proxy.append(p)
                        sc_final.append(fi)
                        all_proxy.append(p)
                        all_final.append(fi)
                except (ValueError, TypeError):
                    pass

        if len(sc_proxy) >= 2:
            rho, pval = stats.spearmanr(sc_proxy, sc_final)
            per_scenario[sc] = {
                "n": len(sc_proxy),
                "rho": round(float(rho), 4),
                "pval": round(float(pval), 6),
                "proxy": sc_proxy,
                "final": sc_final,
            }
        else:
            per_scenario[sc] = {"n": len(sc_proxy), "rho": None, "pval": None}

    # correlação global (todos os cenários e seeds)
    global_rho, global_pval = (None, None)
    if len(all_proxy) >= 2:
        global_rho, global_pval = stats.spearmanr(all_proxy, all_final)
        global_rho = round(float(global_rho), 4)
        global_pval = round(float(global_pval), 6)

    return {
        "global": {
            "n": len(all_proxy),
            "rho": global_rho,
            "pval": global_pval,
        },
        "per_scenario": per_scenario,
    }


# ════════════════════════════════════════════════════════════════════
# 2.  FEATURE USAGE FREQUENCY
# ════════════════════════════════════════════════════════════════════


def compute_feature_usage(
    base_dir: str,
    scenarios: list,
    n_features: int,
) -> dict:
    """
    Para cada cenário, conta quantas vezes cada feature aparece
    nos ENC gates dos arch_mat das 5 seeds.

    Retorna frequência normalizada (soma = 1.0 por cenário).
    """
    base = Path(base_dir)
    result = {}

    for sc in scenarios:
        [
            base / sc / f"results_{sc}.json",
        ]
        found = list((base / sc).rglob("results_*.json"))
        f = found[0] if found else None
        if f is None:
            print(f"  [SKIP] {sc} — JSON não encontrado")
            continue

        data = json.loads(f.read_text())
        counts = np.zeros(n_features, dtype=int)
        total_seeds = 0

        for run in data.get("runs", []):
            best = run.get("best_rlqcv") or {}
            arch = best.get("arch_mat")
            if arch is None:
                continue
            total_seeds += 1
            arr = np.array(arch, dtype=np.int64)  # (5, L)
            ops = arr[2, :]  # OpType
            feat_idx = arr[4, :]  # feature (1-indexed)

            enc_cols = np.where(ops == ENC_VAL)[0]
            for col in enc_cols:
                fi = int(feat_idx[col]) - 1  # 0-indexed
                if 0 <= fi < n_features:
                    counts[fi] += 1

        total = counts.sum()
        freq = (counts / total).tolist() if total > 0 else [0.0] * n_features

        result[sc] = {
            "counts": counts.tolist(),
            "freq": [round(x, 4) for x in freq],
            "total_enc": int(total),
            "total_seeds": total_seeds,
        }

    return result


# ════════════════════════════════════════════════════════════════════
# 3.  VISUALIZAÇÃO — barplots de feature usage
# ════════════════════════════════════════════════════════════════════


def plot_feature_usage(
    usage: dict,
    scenarios: list,
    feature_names: list,
    dataset_name: str,
    out_path: str,
    key_features: list,
) -> None:
    """
    Barplot de feature usage frequency por cenário.
    Features-chave são destacadas em vermelho.
    """
    n_sc = len(scenarios)
    n_feat = len(feature_names)
    if n_sc == 0:
        return

    fig, axes = plt.subplots(1, n_sc, figsize=(4 * n_sc, 4), sharey=True)
    if n_sc == 1:
        axes = [axes]

    fig.suptitle(
        f"Feature Usage Frequency — {dataset_name}\n"
        f"(ENC gates in discovered circuits, 5 seeds each)",
        fontsize=12,
        fontweight="bold",
    )

    colors_default = "#4C72B0"
    colors_key = "#C44E52"

    for ax, sc in zip(axes, scenarios):
        sc_data = usage.get(sc)
        if sc_data is None:
            ax.set_title(sc.split("_", 2)[-1], fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue

        freq = sc_data["freq"]
        colors = [colors_key if i in key_features else colors_default for i in range(n_feat)]
        bars = ax.bar(feature_names, freq, color=colors, edgecolor="black", linewidth=0.7)

        for bar, v in zip(bars, freq):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(v) + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        sc_label = sc.split("_s")[1] if "_s" in sc else sc
        ax.set_title(f"S{sc_label[0]}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Frequency" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    # legenda
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors_key, label="Key feature (expected)"),
        Patch(facecolor=colors_default, label="Other feature"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barplot salvo: {out_path}")


# ════════════════════════════════════════════════════════════════════
# 4.  PRINT — tabelas no terminal
# ════════════════════════════════════════════════════════════════════


def print_spearman_table(spearman_results: dict) -> None:
    print(f"\n{'=' * 68}")
    print("  SPEARMAN ρ — proxy-to-final correlation")
    print("  (best_proxy_rl vs nested_cv.auc_mean)")
    print(f"{'=' * 68}")
    print(f"  {'Dataset':<20} {'Scenario':<35} {'n':>4} {'ρ':>7} {'p':>10}")
    print(f"  {'─' * 66}")

    for ds_name, ds_res in spearman_results.items():
        g = ds_res["global"]
        print(
            f"  {ds_name:<20} {'[ALL SCENARIOS]':<35} "
            f"{g['n']:>4} {g['rho']:>7.4f} {g['pval']:>10.6f}"
            f"  {'✓' if g['rho'] and g['rho'] > 0.7 else '✗'}"
        )
        for sc, res in ds_res["per_scenario"].items():
            if res.get("rho") is None:
                continue
            sc_short = sc.split("_s")[1][:12] if "_s" in sc else sc[:12]
            sig = "**" if res["pval"] < 0.01 else ("*" if res["pval"] < 0.05 else "")
            print(
                f"  {'':20} S{sc_short:<34} "
                f"{res['n']:>4} {res['rho']:>7.4f} "
                f"{res['pval']:>10.6f} {sig}"
            )
        print()


def print_feature_usage_table(
    usage_results: dict,
    feature_names: list,
    dataset_name: str,
    key_features: list,
) -> None:
    print(f"\n{'=' * 68}")
    print(f"  FEATURE USAGE FREQUENCY — {dataset_name}")
    print("  (normalized frequency in ENC gates of discovered circuits)")
    print(f"  Key features (expected): {[feature_names[i] for i in key_features]}")
    print(f"{'=' * 68}")
    header = f"  {'Scenario':<35}" + "".join(f"{n:>12}" for n in feature_names)
    print(header)
    print(f"  {'─' * 66}")

    for sc, data in usage_results.items():
        sc_short = sc[-20:] if len(sc) > 20 else sc
        freq_str = "".join(
            f"{'→' + str(round(v, 2)):>12}" if i in key_features else f"{round(v, 2):>12}"
            for i, v in enumerate(data["freq"])
        )
        print(f"  {sc_short:<35}{freq_str}  (n={data['total_enc']} ENC gates)")


# ════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════════


def main():
    all_spearman = {}
    all_feature_usage = {}

    for ds in DATASETS:
        name = ds["name"]
        base_dir = ds["base_dir"]
        scenarios = ds["scenarios"]
        n_features = ds["n_features"]
        feature_names = ds["feature_names"]
        key_features = ds["key_features"]

        base = Path(base_dir)
        if not base.exists():
            print(f"\n[SKIP] {name} — diretório '{base_dir}' não encontrado")
            continue

        print(f"\n{'█' * 68}")
        print(f"  Processando: {name}")
        print(f"{'█' * 68}")

        # ── Spearman ─────────────────────────────────────────────────
        print("\n  [1/2] Calculando Spearman ρ...")
        spearman = compute_spearman(base_dir, scenarios)
        all_spearman[name] = spearman
        g = spearman["global"]
        print(
            f"  → Global: n={g['n']}  ρ={g['rho']}  p={g['pval']}"
            f"  {'✓ forte' if g['rho'] and g['rho'] > 0.7 else '⚠ fraco'}"
        )

        # ── Feature usage ─────────────────────────────────────────────
        print("\n  [2/2] Calculando feature usage frequency...")
        usage = compute_feature_usage(base_dir, scenarios, n_features)
        all_feature_usage[name] = {
            "feature_names": feature_names,
            "key_features": key_features,
            "usage": usage,
        }

        # barplot
        plot_feature_usage(
            usage=usage,
            scenarios=scenarios,
            feature_names=feature_names,
            dataset_name=name,
            out_path=f"{REPORT_PATH}/feature_usage_{name.lower().replace(' ', '_').replace('/', '_')}.png",
            key_features=key_features,
        )

    # ── Tabelas no terminal ───────────────────────────────────────────
    print_spearman_table(all_spearman)

    for name, res in all_feature_usage.items():
        print_feature_usage_table(
            usage_results=res["usage"],
            feature_names=res["feature_names"],
            dataset_name=name,
            key_features=res["key_features"],
        )

    # ── Salva JSONs ───────────────────────────────────────────────────
    Path("spearman_results.json").write_text(json.dumps(all_spearman, indent=2, default=str))
    print("\n[OK] spearman_results.json")

    Path("feature_usage_results.json").write_text(
        json.dumps(all_feature_usage, indent=2, default=str)
    )
    print("[OK] feature_usage_results.json")

    # ── Resumo para o paper ───────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  RESUMO PARA O PAPER")
    print(f"{'=' * 68}")

    print("\n  §4.4 Proxy-to-final correlation:")
    for name, res in all_spearman.items():
        g = res["global"]
        if g["rho"] is not None:
            sig = "p < 0.01" if g["pval"] < 0.01 else f"p = {g['pval']:.4f}"
            print(f"    {name:<20}: Spearman ρ = {g['rho']:.3f} ({sig}, n = {g['n']})")

    print("\n  §4.5 Feature usage — S1 (enc param):")
    for name, res in all_feature_usage.items():
        fnames = res["feature_names"]
        kf = res["key_features"]
        s1_key = [k for k in res["usage"] if "s1" in k.lower() or "enc_param" in k.lower()]
        if not s1_key:
            continue
        s1_usage = res["usage"][s1_key[0]]
        freq = s1_usage["freq"]
        key_str = ", ".join(f"{fnames[i]}={freq[i]:.2f}" for i in kf)
        other_str = ", ".join(
            f"{fnames[i]}={freq[i]:.2f}" for i in range(len(fnames)) if i not in kf
        )
        print(f"    {name:<20}: key=[{key_str}]  other=[{other_str}]")

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
