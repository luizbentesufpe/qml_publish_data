"""
generate_paper_figures.py
==========================
Gera os 3 gráficos para o paper em PNGs separados:

  figure1_alpha_convergence.png   — Barplot de α convergido por dataset
  figure2_learning_curves.png     — Curvas de aprendizado S0 vs S1 vs S3
  figure3_feature_usage.png       — Heatmap de feature usage frequency

Uso:
  python generate_paper_figures.py
  python generate_paper_figures.py --dpi 300
  python generate_paper_figures.py --fig 1        # só figura 1
  python generate_paper_figures.py --fig 1 2 3    # todas

Requer: matplotlib, numpy, scipy, json
"""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── estilo global — publicação científica ───────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# ── OpType ──────────────────────────────────────────────────────────
ENC_VAL = 2
ROT_VAL = 1
CNOT_VAL = 3

# ── caminhos dos diretórios ─────────────────────────────────────────
DIRS = {
    "cross_circle": "publication_out_cross_circle",
    "make_moons": "publication_out_make_moons",
    "banknote": "publication_out_banknote",
}

SCENARIOS = {
    "cross_circle": [
        "cc_s0_baseline_minimum",
        "cc_s1_plus_enc_parametric",
        "cc_s2_plus_thr_soft_youden",
        "cc_s3_plus_hard_budget",
        "cc_s4_plus_focal",
    ],
    "make_moons": [
        "mm_s0_baseline",
        "mm_s1_enc_param",
        "mm_s2_thr_soft_youden",
        "mm_s3_hard_budget",
        "mm_s4_focal",
    ],
    "banknote": [
        "bn_s0_baseline",
        "bn_s1_enc_param",
        "bn_s2_thr_soft_youden",
        "bn_s3_hard_budget",
        "bn_s4_focal",
    ],
}

SCENARIO_LABELS = ["S0", "S1", "S2", "S3", "S4"]

FEATURE_NAMES = {
    "cross_circle": [f"$f_{i}$" for i in range(9)],
    "make_moons": ["$x_0$", "$x_1$"],
    "banknote": ["var", "skew", "curt", "entr"],
}

KEY_FEATURES = {
    "cross_circle": [4, 0],
    "make_moons": [1],  # x1 é o dominante (α=1.04 vs x0=0.56)
    "banknote": [0, 2],  # variance e curtosis
}

# ════════════════════════════════════════════════════════════════════
# UTILITÁRIOS — leitura dos JSONs
# ════════════════════════════════════════════════════════════════════


def load_runs(dataset: str, scenario: str) -> list:
    base = Path(DIRS[dataset])
    found = list((base / scenario).rglob("results_*.json"))
    if not found:
        print(f"  [DEBUG] Não encontrou JSON em: {base / scenario}")
        # tenta busca alternativa
        found = list((base / scenario).glob("*.json"))
        found = [f for f in found if "results_" in f.name]
    if not found:
        return []
    data = json.loads(found[0].read_text())
    runs = data.get("runs", [])
    seen, deduped = set(), []
    for run in runs:
        seed = (run.get("best_rlqcv") or {}).get("seed")
        if seed not in seen:
            seen.add(seed)
            deduped.append(run)
    return deduped


# Substitua a função extract_alphas no script por esta versão
# que tenta .pt primeiro, depois cai no JSON


def extract_alphas(runs: list, dataset: str = "", scenario: str = "") -> np.ndarray | None:
    """
    Extrai α por feature.
    Cross/Circle: lê dos .pt individuais por seed.
    Outros: lê do JSON agregado.
    """
    import torch

    # ── tenta .pt (Cross/Circle) ────────────────────────────────────
    if dataset and scenario:
        pt_dir = Path(DIRS[dataset]) / scenario / "nq4"
        if pt_dir.exists():
            alphas = []
            seen = set()
            for seed_dir in sorted(pt_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed_name = seed_dir.name
                if seed_name in seen:
                    continue
                pt_file = seed_dir / "logs" / "enc_params_final.pt"
                if not pt_file.exists():
                    continue
                try:
                    params = torch.load(
                        str(pt_file),
                        weights_only=False,
                        map_location="cpu",
                    )
                    a = params.get("alpha")
                    if a is not None:
                        alphas.append(list(a) if hasattr(a, "__iter__") else [float(a)])
                        seen.add(seed_name)
                except Exception:
                    pass
            if alphas:
                return np.array(alphas)

    # ── fallback: JSON agregado (Make Moons, Banknote) ───────────────
    alphas = []
    for run in runs:
        best = run.get("best_rlqcv") or {}
        ep = best.get("enc_params", {})
        a = ep.get("alpha")
        if a is not None:
            alphas.append(list(a))
    return np.array(alphas) if alphas else None


def extract_feature_usage(runs: list, n_features: int) -> np.ndarray:
    """Frequência normalizada de uso de cada feature nos ENC gates."""
    counts = np.zeros(n_features, dtype=int)
    for run in runs:
        best = run.get("best_rlqcv") or {}
        arch = best.get("arch_mat")
        if arch is None:
            continue
        arr = np.array(arch, dtype=np.int64)
        ops = arr[2, :]
        feat_idx = arr[4, :]
        for col in np.where(ops == ENC_VAL)[0]:
            fi = int(feat_idx[col]) - 1
            if 0 <= fi < n_features:
                counts[fi] += 1
    total = counts.sum()
    return counts / max(total, 1)


def extract_auc_per_seed(runs: list, dataset: str = "", scenario: str = "") -> list:
    import torch
    if dataset == "cross_circle" and scenario:
        pt_dir = Path(DIRS[dataset]) / scenario / "nq4"
        if pt_dir.exists():
            aucs, seen = [], set()
            for seed_dir in sorted(pt_dir.iterdir()):
                if not seed_dir.is_dir() or seed_dir.name in seen:
                    continue
                pt_file = seed_dir / "logs" / "enc_params_final.pt"
                if not pt_file.exists():
                    continue
                try:
                    p = torch.load(str(pt_file),
                                   weights_only=False,
                                   map_location="cpu")
                    auc = p.get("auc_te")
                    if auc is not None:
                        aucs.append(float(auc))
                        seen.add(seed_dir.name)
                except Exception:
                    pass
            if aucs:
                return aucs
    return [
        float((run.get("best_rlqcv") or {}).get("holdout", {}).get("auc", np.nan))
        for run in runs
    ]

# ════════════════════════════════════════════════════════════════════
# FIGURA 1 — Barplot de α convergido por dataset
# ════════════════════════════════════════════════════════════════════


def figure1_alpha_convergence(dpi: int = 300, out: str = "figure1_alpha_convergence.png"):
    """
    3 subplots: um por dataset.
    Barras mostram α mean ± std por feature para o cenário S1.
    Features-chave destacadas em vermelho.
    Linha tracejada em α=0.5 (inicialização / barren plateau).
    """
    # S1 é o cenário mais informativo para α
    sc_keys = {
        "cross_circle": "cc_s1_plus_enc_parametric",
        "make_moons": "mm_s1_enc_param",
        "banknote": "bn_s1_enc_param",
    }
    ds_titles = {
        "cross_circle": "Cross/Circle ($d=9$)",
        "make_moons": "Make Moons ($d=2$)",
        "banknote": "Banknote ($d=4$)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    fig.suptitle(
        r"Learned encoding parameters $\alpha_i$ under S1 (+Parametric Encoding)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    color_default = "#4C72B0"
    color_key = "#C44E52"
    color_bp = "#888888"

    for ax, (ds, sc_key) in zip(axes, sc_keys.items()):
        runs = load_runs(ds, sc_key)
        alphas = extract_alphas(runs, dataset=ds, scenario=sc_key)
        fnames = FEATURE_NAMES[ds]
        kf = KEY_FEATURES[ds]
        n_feat = len(fnames)

        if alphas is None or alphas.shape[1] != n_feat:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(ds_titles[ds])
            continue

        means = alphas.mean(axis=0)
        stds = alphas.std(axis=0)
        x = np.arange(n_feat)

        colors = [color_key if i in kf else color_default for i in range(n_feat)]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

        # linha de inicialização / barren plateau
        ax.axhline(
            0.5,
            color=color_bp,
            linestyle="--",
            linewidth=1.0,
            label=r"$\alpha_{\mathrm{init}}$ (barren plateau)",
        )

        # anotação do valor em cada barra
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(m) + float(s) + 0.02,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(fnames, fontsize=8)
        ax.set_ylim(0, max(means + stds) * 1.35 + 0.1)
        ax.set_ylabel(r"$\alpha_i$ (mean $\pm$ std)" if ax == axes[0] else "")
        ax.set_title(ds_titles[ds], fontsize=10)

        # legenda só no último subplot
        if ax == axes[-1]:
            legend_elements = [
                mpatches.Patch(facecolor=color_key, label="Key feature"),
                mpatches.Patch(facecolor=color_default, label="Other feature"),
                plt.Line2D(
                    [0], [0], color=color_bp, linestyle="--", label=r"$\alpha_{\mathrm{init}}=0.5$"
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ════════════════════════════════════════════════════════════════════
# FIGURA 2 — Curvas de aprendizado S0 vs S1 vs S3
# ════════════════════════════════════════════════════════════════════


def figure2_learning_curves(dpi: int = 300, out: str = "figure2_learning_curves.png"):
    """
    AUC holdout por seed para S0, S1, S3 nos 3 datasets.
    Stripplot + boxplot para mostrar distribuição entre seeds.
    """
    sc_map = {
        "cross_circle": {
            "S0": "cc_s0_baseline_minimum",
            "S1": "cc_s1_plus_enc_parametric",
            "S3": "cc_s3_plus_hard_budget",
        },
        "make_moons": {
            "S0": "mm_s0_baseline",
            "S1": "mm_s1_enc_param",
            "S3": "mm_s3_hard_budget",
        },
        "banknote": {
            "S0": "bn_s0_baseline",
            "S1": "bn_s1_enc_param",
            "S3": "bn_s3_hard_budget",
        },
    }

    ds_titles = {
        "cross_circle": "Cross/Circle ($d=9$)",
        "make_moons": "Make Moons ($d=2$)",
        "banknote": "Banknote ($d=4$)",
    }

    colors = {"S0": "#888888", "S1": "#4C72B0", "S3": "#2CA02C"}
    scenarios = ["S0", "S1", "S3"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)
    fig.suptitle(
        "Holdout AUC distribution across 5 seeds: S0 vs S1 vs S3",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    for ax, ds in zip(axes, ["cross_circle", "make_moons", "banknote"]):
        data_by_sc = {}
        for sc_label, sc_key in sc_map[ds].items():
            runs = load_runs(ds, sc_key)
            aucs = extract_auc_per_seed(runs, dataset=ds, scenario=sc_key)
            data_by_sc[sc_label] = [a for a in aucs if not np.isnan(a)]

        x_pos = np.arange(len(scenarios))
        bp_data = [data_by_sc[sc] for sc in scenarios]

        # boxplot
        bp = ax.boxplot(
            bp_data,
            positions=x_pos,
            widths=0.35,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, sc in zip(bp["boxes"], scenarios):
            patch.set_facecolor(colors[sc])
            patch.set_alpha(0.6)

        # stripplot (pontos individuais)
        rng = np.random.default_rng(42)
        for i, (sc, vals) in enumerate(zip(scenarios, bp_data)):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                x_pos[i] + jitter,
                vals,
                color=colors[sc],
                s=28,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_ylabel("Holdout AUC" if ax == axes[0] else "")
        ax.set_title(ds_titles[ds], fontsize=10)
        ax.set_ylim(
            max(0.5, min(min(v) for v in bp_data if v) - 0.05),
            1.02,
        )

        # linha de referência AUC=1.0
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # legenda global
    legend_elements = [
        mpatches.Patch(facecolor=colors[sc], alpha=0.7, label=sc) for sc in scenarios
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
        title="Ablation scenario",
    )

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ════════════════════════════════════════════════════════════════════
# FIGURA 3 — Heatmap de feature usage frequency
# ════════════════════════════════════════════════════════════════════


def figure3_feature_usage(dpi: int = 300, out: str = "figure3_feature_usage.png"):
    """
    Heatmap: linhas = cenários S0→S4, colunas = features.
    Intensidade = frequência normalizada de uso nos ENC gates.
    3 subplots: um por dataset.
    """
    ds_titles = {
        "cross_circle": "Cross/Circle ($d=9$)",
        "make_moons": "Make Moons ($d=2$)",
        "banknote": "Banknote ($d=4$)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    fig.suptitle(
        "Feature usage frequency in ENC gates (normalized, per scenario)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    for ax, ds in zip(axes, ["cross_circle", "make_moons", "banknote"]):
        fnames = FEATURE_NAMES[ds]
        kf = KEY_FEATURES[ds]
        n_feat = len(fnames)
        sc_keys = SCENARIOS[ds]
        usage_mat = np.zeros((len(sc_keys), n_feat))

        for i, sc_key in enumerate(sc_keys):
            runs = load_runs(ds, sc_key)
            if runs:
                usage_mat[i] = extract_feature_usage(runs, n_feat)

        im = ax.imshow(
            usage_mat,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=usage_mat.max() * 1.05 + 1e-6,
        )

        # anotações de valor em cada célula
        for i in range(len(sc_keys)):
            for j in range(n_feat):
                val = usage_mat[i, j]
                color = "white" if val > usage_mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        # destaque nas features-chave — borda vermelha
        for j in kf:
            for i in range(len(sc_keys)):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#C44E52",
                        linewidth=1.5,
                        zorder=3,
                    )
                )

        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(fnames, fontsize=8)
        ax.set_yticks(range(len(sc_keys)))
        ax.set_yticklabels(SCENARIO_LABELS, fontsize=8)
        ax.set_title(ds_titles[ds], fontsize=10)
        ax.set_xlabel("Feature", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Scenario", fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Freq.")

    # nota sobre destaque
    fig.text(
        0.5,
        -0.04,
        "Red borders indicate key features expected from domain knowledge.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Gera figuras do paper RL-QAS.")
    parser.add_argument(
        "--fig",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Figuras a gerar (default: 1 2 3).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolução dos PNGs (default: 300).")
    args = parser.parse_args()

    print(f"\n{'=' * 56}")
    print("  Gerando figuras do paper")
    print(f"  dpi={args.dpi}  figuras={args.fig}")
    print(f"{'=' * 56}\n")

    if 1 in args.fig:
        print("[FIG 1] α convergence barplot...")
        figure1_alpha_convergence(dpi=args.dpi)

    if 2 in args.fig:
        print("[FIG 2] AUC distribution S0/S1/S3...")
        figure2_learning_curves(dpi=args.dpi)

    if 3 in args.fig:
        print("[FIG 3] Feature usage heatmap...")
        figure3_feature_usage(dpi=args.dpi)

    print("\n[DONE]")
    print("Arquivos gerados:")
    for f in [
        "figure1_alpha_convergence.png",
        "figure2_learning_curves.png",
        "figure3_feature_usage.png",
    ]:
        p = Path(f)
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
