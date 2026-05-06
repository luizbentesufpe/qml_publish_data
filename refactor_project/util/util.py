from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch

from refactor_project.data.banknote import FEATURE_NAMES


def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log_to_file(self, filename: str, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = self.log_dir / f"{filename}-{self.run_tag}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")

    def dump_json(self, name: str, obj: Any) -> str:
        path = self.log_dir / f"{name}-{self.run_tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return str(path)

    def return_log_dir(self, filename: str) -> str:
        path = self.log_dir
        return str(path)


def dump_run_metadata(
    logger: Logger, cfg, extra: Optional[Dict[str, Any]] = None, device: str = "cpu"
) -> str:
    import pennylane as qml

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "pennylane_version": getattr(qml, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "cfg": cfg.__dict__,
    }
    if extra:
        meta.update(extra)
    path = logger.dump_json("run_meta", meta)
    logger.log_to_file("results", f"[META] {path}")
    return path


def save_circuit_image(model, x_ref: np.ndarray, out_path: str, title: str = ""):
    """
    Salva o circuito do QNode do modelo como imagem (PNG/PDF).

    FIX-E: três bugs corrigidos:
      1. _q_single → _qnode  (atributo real do CQV_End2End)
      2. drawer recebe os 4 argumentos de circuit():
         (xi, theta_vec, enc_alpha_raw, enc_beta_raw)
      3. o model deve usar diff_method='backprop' (default.qubit);
         lightning não suporta draw_mpl — ver train.py para o model
         temporário que garante isso.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pennylane as qml
    import torch

    # --- x em CPU ---
    x = np.asarray(x_ref, dtype=np.float32).reshape(-1)
    x_t = torch.as_tensor(x, dtype=torch.float32, device="cpu")

    # --- theta (parâmetros variacionais ROT) ---
    theta = getattr(model, "theta", None)
    if theta is None:
        raise AttributeError("Model has no attribute 'theta'")
    theta_t = theta.detach().cpu()

    # --- enc affine params (alpha_raw, beta_raw) ---
    enc_alpha_raw = getattr(model, "enc_alpha_raw", None)
    enc_beta_raw = getattr(model, "enc_beta_raw", None)
    if enc_alpha_raw is None or enc_beta_raw is None:
        raise AttributeError("Model has no enc_alpha_raw / enc_beta_raw")
    alpha_t = enc_alpha_raw.detach().cpu()
    beta_t = enc_beta_raw.detach().cpu()

    # --- FIX-E bug 1: _q_single não existe; usar _qnode ---
    qnode = getattr(model, "_q_single", None) or getattr(model, "_qnode", None)
    if qnode is None:
        raise AttributeError(
            "Model has no drawable QNode (_q_single or _qnode). "
            "Instancie o model com diff_method='backprop' (default.qubit)."
        )

    # --- FIX-E bug 3: passar os 4 argumentos de circuit() ---
    drawer = qml.draw_mpl(qnode, decimals=2, max_length=200)
    out = drawer(x_t, theta_t, alpha_t, beta_t)

    # --- normaliza retorno ---
    figs = []

    if isinstance(out, tuple):
        # caso clássico: (fig, ax)
        figs = [out[0]]

    elif isinstance(out, list):
        # lista de (fig, ax) OU lista de fig
        for item in out:
            if isinstance(item, tuple):
                figs.append(item[0])
            else:
                figs.append(item)

    else:
        # retorno direto de Figure
        figs = [out]

    # --- salva todas (ou só a primeira, se preferir) ---
    for i, fig in enumerate(figs):
        if title:
            try:
                fig.suptitle(title)
            except Exception:
                pass

        fig.tight_layout()

        # se houver múltiplas figuras, adiciona sufixo
        if len(figs) > 1:
            path = out_path.replace(".png", f"_part{i}.png")
        else:
            path = out_path

        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-12
    return float((x * y).sum() / den)


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx, ry)


def _plot_alpha_bar_moons(
    alpha_per_seed: List[Optional[np.ndarray]],
    out_path: str,
    title: str = "α convergido — Make Moons (2 features)",
    feature_names: Tuple[str, str] = ("x0", "x1"),
) -> None:
    """
    Barplot do α convergido para 2 features, com erro entre seeds.
    Análogo ao _plot_alpha_heatmap_3x3 mas para inputs 1D.

    Salva PNG em out_path.
    """
    arrays = [np.asarray(a, dtype=np.float32).flatten() for a in alpha_per_seed if a is not None]
    if not arrays:
        print("[WARN] Nenhum α disponível para barplot.")
        return

    stacked = np.stack(arrays, axis=0)  # (n_seeds, 2)
    alpha_mean = stacked.mean(axis=0)  # (2,)
    alpha_std = stacked.std(axis=0)  # (2,)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── barplot α médio ──────────────────────────────────────────────
    ax = axes[0]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(
        feature_names,
        alpha_mean,
        color=colors,
        yerr=alpha_std,
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="init (α=1.0)")
    ax.set_ylim(0.0, max(alpha_mean.max() * 1.3, 0.8))
    ax.set_title("α médio (entre seeds)", fontsize=11)
    ax.set_xlabel("Feature")
    ax.set_ylabel("α convergido")
    ax.legend(fontsize=9)
    for bar, m, s in zip(bars, alpha_mean, alpha_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.02,
            f"{m:.3f}±{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ── std entre seeds ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(
        feature_names, alpha_std, color=["#9ecae1", "#fdae6b"], edgecolor="black", linewidth=0.8
    )
    ax2.set_title("std(α) entre seeds", fontsize=11)
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("std(α)")
    for i, s in enumerate(alpha_std):
        ax2.text(i, s + 0.002, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barplot α salvo: {out_path}")


def _plot_alpha_bar_banknote(
    alpha_per_seed: List[Optional[np.ndarray]],
    out_path: str,
    title: str = "α convergido — Banknote Authentication (4 features)",
    feature_names: List[str] = FEATURE_NAMES,
) -> None:
    """
    Barplot do α convergido para 4 features wavelet, com erro entre seeds.

    Layout:
      - Painel esquerdo : α médio com barra de erro (std entre seeds)
                          Linha tracejada em α=0.5 (valor de inicialização)
      - Painel direito  : std(α) entre seeds por feature

    Salva PNG em out_path.
    """
    arrays = [
        np.asarray(a, dtype=np.float32).flatten()[:4] for a in alpha_per_seed if a is not None
    ]
    if not arrays:
        print("[WARN] Nenhum α disponível para barplot Banknote.")
        return

    stacked = np.stack(arrays, axis=0)  # (n_seeds, 4)
    alpha_mean = stacked.mean(axis=0)  # (4,)
    alpha_std = stacked.std(axis=0)  # (4,)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x_pos = np.arange(len(feature_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── painel esquerdo: α médio ──────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        x_pos,
        alpha_mean,
        color=colors,
        yerr=alpha_std,
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="init (α=1.0)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.0, max(float(alpha_mean.max()) * 1.35, 0.8))
    ax.set_title("α médio (entre seeds)", fontsize=11)
    ax.set_ylabel("α convergido")
    ax.legend(fontsize=9)
    for bar, m, s in zip(bars, alpha_mean, alpha_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(m) + float(s) + 0.02,
            f"{m:.3f}±{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ── painel direito: std entre seeds ───────────────────────────────
    ax2 = axes[1]
    light_colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#fc9272"]
    ax2.bar(
        x_pos,
        alpha_std,
        color=light_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=15, ha="right", fontsize=9)
    ax2.set_title("std(α) entre seeds", fontsize=11)
    ax2.set_ylabel("std(α)")
    for i, s in enumerate(alpha_std):
        ax2.text(i, float(s) + 0.002, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barplot α salvo: {out_path}")

def _plot_alpha_bar_breast_cancer(
    alpha_per_seed: List[Optional[np.ndarray]],
    out_path: str,
    title: str = "α convergido — Breast Cancer Wisconsin",
    feature_names: List[str] = None,
) -> None:
    """
    Barplot do α convergido com suporte a qualquer número de features.
    
    Layout:
      - Painel esquerdo : α médio com barra de erro (std entre seeds)
                          Linha tracejada em α=0.5 (valor de inicialização)
      - Painel direito  : std(α) entre seeds por feature
    
    Salva PNG em out_path.
    
    Parâmetros
    ----------
    alpha_per_seed : List[Optional[np.ndarray]]
        Lista de arrays α para cada seed. Cada array tem shape (n_features,).
    out_path : str
        Caminho para salvar a figura PNG.
    title : str
        Título da figura.
    feature_names : List[str], optional
        Nomes das features. Se None, usa FEATURE_NAMES ou gera automaticamente.
    """
    # Extrai arrays α de cada seed
    arrays = [
        np.asarray(a, dtype=np.float32).flatten()
        for a in alpha_per_seed
        if a is not None
    ]
 
    if not arrays:
        print("[WARN] Nenhum α disponível para barplot Breast Cancer.")
        return
 
    # Determina número de features a partir dos dados
    n_features = len(arrays[0])
    
    # Extrai apenas n_features de cada array (em caso de mismatch)
    arrays = [a[:n_features] for a in arrays]
    
    # Ajusta feature_names se necessário
    if feature_names is None:
        if n_features <= len(FEATURE_NAMES):
            feature_names = FEATURE_NAMES[:n_features]
        else:
            feature_names = [f"f{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        feature_names = [f"f{i}" for i in range(n_features)]
    
    stacked = np.stack(arrays, axis=0)  # (n_seeds, n_features)
    alpha_mean = stacked.mean(axis=0)  # (n_features,)
    alpha_std = stacked.std(axis=0)  # (n_features,)
 
    # Paleta de cores: começa com 9, repete se necessário
    base_colors = [
        "#4C72B0",  # blue
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B3",  # purple
        "#937860",  # brown
        "#DA8BC3",  # pink
        "#8C6D31",  # olive
        "#466592",  # dark blue
    ]
    
    # Repete paleta se tiver mais features que cores
    colors = [base_colors[i % len(base_colors)] for i in range(n_features)]
    
    # Paleta de cores claras
    base_light_colors = [
        "#9ecae1",  # light blue
        "#fdae6b",  # light orange
        "#a1d99b",  # light green
        "#fc9272",  # light red
        "#c7b9d4",  # light purple
        "#c8b8a0",  # light brown
        "#f1b9d9",  # light pink
        "#d4c0a0",  # light olive
        "#a8c2e1",  # light dark blue
    ]
    
    light_colors = [base_light_colors[i % len(base_light_colors)] for i in range(n_features)]
 
    x_pos = np.arange(n_features)
    
    # Ajusta figsize baseado no número de features
    figwidth = max(12, 2 * n_features)
    fig, axes = plt.subplots(1, 2, figsize=(figwidth, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    # ── painel esquerdo: α médio ──────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        x_pos,
        alpha_mean,
        color=colors,
        yerr=alpha_std,
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="init (α=1.0)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0.0, max(float(alpha_mean.max()) * 1.35, 0.8))
    ax.set_title("α médio (entre seeds)", fontsize=11)
    ax.set_ylabel("α convergido", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
 
    # Adiciona valores nas barras
    for bar, m, s in zip(bars, alpha_mean, alpha_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(m) + float(s) + 0.02,
            f"{m:.3f}±{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
 
    # ── painel direito: std entre seeds ───────────────────────────────
    ax2 = axes[1]
    ax2.bar(
        x_pos,
        alpha_std,
        color=light_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax2.set_title("std(α) entre seeds", fontsize=11)
    ax2.set_ylabel("std(α)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
 
    # Adiciona valores de std
    for i, s in enumerate(alpha_std):
        ax2.text(
            i,
            float(s) + 0.003,
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
 
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barplot α salvo: {out_path}")


def _plot_alpha_bar_higgs(
    alpha_per_seed: List[Optional[np.ndarray]],
    out_path: str,
    title: str = "α convergido — HIGGS Boson (28 features físicas)",
    feature_names: Optional[List[str]] = None,
    low_level_indices: Optional[List[int]] = None,
    high_level_indices: Optional[List[int]] = None,
    alpha_init: float = 0.5,
) -> None:
    """
    Barplot do α convergido para 28 features físicas do HIGGS Boson,
    com erro entre seeds e separação visual low-level vs high-level.

    Layout (barras horizontais para acomodar d=28):
      - Painel esquerdo : α médio com barra de erro (std entre seeds)
                          Linha tracejada em α=alpha_init (default 0.5)
                          Coloração por grupo: cinza=low-level, vermelho=high-level
      - Painel direito  : std(α) entre seeds por feature

    Caixa de estatísticas agregadas no painel esquerdo:
        mean(α[LL])  vs  mean(α[HL])  e  ratio HL/LL.
    Esta razão é o diagnóstico físico-chave: HL/LL >> 1 indica que o
    agente recuperou (sem supervisão) a hierarquia conhecida em que
    massas invariantes derivadas (high-level) são mais discriminativas
    que cinemáticas diretas (low-level) — Baldi et al. 2014.

    Parâmetros
    ----------
    alpha_per_seed       : lista de arrays (28,) — um α por seed
    out_path             : path do PNG de saída
    title                : título do plot
    feature_names        : nomes das 28 features (default: importa de higgs.py)
    low_level_indices    : índices low-level (default: [0..20])
    high_level_indices   : índices high-level (default: [21..27])
    alpha_init           : valor de inicialização (default 0.5 para S1-S4 do Higgs;
                           use 1.0 apenas se plotar S0 com encoding congelado)
    """
    # Import tardio para evitar dependência circular se util.py for
    # importado antes de higgs.py
    if feature_names is None:
        try:
            from refactor_project.data.higgs import FEATURE_NAMES as _HIGGS_FN
            feature_names = list(_HIGGS_FN)
        except Exception:
            feature_names = [f"f{i}" for i in range(28)]

    if low_level_indices is None:
        low_level_indices = list(range(0, 21))
    if high_level_indices is None:
        high_level_indices = list(range(21, 28))

    arrays = [
        np.asarray(a, dtype=np.float32).flatten()[:28]
        for a in alpha_per_seed
        if a is not None
    ]
    if not arrays:
        print("[WARN] Nenhum α disponível para barplot Higgs.")
        return

    # Filtra arrays que não têm exatamente 28 elementos (fail-soft)
    arrays = [a for a in arrays if a.size == 28]
    if not arrays:
        print("[WARN] Nenhum α com d=28 disponível — pulando barplot Higgs.")
        return

    stacked = np.stack(arrays, axis=0)  # (n_seeds, 28)
    alpha_mean = stacked.mean(axis=0)   # (28,)
    alpha_std = stacked.std(axis=0)     # (28,)
    n_seeds = stacked.shape[0]

    # Paleta por grupo físico
    LL_COLOR = "#6c7a89"   # cinza-azulado para low-level
    HL_COLOR = "#c0392b"   # vermelho-quente para high-level
    LL_LIGHT = "#aab2bd"   # versão clara para painel std
    HL_LIGHT = "#e8826b"

    n_feat = len(feature_names)
    bar_colors = [
        HL_COLOR if i in set(high_level_indices) else LL_COLOR
        for i in range(n_feat)
    ]
    bar_colors_light = [
        HL_LIGHT if i in set(high_level_indices) else LL_LIGHT
        for i in range(n_feat)
    ]
    y_pos = np.arange(n_feat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── painel esquerdo: α médio (barras horizontais) ─────────────────
    ax = axes[0]
    bars = ax.barh(
        y_pos,
        alpha_mean,
        color=bar_colors,
        xerr=alpha_std,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "alpha": 0.7},
    )
    ax.axvline(
        alpha_init,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"init (α={alpha_init})",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.invert_yaxis()  # idx 0 no topo (convenção física)
    ax.set_xlabel("α convergido")
    ax.set_title(f"α médio  ·  {n_seeds} seeds", fontsize=11)

    # Sombreamento de fundo separando grupos LL e HL
    if high_level_indices and len(high_level_indices) > 0:
        hl_min = min(high_level_indices) - 0.5
        hl_max = max(high_level_indices) + 0.5
        ax.axhspan(hl_min, hl_max, color=HL_COLOR, alpha=0.06, zorder=0)

    x_max = max(float((alpha_mean + alpha_std).max()) * 1.15, 1.0)
    ax.set_xlim(0.0, x_max)

    # Legenda manual de grupos
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=LL_COLOR, edgecolor="black", label="low-level (kinematic)"),
        Patch(facecolor=HL_COLOR, edgecolor="black", label="high-level (invariant mass)"),
    ]
    ax.legend(
        handles=legend_handles + [
            plt.Line2D([0], [0], color="gray", linestyle="--",
                       label=f"init (α={alpha_init})"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    # ── Caixa de estatísticas agregadas LL vs HL (resultado-chave) ────
    # Posicionada no canto inferior direito (longe da legenda upper-right)
    ll_alpha = alpha_mean[low_level_indices]
    hl_alpha = alpha_mean[high_level_indices]
    ll_m, ll_s = float(ll_alpha.mean()), float(ll_alpha.std())
    hl_m, hl_s = float(hl_alpha.mean()), float(hl_alpha.std())
    ratio = hl_m / ll_m if ll_m > 1e-6 else float("nan")

    stats_text = (
        f"$\\overline{{\\alpha}}_{{LL}}$ = {ll_m:.3f} ± {ll_s:.3f}\n"
        f"$\\overline{{\\alpha}}_{{HL}}$ = {hl_m:.3f} ± {hl_s:.3f}\n"
        f"ratio HL/LL = {ratio:.2f}"
    )
    ax.text(
        0.98, 0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
            boxstyle="round,pad=0.4",
        ),
        zorder=10,
    )

    # ── painel direito: std entre seeds ───────────────────────────────
    ax2 = axes[1]
    ax2.barh(
        y_pos,
        alpha_std,
        color=bar_colors_light,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("std(α)")
    ax2.set_title(f"std(α) entre seeds  ·  {n_seeds} seeds", fontsize=11)

    if high_level_indices and len(high_level_indices) > 0:
        hl_min = min(high_level_indices) - 0.5
        hl_max = max(high_level_indices) + 0.5
        ax2.axhspan(hl_min, hl_max, color=HL_COLOR, alpha=0.06, zorder=0)

    # Diagnóstico de barren plateau: features com std≈0 E mean≈init
    # são candidatas a colapso de gradiente.
    bp_threshold_std = 0.01
    bp_threshold_mean = 0.02
    for i, (m, s) in enumerate(zip(alpha_mean, alpha_std)):
        if s < bp_threshold_std and abs(m - alpha_init) < bp_threshold_mean:
            ax2.scatter(
                s + 0.001, i,
                marker="o",
                color="black",
                s=18,
                zorder=5,
            )
    # Anota o marcador de barren plateau na legenda (apenas se houver)
    bp_count = int(
        np.sum(
            (alpha_std < bp_threshold_std)
            & (np.abs(alpha_mean - alpha_init) < bp_threshold_mean)
        )
    )
    if bp_count > 0:
        ax2.scatter(
            [], [],
            marker="o",
            color="black",
            s=18,
            label=f"barren plateau candidate ({bp_count})",
        )
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Barplot α (Higgs) salvo: {out_path}")

    
class RunningStd:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 1e-6
        return math.sqrt(self.M2 / (self.n - 1))


class RunningPctl:
    """
    Simple running percentile using a fixed-size buffer.
    Good enough for reward normalization (p95 depth).

    Usage:
    rp = RunningPctl(p=95, maxlen=512)
    rp.update(x)
    ref = rp.value(default=1.0)
    """

    def __init__(self, p: float = 95.0, maxlen: int = 512):
        self.p = float(p)
        self.buf = deque(maxlen=int(maxlen))

    def update(self, x: float):
        self.buf.append(float(x))

    def value(self, default: float = 1.0) -> float:
        if len(self.buf) < 8:
            return float(default)
        arr = np.asarray(self.buf, dtype=np.float32)
        v = float(np.percentile(arr, self.p))
        if not np.isfinite(v) or v <= 1e-9:
            return float(default)
        return float(v)
