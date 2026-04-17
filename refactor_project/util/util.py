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
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="init (α=0.5)")
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
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="init (α=0.5)")
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
