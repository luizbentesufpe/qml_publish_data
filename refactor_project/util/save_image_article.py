import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.pyplot as plt

# ── Cores do paper ────────────────────────────────────────────────────────────
COLOR_ENC = "#4C72B0"  # azul  — encoding data-dependent
COLOR_ROT = "#DD8452"  # laranja — rotação variacional
COLOR_CNOT = "#2d2d2d"  # preto  — CNOT
COLOR_WIRE = "#888888"  # cinza  — fio quântico
COLOR_BG = "#FAFAFA"  # fundo
COLOR_LABEL = "white"  # texto nas caixas

# ── OpType values (conforme actions.py) ───────────────────────────────────────
_NOP = 0
_ROT = 1
_ENC = 2
_CNOT = 3

# ── Axis names ────────────────────────────────────────────────────────────────
_AXIS = {1: "X", 2: "Y", 3: "Z"}


def _get_enc_angle(model, tgt: int, feat_idx: int, x: np.ndarray) -> float:
    """Calcula ângulo real do encoding para label da caixa."""
    try:
        enc_lambda = float(getattr(model, "enc_lambda", 1.0))
        enc_beta_max = float(getattr(model, "enc_beta_max", 1.0))
        mode = str(getattr(model, "enc_affine_mode", "per_feature"))

        alpha_raw = model.enc_alpha_raw.detach().cpu()
        beta_raw = model.enc_beta_raw.detach().cpu()

        if mode == "per_feature_qubit":
            a = F.softplus(alpha_raw[tgt, feat_idx]).item() + 1e-6
            b = torch.tanh(beta_raw[tgt, feat_idx]).item() * enc_beta_max
        elif mode == "per_feature":
            a = F.softplus(alpha_raw[feat_idx]).item() + 1e-6
            b = torch.tanh(beta_raw[feat_idx]).item() * enc_beta_max
        elif mode == "per_qubit":
            a = F.softplus(alpha_raw[tgt]).item() + 1e-6
            b = torch.tanh(beta_raw[tgt]).item() * enc_beta_max
        else:  # global
            a = F.softplus(alpha_raw).item() + 1e-6
            b = torch.tanh(beta_raw).item() * enc_beta_max

        xval = float(x[feat_idx]) if feat_idx < len(x) else 0.0
        angle = enc_lambda * (a * xval + b)
        return float(angle)
    except Exception:
        return float("nan")


def _get_rot_angle(model, layer_idx: int) -> float:
    """Retorna ângulo theta do ROT para label da caixa."""
    try:
        idx = model._rot_param_index.get(int(layer_idx), None)
        if idx is None:
            return float("nan")
        return float(model.theta[idx].detach().cpu().item())
    except Exception:
        return float("nan")


def save_circuit_image_paper(
    model,
    x_ref: np.ndarray,
    out_path: str,
    title: str = "",
    show_angle: bool = True,
    show_feature_idx: bool = True,
    figsize_per_col: float = 0.55,
    figsize_per_row: float = 1.0,
    dpi: int = 220,
    max_cols_per_fig: int = 40,
):
    """
    Salva PNG do circuito para publicação com ENC (azul) e ROT (laranja).

    Parâmetros
    ----------
    model            : BinaryCQV_End2End (ou compatível)
    x_ref            : array 1D com uma amostra de referência (para calcular ângulos ENC)
    out_path         : caminho do PNG de saída
    title            : título da figura
    show_angle       : exibe o ângulo numérico dentro da caixa
    show_feature_idx : exibe o índice da feature dentro da caixa ENC
    figsize_per_col  : largura por coluna (polegadas)
    figsize_per_row  : altura por qubit (polegadas)
    dpi              : resolução do PNG
    max_cols_per_fig : número máximo de colunas por figura (quebra em múltiplos PNGs)
    """
    arch = model.arch_mat.detach().cpu().numpy()  # (5, L)
    L = int(arch.shape[1])
    n_qubits = int(model.n_qubits)
    x = np.asarray(x_ref, dtype=np.float32).reshape(-1)

    # ── 1. Remove colunas NOP do final ────────────────────────────────────────
    last_active = L - 1
    while last_active > 0 and int(arch[2, last_active]) == _NOP:
        last_active -= 1
    arch = arch[:, : last_active + 1]
    L = int(arch.shape[1])

    # ── 2. Divide em blocos se circuito for muito largo ────────────────────────
    n_parts = max(1, int(np.ceil(L / max_cols_per_fig)))
    out_paths = []

    for part in range(n_parts):
        col_start = part * max_cols_per_fig
        col_end = min(L, col_start + max_cols_per_fig)
        arch_part = arch[:, col_start:col_end]

        part_path = out_path.replace(".png", f"_part{part}.png") if n_parts > 1 else out_path
        out_paths.append(part_path)

        _draw_part(
            arch_part=arch_part,
            col_offset=col_start,
            model=model,
            x=x,
            n_qubits=n_qubits,
            out_path=part_path,
            title=title if part == 0 else f"{title} (cont.)" if title else "",
            show_angle=show_angle,
            show_feature_idx=show_feature_idx,
            figsize_per_col=figsize_per_col,
            figsize_per_row=figsize_per_row,
            dpi=dpi,
            total_parts=n_parts,
            part_idx=part,
        )

    return out_paths


def _draw_part(
    arch_part,
    col_offset,
    model,
    x,
    n_qubits,
    out_path,
    title,
    show_angle,
    show_feature_idx,
    figsize_per_col,
    figsize_per_row,
    dpi,
    total_parts,
    part_idx,
):
    Lp = int(arch_part.shape[1])

    # ── Layout ────────────────────────────────────────────────────────────────
    margin_left = 0.8  # polegadas para labels dos qubits
    margin_right = 0.3
    margin_top = 0.6 + (0.3 if title else 0.0)
    margin_bot = 0.7  # legenda

    fig_w = margin_left + margin_right + Lp * figsize_per_col
    fig_h = margin_top + margin_bot + n_qubits * figsize_per_row

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(-0.5, Lp - 0.5)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ajusta margens em fração
    left_frac = margin_left / fig_w
    right_frac = 1.0 - margin_right / fig_w
    bot_frac = margin_bot / fig_h
    top_frac = 1.0 - margin_top / fig_h
    ax.set_position([left_frac, bot_frac, right_frac - left_frac, top_frac - bot_frac])

    # ── Coordenadas: qubit q → y = n_qubits - 1 - q  (q0 no topo) ──────────
    def qy(q):
        return float(n_qubits - 1 - q)

    BOX_W = 0.78
    BOX_H = 0.52
    FONT = max(5.5, 7.5 - 0.15 * Lp)  # escala fonte com nº de colunas

    # ── Fios horizontais ──────────────────────────────────────────────────────
    for q in range(n_qubits):
        y = qy(q)
        ax.plot(
            [-0.5, Lp - 0.5], [y, y], color=COLOR_WIRE, lw=1.2, zorder=1, solid_capstyle="round"
        )

    # ── Labels dos qubits ─────────────────────────────────────────────────────
    for q in range(n_qubits):
        ax.text(
            -0.55,
            qy(q),
            f"$q_{{{q}}}$",
            ha="right",
            va="center",
            fontsize=FONT + 1,
            color="#333333",
            fontweight="bold",
        )

    # ── Operações ─────────────────────────────────────────────────────────────
    for col in range(Lp):
        col_abs = col_offset + col  # índice absoluto (para _rot_param_index)
        op = int(arch_part[2, col])
        c = int(arch_part[0, col])
        t = int(arch_part[1, col])
        ax_ = int(arch_part[3, col])
        f1 = int(arch_part[4, col])

        x_pos = float(col)

        # ── ENC ───────────────────────────────────────────────────────────────
        if op == _ENC and t > 0 and ax_ > 0 and f1 > 0:
            tgt = t - 1
            feat_idx = f1 - 1
            axis_name = _AXIS.get(ax_, "?")

            # linha 1: "R{X/Y/Z}" ou "ENC·R{X/Y/Z}"
            label_top = f"R{axis_name}"

            # linha 2: feature index + ângulo (ambos)
            if show_angle and show_feature_idx:
                angle = _get_enc_angle(model, tgt, feat_idx, x)
                angle_str = f"{angle:.2f}" if np.isfinite(angle) else "?"
                label_bot = f"f{feat_idx} | {angle_str}"
            elif show_feature_idx:
                label_bot = f"f{feat_idx}"
            elif show_angle:
                angle = _get_enc_angle(model, tgt, feat_idx, x)
                label_bot = f"{angle:.2f}" if np.isfinite(angle) else "?"
            else:
                label_bot = ""

            _draw_box(
                ax,
                x_pos,
                qy(tgt),
                label_top=label_top,
                label_bot=label_bot,
                color=COLOR_ENC,
                box_w=BOX_W,
                box_h=BOX_H,
                font=FONT,
            )

            # mini-tag "ENC" no canto superior esquerdo da caixa
            ax.text(
                x_pos - BOX_W / 2 + 0.03,
                qy(tgt) + BOX_H / 2 - 0.04,
                "ENC",
                fontsize=FONT - 2.5,
                color="white",
                ha="left",
                va="top",
                style="italic",
                zorder=6,
            )

        # ── ROT ───────────────────────────────────────────────────────────────
        elif op == _ROT and t > 0 and ax_ > 0:
            tgt = t - 1
            axis_name = _AXIS.get(ax_, "?")
            label_top = f"R{axis_name}"

            if show_angle:
                angle = _get_rot_angle(model, col_abs)
                label_bot = f"{angle:.2f}" if np.isfinite(angle) else "?"
            else:
                label_bot = ""

            _draw_box(
                ax,
                x_pos,
                qy(tgt),
                label_top=label_top,
                label_bot=label_bot,
                color=COLOR_ROT,
                box_w=BOX_W,
                box_h=BOX_H,
                font=FONT,
            )

            # mini-tag "ROT"
            ax.text(
                x_pos - BOX_W / 2 + 0.03,
                qy(tgt) + BOX_H / 2 - 0.04,
                "ROT",
                fontsize=FONT - 2.5,
                color="white",
                ha="left",
                va="top",
                style="italic",
                zorder=6,
            )

        # ── CNOT ──────────────────────────────────────────────────────────────
        elif op == _CNOT and c > 0 and t > 0 and c != t:
            ctrl = c - 1
            tgt = t - 1
            yc = qy(ctrl)
            yt = qy(tgt)

            # linha vertical ligando controle ao alvo
            ax.plot([x_pos, x_pos], [yc, yt], color=COLOR_CNOT, lw=1.5, zorder=3)

            # ponto de controle (filled circle)
            ax.add_patch(Circle((x_pos, yc), radius=0.10, color=COLOR_CNOT, zorder=4))

            # alvo (⊕ = círculo vazio com +)
            ax.add_patch(
                Circle(
                    (x_pos, yt), radius=0.22, fill=False, edgecolor=COLOR_CNOT, lw=1.5, zorder=4
                )
            )
            ax.plot([x_pos - 0.22, x_pos + 0.22], [yt, yt], color=COLOR_CNOT, lw=1.2, zorder=5)
            ax.plot([x_pos, x_pos], [yt - 0.22, yt + 0.22], color=COLOR_CNOT, lw=1.2, zorder=5)

    # ── Título ────────────────────────────────────────────────────────────────
    if title:
        suffix = f" [{part_idx + 1}/{total_parts}]" if total_parts > 1 else ""
        fig.text(
            0.5,
            0.97,
            title + suffix,
            ha="center",
            va="top",
            fontsize=FONT + 2,
            fontweight="bold",
            color="#222222",
        )

    # ── Legenda ───────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(
            facecolor=COLOR_ENC, edgecolor="none", label="ENC — encoding parametrizado (α·x + β)"
        ),
        mpatches.Patch(
            facecolor=COLOR_ROT, edgecolor="none", label="ROT — rotação variacional (θ)"
        ),
        mpatches.Patch(facecolor=COLOR_CNOT, edgecolor="none", label="CNOT — entrelaçamento"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=FONT - 1,
        framealpha=0.0,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.2,
        handleheight=0.9,
    )

    # ── Salva ─────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)


def _draw_box(ax, x, y, label_top, label_bot, color, box_w, box_h, font):
    """Desenha uma caixa arredondada com duas linhas de texto."""
    box = FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.add_patch(box)

    if label_bot:
        # duas linhas: gate em cima, valor embaixo
        ax.text(
            x,
            y + box_h * 0.13,
            label_top,
            ha="center",
            va="center",
            fontsize=font,
            color=COLOR_LABEL,
            fontweight="bold",
            zorder=6,
        )
        ax.text(
            x,
            y - box_h * 0.22,
            label_bot,
            ha="center",
            va="center",
            fontsize=font - 1.5,
            color=COLOR_LABEL,
            zorder=6,
        )
    else:
        ax.text(
            x,
            y,
            label_top,
            ha="center",
            va="center",
            fontsize=font,
            color=COLOR_LABEL,
            fontweight="bold",
            zorder=6,
        )


def _plot_alpha_heatmap_3x3(
    alpha_per_seed: list,  # list of np.ndarray (input_dim,), um por seed
    out_path: str,
    title: str = "α finais — Cross/Circle (3×3 patches)",
    grid_shape: tuple = (3, 3),  # para Cross/Circle com 9 features
):
    """
    Gera heatmap 3×3 dos α médios entre seeds.
    Salva PNG em out_path.
    """
    import numpy as np

    arrays = [np.asarray(a, dtype=np.float32) for a in alpha_per_seed if a is not None]
    if len(arrays) == 0:
        print("[WARN] Nenhum α disponível para heatmap.")
        return

    # média entre seeds
    alpha_mean = np.mean(np.stack(arrays, axis=0), axis=0)  # (input_dim,)
    alpha_std = np.std(np.stack(arrays, axis=0), axis=0)

    rows, cols = grid_shape
    expected = rows * cols
    if alpha_mean.shape[0] != expected:
        print(f"[WARN] alpha shape {alpha_mean.shape[0]} ≠ grid {expected}. Truncando/padando.")
        alpha_mean = np.resize(alpha_mean, expected)
        alpha_std = np.resize(alpha_std, expected)

    alpha_grid = alpha_mean.reshape(rows, cols)
    std_grid = alpha_std.reshape(rows, cols)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- heatmap α médio ---
    ax = axes[0]
    im = ax.imshow(
        alpha_grid,
        cmap="Blues",
        aspect="auto",
        vmin=float(alpha_grid.min()),
        vmax=float(alpha_grid.max()),
    )
    ax.set_title("α médio (entre seeds)", fontsize=11)
    ax.set_xlabel("Coluna do patch")
    ax.set_ylabel("Linha do patch")
    for r in range(rows):
        for c in range(cols):
            ax.text(
                c,
                r,
                f"{alpha_grid[r, c]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if alpha_grid[r, c] > alpha_grid.mean() else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- heatmap std ---
    ax2 = axes[1]
    im2 = ax2.imshow(
        std_grid, cmap="Oranges", aspect="auto", vmin=0.0, vmax=float(std_grid.max()) + 1e-6
    )
    ax2.set_title("std(α) entre seeds", fontsize=11)
    ax2.set_xlabel("Coluna do patch")
    ax2.set_ylabel("Linha do patch")
    for r in range(rows):
        for c in range(cols):
            ax2.text(
                c,
                r,
                f"{std_grid[r, c]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if std_grid[r, c] > std_grid.mean() else "black",
            )
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Heatmap salvo: {out_path}")
