"""
benchmark_minimal_circuit.py
==============================
Compara três circuitos baseline handcrafted contra o circuito produzido pelo agente RL
no dataset Cross/Circle.

Objetivo: estabelecer o "chão" — o menor circuito que um humano consegue
construir com conhecimento do problema — e usar isso como:
  1. Referência de tamanho (budget_k para o task encoder)
  2. Evidência de que o agente está superdimensionando o circuito

Uso:
    python benchmark_minimal_circuit.py

    # Para comparar com o arch_mat salvo pelo agente:
    python benchmark_minimal_circuit.py --agent_arch caminho/para/arch.npy

Outputs:
    - Tabela comparativa no terminal (AUC, n_cnots, n_gates, depth)
    - benchmark_results.json com os resultados completos
    - circuit_comparison.png com os circuitos desenhados (requer matplotlib)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# ── imports do projeto ────────────────────────────────────────────────────────
from rl_and_qml_in_clinical_images.dataset import create_circle_cross_dataset
from rl_and_qml_in_clinical_images.modeling.model import BinaryCQV_End2End
from rl_and_qml_in_clinical_images.rl.rl_config import Config
from rl_and_qml_in_clinical_images.rl.env import sanitize_architecture, find_threshold
from rl_and_qml_in_clinical_images.rl.actions import OpType


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTRUÇÃO DOS ARCH_MAT BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def make_arch_mat(ops: list[dict], n_qubits: int, L_max: int = 16) -> torch.Tensor:
    """
    Constrói um arch_mat a partir de uma lista de operações legível por humanos.

    Cada operação é um dict com:
        type: "ENC" | "ROT" | "CNOT" | "NOP"
        qubit: int (0-indexed externamente, convertido para 1-indexed internamente)
        axis: 1=X, 2=Y, 3=Z  (só para ENC e ROT)
        feature: int 0-indexed (só para ENC)
        ctrl: int 0-indexed   (só para CNOT)

    arch_mat rows:
        [0] ctrl qubit  (1-indexed, 0 = sem controle)
        [1] target qubit (1-indexed)
        [2] OpType value
        [3] axis (1=X, 2=Y, 3=Z)
        [4] feature index (1-indexed, 0 = sem feature)
    """
    arch = np.zeros((5, L_max), dtype=np.int64)
    for col, op in enumerate(ops):
        if col >= L_max:
            break
        kind = op["type"]
        if kind == "ENC":
            arch[0, col] = 0
            arch[1, col] = op["qubit"] + 1          # 1-indexed
            arch[2, col] = OpType.ENC.value
            arch[3, col] = op.get("axis", 2)         # default RY
            arch[4, col] = op["feature"] + 1         # 1-indexed
        elif kind == "ROT":
            arch[0, col] = 0
            arch[1, col] = op["qubit"] + 1
            arch[2, col] = OpType.ROT.value
            arch[3, col] = op.get("axis", 2)
            arch[4, col] = 0
        elif kind == "CNOT":
            arch[0, col] = op["ctrl"] + 1
            arch[1, col] = op["qubit"] + 1
            arch[2, col] = OpType.CNOT.value
            arch[3, col] = 0
            arch[4, col] = 0
    mat = torch.tensor(arch, dtype=torch.int64)
    return sanitize_architecture(mat, n_qubits)


def build_baselines(n_qubits: int = 4, L_max: int = 16) -> dict[str, torch.Tensor]:
    """
    Três baseline progressivos handcrafted para Cross/Circle.

    O dataset tem 9 features (3x3 grid). A ideia de cada baseline:

    B0 — Ultra-mínimo (2 CNOTs):
        Codifica só as 2 features mais discriminativas (pixel central
        e um pixel de borda), aplica 1 camada de ROT, 1 CNOT.
        Raciocínio: circle tem pixel [1,1] ativo, cross também — mas
        os cantos diferem. 2 features são suficientes para separação linear.

    B1 — Mínimo estrutural (3 CNOTs):
        Codifica as 4 features de borda (pixels [0,1],[1,0],[1,2],[2,1]),
        que definem o "+" do cross vs o "o" do circle. 1 CNOT por par.
        Inspirado na simetria do problema.

    B2 — Ansatz HW-efficient completo (1 CNOT por qubit adjacente):
        Codifica todas as 9 features (uma por qubit se n_q>=9, senão
        faz round-robin). Usa hardware-efficient ansatz padrão da
        literatura (Kandala et al., Nature 2017).
        Este é o "baseline justo" — o que a maioria dos papers usaria.
    """
    baselines = {}

    # ── B0: ultra-mínimo ──────────────────────────────────────────────────────
    # Features: 4=centro (mais informativo), 1=canto superior esquerdo
    # Só 2 qubits ativos, 1 CNOT
    baselines["B0_ultra_minimal"] = make_arch_mat([
        {"type": "ENC",  "qubit": 0, "axis": 2, "feature": 4},  # pixel centro
        {"type": "ENC",  "qubit": 1, "axis": 2, "feature": 1},  # pixel borda
        {"type": "ROT",  "qubit": 0, "axis": 2},
        {"type": "ROT",  "qubit": 1, "axis": 2},
        {"type": "CNOT", "qubit": 1, "ctrl": 0},
        {"type": "ROT",  "qubit": 0, "axis": 3},
        {"type": "ROT",  "qubit": 1, "axis": 3},
    ], n_qubits=n_qubits, L_max=L_max)

    # ── B1: mínimo estrutural ─────────────────────────────────────────────────
    # Features: bordas do 3x3 que definem circle vs cross
    # pixels: índices 1(top), 3(left), 5(right), 7(bottom)
    baselines["B1_structural_min"] = make_arch_mat([
        {"type": "ENC",  "qubit": 0, "axis": 2, "feature": 1},  # pixel top
        {"type": "ENC",  "qubit": 1, "axis": 2, "feature": 3},  # pixel left
        {"type": "ENC",  "qubit": 2, "axis": 2, "feature": 5},  # pixel right
        {"type": "ENC",  "qubit": 3, "axis": 2, "feature": 7},  # pixel bottom
        {"type": "ROT",  "qubit": 0, "axis": 2},
        {"type": "ROT",  "qubit": 1, "axis": 2},
        {"type": "ROT",  "qubit": 2, "axis": 2},
        {"type": "ROT",  "qubit": 3, "axis": 2},
        {"type": "CNOT", "qubit": 1, "ctrl": 0},
        {"type": "CNOT", "qubit": 3, "ctrl": 2},
        {"type": "ROT",  "qubit": 0, "axis": 3},
        {"type": "ROT",  "qubit": 1, "axis": 3},
        {"type": "ROT",  "qubit": 2, "axis": 3},
        {"type": "ROT",  "qubit": 3, "axis": 3},
    ], n_qubits=n_qubits, L_max=L_max)

    # ── B2: HW-efficient ansatz (Kandala et al. 2017) ─────────────────────────
    # Encoding round-robin de todas as 9 features nos 4 qubits
    # depois 1 camada de CNOT em cadeia (nearest-neighbor)
    hw_ops = []
    n_feat = 9
    for q in range(n_qubits):
        feat = q % n_feat
        hw_ops.append({"type": "ENC", "qubit": q, "axis": 2, "feature": feat})
    for q in range(n_qubits):
        hw_ops.append({"type": "ROT", "qubit": q, "axis": 1})
        hw_ops.append({"type": "ROT", "qubit": q, "axis": 2})
    for q in range(n_qubits - 1):
        hw_ops.append({"type": "CNOT", "qubit": q + 1, "ctrl": q})
    for q in range(n_qubits):
        hw_ops.append({"type": "ROT", "qubit": q, "axis": 2})
        hw_ops.append({"type": "ROT", "qubit": q, "axis": 3})

    baselines["B2_hw_efficient"] = make_arch_mat(hw_ops, n_qubits=n_qubits, L_max=L_max)

    return baselines


# ══════════════════════════════════════════════════════════════════════════════
# 2. AVALIAÇÃO DO CIRCUITO
# ══════════════════════════════════════════════════════════════════════════════

def count_ops(arch_mat: torch.Tensor) -> dict:
    ops = arch_mat[2, :].numpy()
    # Conta só colunas não-zero (operações reais)
    n_cnot = int(np.sum(ops == OpType.CNOT.value))
    n_enc  = int(np.sum(ops == OpType.ENC.value))
    n_rot  = int(np.sum(ops == OpType.ROT.value))
    n_total = n_cnot + n_enc + n_rot
    # Depth = número de colunas até o último gate não-NOP
    nonzero_cols = np.where(ops != OpType.NOP.value)[0]
    depth = int(nonzero_cols[-1] + 1) if len(nonzero_cols) > 0 else 0
    return {
        "n_cnot": n_cnot,
        "n_enc": n_enc,
        "n_rot": n_rot,
        "n_gates_total": n_total,
        "depth": depth,
    }


def train_and_eval(
    arch_mat: torch.Tensor,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    n_qubits: int,
    cfg: Config,
    n_epochs: int = 150,
    lr: float = 0.02,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """
    Treina o modelo com o arch_mat fixo (end-to-end: theta + alpha/beta)
    e retorna AUC no conjunto de validação.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = int(X_tr.shape[1])

    model = BinaryCQV_End2End(
        arch_mat=arch_mat,
        n_qubits=n_qubits,
        enc_lambda=float(cfg.enc_lambda),
        diff_method="backprop",          # mais rápido para benchmark
        input_dim=input_dim,
        enc_affine_mode=str(cfg.enc_affine_mode),
        enc_alpha_init=float(cfg.enc_alpha_init),
        enc_beta_init=float(cfg.enc_beta_init),
        enc_beta_max=float(cfg.enc_beta_max),
    ).to(device)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(Y_tr.reshape(-1), dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    best_epoch = 0
    loss_history = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr_t).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, Ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

        if (epoch + 1) % 25 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva_t).squeeze(-1)
                probs_va = torch.sigmoid(logits_va).cpu().numpy()
            y_va_int = (Y_va.reshape(-1) > 0.5).astype(int)
            if len(np.unique(y_va_int)) > 1:
                auc = float(roc_auc_score(y_va_int, probs_va))
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1

    # Threshold calibrado no treino
    model.eval()
    with torch.no_grad():
        probs_tr = torch.sigmoid(model(Xtr_t).squeeze(-1)).cpu().numpy()
        probs_va = torch.sigmoid(model(Xva_t).squeeze(-1)).cpu().numpy()

    y_tr_int = (Y_tr.reshape(-1) > 0.5).astype(int)
    y_va_int = (Y_va.reshape(-1) > 0.5).astype(int)

    thr_star = find_threshold(y_tr_int, probs_tr, mode="soft")
    auc_final = float(roc_auc_score(y_va_int, probs_va)) if len(np.unique(y_va_int)) > 1 else 0.5
    sens_thr = float(np.mean((probs_va >= thr_star) == (y_va_int == 1)))

    # Alpha convergido (evidência de aprendizado do encoding)
    alpha_vals = None
    if hasattr(model, "enc_alpha_raw"):
        alpha_vals = F.softplus(model.enc_alpha_raw).detach().cpu().numpy().tolist()

    return {
        "auc": round(auc_final, 4),
        "best_auc": round(best_auc, 4),
        "best_epoch": best_epoch,
        "thr_star": round(float(thr_star), 4),
        "sens_at_thr": round(sens_thr, 4),
        "final_loss": round(loss_history[-1], 4),
        "alpha_converged": alpha_vals,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_arch", type=str, default=None,
                        help="Caminho para arch_mat.npy salvo pelo agente RL")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Amostras por classe no dataset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("  BENCHMARK — Circuito mínimo vs agente RL")
    print(f"  Device: {device} | n_qubits: {args.n_qubits} | seeds: {args.seeds}")
    print(f"{'='*60}\n")

    cfg = Config()

    # ── Dataset ───────────────────────────────────────────────────────────────
    X, Y = create_circle_cross_dataset(
        n_samples_per_class=args.n_samples,
        noise_std=0.1,
        seed=42,
    )
    print(f"[DATA] X={X.shape}, Y={Y.shape}  "
          f"pos={int(Y.sum())}  neg={int((1-Y).sum())}  "
          f"ratio={float(Y.mean()):.2f}\n")

    # ── Baselines handcrafted ─────────────────────────────────────────────────
    baselines = build_baselines(n_qubits=args.n_qubits)

    # Adiciona arch do agente se fornecida
    if args.agent_arch:
        p = Path(args.agent_arch)
        if p.exists():
            agent_arr = np.load(str(p))
            agent_mat = torch.tensor(agent_arr, dtype=torch.int64)
            agent_mat = sanitize_architecture(agent_mat, args.n_qubits)
            baselines["AGENT_RL"] = agent_mat
            print(f"[AGENT] Carregado arch_mat de {p}\n")
        else:
            print(f"[WARN] Arquivo não encontrado: {p}\n")

    # ── Avaliação cross-seed ──────────────────────────────────────────────────
    results = {}

    for name, arch_mat in baselines.items():
        ops_info = count_ops(arch_mat)
        print(f"{'─'*50}")
        print(f"[{name}]")
        print(f"  Gates: ENC={ops_info['n_enc']}  ROT={ops_info['n_rot']}  "
              f"CNOT={ops_info['n_cnot']}  total={ops_info['n_gates_total']}  "
              f"depth={ops_info['depth']}")

        seed_results = []
        for seed in args.seeds:
            # Split estratificado 80/20
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            y_int = (Y.reshape(-1) > 0.5).astype(int)
            tr_idx, va_idx = next(sss.split(X, y_int))
            X_tr, Y_tr = X[tr_idx], Y[tr_idx]
            X_va, Y_va = X[va_idx], Y[va_idx]

            res = train_and_eval(
                arch_mat, X_tr, Y_tr, X_va, Y_va,
                n_qubits=args.n_qubits,
                cfg=cfg,
                n_epochs=args.n_epochs,
                device=device,
                seed=seed,
            )
            seed_results.append(res)
            print(f"  seed={seed}  AUC={res['auc']:.4f}  "
                  f"best_AUC={res['best_auc']:.4f}@ep{res['best_epoch']}  "
                  f"loss={res['final_loss']:.4f}  thr*={res['thr_star']:.3f}")

        aucs = [r["auc"] for r in seed_results]
        best_aucs = [r["best_auc"] for r in seed_results]
        print(f"\n  → AUC média={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  → best AUC média={np.mean(best_aucs):.4f} ± {np.std(best_aucs):.4f}")

        # Alpha convergido da última seed (evidência de encoding)
        last_alpha = seed_results[-1].get("alpha_converged")
        if last_alpha is not None:
            alpha_arr = np.array(last_alpha)
            print(f"  → alpha convergido: mean={alpha_arr.mean():.3f}  "
                  f"std={alpha_arr.std():.3f}  "
                  f"min={alpha_arr.min():.3f}  max={alpha_arr.max():.3f}")

        results[name] = {
            "circuit": ops_info,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "best_auc_mean": float(np.mean(best_aucs)),
            "best_auc_std": float(np.std(best_aucs)),
            "per_seed": seed_results,
        }

    # ── Tabela comparativa ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TABELA COMPARATIVA")
    print(f"{'='*60}")
    header = f"{'Circuito':<24} {'CNOTs':>6} {'Gates':>6} {'Depth':>6} {'AUC':>8} {'±':>6}"
    print(header)
    print("─" * len(header))

    for name, r in results.items():
        c = r["circuit"]
        print(f"{name:<24} {c['n_cnot']:>6} {c['n_gates_total']:>6} {c['depth']:>6} "
              f"{r['auc_mean']:>8.4f} {r['auc_std']:>6.4f}")

    # ── Recomendação de budget_k ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RECOMENDAÇÃO DE budget_k PARA TASK ENCODER")
    print(f"{'='*60}")

    # Encontra o menor circuito que atinge AUC >= 0.90
    min_cnot_above_threshold = None
    auc_threshold = 0.90
    for name, r in results.items():
        if name == "AGENT_RL":
            continue
        if r["auc_mean"] >= auc_threshold:
            c = r["circuit"]["n_cnot"]
            if min_cnot_above_threshold is None or c < min_cnot_above_threshold:
                min_cnot_above_threshold = c
                best_baseline = name

    if min_cnot_above_threshold is not None:
        print(f"\n  Menor circuito com AUC >= {auc_threshold}: {best_baseline}")
        print(f"  → budget_k recomendado para Cross/Circle = {min_cnot_above_threshold} CNOTs")
        print(f"\n  Use isso como supervisão direta do task encoder:")
        print(f"    encoder(stats_cross_circle) → budget_k = {min_cnot_above_threshold}")
    else:
        print(f"  Nenhum baseline atingiu AUC >= {auc_threshold} nos {args.seeds} seeds testados.")
        print("  Aumente n_epochs ou n_samples e re-execute.")

    if "AGENT_RL" in results:
        agent_cnots = results["AGENT_RL"]["circuit"]["n_cnot"]
        agent_auc = results["AGENT_RL"]["auc_mean"]
        print(f"\n  Agente RL: {agent_cnots} CNOTs, AUC={agent_auc:.4f}")
        if min_cnot_above_threshold and agent_cnots > min_cnot_above_threshold:
            excess = agent_cnots - min_cnot_above_threshold
            print(f"  → Circuito inflado: {excess} CNOTs a mais que o mínimo necessário")
            print(f"  → Isso confirma a necessidade do budget adaptativo no Meta-RL")
        else:
            print(f"  → Circuito do agente é eficiente para este problema")

    # ── Salva JSON ────────────────────────────────────────────────────────────
    out_path = Path("benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Resultados salvos em {out_path}")

    # ── Tenta salvar imagem dos circuitos ─────────────────────────────────────
    try:
        import pennylane as qml
        import matplotlib.pyplot as plt
        from rl_and_qml_in_clinical_images.util import save_circuit_image

        X_ref = X[0]  # amostra de referência para o drawer

        fig, axes = plt.subplots(len(baselines), 1,
                                 figsize=(14, 4 * len(baselines)))
        if len(baselines) == 1:
            axes = [axes]

        for ax, (name, arch_mat) in zip(axes, baselines.items()):
            try:
                model_draw = BinaryCQV_End2End(
                    arch_mat=arch_mat,
                    n_qubits=args.n_qubits,
                    enc_lambda=float(cfg.enc_lambda),
                    diff_method="backprop",
                    input_dim=int(X.shape[1]),
                    enc_affine_mode=str(cfg.enc_affine_mode),
                )
                ops_info = count_ops(arch_mat)
                title = (f"{name} | CNOTs={ops_info['n_cnot']} "
                         f"gates={ops_info['n_gates_total']} "
                         f"AUC={results[name]['auc_mean']:.3f}")

                fig_circ = qml.draw_mpl(model_draw._qnode)(
                    torch.tensor(X_ref, dtype=torch.float32),
                    model_draw.theta.detach(),
                    model_draw.enc_alpha_raw.detach(),
                    model_draw.enc_beta_raw.detach(),
                )
                fig_circ.suptitle(title, fontsize=10)
                fig_circ.savefig(f"circuit_{name}.png", dpi=120, bbox_inches="tight")
                plt.close(fig_circ)
                print(f"[CIRCUIT] Salvo: circuit_{name}.png")
            except Exception as e:
                print(f"[WARN] Não foi possível desenhar {name}: {e}")

    except Exception as e:
        print(f"\n[INFO] Imagem dos circuitos não gerada: {e}")

    print("\n[DONE] Benchmark completo.\n")
    return results


if __name__ == "__main__":
    main()