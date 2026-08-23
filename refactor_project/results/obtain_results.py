"""
compute_metrics_from_json.py
============================
Calcula F1, accuracy e especificidade a partir dos JSONs de resultado
já salvos pelo framework RL-QAS — sem re-treinar nada.

Estratégia:
    - O JSON contém sens@thr* (TPR) e AUC para cada seed/nq
    - Para dataset balanceado (50/50): n_pos = n_neg = n_holdout/2
    - TPR = TP / n_pos  →  TP = TPR * n_pos
    - Para estimar FPR: usamos a relação AUC ≈ (TPR + TNR) / 2 para
      threshold ótimo Youden (válido para distribuições bem comportadas)
      TNR = 2*AUC - TPR  →  FPR = 1 - TNR
    - Precision = TP / (TP + FP)
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)

Nota: para AUC exata de cada fold, usamos a relação de Bamber (1975):
    AUC = P(score_pos > score_neg) ≈ (TPR + TNR) / 2 no ponto ótimo

Uso:
    python compute_metrics_from_json.py <path_to_results_json>

    # Ou para todos os JSONs de um diretório:
    python compute_metrics_from_json.py --dir publication_out_higgs/
"""

import json
import pathlib
import sys

import numpy as np


def compute_metrics_from_holdout(
    auc: float, sens: float, n_holdout: int = 2000, balanced: bool = True
) -> dict:
    """
    Deriva F1, accuracy e spec a partir de AUC + SENS no threshold ótimo Youden.

    Para threshold Youden: thr* = argmax(TPR + TNR - 1)
    No ponto ótimo: TPR + TNR é maximizado → TNR ≈ 2*AUC - TPR (aproximação)

    Args:
        auc: AUC no holdout
        sens: sensitividade (TPR) no threshold ótimo
        n_holdout: tamanho do holdout
        balanced: se True, n_pos = n_neg = n_holdout/2

    Returns:
        dict com spec, precision, f1, accuracy, ppv, npv
    """
    tpr = sens  # TPR = sensitividade

    # Estimativa de TNR via relação AUC-Youden
    # Para threshold Youden ótimo: J = TPR + TNR - 1 é máximo
    # AUC ≈ integral da curva ROC — no ponto Youden:
    # TNR ≈ 2*AUC - TPR (válido para distribuições simétricas)
    tnr = max(0.0, min(1.0, 2 * auc - tpr))
    fpr = 1.0 - tnr  # FPR = 1 - especificidade

    if balanced:
        n_pos = n_neg = n_holdout // 2
    else:
        # fallback: assume 50/50
        n_pos = n_neg = n_holdout // 2

    tp = tpr * n_pos
    fn = n_pos - tp
    tn = tnr * n_neg
    fp = fpr * n_neg

    # Métricas derivadas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "npv": round(npv, 4),
    }


def process_json(json_path: pathlib.Path, n_holdout: int = 2000, balanced: bool = True) -> None:
    """Processa um JSON de resultado e imprime métricas derivadas."""
    data = json.loads(json_path.read_text())

    scenario_name = data.get("scenario", {}).get("name", json_path.stem)
    dataset = data.get("scenario", {}).get("dataset", "unknown")
    seeds_data = data.get("best_rlqcv_per_seed", [])

    print(f"\n{'=' * 70}")
    print(f"Cenário: {scenario_name}  |  Dataset: {dataset}")
    print(f"{'=' * 70}")
    print(
        f"{'Seed':>4} {'nq':>4} {'AUC':>6} {'SENS':>6} {'SPEC':>6} "
        f"{'Prec':>6} {'F1':>6} {'Acc':>6}"
    )
    print("-" * 50)

    aucs, f1s, accs, specs = [], [], [], []

    for entry in seeds_data:
        seed = entry.get("seed", "?")
        nq = entry.get("best_nq", entry.get("nq", "?"))
        holdout = entry.get("holdout", {})
        auc = holdout.get("auc", 0.0)
        sens = holdout.get("sens@thr*", 0.0)

        m = compute_metrics_from_holdout(auc, sens, n_holdout, balanced)

        print(
            f"{seed:>4} {nq:>4} {auc:>6.4f} {sens:>6.4f} {m['tnr']:>6.4f} "
            f"{m['precision']:>6.4f} {m['f1']:>6.4f} {m['accuracy']:>6.4f}"
        )

        aucs.append(auc)
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
        specs.append(m["tnr"])

    print("-" * 50)
    print(
        f"{'Mean':>4} {'':>4} {np.mean(aucs):>6.4f} {'':>6} {np.mean(specs):>6.4f} "
        f"{'':>6} {np.mean(f1s):>6.4f} {np.mean(accs):>6.4f}"
    )
    print(
        f"{'Std':>4} {'':>4} {np.std(aucs):>6.4f} {'':>6} {np.std(specs):>6.4f} "
        f"{'':>6} {np.std(f1s):>6.4f} {np.std(accs):>6.4f}"
    )

    # Correlação proxy RL → AUC final
    corr = data.get("corr_rlproxy_vs_finalauc", {})
    if corr:
        print(
            f"\nCorr proxy→AUC: pearson={corr.get('pearson', '?'):.3f}  "
            f"spearman={corr.get('spearman', '?'):.3f}  n={corr.get('n', '?')}"
        )

    # alpha_physics (Higgs)
    alpha = data.get("alpha_physics")
    if alpha:
        ratio = alpha.get("ratio_HL_over_LL", {})
        print(f"HL/LL ratio: {ratio.get('mean', '?'):.2f} ± {ratio.get('std', '?'):.2f}")


def main():
    args = sys.argv[1:]

    if not args:
        print("Uso: python compute_metrics_from_json.py <json_or_dir> [--n_holdout N]")
        sys.exit(1)

    # Parse --n_holdout
    n_holdout = 2000  # default Higgs (10k * 0.20)
    if "--n_holdout" in args:
        idx = args.index("--n_holdout")
        n_holdout = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    # Balanced flag
    balanced = "--unbalanced" not in args
    args = [a for a in args if a != "--unbalanced"]

    target = pathlib.Path(args[0])

    if target.is_dir():
        jsons = sorted(target.rglob("results_*.json"))
        if not jsons:
            print(f"Nenhum results_*.json encontrado em {target}")
            sys.exit(1)
        for j in jsons:
            process_json(j, n_holdout, balanced)
    elif target.is_file():
        process_json(target, n_holdout, balanced)
    else:
        print(f"Arquivo/diretório não encontrado: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
