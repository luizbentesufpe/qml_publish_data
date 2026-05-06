"""
Gera ALL_SCENARIOS_consolidated.json consolidando todos os cenários
mantendo o formato com metadata, scenario e results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    """Carrega JSON com fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_aggregated_json(dataset_root: Path) -> Path:
    """Busca arquivo agregado JSON no root do dataset."""
    candidates = [
        dataset_root / "aggregated.json",
        dataset_root / "results.json",
        dataset_root / "ALL_SCENARIOS_1.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Se não encontrar, busca qualquer JSON no root
    for json_file in sorted(dataset_root.glob("*.json")):
        if json_file.is_file() and json_file.name not in [
            "ALL_SCENARIOS.json",
            "ALL_SCENARIOS_consolidated.json",
        ]:
            return json_file

    return None


def load_scenario_results(scenario_dir: Path) -> List[Dict]:
    """
    Tenta carregar resultados individuais de uma scenario.
    """
    result_files = list(scenario_dir.glob("results_*.json"))
    if result_files:
        data = load_json(result_files[0])
        if isinstance(data, dict) and "results" in data:
            return data.get("results", [])
        elif isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and "results" in value:
                    return value.get("results", [])

    return []


def consolidate_dataset(aggregated_json: Path, dataset_root: Path) -> Dict[str, Any]:
    """
    Consolida um dataset inteiro em formato único.
    """
    aggregated = load_json(aggregated_json)

    if not aggregated or "scenarios" not in aggregated:
        return {}

    consolidated = {"meta": aggregated.get("meta", {}), "scenarios": []}

    for scenario_item in aggregated["scenarios"]:
        if not isinstance(scenario_item, dict):
            continue

        scenario_meta = scenario_item.get("scenario", {})
        scenario_name = scenario_meta.get("name")

        if not scenario_name:
            continue

        # Tenta carregar resultados individuais
        scenario_dir = dataset_root / scenario_name
        results = []

        if scenario_dir.exists():
            results = load_scenario_results(scenario_dir)

        # Se não encontrou, usa dados do summary como fallback
        if not results and "summary" in scenario_item:
            summary = scenario_item.get("summary", {})
            perf_data = summary.get("rlqcv_perf_ci95", {})
            per_seed = perf_data.get("per_seed", [])

            if per_seed:
                results = [
                    {
                        "seed": i,
                        "perf": p,
                        "cost": scenario_item.get("summary", {})
                        .get("pareto_front", [{}])[0]
                        .get("cost", 0),
                    }
                    for i, p in enumerate(per_seed)
                ]

        # Monta item do cenário
        scenario_consolidated = {
            "scenario": scenario_meta,
            "summary": scenario_item.get("summary", {}),
            "results": results,
            "results_file": scenario_item.get("results_file", ""),
        }

        consolidated["scenarios"].append(scenario_consolidated)

    return consolidated


def consolidate_all_datasets(
    base_path: Path = Path("."),
    datasets: List[str] = None,
) -> Dict[str, Path]:
    """
    Processa todos os datasets e gera ALL_SCENARIOS_consolidated.json
    """

    if datasets is None:
        datasets = [
            "publication_out_banknote",
            "publication_out_breast_cancer",
            "publication_out_cross_circle",
            "publication_out_make_moons",
        ]

    print("=" * 70)
    print("🚀 GERADOR ALL_SCENARIOS_CONSOLIDATED")
    print("=" * 70)

    all_results_paths = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"📊 Processando: {dataset_name}")
        print(f"{'=' * 70}")

        dataset_root = base_path / dataset_name

        if not dataset_root.exists():
            print(f"❌ Não encontrado: {dataset_root}")
            continue

        # Busca arquivo agregado
        aggregated_json = find_aggregated_json(dataset_root)

        if not aggregated_json:
            print("❌ Nenhum JSON agregado encontrado")
            continue

        print(f"📋 Encontrado: {aggregated_json.name}")

        # Consolida dataset
        consolidated = consolidate_dataset(aggregated_json, dataset_root)

        if not consolidated:
            print("❌ Falha ao consolidar")
            continue

        n_scenarios = len(consolidated.get("scenarios", []))
        print(f"📊 {n_scenarios} cenários consolidados")

        # Salva ALL_SCENARIOS_consolidated.json
        output_path = dataset_root / "ALL_SCENARIOS_consolidated.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        print("✅ ALL_SCENARIOS_consolidated.json salvo")
        all_results_paths[dataset_name] = output_path

    return all_results_paths


def print_summary(all_results_paths: Dict[str, Path]) -> None:
    """Exibe resumo final."""

    print("\n" + "=" * 70)
    print("✨ CONSOLIDAÇÃO CONCLUÍDA")
    print("=" * 70)

    for dataset_name, path in sorted(all_results_paths.items()):
        print(f"\n✅ {dataset_name}")
        print(f"   📁 {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        n_scenarios = len(data.get("scenarios", []))
        print(f"   📊 Cenários: {n_scenarios}")

        for scenario_item in data.get("scenarios", []):
            scenario_name = scenario_item.get("scenario", {}).get("name", "?")
            n_seeds = len(scenario_item.get("results", []))
            print(f"      • {scenario_name}: {n_seeds} seeds")

    print("\n" + "=" * 70)


def print_detailed_summary(all_results_paths: Dict[str, Path]) -> None:
    """Exibe resumo detalhado com métricas."""

    print("\n" + "=" * 70)
    print("📈 RESUMO DETALHADO")
    print("=" * 70)

    for dataset_name, consolidated_path in sorted(all_results_paths.items()):
        print(f"\n{dataset_name.upper()}")
        print("-" * 70)

        if not consolidated_path.exists():
            continue

        with open(consolidated_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for scenario_item in data.get("scenarios", []):
            scenario_name = scenario_item.get("scenario", {}).get("name")
            results = scenario_item.get("results", [])
            summary = scenario_item.get("summary", {})

            # Extrai de summary
            perf_data = summary.get("rlqcv_perf_ci95", {})
            perf_mean = perf_data.get("mean", 0)
            ci95 = perf_data.get("ci95", [0, 0])

            pareto = summary.get("pareto_front", [])
            costs = [p.get("cost", 0) for p in pareto]
            avg_cost = sum(costs) / len(costs) if costs else 0

            print(f"\n   {scenario_name}")
            print(f"      Seeds:  {len(results)}")
            print(f"      Perf:   {perf_mean:.4f} (CI95: [{ci95[0]:.4f}, {ci95[1]:.4f}])")
            print(f"      Cost:   {avg_cost:.2f} (avg pareto)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    current_dir = Path(".")

    all_results_paths = consolidate_all_datasets(base_path=current_dir)

    if all_results_paths:
        print_summary(all_results_paths)
        print_detailed_summary(all_results_paths)
        print(f"\n🎉 Sucesso! {len(all_results_paths)} datasets processados")
    else:
        print("\n❌ Nenhum dataset processado com sucesso")
