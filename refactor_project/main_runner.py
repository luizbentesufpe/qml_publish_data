import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


from main.main_banknote import main_banknote  # noqa: I001
from main.main_blobs import main_blobs
from main.main_breast_cancer import main_breast_cancer
from main.main_concentric_circles import main_concentric_circles
from main.main_cross_circle import main_cross_circle
# from main.main_higgs import main_higgs
from main.main_iris import main_iris
from main.main_mnist import main_mnist
from main.main_moons import main_make_moons
from main.main_stripes import main_stripes

# Higgs: já foi feito — import comentado


# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Runner")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "cross_circle",
            "make_moons",
            "banknote",
            "breast_cancer",
            "iris",
            "blobs",
            "concentric_circles",
            "stripes",
            "mnist",
            # "higgs",  # já foi feito
            "all",
        ],
        required=True,
        help="Which experiment to run",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in DEBUG mode (fast, no guarantees)",
    )

    return parser.parse_args()


# =========================
# Experiment dispatcher
# =========================
def run_experiment(name: str, debug: bool):
    print(f"\n[RUN] Experiment: {name} | DEBUG={debug}")

    if name == "cross_circle":
        main_cross_circle(
            DEBUG=debug,
            scenario_filter=None,
            total_core_budget=28,
            outer_n_jobs=2,
        )

    elif name == "make_moons":
        main_make_moons(
            DEBUG=debug,
            scenario_filter=None,
            total_core_budget=28,
            outer_n_jobs=2,
        )

    elif name == "banknote":
        main_banknote(
            DEBUG=debug,
            scenario_filter=None,
            total_core_budget=28,
            outer_n_jobs=2,
        )

    elif name == "breast_cancer":
        main_breast_cancer(
            DEBUG=debug,
            scenario_filter=None,
            total_core_budget=28,
            outer_n_jobs=2,
        )

    elif name == "iris":
        # main_iris ainda não tem scenario_filter/core_budget — só DEBUG/data_dir
        main_iris(DEBUG=debug)

    elif name == "blobs":
        # main_blobs tem scenario_filter (S0-S4 + varredura high-dim),
        # mas ainda não tem total_core_budget/outer_n_jobs
        main_blobs(DEBUG=debug, scenario_filter=None)

    elif name == "concentric_circles":
        main_concentric_circles(DEBUG=debug)

    elif name == "stripes":
        main_stripes(DEBUG=debug)

    elif name == "mnist":
        main_mnist(DEBUG=debug)

    # ── Higgs: já foi feito — dispatch comentado ──────────────────────
    # elif name == "higgs":
    #     main_higgs(
    #         DEBUG=debug,
    #         scenario_filter=None,
    #         total_core_budget=28,
    #         outer_n_jobs=2,
    #     )

    else:
        raise ValueError(f"Unknown experiment: {name}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Clean cache (mais seguro que popen)
    os.system('find . | grep -E "(__pycache__|\\.pyc|\\.pyo$)" | xargs rm -rf')

    args = parse_args()

    if args.experiment == "all":
        experiments = [
            "cross_circle",
            "make_moons",
            "banknote",
            "breast_cancer",
            "iris",
            "blobs",
            "concentric_circles",
            "stripes",
            "mnist",
            # "higgs",  # já foi feito — fora do "all"
        ]

        for exp in experiments:
            run_experiment(exp, args.debug)

    else:
        run_experiment(args.experiment, args.debug)