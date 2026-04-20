import argparse
import os

from main.main_banknote import main_banknote
from main.main_cross_circle import main_cross_circle
from main.main_moons import main_make_moons
from refactor_project.main.main_breat_cancer import main_breast_cancer


# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Runner")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["cross_circle", "make_moons", "banknote", "breast_cancer", "all"],
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
        main_cross_circle(DEBUG=debug)

    elif name == "make_moons":
        main_make_moons(DEBUG=debug)

    elif name == "banknote":
        main_banknote(DEBUG=debug)

    elif name == "breast_cancer":
        main_breast_cancer(DEBUG=debug)

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
        experiments = ["cross_circle", "make_moons", "banknote", "breast_cancer"]

        for exp in experiments:
            run_experiment(exp, args.debug)

    else:
        run_experiment(args.experiment, args.debug)
