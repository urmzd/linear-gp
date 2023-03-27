#!/usr/bin/env python

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import time
from typing import Any, Callable, Dict, List
from subprocess import Popen, PIPE
import optuna
from optuna.visualization import plot_intermediate_values, plot_optimization_history

from pathlib import Path
import json
from typing import Any, Dict

STORAGE = "postgresql://user:password@localhost:5432/database"
ENV = [
    "mountain-car-q",
    "mountain-car-lgp",
    "iris",
    "cart-pole-q",
    "cart-pole-lgp",
]


def save_best_hyperparameters(
    hyper_parameters: Dict[str, Any], best_score: float, study_name: str
) -> None:
    program_parameters = {
        "max_instructions": hyper_parameters["max_instructions"],
        "instruction_generator_parameters": {
            "n_extras": 1,
            "external_factor": hyper_parameters["external_factor"],
        },
    }

    if "alpha" in hyper_parameters:
        program_parameters = {
            "program_parameters": program_parameters,
            "consts": {
                "alpha": hyper_parameters["alpha"],
                "epsilon": hyper_parameters["epsilon"],
                "gamma": hyper_parameters["gamma"],
                "alpha_decay": hyper_parameters["alpha_decay"],
                "epsilon_decay": hyper_parameters["epsilon_decay"],
            },
        }

    full_hp = {
        "n_generations": 100,
        "population_size": 100,
        "gap": 0.5,
        "mutation_percent": 0.5,
        "crossover_percent": 0.5,
        "fitness_parameters": {
            "n_generations": 100,
            "n_trials": 5,
        },
        "program_parameters": program_parameters,
    }

    path_to_save = f"assets/parameters/{best_score}_{study_name}.json"

    Path(path_to_save).parent.mkdir(exist_ok=True)

    with open(path_to_save, "w") as f:
        json.dump(full_hp, f, indent=4)


def load_study(study_name: str) -> optuna.Study:
    study = optuna.load_study(
        study_name=study_name,
        storage=STORAGE,
    )

    return study


def create_study(env: str) -> str:
    study_name = f"{env}_{int(time.time())}"
    optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=STORAGE,
    )

    return study_name


def run_optimization(
    study_name: str, objective: Callable[[optuna.Trial], float], n_trials: int
):
    study = load_study(study_name)
    study.optimize(objective, n_trials=n_trials)


def build_objective(study_name: str, trial: optuna.Trial) -> float:
    # Define the hyperparameters to optimize
    population_size = 100
    gap = 0.5
    mutation_percent = 0.5
    crossover_percent = 0.5
    n_generations = 100
    n_extras = 1
    n_trials = 5
    max_instructions = trial.suggest_int("max_instructions", 1, 64)
    external_factor = trial.suggest_float("external_factor", 0.0, 100.0)

    env, _timestamp = study_name.split("_")

    # Define the command to run with the CLI
    command = [
        "./target/release/lgp",
        env,
        "--n-trials",
        n_trials,
        "--population-size",
        population_size,
        "--gap",
        gap,
        "--mutation-percent",
        mutation_percent,
        "--crossover-percent",
        crossover_percent,
        "--n-generations",
        n_generations,
        "--max-instructions",
        max_instructions,
        "--n-extras",
        n_extras,
        "--external-factor",
        external_factor,
    ]

    if "q" in env:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        alpha_decay = trial.suggest_float("alpha_decay", 0.0, 1.0)
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.0, 1.0)
        q_command = [
            "--alpha",
            alpha,
            "--alpha-decay",
            alpha_decay,
            "--gamma",
            gamma,
            "--epsilon",
            epsilon,
            "--epsilon-decay",
            epsilon_decay,
        ]
        command.extend(q_command)

    command = list(map(lambda x: str(x), command))
    print(" ".join(command))
    # Run the command and capture the output
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error running command: {error.decode('utf-8')}")

    # Get the best score from the output
    parsed_output = output.decode("utf-8").strip().split("\n")
    scores = [float(score) for score in parsed_output]
    print(f"Output: {parsed_output}")

    best_score = scores[-1]
    for score_idx, score in enumerate(scores[:-1]):
        trial.report(score, score_idx)

    if best_score == float("nan"):
        raise optuna.TrialPruned

    if "cart" in env:
        if best_score < 400:
            raise optuna.TrialPruned()
    elif "iris" in env:
        if best_score < 0.9:
            raise optuna.TrialPruned()
    else:
        if best_score < -100:
            raise optuna.TrialPruned()

    return best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimizer")
    parser.add_argument(
        "--env",
        type=str,
        choices=ENV,
        required=True,
        help="The name of the environment to run simulation in.",
    )
    parser.add_argument(
        "--n-trials",
        default=150,
        type=int,
        help="The number of trials to run per study",
    )
    parser.add_argument(
        "--n-threads",
        default=4,
        type=int,
        help="The number of threads to use per study",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    study_name = create_study(args.env)
    objective = partial(build_objective, study_name)

    n_trials = args.n_trials
    results: List[Future[Any]] = []

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        for _ in range(args.n_threads):
            future = executor.submit(
                run_optimization,
                study_name=study_name,
                objective=objective,
                n_trials=n_trials,
            )
            results.append(future)

    for future in results:
        future.result()

    study = load_study(study_name)
    save_best_hyperparameters(study.best_params, study.best_value, study_name)
    plot_optimization_history(study)
    plot_intermediate_values(study)


if __name__ == "__main__":
    args = parse_args()

    main(args)
