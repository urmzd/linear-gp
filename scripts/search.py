#!/usr/bin/env python

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import json
from loguru import logger
from pathlib import Path
import time
from typing import Any, Callable, List

import optuna
from optuna.visualization import plot_intermediate_values, plot_optimization_history
from subprocess import Popen, PIPE
from threading import Lock

STORAGE = "postgresql://user:password@localhost:5432/database"
ENV = [
    "iris-lgp",
    "mountain-car-lgp",
    "cart-pole-lgp",
    "mountain-car-q",
    "cart-pole-q",
]

global_best_score = None
global_hyper_parameters = None
score_lock = Lock()


def update_best_hyperparameters(score: float, hyper_parameters: str) -> None:
    global global_best_score, global_hyper_parameters, score_lock

    with score_lock:
        if global_best_score is None or score > global_best_score:
            global_best_score = score
            global_hyper_parameters = hyper_parameters


def save_best_hyperparameters(study_name: str) -> None:
    with score_lock:
        env = study_name.split("_")[0]
        path_to_save = f"assets/parameters/{env}.json"
        Path(path_to_save).parent.mkdir(exist_ok=True)

        with open(path_to_save, "w") as f:
            if global_hyper_parameters is not None:
                f.write(global_hyper_parameters)


def load_study(study_name: str) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=STORAGE)


def create_study(env: str) -> str:
    study_name = f"{env}_{int(time.time())}"
    optuna.create_study(study_name=study_name, direction="maximize", storage=STORAGE)
    return study_name


def run_optimization(
    study_name: str, objective: Callable[[optuna.Trial], float], n_trials: int
):
    study = load_study(study_name)
    study.optimize(objective, n_trials=n_trials)


def build_objective(
    study_name: str, trial: optuna.Trial, lgp_parameters: dict[str, Any] | None = None
) -> float:

    env, _timestamp = study_name.split("_")

    max_instructions = None
    external_factor = None

    if lgp_parameters is None:
        max_instructions = trial.suggest_int("max_instructions", 1, 100)
        external_factor = trial.suggest_float("external_factor", 0.0, 100.0)
    else:
        program_parameters = lgp_parameters["program_parameters"]
        max_instructions = program_parameters["max_instructions"]
        external_factor = program_parameters["external_factor"]

    # Define the command to run with the CLI
    base_command = [
        "./target/release/lgp",
        env,
        "--max-instructons",
        max_instructions,
        "--n-extras",
        1,
        "--external-factor",
        external_factor,
    ]

    if lgp_parameters:
        q_cli_parameters = [
            "--alpha",
            trial.suggest_float("alpha", 0.0, 1.0),
            "--alpha-decay",
            trial.suggest_float("alpha_decay", 0.0, 1.0),
            "--gamma",
            trial.suggest_float("gamma", 0.0, 1.0),
            "--epsilon",
            trial.suggest_float("epsilon", 0.0, 1.0),
            "--epsilon-decay",
            trial.suggest_float("epsilon_decay", 0.0, 1.0),
        ]

        base_command.extend(q_cli_parameters)

    command = list(map(str, base_command))
    logger.trace(" ".join(command))

    # Run the command and capture the output
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error running command: {error.decode('utf-8')}")

    # Get the best score from the output
    parsed_output = output.decode("utf-8").strip().split("\n")
    scores = [float(score) for score in parsed_output[:-1]]

    # Save hyperparameters
    hyperparameters = parsed_output[-1]

    champion = scores[-1]
    for score_idx, score in enumerate(scores[:-1]):
        trial.report(score, score_idx)

    if champion == float("nan"):
        raise optuna.TrialPruned

    prune_thresholds = {
        "cart": 400,
        "iris": 0.9,
        "mountain": -150,
        "default": 0
    }
    threshold = prune_thresholds.get(env.split("-")[0], prune_thresholds["default"])

    update_best_hyperparameters(champion, hyperparameters)

    if champion < threshold:
        raise optuna.TrialPruned()

    return champion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter Searcher")
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

    env_tokens = study_name.split("-")
    learning_type = env_tokens[-1]
    env_name = "-".join(env_tokens[:-1])

    if learning_type == "q":
        # todo: check for assets/parameters/{env_name}-lgp.json
        # load the parameters and set max_instructions and external_factor
        # to the values found in the file
        lgp_params = f"assets/parameters/{env_name}-q.json"
        assert Path(lgp_params).exists()
        with open(lgp_params, "r"):
            parameters = json.loads(lgp_params)
            objective = partial(build_objective, study_name, lgp_parameters=parameters)
    else:
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
    save_best_hyperparameters(study_name)
    plot_optimization_history(study)
    plot_intermediate_values(study)
    logger.info(
        f"best_score={global_best_score}, best_params={global_hyper_parameters}"
    )


if __name__ == "__main__":
    args = parse_args()

    main(args)
