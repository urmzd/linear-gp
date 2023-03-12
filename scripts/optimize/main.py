#!/usr/bin/env python

import optuna
import time
import argparse
from subprocess import Popen, PIPE
from functools import partial
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGP Optimizer")
    parser.add_argument(
        "game",
        type=str,
        choices=["cart-pole", "mountain-car"],
        help="The name of the game to optimize for",
    )
    parser.add_argument(
        "--n_trials",
        default=150,
        required=False,
        type=int,
        help="The number of trials",
    )
    return parser.parse_args()


def objective(game: str, trial: optuna.Trial) -> float:
    # Define the hyperparameters to optimize
    population_size = 100
    gap = 0.5
    mutation_percent = 0.5
    crossover_percent = 0.5
    n_generations = 100
    max_instructions = trial.suggest_int("max_instructions", 1, 32)
    n_extras = 1
    external_factor = trial.suggest_float("external_factor", 0.0, 100.0)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    alpha_decay = trial.suggest_float("alpha_decay", 0.0, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.0, 1.0)
    n_trials = 5

    # Define the command to run with the CLI
    command = [
        "./target/release/lgp",
        game,
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

    command = list(map(lambda x: str(x), command))
    print(" ".join(command))

    # Run the command and capture the output
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error running command: {error.decode('utf-8')}")

    # Get the best score from the output
    print(f"Output: {output}")
    parsed_output = output.decode("utf-8").strip().split("\n")
    scores = [float(score) for score in parsed_output]

    for score_idx, score in enumerate(scores[:-1]):
        trial.report(score, score_idx)

    best_score = scores[-1]

    if game == "cart-pole":
        if best_score < 100:
            raise optuna.TrialPruned()
    else:
        if best_score < -180:
            raise optuna.TrialPruned()

    return best_score


if __name__ == "__main__":
    args = parse_args()

    # Define the study and run the optimization
    study = optuna.create_study(
        study_name=f"{args.game}-{int(time.time())}",
        direction="maximize",
        storage="sqlite:///db.sqlite3",
    )
    objective_partial = partial(objective, args.game)
    study.optimize(objective_partial, n_trials=150)

    plot_optimization_history(study)
    plot_parallel_coordinate(study)

    # Print the best hyperparameters and score
    best_hyperparams = study.best_params
    best_score = study.best_value
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best score: {best_score}")
