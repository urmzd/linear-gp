#!/usr/bin/env python

import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
import time
import argparse
from subprocess import Popen, PIPE
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGP Optimizer")
    # parser.add_argument(
        # "game",
        # type=str,
        # choices=["cart-pole", "mountain-car"],
        # help="The name of the game to optimize for",
    # )

    # parser.add_argument(
        # "learning_type",
       # type=str,
       # choices=["q", "norm"],
       # help="The type of learning to be done."
    # )

    parser.add_argument(
        "--n_trials",
        default=150,
        required=False,
        type=int,
        help="The number of trials",
    )
    return parser.parse_args()


def objective(game: str, learning_type: str, trial: optuna.Trial) -> float:
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

    # Define the command to run with the CLI
    command = [
        "./target/release/lgp",
        game,
        learning_type,
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

    if learning_type == "q":
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

    if game == "cart-pole":
        if best_score < 100:
            raise optuna.TrialPruned()
    else:
        if best_score < -100:
            raise optuna.TrialPruned()

    return best_score

def run_optimization(game, learning_type, n_trials):
    # Define the study and run the optimization
    study = optuna.create_study(
        study_name=f"{game}-{learning_type}-{int(time.time())}",
        direction="maximize",
    )
    objective_partial = partial(objective, game, learning_type)
    study.optimize(objective_partial, n_trials=n_trials)

    plot_optimization_history(study)
    plot_parallel_coordinate(study)

    # Print the best hyperparameters and score
    best_hyperparams = study.best_params
    best_score = study.best_value
    print(f"Best hyperparameters for {game} with {learning_type}: {best_hyperparams}")
    print(f"Best score: {best_score}")

if __name__ == "__main__":
    args = parse_args()

    games = ["cart-pole", "mountain-car"]
    learning_types = ["q", "norm"]

    # Run all possible variations in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for game in games:
            for learning_type in learning_types:
                futures.append(executor.submit(run_optimization, game, learning_type, args.n_trials))

        # Wait for all tasks to complete
        for future in futures:
            future.result()
