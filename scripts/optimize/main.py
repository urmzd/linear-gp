#!/usr/bin/env python

import optuna
from subprocess import Popen, PIPE
from typing import Tuple, Any
from functools import partial

def objective(trial: optuna.Trial) -> float:
    # Define the hyperparameters to optimize
    population_size = 100
    gap = 0.5
    mutation_percent = 0.5
    crossover_percent =  0.5
    n_generations = 100
    max_instructions = trial.suggest_int("max_instructions", 1, 32)
    n_extras = 1
    external_factor = trial.suggest_float("external_factor", 0.0, 100.)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    alpha_decay = trial.suggest_float("alpha_decay", 0.0, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.0, 1.0)
    n_trials = 5

    game = "cart-pole"

    # Define the command to run with the CLI
    command = ["./lgp", game,
                "--alpha", alpha, "--alpha-decay", alpha_decay, "--gamma", gamma,
                "--epsilon", epsilon, "--epsilon-decay", epsilon_decay,
                "--n-trials", n_trials,
                "--population-size", population_size, "--gap", gap,
                "--mutation-percent", mutation_percent, "--crossover-percent", crossover_percent,
                "--n-generations", n_generations, "--max-instructions", max_instructions,
                "--n-extras", n_extras,
               "--external-factor", external_factor]

    command = list(map(lambda x: str(x), command))
    print(" ".join(command))

    try:
        # Run the command and capture the output
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, error = process.communicate()

        if error:
            raise Exception(f"Error running command: {error.decode('utf-8')}")

        # Get the best score from the output
        print(f"OUTPUT: {output}")
        best_score = float(output.decode('utf-8').strip())
    except Exception as e:
        print(f"Error: {e}")
        best_score = float('nan')

    return best_score

# Define the study and run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and score
best_hyperparams = study.best_params
best_score = study.best_value
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best score: {best_score}")
