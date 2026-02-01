"""Hyperparameter search commands."""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Lock
from typing import Any, Callable

import optuna
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="search", help="Hyperparameter optimization commands")
console = Console()

STORAGE = "postgresql://user:password@localhost:5432/database"
ENVIRONMENTS = [
    "iris-lgp",
    "mountain-car-lgp",
    "cart-pole-lgp",
    "mountain-car-q",
    "cart-pole-q",
]

# Global state for tracking best hyperparameters
_best_score: float | None = None
_best_hyperparameters: str | None = None
_score_lock = Lock()


def _update_best_hyperparameters(score: float, hyperparameters: str) -> None:
    """Update the global best score and hyperparameters if better."""
    global _best_score, _best_hyperparameters

    with _score_lock:
        if _best_score is None or score > _best_score:
            _best_score = score
            _best_hyperparameters = hyperparameters


def _reset_best_hyperparameters() -> None:
    """Reset the global best score tracking."""
    global _best_score, _best_hyperparameters

    with _score_lock:
        _best_score = None
        _best_hyperparameters = None


def _save_best_hyperparameters(study_name: str) -> None:
    """Save the best hyperparameters to a JSON file."""
    with _score_lock:
        env = study_name.split("_")[0]
        path_to_save = Path(f"experiments/assets/parameters/{env}.json")
        path_to_save.parent.mkdir(exist_ok=True, parents=True)

        if _best_hyperparameters is not None:
            path_to_save.write_text(_best_hyperparameters)
            console.print(f"[green]Saved parameters to {path_to_save}[/green]")


def _load_study(study_name: str) -> optuna.Study:
    """Load an existing Optuna study."""
    return optuna.load_study(study_name=study_name, storage=STORAGE)


def _create_study(env: str) -> str:
    """Create a new Optuna study with a timestamped name."""
    study_name = f"{env}_{int(time.time())}"
    optuna.create_study(study_name=study_name, direction="maximize", storage=STORAGE)
    return study_name


def _run_optimization(
    study_name: str, objective: Callable[[optuna.Trial], float], n_trials: int
) -> None:
    """Run optimization on a study."""
    study = _load_study(study_name)
    study.optimize(objective, n_trials=n_trials)


def _build_objective(
    study_name: str,
    median_trials: int,
    trial: optuna.Trial,
    lgp_parameters: dict[str, Any] | None = None,
) -> float:
    """Build the objective function for Optuna optimization."""
    env, _ = study_name.split("_")

    if lgp_parameters is None:
        max_instructions = trial.suggest_int("max_instructions", 1, 100)
        external_factor = trial.suggest_float("external_factor", 0.0, 100.0)
    else:
        program_parameters = lgp_parameters["program_parameters"]
        max_instructions = program_parameters["max_instructions"]
        external_factor = program_parameters["instruction_generator_parameters"][
            "external_factor"
        ]

    base_command = [
        "./target/release/lgp",
        env,
        f"--max-instructions={max_instructions}",
        f"--external-factor={external_factor}",
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
    pairings = []

    for _ in range(median_trials):
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, error = process.communicate()

        if error:
            raise Exception(f"Error running command: {error.decode('utf-8')}")

        parsed_output = output.decode("utf-8").strip().split("\n")
        scores = [float(score) for score in parsed_output[:-1]]
        hyperparameters = parsed_output[-1]
        champion = scores[-1]

        pairings.append((champion, hyperparameters))

    pairings.sort(key=lambda x: x[0])
    champion, hyperparameters = pairings[len(pairings) // 2]

    if champion != champion:  # NaN check
        raise optuna.TrialPruned()

    prune_thresholds = {"cart": 400, "iris": 0.9, "mountain": -150, "default": 0}
    threshold = prune_thresholds.get(env.split("-")[0], prune_thresholds["default"])

    _update_best_hyperparameters(champion, hyperparameters)

    if champion < threshold:
        raise optuna.TrialPruned()

    return champion


@app.command()
def single(
    env: str = typer.Argument(..., help="Environment to optimize"),
    n_trials: int = typer.Option(10, "--n-trials", "-t", help="Number of trials per thread"),
    n_threads: int = typer.Option(4, "--n-threads", "-j", help="Number of parallel threads"),
    median_trials: int = typer.Option(
        10, "--median-trials", "-m", help="Number of runs per trial for median calculation"
    ),
) -> None:
    """Search hyperparameters for a single environment."""
    if env not in ENVIRONMENTS:
        console.print(f"[red]Invalid environment: {env}[/red]")
        console.print(f"[yellow]Valid environments: {', '.join(ENVIRONMENTS)}[/yellow]")
        raise typer.Exit(1)

    _reset_best_hyperparameters()

    console.print(f"[bold blue]Starting hyperparameter search for {env}[/bold blue]")
    console.print(f"  Trials per thread: {n_trials}")
    console.print(f"  Threads: {n_threads}")
    console.print(f"  Median trials: {median_trials}")

    study_name = _create_study(env)
    console.print(f"  Study name: {study_name}")

    env_tokens = env.split("-")
    learning_type = env_tokens[-1]
    env_name = "-".join(env_tokens[:-1])

    if learning_type == "q":
        lgp_params_path = Path(f"experiments/assets/parameters/{env_name}-lgp.json")
        if not lgp_params_path.exists():
            console.print(
                f"[red]LGP parameters not found: {lgp_params_path}[/red]"
            )
            console.print(
                "[yellow]Run LGP search first before Q-learning search[/yellow]"
            )
            raise typer.Exit(1)

        parameters = json.loads(lgp_params_path.read_text())
        objective = partial(
            _build_objective,
            study_name,
            median_trials,
            lgp_parameters=parameters,
        )
    else:
        objective = partial(_build_objective, study_name, median_trials)

    results: list[Future[Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Optimizing {env}...", total=None)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for _ in range(n_threads):
                future = executor.submit(
                    _run_optimization,
                    study_name=study_name,
                    objective=objective,
                    n_trials=n_trials,
                )
                results.append(future)

            for future in results:
                future.result()

        progress.update(task, completed=True)

    _load_study(study_name)
    _save_best_hyperparameters(study_name)

    console.print(f"\n[bold green]Search complete![/bold green]")
    console.print(f"  Best score: {_best_score}")


@app.command()
def all(
    n_trials: int = typer.Option(40, "--n-trials", "-t", help="Number of trials per thread"),
    n_threads: int = typer.Option(4, "--n-threads", "-j", help="Number of parallel threads"),
    median_trials: int = typer.Option(
        10, "--median-trials", "-m", help="Number of runs per trial for median calculation"
    ),
) -> None:
    """Search hyperparameters for all environments.

    Runs LGP environments first, then Q-learning environments
    (Q-learning requires LGP parameters).
    """
    lgp_environments = ["iris-lgp", "cart-pole-lgp", "mountain-car-lgp"]
    q_environments = ["cart-pole-q", "mountain-car-q"]

    console.print("[bold blue]Starting hyperparameter search for all environments[/bold blue]")

    console.print("\n[bold]Phase 1: LGP environments[/bold]")
    for env in lgp_environments:
        console.print(f"\n[cyan]{'=' * 50}[/cyan]")
        single(
            env=env,
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )

    console.print("\n[bold]Phase 2: Q-learning environments[/bold]")
    for env in q_environments:
        console.print(f"\n[cyan]{'=' * 50}[/cyan]")
        single(
            env=env,
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )

    console.print("\n[bold green]All environments completed![/bold green]")
