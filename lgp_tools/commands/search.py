"""Search command for hyperparameter optimization."""

import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Lock
from typing import Any

import optuna
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lgp_tools.config import (
    PRUNE_THRESHOLDS,
    STORAGE,
    discover_configs,
    get_configs_dir,
)
from lgp_tools.models import ExperimentConfig

console = Console()

# === Global state for tracking best hyperparameters ===

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


def _save_best_hyperparameters(study_name: str, config_name: str) -> None:
    """Save the best hyperparameters to a JSON file."""
    with _score_lock:
        path_to_save = Path(f"outputs/parameters/{config_name}.json")
        path_to_save.parent.mkdir(exist_ok=True, parents=True)

        if _best_hyperparameters is not None:
            path_to_save.write_text(_best_hyperparameters)
            console.print(f"[green]Saved parameters to {path_to_save}[/green]")

            # Generate optimal.toml files
            params = json.loads(_best_hyperparameters)
            _generate_optimal_config(config_name, params)


def _generate_optimal_config(config_name: str, params: dict) -> None:
    """Generate optimal.toml from search results."""
    configs_dir = get_configs_dir()
    folder_path = configs_dir / config_name
    default_path = folder_path / "default.toml"
    optimal_path = folder_path / "optimal.toml"

    if not default_path.exists():
        console.print(f"[yellow]No default.toml found for {config_name}[/yellow]")
        return

    # Load and validate config with Pydantic
    exp_config = ExperimentConfig.from_toml(default_path)

    # Update with optimal parameters
    if "program_parameters" in params:
        prog = params["program_parameters"]

        # Handle nested structure for Q-learning params
        if "program_parameters" in prog:
            # Q-learning format: params["program_parameters"]["program_parameters"]
            inner_prog = prog["program_parameters"]
            if "max_instructions" in inner_prog:
                exp_config.hyperparameters.program.max_instructions = inner_prog["max_instructions"]
            if "instruction_generator_parameters" in inner_prog:
                igp = inner_prog["instruction_generator_parameters"]
                if "external_factor" in igp:
                    exp_config.hyperparameters.program.external_factor = igp["external_factor"]

            # Q-learning consts
            if "consts" in prog:
                for op in exp_config.operations:
                    if op.name == "q_learning":
                        op.parameters.update(prog["consts"])
                        break
        else:
            # Standard LGP format: params["program_parameters"]
            if "max_instructions" in prog:
                exp_config.hyperparameters.program.max_instructions = prog["max_instructions"]
            if "instruction_generator_parameters" in prog:
                igp = prog["instruction_generator_parameters"]
                if "external_factor" in igp:
                    exp_config.hyperparameters.program.external_factor = igp["external_factor"]

    # Write optimal.toml
    exp_config.to_toml(optimal_path)
    console.print(f"[green]Generated: {optimal_path}[/green]")


def _load_study(study_name: str) -> optuna.Study:
    """Load an existing Optuna study."""
    return optuna.load_study(study_name=study_name, storage=STORAGE)


def _create_study(config_name: str) -> str:
    """Create a new Optuna study with a timestamped name."""
    study_name = f"{config_name}_{int(time.time())}"
    optuna.create_study(study_name=study_name, direction="maximize", storage=STORAGE)
    return study_name


def _run_optimization(
    study_name: str, objective: Callable[[optuna.Trial], float], n_trials: int
) -> None:
    """Run optimization on a study."""
    study = _load_study(study_name)
    study.optimize(objective, n_trials=n_trials)


def _is_q_config(config_name: str) -> bool:
    """Check if a config is a Q-learning config."""
    return "with_q" in config_name or config_name.endswith("_q")


def _get_lgp_config_for_q(config_name: str) -> str | None:
    """Get the corresponding LGP config for a Q-learning config."""
    # e.g., cart_pole_with_q -> cart_pole_lgp
    if "with_q" in config_name:
        return config_name.replace("with_q", "lgp")
    return None


def _build_objective(
    study_name: str,
    config_name: str,
    median_trials: int,
    trial: optuna.Trial,
    lgp_parameters: dict[str, Any] | None = None,
) -> float:
    """Build the objective function for Optuna optimization."""
    if lgp_parameters is None:
        max_instructions = trial.suggest_int("max_instructions", 1, 100)
        external_factor = trial.suggest_float("external_factor", 0.0, 100.0)
    else:
        program_parameters = lgp_parameters["program_parameters"]
        max_instructions = program_parameters["max_instructions"]
        external_factor = program_parameters["instruction_generator_parameters"]["external_factor"]

    # Use lgp-cli with override flags
    base_command = [
        "lgp",
        "experiment",
        "run",
        config_name,
        "--override",
        f"hyperparameters.program.max_instructions={max_instructions}",
        "--override",
        f"hyperparameters.program.external_factor={external_factor}",
    ]

    if lgp_parameters:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        alpha_decay = trial.suggest_float("alpha_decay", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.0, 1.0)

        base_command.extend(
            [
                "--override",
                f"operations.q_learning.alpha={alpha}",
                "--override",
                f"operations.q_learning.alpha_decay={alpha_decay}",
                "--override",
                f"operations.q_learning.gamma={gamma}",
                "--override",
                f"operations.q_learning.epsilon={epsilon}",
                "--override",
                f"operations.q_learning.epsilon_decay={epsilon_decay}",
            ]
        )

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

    # Get threshold based on config prefix
    prefix = config_name.split("_")[0]
    threshold = PRUNE_THRESHOLDS.get(prefix, PRUNE_THRESHOLDS["default"])

    _update_best_hyperparameters(champion, hyperparameters)

    if champion < threshold:
        raise optuna.TrialPruned()

    return champion


def _search_single_config(
    config_name: str,
    n_trials: int,
    n_threads: int,
    median_trials: int,
) -> None:
    """Search hyperparameters for a single config."""
    configs = [c.name for c in discover_configs()]

    if config_name not in configs:
        console.print(f"[red]Invalid config: {config_name}[/red]")
        console.print(f"[yellow]Valid configs: {', '.join(configs)}[/yellow]")
        raise typer.Exit(1)

    _reset_best_hyperparameters()

    console.print(f"[bold blue]Starting hyperparameter search for {config_name}[/bold blue]")
    console.print(f"  Trials per thread: {n_trials}")
    console.print(f"  Threads: {n_threads}")
    console.print(f"  Median trials: {median_trials}")

    study_name = _create_study(config_name)
    console.print(f"  Study name: {study_name}")

    if _is_q_config(config_name):
        lgp_config = _get_lgp_config_for_q(config_name)
        if lgp_config:
            lgp_params_path = Path(f"outputs/parameters/{lgp_config}.json")
            if not lgp_params_path.exists():
                console.print(f"[red]LGP parameters not found: {lgp_params_path}[/red]")
                console.print("[yellow]Run LGP search first before Q-learning search[/yellow]")
                raise typer.Exit(1)

            parameters = json.loads(lgp_params_path.read_text())
            objective = partial(
                _build_objective,
                study_name,
                config_name,
                median_trials,
                lgp_parameters=parameters,
            )
        else:
            objective = partial(_build_objective, study_name, config_name, median_trials)
    else:
        objective = partial(_build_objective, study_name, config_name, median_trials)

    results: list[Future[Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Optimizing {config_name}...", total=None)

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
    _save_best_hyperparameters(study_name, config_name)

    console.print("\n[bold green]Search complete![/bold green]")
    console.print(f"  Best score: {_best_score}")


def _search_all_configs(
    n_trials: int,
    n_threads: int,
    median_trials: int,
) -> None:
    """Search hyperparameters for all configs."""
    configs = [c.name for c in discover_configs()]
    lgp_configs = [c for c in configs if not _is_q_config(c)]
    q_configs = [c for c in configs if _is_q_config(c)]

    console.print("[bold blue]Starting hyperparameter search for all configs[/bold blue]")

    console.print("\n[bold]Phase 1: LGP configs[/bold]")
    for config_name in lgp_configs:
        console.print(f"\n[cyan]{'=' * 50}[/cyan]")
        _search_single_config(
            config_name=config_name,
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )

    console.print("\n[bold]Phase 2: Q-learning configs[/bold]")
    for config_name in q_configs:
        console.print(f"\n[cyan]{'=' * 50}[/cyan]")
        _search_single_config(
            config_name=config_name,
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )

    console.print("\n[bold green]All configs completed![/bold green]")


def search(
    config: str = typer.Argument(
        None, help="Config to optimize. If not specified, searches all configs."
    ),
    n_trials: int = typer.Option(40, "--n-trials", "-t", help="Trials per thread"),
    n_threads: int = typer.Option(4, "--n-threads", "-j", help="Parallel threads"),
    median_trials: int = typer.Option(10, "--median-trials", "-m", help="Runs for median"),
) -> None:
    """Search hyperparameters for experiment configs.

    Creates optimal.toml files with best found parameters.

    Examples:
        lgp-tools search                    # Search all configs
        lgp-tools search iris_baseline      # Search specific config
        lgp-tools search -t 20 -j 8         # Custom trials and threads
    """
    if config is None:
        _search_all_configs(
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )
    else:
        _search_single_config(
            config_name=config,
            n_trials=n_trials,
            n_threads=n_threads,
            median_trials=median_trials,
        )
