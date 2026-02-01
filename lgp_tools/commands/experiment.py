"""Experiment command for running end-to-end experiment pipelines."""

from pathlib import Path
from subprocess import PIPE, Popen

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lgp_tools.commands.analyze import analyze
from lgp_tools.commands.search import _search_all_configs, _search_single_config
from lgp_tools.config import discover_configs, get_configs_dir

console = Console()


def _run_single_experiment(config_name: str, iteration: int) -> bool:
    """Run a single experiment iteration."""
    configs_dir = get_configs_dir()
    config_dir = configs_dir / config_name

    # Use optimal.toml if it exists, otherwise default.toml
    optimal_path = config_dir / "optimal.toml"
    default_path = config_dir / "default.toml"

    if optimal_path.exists():
        config_file = "optimal.toml"
    elif default_path.exists():
        config_file = "default.toml"
    else:
        console.print(f"[red]No config found for {config_name}[/red]")
        return False

    command = ["lgp", "experiment", "run", config_name, "--config", config_file]
    logger.trace(f"Running iteration {iteration}: {' '.join(command)}")

    process = Popen(command, stdout=PIPE, stderr=PIPE)
    _, error = process.communicate()

    if process.returncode != 0:
        console.print(f"[red]Iteration {iteration} failed: {error.decode('utf-8')}[/red]")
        return False

    return True


def _run_experiments(config_name: str | None, iterations: int) -> bool:
    """Run experiments for one or all configs."""
    configs = [c.name for c in discover_configs()]

    if config_name:
        if config_name not in configs:
            console.print(f"[red]Invalid config: {config_name}[/red]")
            console.print(f"[yellow]Valid configs: {', '.join(configs)}[/yellow]")
            return False
        target_configs = [config_name]
    else:
        target_configs = configs

    console.print(
        f"[bold blue]Running {iterations} iterations for {len(target_configs)} config(s)[/bold blue]"
    )

    for cfg in target_configs:
        console.print(f"\n[cyan]{'=' * 50}[/cyan]")
        console.print(f"[bold]Config: {cfg}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {cfg}...", total=iterations)

            for i in range(iterations):
                success = _run_single_experiment(cfg, i + 1)
                if not success:
                    console.print(f"[yellow]Warning: iteration {i + 1} failed[/yellow]")
                progress.update(task, advance=1)

        console.print(f"[green]Completed {iterations} iterations for {cfg}[/green]")

    return True


def experiment(
    config: str = typer.Argument(None, help="Config to run. If not specified, runs all."),
    iterations: int = typer.Option(
        10, "--iterations", "-n", help="Number of experiment iterations"
    ),
    skip_search: bool = typer.Option(False, "--skip-search", help="Skip hyperparameter search"),
    skip_analyze: bool = typer.Option(False, "--skip-analyze", help="Skip analysis"),
    n_trials: int = typer.Option(40, "--n-trials", "-t", help="Search trials per thread"),
    n_threads: int = typer.Option(4, "--n-threads", "-j", help="Search threads"),
    median_trials: int = typer.Option(10, "--median-trials", "-m", help="Runs for median"),
) -> None:
    """Run end-to-end experiment pipeline.

    Steps:
    1. Search: Find optimal hyperparameters (creates optimal.toml)
    2. Run: Execute experiments N times with optimal params
    3. Analyze: Generate tables and figures

    Examples:
        lgp-tools experiment                    # Full pipeline, all configs
        lgp-tools experiment iris_baseline      # Single config
        lgp-tools experiment --skip-search      # Use existing optimal.toml
        lgp-tools experiment -n 20              # 20 iterations
    """
    console.print("[bold blue]Starting experiment pipeline[/bold blue]")

    # Phase 1: Search
    if not skip_search:
        console.print("\n[bold]Phase 1: Hyperparameter Search[/bold]")
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
    else:
        console.print("\n[yellow]Skipping hyperparameter search[/yellow]")

    # Phase 2: Run experiments
    console.print("\n[bold]Phase 2: Running Experiments[/bold]")
    success = _run_experiments(config, iterations)
    if not success:
        console.print("[red]Experiment runs failed[/red]")
        raise typer.Exit(1)

    # Phase 3: Analyze
    if not skip_analyze:
        console.print("\n[bold]Phase 3: Analysis[/bold]")
        # Call analyze with defaults
        analyze(
            input_dir=Path("outputs/output"),
            output_dir=Path("outputs"),
        )
    else:
        console.print("\n[yellow]Skipping analysis[/yellow]")

    console.print("\n[bold green]Pipeline complete![/bold green]")
