"""Composable workflow pipelines."""

import typer
from rich.console import Console

app = typer.Typer(name="pipeline", help="Composable workflow pipelines")
console = Console()


@app.command()
def full(
    n_trials: int = typer.Option(40, "--n-trials", "-t", help="Search trials per thread"),
    n_threads: int = typer.Option(4, "--n-threads", "-j", help="Search threads"),
    median_trials: int = typer.Option(10, "--median-trials", "-m", help="Median calculation runs"),
    n_iterations: int = typer.Option(10, "--iterations", "-n", help="Experiment iterations"),
) -> None:
    """Run full pipeline: search-all -> experiments -> analyze.

    This pipeline:
    1. Searches hyperparameters for all environments
    2. Runs experiment iterations with the found parameters
    3. Generates aggregated tables and figures
    """
    from lgp_tools import search, run

    console.print("[bold magenta]=" * 60)
    console.print("[bold magenta]FULL PIPELINE[/bold magenta]")
    console.print("[bold magenta]=" * 60)

    console.print("\n[bold]Step 1/2: Hyperparameter Search[/bold]")
    console.print("[cyan]-" * 40)
    search.all(
        n_trials=n_trials,
        n_threads=n_threads,
        median_trials=median_trials,
    )

    console.print("\n[bold]Step 2/2: Running Experiments[/bold]")
    console.print("[cyan]-" * 40)
    run.experiments(n_iterations=n_iterations)

    console.print("\n[bold magenta]=" * 60)
    console.print("[bold green]FULL PIPELINE COMPLETE![/bold green]")
    console.print("[bold magenta]=" * 60)


@app.command()
def quick(
    n_iterations: int = typer.Option(10, "--iterations", "-n", help="Number of experiment iterations"),
    keep_artifacts: bool = typer.Option(False, "--keep-artifacts", "-k", help="Keep intermediate artifacts"),
) -> None:
    """Run quick pipeline: experiments -> analyze (skip search).

    Uses existing parameters from experiments/assets/parameters/.
    """
    from lgp_tools import run

    console.print("[bold magenta]=" * 60)
    console.print("[bold magenta]QUICK PIPELINE (skip search)[/bold magenta]")
    console.print("[bold magenta]=" * 60)

    console.print("\n[bold]Running Experiments[/bold]")
    console.print("[cyan]-" * 40)
    run.experiments(n_iterations=n_iterations, keep_artifacts=keep_artifacts)

    console.print("\n[bold magenta]=" * 60)
    console.print("[bold green]QUICK PIPELINE COMPLETE![/bold green]")
    console.print("[bold magenta]=" * 60)


@app.command()
def baseline() -> None:
    """Run baseline pipeline: iris experiments with analysis.

    Runs iris variant experiments and generates tables and figures.
    """
    from lgp_tools import run

    console.print("[bold magenta]=" * 60)
    console.print("[bold magenta]BASELINE PIPELINE[/bold magenta]")
    console.print("[bold magenta]=" * 60)

    run.baseline()

    console.print("\n[bold magenta]=" * 60)
    console.print("[bold green]BASELINE PIPELINE COMPLETE![/bold green]")
    console.print("[bold magenta]=" * 60)
