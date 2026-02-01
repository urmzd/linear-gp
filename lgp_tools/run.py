"""Experiment execution commands."""

import os
import shutil
import subprocess
from glob import glob
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(name="run", help="Experiment execution commands")
console = Console()


def _get_max_fitness(df: pd.DataFrame) -> float:
    """Get the maximum fitness from the last generation."""
    return df.iloc[-1]["Max Fitness"]


@app.command()
def baseline(
    output_dir: str = typer.Option(
        "experiments/assets/baseline",
        "--output",
        "-o",
        help="Output directory for baseline results",
    ),
    temp_dir: str = typer.Option(
        "experiments/assets/tmp",
        "--temp",
        help="Temporary directory for benchmark output",
    ),
) -> None:
    """Run baseline experiments (iris variants).

    Runs iris experiments and generates tables and figures.
    """
    from lgp_tools import analyze

    console.print("[bold blue]Running baseline experiments[/bold blue]")

    output_path = Path(output_dir)
    temp_path = Path(temp_dir)
    figures_dir = output_path / "figures"

    # Create directories
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for benchmark output
    os.environ["BENCHMARK_PREFIX"] = str(temp_path)

    console.print("[cyan]Running iris tests...[/cyan]")
    result = subprocess.run(
        ["cargo", "nextest", "run", "iris", "--no-capture", "--release"],
        capture_output=False,
    )

    if result.returncode != 0:
        console.print("[red]Tests failed![/red]")
        raise typer.Exit(1)

    console.print("[cyan]Generating tables...[/cyan]")
    analyze.tables(input_dir=str(temp_path), output_dir=str(output_path))

    console.print("[cyan]Generating figures...[/cyan]")
    analyze.figures(input_dir=str(output_path), output_dir=str(figures_dir))

    # Cleanup temp directory
    if temp_path.exists():
        shutil.rmtree(temp_path)

    console.print(f"\n[bold green]Baseline experiments complete![/bold green]")
    console.print(f"  Tables: {output_path}")
    console.print(f"  Figures: {figures_dir}")


@app.command()
def experiments(
    n_iterations: int = typer.Argument(10, help="Number of iterations to run"),
    base_dir: str = typer.Option(
        "experiments/assets/experiments",
        "--output",
        "-o",
        help="Base output directory for experiments",
    ),
    keep_artifacts: bool = typer.Option(
        False, "--keep-artifacts", "-k", help="Keep intermediate artifacts"
    ),
    test_filter: str = typer.Option(
        "mountain_car cart_pole",
        "--filter",
        "-f",
        help="Test filter pattern for cargo nextest",
    ),
) -> None:
    """Run N iterations of experiments with aggregation.

    Runs the specified tests multiple times, aggregates results,
    and generates figures.
    """
    from lgp_tools import analyze

    console.print(f"[bold blue]Running {n_iterations} experiment iterations[/bold blue]")

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running iterations", total=n_iterations)

        for i in range(n_iterations):
            current_folder = base_path / f"iteration_{i + 1}"
            population_file = current_folder / "benchmarks"
            table_output_dir = current_folder / "tables"

            os.environ["BENCHMARK_PREFIX"] = str(population_file)

            progress.update(task, description=f"Iteration {i + 1}/{n_iterations}")

            # Run tests
            subprocess.run(
                ["cargo", "nextest", "run"]
                + test_filter.split()
                + ["--no-capture", "--release"],
                capture_output=True,
            )

            # Generate tables for this iteration
            analyze.tables(input_dir=str(population_file), output_dir=str(table_output_dir))

            current_folder.mkdir(parents=True, exist_ok=True)
            progress.advance(task)

    # Aggregate CSV files
    console.print("[cyan]Aggregating results...[/cyan]")

    csv_files = glob(str(base_path / "iteration_*" / "tables" / "*.csv"))
    aggregated_data: dict[str, list[pd.DataFrame]] = {}

    for csv_file in csv_files:
        file_name = Path(csv_file).name
        df = pd.read_csv(csv_file)

        if file_name not in aggregated_data:
            aggregated_data[file_name] = []

        aggregated_data[file_name].append(df)

    # Compute aggregate statistics
    aggregate_folder = base_path / "aggregate_results"
    aggregate_folder.mkdir(parents=True, exist_ok=True)

    for file_name, data_frames in aggregated_data.items():
        agg_df = pd.concat(data_frames)
        agg_df = agg_df.groupby("Generation", as_index=False).mean()
        agg_df.to_csv(aggregate_folder / file_name, index=False)

    # Generate figures from aggregated results
    console.print("[cyan]Generating figures...[/cyan]")
    figure_output_dir = aggregate_folder / "figures"
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    analyze.figures(input_dir=str(aggregate_folder), output_dir=str(figure_output_dir))

    # Cleanup if requested
    if not keep_artifacts:
        console.print("[cyan]Cleaning up intermediate artifacts...[/cyan]")
        for folder_name in os.listdir(base_path):
            folder_path = base_path / folder_name
            if folder_path != aggregate_folder and folder_path.is_dir():
                shutil.rmtree(folder_path)

    console.print(f"\n[bold green]Experiments complete![/bold green]")
    console.print(f"  Aggregated results: {aggregate_folder}")
    console.print(f"  Figures: {figure_output_dir}")
