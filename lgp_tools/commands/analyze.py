"""Analyze command for generating tables and figures from experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from lgp_tools.config import get_configs_dir
from lgp_tools.models import ExperimentConfig, Population

console = Console()


def _load_config_metadata(config_name: str) -> ExperimentConfig | None:
    """Load experiment config to get metadata."""
    configs_dir = get_configs_dir()
    config_path = configs_dir / config_name / "default.toml"
    if config_path.exists():
        return ExperimentConfig.from_toml(config_path)
    return None


def _generate_table_from_population(path: Path, output_dir: Path) -> None:
    """Generate a statistics table from a population.json file."""
    basename = path.name

    population_file = path / "population.json"
    if not population_file.exists():
        console.print(f"[yellow]Skipping {path}: no population.json found[/yellow]")
        return

    # Use Pydantic model for validation
    population = Population.from_json(population_file)
    fitness_scores = population.get_fitness_by_generation()

    mean_fitness = [np.mean(gen) for gen in fitness_scores]
    max_fitness = [np.max(gen) for gen in fitness_scores]
    min_fitness = [np.min(gen) for gen in fitness_scores]
    median_fitness = [np.median(gen) for gen in fitness_scores]

    data = {
        "Max Fitness": max_fitness,
        "Mean Fitness": mean_fitness,
        "Median Fitness": median_fitness,
        "Min Fitness": min_fitness,
    }

    df = pd.DataFrame(data)
    df.index.name = "Generation"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{basename}.csv"
    df.to_csv(output_path)

    console.print(f"  [green]Generated[/green] {output_path}")


def _generate_figure_from_table(table_path: Path, output_dir: Path) -> None:
    """Generate a figure from a statistics table CSV."""
    df = pd.read_csv(table_path, index_col="Generation")
    config_name = table_path.stem

    # Try to load metadata from config
    config = _load_config_metadata(config_name)

    # Get title and labels from metadata or use defaults
    if config and config.metadata.title:
        title = config.metadata.title
    else:
        title = config_name.replace("_", " ").title()

    x_label = config.metadata.x_label if config else "Generation"
    y_label = config.metadata.y_label if config else "Fitness"

    fig, ax = plt.subplots()

    # Handle both old and new column naming conventions
    max_col = "Max Fitness" if "Max Fitness" in df.columns else "Max"
    mean_col = "Mean Fitness" if "Mean Fitness" in df.columns else "Mean"
    median_col = "Median Fitness" if "Median Fitness" in df.columns else "Median"
    min_col = "Min Fitness" if "Min Fitness" in df.columns else "Min"

    ax.plot(df.index, df[max_col], label="max")
    ax.plot(df.index, df[mean_col], label=r"$\mu$")
    ax.plot(df.index, df[median_col], label="median")
    ax.plot(df.index, df[min_col], label="min")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_path.stem}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    console.print(f"  [green]Generated[/green] {output_path}")


def analyze(
    input_dir: Path = typer.Option(
        Path("outputs/output"),
        "--input",
        "-i",
        help="Directory with population.json files",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output",
        "-o",
        help="Output directory for tables and figures",
    ),
) -> None:
    """Generate tables and figures from experiment results.

    Reads population.json files, generates CSV statistics tables,
    then creates PNG plots from the tables.

    Examples:
        lgp-tools analyze
        lgp-tools analyze --input outputs/output --output outputs
    """
    # Generate tables
    tables_dir = output_dir / "tables"

    console.print(f"[bold blue]Generating tables from {input_dir}[/bold blue]")

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    table_count = 0
    for item in input_dir.iterdir():
        if item.is_dir():
            _generate_table_from_population(item, tables_dir)
            table_count += 1

    if table_count == 0:
        console.print("[yellow]No benchmark directories found[/yellow]")
        raise typer.Exit(1)
    else:
        console.print(f"\n[bold green]Generated {table_count} tables[/bold green]")

    # Generate figures
    figures_dir = output_dir / "figures"

    console.print(f"\n[bold blue]Generating figures from {tables_dir}[/bold blue]")

    csv_files = list(tables_dir.glob("*.csv"))

    if not csv_files:
        console.print("[yellow]No CSV files found[/yellow]")
        raise typer.Exit(1)

    for csv_file in csv_files:
        _generate_figure_from_table(csv_file, figures_dir)

    console.print(f"\n[bold green]Generated {len(csv_files)} figures[/bold green]")
    console.print("\n[bold]Analysis complete![/bold]")
    console.print(f"  Tables: {tables_dir}")
    console.print(f"  Figures: {figures_dir}")
