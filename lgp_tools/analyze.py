"""Analysis commands for generating tables and figures."""

import json
from glob import glob
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(name="analyze", help="Analysis commands for tables and figures")
console = Console()

# Label mappings for figure titles
LABELS = {
    "iris_baseline": "Iris without Crossover or Mutation",
    "iris_crossover": "Iris with Crossover",
    "iris_mutation": "Iris with Mutation",
    "iris_full": "Iris with Crossover and Mutation",
    "cart_pole_lgp": "Cart Pole GP",
    "cart_pole_q": "Cart Pole Q-Learning",
    "mountain_car_lgp": "Mountain Car GP",
    "mountain_car_q": "Mountain Car Q-Learning",
}


def _generate_table_from_population(path: Path, output_dir: Path) -> None:
    """Generate a statistics table from a population.json file."""
    basename = path.name

    population_file = path / "population.json"
    if not population_file.exists():
        console.print(f"[yellow]Skipping {path}: no population.json found[/yellow]")
        return

    with open(population_file, "r") as f:
        programs: list[list[dict[str, Any]]] = json.load(f)

    fitness_scores: list[list[float]] = []

    for i, program_group in enumerate(programs):
        generation_fitness: list[float] = []
        for program in program_group:
            if "program" in program:
                program = program["program"]
            generation_fitness.append(program["fitness"])
        fitness_scores.append(generation_fitness)

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


def _generate_figure_from_table(table_path: Path, output_dir: Path, label: str = "") -> None:
    """Generate a figure from a statistics table CSV."""
    df = pd.read_csv(table_path, index_col="Generation")

    fig, ax = plt.subplots()

    title = "Fitness Evolution"
    if label:
        title = f"{title} ({label})"

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
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_path.stem}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    console.print(f"  [green]Generated[/green] {output_path}")


@app.command()
def tables(
    input_dir: str = typer.Option(
        "experiments/assets/output",
        "--input",
        "-i",
        help="Directory containing population benchmark output",
    ),
    output_dir: str = typer.Option(
        "experiments/assets/tables",
        "--output",
        "-o",
        help="Output directory for CSV tables",
    ),
) -> None:
    """Generate CSV tables from experiment results.

    Reads population.json files from benchmark directories and
    generates statistics tables (max, mean, median, min fitness per generation).
    """
    console.print(f"[bold blue]Generating tables from {input_dir}[/bold blue]")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    count = 0
    for item in input_path.iterdir():
        if item.is_dir():
            _generate_table_from_population(item, output_path)
            count += 1

    if count == 0:
        console.print("[yellow]No benchmark directories found[/yellow]")
    else:
        console.print(f"\n[bold green]Generated {count} tables[/bold green]")


@app.command()
def figures(
    input_dir: str = typer.Option(
        "experiments/assets/tables",
        "--input",
        "-i",
        help="Directory containing CSV tables",
    ),
    output_dir: str = typer.Option(
        "experiments/assets/figures",
        "--output",
        "-o",
        help="Output directory for PNG figures",
    ),
) -> None:
    """Generate PNG figures from CSV tables.

    Reads statistics tables and generates fitness evolution plots.
    """
    console.print(f"[bold blue]Generating figures from {input_dir}[/bold blue]")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        console.print("[yellow]No CSV files found[/yellow]")
        raise typer.Exit(1)

    for csv_file in csv_files:
        basename = csv_file.stem
        label = LABELS.get(basename, "")
        _generate_figure_from_table(csv_file, output_path, label)

    console.print(f"\n[bold green]Generated {len(csv_files)} figures[/bold green]")
