#!/usr/bin/env python

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
import argparse

DEFAULTS = {
    "iris_baseline": {
        "label": "Iris without Crossover or Mutation",
        "fallback_fitness": 0,
    },
    "iris_crossover": {"label": "Iris with Crossover", "fallback_fitness": 0},
    "iris_mutation": {"label": "Iris with Mutation", "fallback_fitness": 0},
    "iris_full": {"label": "Iris with Crossover and Mutation", "fallback_fitness": 0},
    "cart_pole_lgp": {"label": "Cart Pole GP", "fallback_fitness": 0},
    "cart_pole_q": {"label": "Cart Pole Q-Learning", "fallback_fitness": 0},
    "mountain_car_lgp": {"label": "Mountain Cart GP", "fallback_fitness": -200},
    "mountain_car_q": {
        "label": "Mountain Car Q-Learning",
        "fallback_fitness": -200,
    },
}


def generate_tables(
    path: str,
    fallback_fitness: float = 0.0,
    output_dir: str = "assets/tables",
) -> None:
    # Load programs from JSON file.
    basename: str = Path(path).name

    with open(Path(path) / "population.json", "r") as f:
        programs: List[List[Dict[str, Any]]] = json.load(f)

    # Extract fitness scores and generation information from programs.
    fitness_scores: List[List[float]] = []
    generations: List[int] = []
    for i, program_group in enumerate(programs):
        generation_fitness: List[float] = []
        for program in program_group:
            if "program" in program:
                program = program["program"]

            if "Valid" in program["fitness"]:
                fitness = program["fitness"]["Valid"]
            else:
                fitness = fallback_fitness
            generation_fitness.append(fitness)

        fitness_scores.append(generation_fitness)
        generations.append(i)

    # Compute statistics of fitness scores.
    mean_fitness: List[np.floating[Any]] = [
        np.mean(generation_fitness) for generation_fitness in fitness_scores
    ]
    max_fitness: List[Any] = [
        np.max(generation_fitness) for generation_fitness in fitness_scores
    ]
    min_fitness: List[Any] = [
        np.min(generation_fitness) for generation_fitness in fitness_scores
    ]
    median_fitness: List[np.floating[Any]] = [
        np.median(generation_fitness) for generation_fitness in fitness_scores
    ]

    # Create a pandas DataFrame with the statistics.
    data: Dict[str, Any] = {
        "Max Fitness": max_fitness,
        "Mean Fitness": mean_fitness,
        "Median Fitness": median_fitness,
        "Min Fitness": min_fitness,
    }

    df: pd.DataFrame = pd.DataFrame(data)
    df.index.name = "Generation"

    # Save the DataFrame as a CSV file in the specified output directory.
    tables_path: Path = Path(output_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_path / f"{basename}.csv")


def generate_figures(
    table_path: str, label: str = "", output_dir: str = "assets/figures"
):
    df = pd.read_csv(table_path, index_col="Generation")

    fig, ax = plt.subplots()

    title: str = "Fitness Evolution"

    if label != "":
        title = f"{title} ({label})"

    ax.plot(df.index, df["Max Fitness"], label="max")
    ax.plot(df.index, df["Mean Fitness"], label=r"$\mu$")
    ax.plot(df.index, df["Median Fitness"], label="median")
    ax.plot(df.index, df["Min Fitness"], label="min")

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig_path: Path = Path(output_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / f"{Path(table_path).stem}.png", bbox_inches="tight", dpi=300)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tables and plots for fitness data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing either a table for figure generation or a population.json file for table generation.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for tables or figures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Tables subcommand
    subparsers.add_parser("tables", help="Generate tables.")

    # Figures subcommand
    subparsers.add_parser("figures", help="Generate figures.")

    args = parser.parse_args()

    if args.command == "tables":
        for stem, values in DEFAULTS.items():
            path = f"{args.input}/{stem}"
            label = values["label"]
            fallback_fitness = values["fallback_fitness"]
            generate_tables(path, float(fallback_fitness), args.output)

    elif args.command == "figures":
        for stem, values in DEFAULTS.items():
            table_path = f"{args.input}/{stem}.csv"
            label = values["label"]
            generate_figures(table_path, label, args.output)


if __name__ == "__main__":
    main()
