#!/usr/bin/env python

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
import argparse
import glob
import os

DEFAULTS = {
    "iris_baseline": {
        "label": "Iris without Crossover or Mutation",
    },
    "iris_crossover": {"label": "Iris with Crossover"},
    "iris_mutation": {"label": "Iris with Mutation"},
    "iris_full": {"label": "Iris with Crossover and Mutation"},
    "cart_pole_lgp": {"label": "Cart Pole GP"},
    "cart_pole_q": {"label": "Cart Pole Q-Learning"},
    "mountain_car_lgp": {"label": "Mountain Cart GP"},
    "mountain_car_q": {
        "label": "Mountain Car Q-Learning",
    },
}


def generate_tables(
    path: str,
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

            generation_fitness.append(program["fitness"])

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
        "Max": max_fitness,
        "Mean": mean_fitness,
        "Median": median_fitness,
        "Min": min_fitness,
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
        for test in os.listdir(args.input):
            test_base = str(Path(test).stem)
            path = f"{args.input}/{test_base}"
            generate_tables(path, args.output)

    elif args.command == "figures":
        for test in glob.glob(f"{args.input}/*.csv"):
            basename = Path(test).stem
            label = DEFAULTS[basename]["label"]
            generate_figures(test, label, args.output)


if __name__ == "__main__":
    main()
