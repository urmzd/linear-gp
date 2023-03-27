#!/usr/bin/env python

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_fitness_benchmarks(
    path: str = "assets/logs/cart-pole-lgp",
    label: str = "",
    fallback_fitness: float = -200.0,
) -> None:
    # Load programs from JSON file.
    basename: str = Path(path).name

    with open(Path(path) / "plot.json", "r") as f:
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
    mean_fitness: List[float] = [
        np.mean(generation_fitness) for generation_fitness in fitness_scores
    ]
    max_fitness: List[float] = [
        np.max(generation_fitness) for generation_fitness in fitness_scores
    ]
    min_fitness: List[float] = [
        np.min(generation_fitness) for generation_fitness in fitness_scores
    ]
    median_fitness: List[float] = [
        np.median(generation_fitness) for generation_fitness in fitness_scores
    ]
    std_fitness: List[float] = [
        np.std(generation_fitness) for generation_fitness in fitness_scores
    ]

    # Create a pandas DataFrame with the statistics.
    data: Dict[str, List[float]] = {
        "max_fitness": max_fitness,
        "mean_fitness": mean_fitness,
        "median_fitness": median_fitness,
        "min_fitness": min_fitness,
        "std_fitness": std_fitness,
    }

    df: pd.DataFrame = pd.DataFrame(data)
    df.index.name = "Generation"

    # Save the DataFrame as a CSV file in the assets/tables directory.
    tables_path: Path = Path("assets/tables/")
    tables_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_path / f"{basename}.csv")

    # Plot fitness scores as lines.
    fig, ax = plt.subplots()

    title: str = "Fitness Evolution"

    if label != "":
        title = f"{title} ({label})"

    ax.plot(generations, max_fitness, label="max", linestyle="-", marker="o")
    ax.plot(generations, mean_fitness, label=r"$\mu$", linestyle="--", marker="v")
    ax.plot(generations, median_fitness, label="median", linestyle="-.", marker="s")
    ax.plot(generations, min_fitness, label="min", linestyle=":", marker="D")
    ax.plot(generations, std_fitness, label=r"$\sigma$", linestyle="-", marker="x")

    ax.set_title(title)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    ax.grid(visible=True, which="both")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig_path: Path = Path("assets/images/")
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / f"{basename}.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_fitness_benchmarks(
        "assets/logs/iris_baseline", "Iris without Crossover or Mutation", 0
    )
    plot_fitness_benchmarks("assets/logs/iris_crossover", "Iris with Crossover", 0)
    plot_fitness_benchmarks("assets/logs/iris_mutation", "Iris with Mutation", 0)
    plot_fitness_benchmarks(
        "assets/logs/iris_full", "Iris with Crossover and Mutation", 0
    )

    plot_fitness_benchmarks("assets/logs/cart_pole_lgp", "Cart Pole GP", 0)
    plot_fitness_benchmarks("assets/logs/cart_pole_q", "Cart Pole Q-Learning", 0)
    plot_fitness_benchmarks("assets/logs/mountain_car_lgp", "Mountain Cart GP", -200)
    plot_fitness_benchmarks(
        "assets/logs/mountain_car_q", "Mountain Car Q-Learning", -200
    )
