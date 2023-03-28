#!/usr/bin/env python

import os
from pathlib import Path
import statistics
import subprocess
from glob import glob
import pandas as pd
import argparse

def get_max_fitness(df: pd.DataFrame):
    return df.iloc[-1]["Max Fitness"]

def main(n_times: int):
    BASE_DIR = "assets/experiments"

    # Create the base directory for storing artifacts
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR, exist_ok=True)

    # Run the commands N_TIMES and collect artifacts
    for i in range(n_times):
        # Run cargo nextest
        current_folder = Path(os.path.join(BASE_DIR, f"iteration_{i+1}"))

        os.environ["BENCHMARK_PREFIX"] = str(current_folder / "benchmarks")

        subprocess.run(["cargo", "nextest", "run", "--no-capture", "--release"])

        population_file = current_folder / "benchmarks"
        table_output_dir = current_folder / "tables"

        subprocess.run(
            [
                "./scripts/asset_generator.py",
                f"--input={population_file}",
                f"--output={table_output_dir}",
                "tables",
            ]
        )

        # Create a folder for the current iteration's artifacts
        os.makedirs(current_folder, exist_ok=True)

    # Aggregate the CSV files with the same name
    csv_files = glob(os.path.join(BASE_DIR, "iteration_*", "tables", "*.csv"))
    aggregated_data = {}

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)

        if file_name not in aggregated_data:
            aggregated_data[file_name] = []

        aggregated_data[file_name].append(df)

    # Compute aggregate information and save to a new folder
    aggregate_folder = os.path.join(BASE_DIR, "aggregate_results")
    os.makedirs(aggregate_folder, exist_ok=True)

    for file_name, data_frames in aggregated_data.items():
        # Compute aggregate information
        max_fitness = {get_max_fitness(df): df for df in data_frames}
        max_fitness_mean = statistics.mean(max_fitness.keys())
        mean_df = max_fitness[max_fitness_mean]
        mean_df.to_csv(os.path.join(aggregate_folder, file_name), index=False)

    # for each file in aggregate folder, using produce_assets.py to generate figures
    for csv_file in glob(os.path.join(aggregate_folder, "*.csv")):
        figure_output_dir = os.path.join(aggregate_folder, "figures")
        os.makedirs(figure_output_dir, exist_ok=True)
        subprocess.run(
            [
                "./scripts/asset_generator.py",
                f"--input={Path(csv_file).parent}",
                f"--output={figure_output_dir}",
                "figures",
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run commands N times and aggregate results"
    )
    parser.add_argument("n_times", type=int, help="Number of times to run the commands")

    args = parser.parse_args()
    main(args.n_times)
