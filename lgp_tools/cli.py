"""Main CLI entry point with Typer."""

import typer

from lgp_tools import search, run, analyze, pipelines

app = typer.Typer(
    name="lgp-tools",
    help="LGP Experimentation Tools - Hyperparameter search, experiments, and analysis",
    no_args_is_help=True,
)

app.add_typer(search.app, name="search")
app.add_typer(run.app, name="run")
app.add_typer(analyze.app, name="analyze")
app.add_typer(pipelines.app, name="pipeline")


@app.callback()
def callback() -> None:
    """LGP Experimentation Tools.

    A unified CLI for managing Linear Genetic Programming experiments:

    - search: Hyperparameter optimization using Optuna
    - run: Execute experiments (baseline, iterations)
    - analyze: Generate tables and figures from results
    - pipeline: Composable workflows combining the above
    """
    pass


if __name__ == "__main__":
    app()
