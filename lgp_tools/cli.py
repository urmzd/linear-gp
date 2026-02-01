"""LGP Tools CLI entrypoint."""

import typer

from lgp_tools.commands import analyze, experiment, search

app = typer.Typer(
    name="lgp-tools",
    help="LGP Tools - Hyperparameter search and result analysis",
    no_args_is_help=True,
)

# Register commands from submodules
app.command()(search)
app.command()(analyze)
app.command()(experiment)

if __name__ == "__main__":
    app()
