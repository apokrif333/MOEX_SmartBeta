import typer

from moex_tools.pipelines.make_clean import run as make_clean_run
from moex_tools.strategies.low_volatility.pipelines import run_weights_for_bot

app = typer.Typer(help="MOEX Tools CLI")


@app.command()
def make_clean():
    make_clean_run()


@app.command()
def low_volatility():
    run_weights_for_bot()


if __name__ == "__main__":
    app()
