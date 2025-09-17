import typer

from moex_tools.pipelines.make_clean import run as make_clean_run
# from moex_tools.strategies.low_volatility.pipeline import run_low_volatility_backtest

app = typer.Typer(help="MOEX Tools CLI")


@app.command()
def make_clean():
    make_clean_run()


# @app.command()
# def low_volatility():
#     """Run low-volatility backtest and print last equity value."""
#     res = run_low_volatility_backtest()
#     # Print final equity and rows count
#     final_equity = res.select("equity").tail(1).to_series()[0]
#     typer.echo(f"Rows: {res.height}, Final equity: {final_equity:.4f}")


if __name__ == "__main__":
    app()
