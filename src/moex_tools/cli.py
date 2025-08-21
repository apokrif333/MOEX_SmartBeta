import typer

from moex_tools.pipelines.make_clean import run as make_clean_run

app = typer.Typer(help="MOEX Tools CLI")


@app.command()
def make_clean():
    """Run the clean pipeline for MOEX data."""
    make_clean_run()


if __name__ == "__main__":
    app()
