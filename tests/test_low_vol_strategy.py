import polars as pl

from moex_tools.strategies.low_volatility.pipeline import run_low_volatility_backtest


def test_run_low_volatility_backtest_smoke(monkeypatch, tmp_path):
    # Create a tiny synthetic dataset that mimics moex_splits_divs.parquet
    dates = pl.date_range(low=pl.datetime(2022, 1, 1), high=pl.datetime(2022, 3, 31), interval="1d").alias("DATE")
    df = (
        pl.DataFrame({
            "DATE": dates,
        })
        .with_columns(
            pl.lit("AAA").alias("SECID"),
            pl.lit("Акции").alias("class"),
            pl.lit(100.0).alias("CLOSE"),
        )
    )
    df2 = df.with_columns(pl.lit("BBB").alias("SECID"), pl.lit(105.0).alias("CLOSE"))
    base = pl.concat([df, df2]).sort(["SECID", "DATE"])\
        .with_columns((pl.arange(0, pl.len()) * 0.001).over("SECID").alias("noise"))\
        .with_columns((pl.col("CLOSE") * (1.0 + pl.col("noise"))).alias("CLOSE"))

    file_path = tmp_path / "moex_splits_divs.parquet"
    base.write_parquet(file_path)

    # Patch loader to read from tmp path
    from moex_tools.strategies.low_volatility import data as data_mod
    monkeypatch.setattr(data_mod, "load_base_df", lambda path=None: pl.read_parquet(file_path))

    res = run_low_volatility_backtest()
    assert isinstance(res, pl.DataFrame)
    assert "equity" in res.columns
    assert res.height > 0
