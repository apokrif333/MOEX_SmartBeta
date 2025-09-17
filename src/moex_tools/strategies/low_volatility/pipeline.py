from __future__ import annotations

import polars as pl


def compute_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["SECID", "DATE"]).with_columns(
            pl.col("CLOSE").pct_change().over("SECID").alias("ret")
        )
    )


def select_low_volatility_universe(
    df: pl.DataFrame,
    lookback_days: int = 252,
    top_n: int = 20,
) -> pl.DataFrame:
    """Monthly rebalance: pick tickers with the lowest realized volatility."""
    df = df.with_columns(pl.col("DATE").dt.truncate("1mo").alias("month"))
    vol = (
        df.group_by(["SECID", "month"]).agg(
            pl.col("ret").rolling_std(window_size=lookback_days, by="SECID").last().alias("vol")
        )
    )
    # rank per month
    ranked = (
        vol.with_columns(pl.col("vol").rank(method="dense").over("month").alias("vol_rank"))
        .sort(["month", "vol_rank", "SECID"])
    )
    picks = ranked.group_by("month").head(top_n)
    return picks


def backtest_equal_weight(
    df: pl.DataFrame,
    picks: pl.DataFrame,
    fee_bps: float = 10.0,
) -> pl.DataFrame:
    """Simple backtest with monthly rebalancing and equal weights."""
    # Merge picks back to returns, keep only selected months
    mdf = df.with_columns(pl.col("DATE").dt.truncate("1mo").alias("month"))
    joined = mdf.join(picks.select(["SECID", "month"]), on=["SECID", "month"], how="inner")

    # Equal weight per month
    weights = joined.group_by("month").agg(pl.len().alias("n")).with_columns((1.0 / pl.col("n")).alias("w"))
    joined = joined.join(weights, on="month", how="left")

    # Apply simple transaction fee on first day of month (approx by subtracting fee once per month)
    daily = (
        joined.sort(["SECID", "DATE"]).with_columns((pl.col("ret") * pl.col("w")).alias("wret"))
    )
    port = (
        daily.group_by("DATE").agg(pl.col("wret").sum().alias("port_ret")).sort("DATE")
        .with_columns(
            pl.when(pl.col("DATE").dt.day() == 1).then(-fee_bps / 10000.0).otherwise(0.0).alias("fee")
        )
        .with_columns((pl.col("port_ret") + pl.col("fee")).alias("port_ret_net"))
        .with_columns((1.0 + pl.col("port_ret_net")).cum_prod().alias("equity"))
    )
    return port


# def run_low_volatility_backtest() -> pl.DataFrame:
#     base = load_base_df()
#     base = compute_daily_returns(base)
#     picks = select_low_volatility_universe(base)
#     result = backtest_equal_weight(base, picks)
#     return result


# run_low_volatility_backtest()