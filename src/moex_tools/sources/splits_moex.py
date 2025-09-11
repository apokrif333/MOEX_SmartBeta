import polars as pl

from moex_tools.config import settings


def detect_splits(df: pl.DataFrame):
    """
    Analyzes a Polars DataFrame to detect stock splits based on specific column values and conditions.
    The function processes the DataFrame by sorting, generating shifted columns, and applying a series
    of filters and transformations. It identifies rows where stock splits occurred, using ratios and
    error calculations, and outputs the resulting DataFrame to a Parquet file.

    :param df: A Polars DataFrame expected to contain the required columns such as "SECID", "BOARDID",
        "CURRENCYID", "OPEN", "HIGH", "LOW", "CLOSE", "ISSUESIZE", and "DATE".
    :type df: polars.DataFrame
    :raises ValueError: If the input DataFrame is missing any of the required columns.
    :return: None. The function saves the processed data containing detected stock splits to a Parquet file
        with columns: SECID, DATE, CLOSE, prev_CLOSE, err_ohlc_mean, rI, rI_rev, rI_frac, rI_rev_frac,
        ISSUESIZE, and prev_ISSUESIZE.
    """
    needed = ["SECID", "BOARDID", "CURRENCYID", "OPEN", "HIGH", "LOW", "CLOSE", "ISSUESIZE", "DATE"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.sort(["SECID", "DATE"]).with_columns([
        pl.col(c).shift(1).over("SECID").alias(f"prev_{c}")
        for c in ["OPEN", "HIGH", "LOW", "CLOSE", "ISSUESIZE", "BOARDID", "CURRENCYID"]
    ])

    valid = (
        (pl.col("CLOSE") > 0)
        & (pl.col("prev_CLOSE") > 0)
        & (pl.col("OPEN") > 0)
        & (pl.col("prev_OPEN") > 0)
        & (pl.col("HIGH") > 0)
        & (pl.col("prev_HIGH") > 0)
        & (pl.col("LOW") > 0)
        & (pl.col("prev_LOW") > 0)
        & (pl.col("ISSUESIZE") != 0)
        & (pl.col("prev_ISSUESIZE") != 0)
        & (pl.col("BOARDID") == pl.col("prev_BOARDID"))
        & (pl.col("CURRENCYID") == pl.col("prev_CURRENCYID"))
        & (pl.col("DATE").is_not_null())
    )
    df = df.filter(valid)

    df = df.with_columns([
        (pl.col("prev_OPEN") / pl.col("OPEN")).alias("rO"),
        (pl.col("prev_HIGH") / pl.col("HIGH")).alias("rH"),
        (pl.col("prev_LOW") / pl.col("LOW")).alias("rL"),
        (pl.col("prev_CLOSE") / pl.col("CLOSE")).alias("rC"),
        (pl.col("ISSUESIZE") / pl.col("prev_ISSUESIZE")).alias("rI"),
        (pl.col("prev_ISSUESIZE") / pl.col("ISSUESIZE")).alias("rI_rev"),
    ])
    df = df.with_columns([
        (abs(pl.col("rO") / pl.col("rI") - 1.0)).alias("errO"),
        (abs(pl.col("rH") / pl.col("rI") - 1.0)).alias("errH"),
        (abs(pl.col("rL") / pl.col("rI") - 1.0)).alias("errL"),
        (abs(pl.col("rC") / pl.col("rI") - 1.0)).alias("errC"),
    ])
    df = df.with_columns(
        ((pl.col("errO") + pl.col("errH") + pl.col("errL") + pl.col("errC")) / 4.0).alias("err_ohlc_mean")
    ).drop(["errO", "errH", "errL", "errC"])

    splits = df.filter(
            ((pl.col("rI") > 1.1) | (pl.col("rI_rev") > 1.1)) &
            (pl.col("err_ohlc_mean") < 0.3)
        ).with_columns([
            pl.col("rI").cast(pl.Int64).alias("rI_int"),
            pl.col("rI_rev").cast(pl.Int64).alias("rI_rev_int"),
        ]).with_columns([
            pl.when(pl.col("rI_int") != 0)
            .then(pl.col("rI") % pl.col("rI_int"))
            .otherwise(pl.col("rI"))
            .alias("rI_frac"),
            pl.when(pl.col("rI_rev_int") != 0)
            .then(pl.col("rI_rev") % pl.col("rI_rev_int"))
            .otherwise(pl.col("rI_rev"))
            .alias("rI_rev_frac"),
        ]).filter(
            (pl.col("rI") / pl.col("rI_int")).is_close(1) | (pl.col("rI_rev") / pl.col("rI_rev_int")).is_close(1) |
            ((pl.col("rI_frac") % 0.01) < 1e-12) | ((pl.col("rI_rev_frac") % 0.01) < 1e-12)
        )

    splits = splits["SECID", "DATE", "CLOSE", "prev_CLOSE", "err_ohlc_mean", "rI", "rI_rev", "rI_frac", "rI_rev_frac",
                    "ISSUESIZE", "prev_ISSUESIZE"]
    splits.write_parquet(settings.data_dir / "auxiliary" / "splits_moex.parquet")


def create_split_base():
    """Creates a base dataframe for detecting splits."""
    path = settings.data_dir / "raw" / "union_raw_moex_descript.parquet"
    base_pl = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Date))
        .filter(pl.col("class") == "Акции")
    )

    detect_splits(base_pl)


def create_split_adjusted_ohlc():
    """
    Creates a split-adjusted OHLC (Open, High, Low, Close) dataset by combining raw data
    with stock split adjustments. The function reads a base dataset and a corresponding
    splits dataset, calculates adjusted factors for OHLC prices based on split data, and
    writes the split-adjusted dataset to an output file.

    :raises FileNotFoundError: If the specified input files do not exist.
    :raises IOError: If there is an I/O issue while reading or writing files.

    :return: None
    """
    base_path = settings.data_dir / "raw" / "union_raw_moex_descript.parquet"
    splits_path = settings.data_dir / "auxiliary" / "splits_moex.parquet"
    out_path = settings.data_dir / "intermediate" / "union_raw_moex_descript_split.parquet"

    base = (
        pl.read_parquet(base_path)
        .with_columns(pl.col("DATE").cast(pl.Date))
    )
    splits = (
        pl.read_parquet(splits_path)
        .with_columns(pl.col("DATE").cast(pl.Date))
        .rename({'rI': 'split'})
    )

    base = (
        base.join(splits["SECID", "DATE", 'split'], on=["SECID", "DATE"], how="left")
        .sort(["SECID", "DATE"])
        .with_columns(split=1 / pl.col("split").fill_null(1.0))
        .with_columns(adj_factor=pl.col('split').shift(-1, fill_value=1).cum_prod(reverse=True).over('SECID'))
    )

    adj_exprs = [(pl.col(c) * pl.col("adj_factor")).alias(f"{c}_adj") for c in ["OPEN", "HIGH", "LOW", "CLOSE"]]
    base = base.with_columns(adj_exprs)

    base.write_parquet(out_path)