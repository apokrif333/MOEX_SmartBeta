from __future__ import annotations
from moex_tools.config import settings
from tqdm import tqdm
from bs4 import BeautifulSoup

import datetime
import polars as pl
import pandas as pd
import requests


def finance_market_download_dividends(stocks_for_parsing: list) -> pd.DataFrame:
    main_url = "https://financemarker.ru/stocks/MOEX/{}/dividends/"

    div_df = pd.DataFrame()
    with requests.Session() as s:
        for t in tqdm(stocks_for_parsing):
            answ = s.get(main_url.format(t))
            if answ.status_code == 500:
                print(f"{t} doesn't exist in the FinanceMarker. Next...")
                continue

            bs = BeautifulSoup(answ.content, "html.parser")
            table = bs.find_all(
                name="table",
                attrs={"class": "table pd-4-rem table-hover text-secondary text-center"},
            )
            if len(table) == 0:
                print(f"{t} doesn't exist any dividend data. Next...")
                continue

            # Pandas
            test_df = pd.read_html(str(table[0]), header=0)[0].iloc[:, :3]
            div_dates = test_df.iloc[:, 1].str.split(r"(\d{1,2}\s\w+(?:\s\d{4})?)", expand=True)
            if len(div_dates.columns) < 7:
                print(f"{t} has dividend just few days ago. Next...")
                continue

            test_df["name"] = (
                (test_df.iloc[:, 0].str.split(t, expand=True).iloc[:, -1]).str.lstrip().values
            )
            test_df["ticker"] = t
            test_df["record_date"] = div_dates[1]
            test_df["exdiv_date"] = div_dates[3]
            test_df["registry_date"] = div_dates[5]

            test_df = test_df.iloc[:, 2:]
            test_df.rename(columns={test_df.columns[0]: "size"}, inplace=True)

            # Concat
            div_df = pd.concat([div_df, test_df])

    return div_df


def fm_date_converter(pl_df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    cur_year = datetime.datetime.now().year
    month_dict = {
        "января": "01",
        "февраля": "02",
        "марта": "03",
        "апреля": "04",
        "мая": "05",
        "июня": "06",
        "июля": "07",
        "августа": "08",
        "сентября": "09",
        "октября": "10",
        "ноября": "11",
        "декабря": "12",
    }
    month_dict_2023 = {
        "января": f"01.{cur_year}",
        "февраля": f"02.{cur_year}",
        "марта": f"03.{cur_year}",
        "апреля": f"04.{cur_year}",
        "мая": f"05.{cur_year}",
        "июня": f"06.{cur_year}",
        "июля": f"07.{cur_year}",
        "августа": f"08.{cur_year}",
        "сентября": f"09.{cur_year}",
        "октября": f"10.{cur_year}",
        "ноября": f"11.{cur_year}",
        "декабря": f"12.{cur_year}",
    }
    for k, v in month_dict.items():
        pl_df = pl_df.with_columns(pl.col(col_name).str.replace(f" {k} ", f".{v}.").alias(col_name))
    for k, v in month_dict_2023.items():
        pl_df = pl_df.with_columns(pl.col(col_name).str.replace(f" {k}", f".{v}").alias(col_name))
    for i in range(1, 10):
        pl_df = pl_df.with_columns(
            pl.col(col_name).str.replace(rf"^{i}\.", rf"0{i}.").alias(col_name)
        )

    pl_df = pl_df.with_columns(pl.col(col_name).str.to_date(format="%d.%m.%Y"))

    return pl_df


def fm_convert_usd_to_rub(fm_divs: pl.DataFrame) -> pl.DataFrame:
    fm_dollars = fm_divs.filter(~pl.col("size").str.contains("₽")).with_columns(
        pl.col("size").str.replace(r" \$", "").cast(pl.Float32).alias("size")
    )

    fm.download_tickers(["RUB=X"], downloader="yahoo")
    usdrub_df = fm.get_tickers(["RUB=X"])["RUB=X"]
    usdrub_pl = pl.from_pandas(usdrub_df.reset_index())
    usdrub_pl = usdrub_pl.with_columns(usdrub_pl["date"].dt.date().alias("date"))

    ctx = pl.SQLContext()
    ctx.register("fm_dollars", fm_dollars)
    ctx.register("usdrub_pl", usdrub_pl)
    fm_dollars = ctx.execute(
        """
        WITH join AS (
            SELECT *
            FROM fm_dollars
            LEFT JOIN usdrub_pl AS usdrub
            ON usdrub.date = fm_dollars.record_date
        )
        SELECT name, ticker, record_date, exdiv_date, registry_date, size * adjClose AS size
        FROM join
    """
    ).collect()

    fm_divs = fm_divs.with_columns(
        pl.col("size").str.replace("₽", "").str.replace("\$", "").alias("size")
    )
    fm_divs = fm_divs.filter(~pl.col("size").str.contains("-")).with_columns(
        pl.col("size").str.replace(r" ", "").str.replace(r" ", "").cast(pl.Float32).alias("size")
    )
    fm_divs = fm_divs.join(
        fm_dollars[["ticker", "record_date", "size"]], on=["ticker", "record_date"], how="left"
    )
    fm_divs = fm_divs.with_columns(
        pl.when(pl.col("size_right").is_null())
        .then(pl.col("size"))
        .otherwise(pl.col("size_right"))
        .alias("size")
    ).drop("size_right")

    return fm_divs


def fm_final_makeup(base_pl: pl.DataFrame, fm_divs: pl.DataFrame) -> pl.DataFrame:
    base_isin = base_pl.unique(subset=["SECID", "isin"])[["SECID", "isin"]]
    fm_divs = fm_divs.join(base_isin, how="left", left_on="ticker", right_on="SECID")

    fm_divs = fm_divs.filter(~fm_divs[["isin", "record_date", "size"]].is_duplicated())
    need_cols = ["name", "ticker", "record_date", "exdiv_date", "registry_date", "isin"]
    fm_divs = fm_divs.group_by(need_cols).agg(pl.col("size").sum())

    return fm_divs


def collect_fm_dividends(stocks_for_parsing: list):
    fm_divs = finance_market_download_dividends(stocks_for_parsing)
    fm_divs = pl.from_pandas(fm_divs)

    fm_divs = fm_divs.filter(~pl.col("exdiv_date").str.contains("23 Даты"))
    fm_divs = fm_date_converter(fm_divs, "record_date")
    fm_divs = fm_date_converter(fm_divs, "exdiv_date")
    fm_divs = fm_date_converter(fm_divs, "registry_date")
    fm_divs = fm_convert_usd_to_rub(fm_divs)
    fm_divs = fm_final_makeup(base_pl, fm_divs)
    fm_divs.write_parquet(f"data/{today}_financemarker_dividends.parquet")
