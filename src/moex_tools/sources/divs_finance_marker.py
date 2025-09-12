from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd
import polars as pl
import requests
import yfinance as yf
from anyio import sleep
from bs4 import BeautifulSoup
from tqdm import tqdm

from moex_tools.config import settings


def _fetch_dividend(ticker: str) -> pd.DataFrame | None:
    """Fetch dividend data for a single ticker from FinanceMarker."""
    main_url = "https://financemarker.ru/stocks/MOEX/{}/dividends/"
    with requests.Session() as s:
        attempt = 0
        while True:
            answ = s.get(main_url.format(ticker), timeout=10)
            if answ.status_code == 500:
                attempt += 1
                if attempt > 5:
                    print(f"{ticker} error 500")
                    return None
                time.sleep(1 * attempt)
            else:
                break

        bs = BeautifulSoup(answ.content, "html.parser")
        table = bs.find_all(
            name="table",
            attrs={"class": "table pd-4-rem table-hover text-secondary text-center"},
        )
        if len(table) == 0:
            print(f"{ticker} doesn't exist any dividend data. Next...")
            return None

        # Pandas
        test_df = pd.read_html(StringIO(str(table[0])), header=0)[0].iloc[:, :3]
        div_dates = test_df.iloc[:, 1].str.split(r"(\d{1,2}\s\w+(?:\s\d{4})?)", expand=True)
        if len(div_dates.columns) < 7:
            print(f"{ticker} has dividend just few days ago. Next...")
            return None

        test_df["name"] = (
            test_df.iloc[:, 0].str.split(ticker, expand=True).iloc[:, -1].str.lstrip().values
        )
        test_df["ticker"] = ticker
        test_df["record_date"] = div_dates[1]
        test_df["exdiv_date"] = div_dates[3]
        test_df["registry_date"] = div_dates[5]

        test_df = test_df.iloc[:, 2:]
        test_df.rename(columns={test_df.columns[0]: "size"}, inplace=True)

        return test_df


def finance_market_download_dividends(stocks_for_parsing: list) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=min(6, len(stocks_for_parsing))) as executor:
        results = [
            df
            for df in tqdm(executor.map(_fetch_dividend, stocks_for_parsing), total=len(stocks_for_parsing),
                           desc="Fetching dividends from Finance Market")
            if df is not None
        ]

    if results:
        return pd.concat(results)
    return pd.DataFrame()


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


def fm_currecy_convert(fm_divs: pl.DataFrame) -> pl.DataFrame:
    fm_divs = (
        fm_divs.with_columns([
            pl.col('exdiv_date').cast(pl.Date),
            pl.when(pl.col("size").str.contains("\$")).then(pl.lit('USD'))
            .when(pl.col("size").str.contains("€")).then(pl.lit('EUR'))
            .when(pl.col("size").str.contains("₽")).then(pl.lit('RUB'))
            .otherwise(None)
            .alias("currency"),
            pl.col("size").str.replace(r" ", "")
            .str.extract(r"([\d.]+)")
            .cast(pl.Float32)
            .alias("size"),
        ])
    )
    check = fm_divs.filter(pl.col("currency").is_null())
    assert len(check) == 0, f"Finance Marker has inappropriate currency: \n{check}"

    currecy_for_load = set(fm_divs['currency'].unique()) - set(['RUB'])
    currecy_for_load = [f'{c}RUB=X' for c in currecy_for_load]

    if currecy_for_load:
        currency_df = yf.download(
            tickers=currecy_for_load, auto_adjust=False, group_by="ticker", period="max"
        )
        currency_df.columns = [f"{c[0]}_{c[1]}" for c in currency_df.columns]
        cls_cols = [c for c in currency_df.columns if c.endswith('_Close')]
        currency_df = pl.from_pandas(currency_df[cls_cols].reset_index()).with_columns(pl.col('Date').dt.date())

        fm_divs = fm_divs.join(currency_df, left_on="exdiv_date", right_on='Date', how="left")
        for c in currecy_for_load:
            short_c = c.replace('RUB=X', '')
            fm_divs = (
                fm_divs.with_columns(
                    pl.when(pl.col("currency") == short_c)
                    .then(pl.col('size') * pl.col(f'{c}_Close'))
                    .otherwise(pl.col('size'))
                    .alias('size')
                ).drop(f'{c}_Close')
            )

    fm_divs = fm_divs.drop('currency')

    return fm_divs


def collect_fm_dividends(stocks_for_parsing: list):
    fm_divs = finance_market_download_dividends(stocks_for_parsing)
    fm_divs = pl.from_pandas(fm_divs)
    fm_divs = fm_date_converter(fm_divs, col_name="record_date")
    fm_divs = fm_date_converter(fm_divs, col_name="exdiv_date")
    fm_divs = fm_date_converter(fm_divs, col_name="registry_date")
    fm_divs = fm_currecy_convert(fm_divs)

    check = fm_divs.filter(fm_divs["ticker", "record_date"].is_duplicated())
    assert len(check) > 0, f"Finance Marker has duplicates: \n{check}"

    fm_divs = fm_divs.group_by(fm_divs.drop(["name", 'size']).columns).agg(pl.col('size').sum())
    fm_divs.write_parquet(
        settings.data_dir / "auxiliary" / "financemarker_dividends.parquet", compression="lz4"
    )
