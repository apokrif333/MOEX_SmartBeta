from __future__ import annotations

import datetime
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import polars as pl
import requests
import urllib3
from tqdm import tqdm

from moex_tools.config import settings
from moex_tools.MOEX_base_functions import (
    get_company_description,
    get_exchange_total_info,
    get_OHLC,
)


def _clean_exchange_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans and preprocesses a Polars DataFrame containing stock or financial exchange
    information. The function filters rows based on specific conditions, renames a
    column, drops unnecessary columns, and casts selected columns to specific data types.

    :param df: Input DataFrame containing stock or financial exchange data to be cleaned.
    :return: A cleaned and preprocessed DataFrame.
    """
    df = df.filter((pl.col("BOARDID") == "MRKT") & (pl.col("NUMTRADES") != 0)).with_columns(
        pl.col("TRADEDATE").alias("DATE")
    )
    if df.is_empty():
        return df

    drop_cols = [
        "WAPRICE",
        "TRENDCLOSE",
        "TRENDWAP",
        "TRENDWAPPR",
        "FACEVALUE",
        "MP2VALTRD",
        "MARKETPRICE3",
        "MARKETPRICE3TRADESVALUE",
        "LISTNAME",
        "SYSTIME",
        "TRADINGSESSION",
        "MARKETPRICE2",
        "ADMITTEDQUOTE",
        "ADMITTEDVALUE",
        "TRADEDATE",
    ]
    df = df.drop(drop_cols)

    int_cols = ["VOLUME", "DAILYCAPITALIZATION", "MONTHLYCAPITALIZATION", "VALUE"]
    df = df.with_columns(pl.col(col).fill_nan(0).cast(pl.Int64) for col in int_cols)

    float_cols = ["OPEN", "HIGH", "LOW", "CLOSE"]
    df = df.with_columns(pl.col(col).cast(pl.Float32) for col in float_cols)

    return df


def fetch_exchange_day(day: str) -> pl.DataFrame:
    """
    Fetches and processes exchange data for a specific day.

    This function attempts to retrieve exchange information for a particular day,
    handling potential connection and request errors. It retries on failures,
    and processes the returned data to ensure proper formatting and cleanliness.

    :param day: The specific day for which the exchange data is being fetched.
    :return: A processed DataFrame containing the exchange data for the given
        day. Returns an empty DataFrame if no relevant data is available or
        if there are no trades.
    """
    df = None
    success = False
    while success is False:
        try:
            df = get_exchange_total_info(day)
            success = True
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.RequestException,
            urllib3.exceptions.ConnectTimeoutError,
        ) as e:
            print(e)
            time.sleep(5)
        except Exception as e:
            print(e)
            raise

    df = pl.DataFrame(df)
    if df.is_empty() or len(df) == 1:
        return pl.DataFrame()

    df = _clean_exchange_df(df)
    if df.is_empty():
        print(f"No trades for BOARDID MRKT on {day}")
        return pl.DataFrame()

    return df


def save_to_parquet(df: pl.DataFrame, name: str) -> None:
    """
    Saves a given Polars DataFrame to a Parquet file with a specified name
    in the raw data directory. This function ensures atomicity of the operation
    by first writing to a temporary file and then replacing the final file.

    :param df: The Polars DataFrame to be saved to a Parquet file.
    :param name: The name of the Parquet file to save, excluding the file's
        extension.
    :return: None
    """
    path = settings.data_dir / "raw" / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def fetch_exchange_range_parallel(end: str | None = None, start: str | None = None) -> None:
    """
    Fetches exchange data for a specified date range in parallel.

    This function retrieves exchange data leveraging parallel processing. Data for each day within the
    specified range is fetched and saved to parquet files, skipping weekends and pre-specified dates.
    If no new data exists to update, the function exits without performing operations.

    :param end: The end date of the range, formatted as a string (YYYYMMDD). Defaults to yesterday's date
                if not provided.
    :param start: The start date of the range, formatted as a string (YYYYMMDD). Defaults to
                  `settings.moex_data_start` if not provided.
    :return: None
    """
    if start is None:
        start = settings.moex_data_start

    if end is None:
        end = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")

    if start > end:
        print("No new data to update.")
        return
    dates = pd.date_range(start, end).strftime("%Y%m%d")

    path = settings.data_dir / "raw"
    already_exist = []
    for f in os.listdir(path):
        pattern = r"^\d{8}_moex_data\.parquet$"
        if bool(re.match(pattern, f)):
            already_exist.append(f.split("_")[0])
    dont_exist = set(dates) - set(already_exist) - settings.war_days
    dont_exist = [d for d in dont_exist if datetime.datetime.strptime(d, "%Y%m%d").weekday() < 5]

    print(f"Fetching data from {start} to {end} in parallel {settings.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        future_to_date = {executor.submit(fetch_exchange_day, d): d for d in dont_exist}

        for future in tqdm(
            as_completed(future_to_date), total=len(dont_exist), desc="Fetching in parallel MOEX data"
        ):
            day = future_to_date[future]
            try:
                result = future.result()
                if not result.is_empty():
                    save_to_parquet(result, name=f"{day}_moex_data.parquet")
            except Exception as e:
                print(f"Error fetching data for {day}: {e}")

    return


def check_loaded_moex_data():
    """
    Checks for missing MOEX data by comparing available dates with dates present in the local database
    and determines if additional data needs to be fetched.

    This function iterates over a list of MOEX tickers and retrieves OHLC data for the defined date range.
    It compares the available exchange dates with files already stored locally and identifies missing dates.
    For any missing data, it triggers the fetching of this data to ensure the dataset is complete.

    :returns: None
    """
    with requests.Session() as s:
        df = pd.DataFrame()
        for t in settings.moex_ticker_for_dates_check:
            candles = get_OHLC(s, start=settings.moex_data_start, ticker=t)
            df = pd.concat([df, candles], ignore_index=True)

        exch_dates = pd.to_datetime(df["begin"]).dt.strftime("%Y%m%d")
        exch_dates = [
            d for d in exch_dates if datetime.datetime.strptime(d, "%Y%m%d").weekday() < 5
        ]

        already_exist = []
        path = settings.data_dir / "raw"
        for f in os.listdir(path):
            pattern = r"^\d{8}_moex_data\.parquet$"
            if bool(re.match(pattern, f)):
                already_exist.append(f.split("_")[0])

        today_str = datetime.datetime.now().strftime("%Y%m%d")
        date_for_load = set(exch_dates) - set(already_exist) - {today_str}
        if len(date_for_load) > 0:
            print(f"Dates to load: {date_for_load}")
            for d in date_for_load:
                fetch_exchange_range_parallel(end=d, start=d, max_workers=1)


def create_union_raw_moex_data():
    """
    Creates or updates a unionized parquet file containing raw MOEX (Moscow Exchange) data from individual files
    located in the raw data directory. It consolidates MOEX data starting from a specified date, processes the
    files to ensure proper data formats, and updates the unionized file.

    If the unionized file does not already exist, it is created with data from all appropriately formatted files
    in the raw directory. Otherwise, the data in the unionized file is augmented with any new data from the raw
    directory that postdates the maximum date in the existing unionized file.

    :returns: None
    """
    union_path = settings.data_dir / "raw" / "union_raw_moex_data.parquet"
    path = settings.data_dir / "raw"

    if not union_path.exists():
        total_df = pl.DataFrame()
        last_date = (pd.to_datetime(settings.moex_data_start) - pd.Timedelta(days=1)).strftime(
            "%Y%m%d"
        )
    else:
        total_df = pl.read_parquet(union_path)
        last_date = pd.to_datetime(total_df["DATE"].max()).strftime("%Y%m%d")

    save_data = False
    files = os.listdir(path)
    print(f"Last date in union: {last_date}")
    for f in tqdm(files, desc="Checking files in raw directory", total=len(files)):
        pattern = r"^\d{8}_moex_data\.parquet$"
        if bool(re.match(pattern, f)):
            cur_date = f.split("_")[0]
            if cur_date <= last_date:
                continue

            save_data = True
            cur_df = pl.read_parquet(path / f)
            float_cols = ["OPENVAL", "CLOSEVAL"]
            cur_df = cur_df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            total_df = pl.concat([total_df, cur_df], how="vertical")

    if save_data:
        total_df = total_df.sort(["SECID", "DATE"])
        save_to_parquet(total_df, name="union_raw_moex_data.parquet")


def create_description_json():
    """
    Generates or updates a JSON file containing descriptions for all stocks by fetching
    their details from an external data source, combining the new data with any
    existing data in the JSON file.

    This function reads a source parquet file containing raw stock data to identify
    the list of stock tickers (SECID) that require descriptions. If a JSON file with
    existing descriptions is found, it will skip fetching data for already described
    tickers, reducing unnecessary requests. For remaining tickers, it fetches detailed
    descriptions, processes the data, and updates the JSON file accordingly.

    :returns: None
    """
    union_path = settings.data_dir / "raw" / "union_raw_moex_data.parquet"
    descipt_path = settings.data_dir / "auxiliary" / "all_stocks_description.json"

    total_df = pl.read_parquet(union_path)
    tickers = set(total_df["SECID"])
    if descipt_path.exists():
        with open(descipt_path) as f:
            description_dict = json.load(f)
        tickers = tickers - description_dict.keys()
    else:
        description_dict = {}

    new_description = {}
    with requests.Session() as s:
        for t in tqdm(tickers, desc="Fetching company descriptions", total=len(tickers)):
            df = get_company_description(s, t)
            cur_dict = df.drop(columns=["name"]).set_index("title").to_dict()["value"]
            new_description[t] = cur_dict

    description_dict = {**description_dict, **new_description}
    with open(descipt_path, "w") as f:
        json.dump(description_dict, f, indent=4, ensure_ascii=False)


def update_description_in_new_prices():
    """
    Updates the description data in the new prices dataset by joining stock description data
    with the union of raw MOEX data. The function reads stock descriptions from a JSON file
    and combines it with raw MOEX price data stored in a Parquet file. The updated data is
    saved as a new Parquet file.

    :return: None
    """
    descipt_path = settings.data_dir / "auxiliary" / "all_stocks_description.json"
    with open(descipt_path) as f:
        description_dict = json.load(f)

    dict_for_connect = {"ticker": [], "name": [], "regNumber": [], "category": [], "codeType": [], "type": [],
                        "class": [], "code": [], "isin": []}
    for v in description_dict.values():
        dict_for_connect["ticker"].append(v["Код ценной бумаги"])
        dict_for_connect["name"].append(v["Полное наименование"])
        dict_for_connect["category"].append(v["Вид/категория ценной бумаги"])
        dict_for_connect["codeType"].append(v["Код ценной бумаги"])
        dict_for_connect["type"].append(v["Код типа инструмента"])
        dict_for_connect["class"].append(v["Типа инструмента"])

        if "Номер государственной регистрации" in v.keys():
            dict_for_connect["regNumber"].append(v["Номер государственной регистрации"])
        else:
            dict_for_connect["regNumber"].append(None)

        if "Код эмитента" in v.keys():
            dict_for_connect["code"].append(v["Код эмитента"])
        else:
            dict_for_connect["code"].append(None)

        if "ISIN код" in v.keys():
            dict_for_connect["isin"].append(v["ISIN код"])
        elif "Код ценной бумаги" in v.keys():
            dict_for_connect["isin"].append(v["Код ценной бумаги"])
        else:
            dict_for_connect["isin"].append(None)
    description_pl = pl.DataFrame(dict_for_connect)

    union_path = settings.data_dir / "raw" / "union_raw_moex_data.parquet"
    total_df = pl.read_parquet(union_path)

    total_df = total_df.join(
        description_pl[["ticker", "category", "type", "class", "isin"]],
        left_on="SECID",
        right_on="ticker",
        how="left",
    )
    total_df.write_parquet(
        settings.data_dir / "raw" / "union_raw_moex_descript.parquet", compression="lz4"
    )


def change_secid():
    """
    Processes a dataset to adjust and correct 'SECID' values for financial instruments.

    This function reads a dataset of financial instruments from a Parquet file, applies
    a series of transformations to detect and resolve discrepancies in 'SECID' values
    based on specific business logic, and writes the corrected dataset back to the file
    system. It ensures consistency of 'SECID' values by identifying and replacing
    incorrect or outdated values.

    :return: None
    """
    path = settings.data_dir / "raw" / "union_raw_moex_descript.parquet"
    total_df = pl.read_parquet(path).with_columns(pl.col('DATE').cast(pl.Date))

    secid_correct = (
        total_df.sort('isin', 'DATE')
        .filter(pl.col('ISSUESIZE') != 0)
        .with_columns(
            date_diff=(pl.col('DATE') - pl.col('DATE').shift(1)).over('isin').dt.total_days()
        ).filter(
            (
                (pl.col('SECID') != pl.col('SECID').shift(1)).over('isin') &
                (pl.col('category') == pl.col('category').shift(1)).over('isin') &
                (pl.col('date_diff') > 0) &
                (pl.col('ISSUESIZE') == pl.col('ISSUESIZE').shift(1)).over('isin')
            )
        ).sort('isin', 'DATE')
        .group_by('isin').last()
    )
    secid_correct = secid_correct['SECID', 'isin'].rename({'SECID': 'main_SECID'})

    total_df = (
        total_df.join(secid_correct, on='isin', how='left')
        .with_columns(
            pl.when(pl.col('main_SECID').is_null()).then(pl.col('SECID')).otherwise(pl.col('main_SECID')).alias('SECID')
        ).sort('SECID', 'DATE')
        .drop('main_SECID')
    )

    total_df.write_parquet(
        settings.data_dir / "raw" / "union_raw_moex_descript.parquet", compression="lz4"
    )

def collect_moex_data():
    """
    Gathers MOEX (Moscow Exchange) data by performing a series of operations to collect,
    validate, combine, and update the data structure for further processing.

    :return: None
    """
    fetch_exchange_range_parallel()
    check_loaded_moex_data()
    create_union_raw_moex_data()
    create_description_json()
    update_description_in_new_prices()
    change_secid()