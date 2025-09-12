import datetime
import time

import polars as pl
from tinkoff.invest import Client
from tinkoff.invest.utils import now
from tqdm import tqdm

from moex_tools.config import settings

MAX_REQUESTS = 199


def tink_get_figi(isin_for_parsing: list) -> pl.DataFrame:
    """
    Retrieves FIGI (Financial Instrument Global Identifier) information from Tinkoff API for given ISINs.
    The function caches results in a parquet file to avoid redundant API calls.

    :param isin_for_parsing: List of ISIN codes to get FIGI information for
    :return: Polars DataFrame containing ISIN, FIGI, ticker and class code information
    """
    tink_figi_path = settings.data_dir / "auxiliary" / "tinkoff_figi.parquet"

    if tink_figi_path.exists():
        tink_figi_pl = pl.read_parquet(tink_figi_path)
        existing_isins = set(tink_figi_pl["isin"].to_list())
        isins_for_parsing = [isin for isin in isin_for_parsing if isin not in existing_isins]
    else:
        tink_figi_pl = pl.DataFrame({"isin": [], "figi": [], "ticker": [], "class_code": []})
        isins_for_parsing = isin_for_parsing
        tink_figi_path.parent.mkdir(parents=True, exist_ok=True)

    if isins_for_parsing:
        tink_figi = {"isin": [], "figi": [], "ticker": [], "class_code": []}
        count = 0
        with Client(settings.tinkoff_api_token) as client:
            for isin in tqdm(isins_for_parsing, desc="Fetching FIGI from Tinkoff"):
                answ = client.instruments.find_instrument(query=isin)
                for i in answ.instruments:
                    tink_figi["isin"].append(i.isin)
                    tink_figi["figi"].append(i.figi)
                    tink_figi["ticker"].append(i.ticker)
                    tink_figi["class_code"].append(i.class_code)

                count += 1
                if count == MAX_REQUESTS:
                    time.sleep(60)
                    count = 0

        new_tink_figi_pl = pl.from_dict(tink_figi)
        tink_figi_pl = pl.concat([tink_figi_pl, new_tink_figi_pl], how="vertical")
        tink_figi_pl.write_parquet(tink_figi_path)

    return tink_figi_pl


def tink_download_dividends(tink_figi_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Downloads dividend information from Tinkoff API for given FIGI instruments.
    The function downloads 30 years of dividend history and saves results to a parquet file.

    :param tink_figi_pl: Polars DataFrame containing FIGI information for instruments
    :return: Polars DataFrame containing dividend information including payment dates, amounts and prices
    """
    tink_divs = {
        "figi": [],
        "currency": [],
        "div_units": [],
        "div_nano": [],
        "payment_date": [],
        "declared_date": [],
        "last_buy_date": [],
        "record_date": [],
        "cls_units": [],
        "cls_nano": [],
    }
    with Client(settings.tinkoff_api_token) as client:
        for figi in tqdm(tink_figi_pl["figi"].unique()):
            while True:
                try:
                    data = client.instruments.get_dividends(
                        figi=figi, from_=now() - datetime.timedelta(days=365 * 30), to=now()
                    )
                    break
                except Exception as e:
                    cur_limit = e.metadata.ratelimit_remaining
                    cur_reset = e.metadata.ratelimit_reset
                    if e.metadata.message != "instrument type is not a share or etf":
                        print(e)

                if cur_limit is None:
                    time.sleep(3)
                    print(f"Limit is {cur_limit}. Will sleep 3 sec")
                elif cur_limit > 0:
                    break
                else:
                    print(f"Limit is {cur_limit}. Will sleep {cur_reset} sec")
                    time.sleep(cur_reset)

            for i in data.dividends:
                tink_divs["figi"].append(figi)
                tink_divs["currency"].append(i.dividend_net.currency)
                tink_divs["div_units"].append(i.dividend_net.units)
                tink_divs["div_nano"].append(i.dividend_net.nano)
                tink_divs["payment_date"].append(i.payment_date)
                tink_divs["declared_date"].append(i.declared_date)
                tink_divs["last_buy_date"].append(i.last_buy_date)
                tink_divs["record_date"].append(i.record_date)
                tink_divs["cls_units"].append(i.close_price.units)
                tink_divs["cls_nano"].append(i.close_price.nano)

    tink_divs_pl = pl.from_dict(tink_divs)
    tink_divs_pl.write_parquet(settings.data_dir / "auxiliary" / "tinkoff_divs.parquet")

    return tink_divs_pl


def tink_filtration(tink_figi_pl: pl.DataFrame, tink_divs_pl: pl.DataFrame):
    """
    Filters and processes dividend data from Tinkoff API.
    Removes USD dividends, zero values, joins with FIGI information,
    filters for TQBR class code, and aggregates dividend amounts.
    Results are saved to a parquet file.

    :param tink_figi_pl: Polars DataFrame containing FIGI information for instruments
    :param tink_divs_pl: Polars DataFrame containing raw dividend information
    :return: None
    """
    tink_divs_pl = (
        tink_divs_pl.filter(
            (tink_divs_pl["currency"] != "usd")
            & ((tink_divs_pl["cls_units"] != 0) | (tink_divs_pl["cls_nano"] != 0))
            & ((tink_divs_pl["div_units"] != 0) | (tink_divs_pl["div_nano"] != 0))
        ).join(
            tink_figi_pl, on="figi", how="left"
        )
    )

    tink_divs_pl = tink_divs_pl.filter(tink_divs_pl["class_code"] == "TQBR").sort(["ticker", "last_buy_date"])

    tink_divs_pl = tink_divs_pl.with_columns(
        (pl.col("div_units") + pl.col("div_nano") / int("1" + "0" * 9)).alias("correct_divs"),
        (pl.col("cls_units") + pl.col("cls_nano") / int("1" + "0" * 9)).alias("ex_div_cls"),
    ).drop(["div_units", "div_nano", "cls_units", "cls_nano"])

    tink_divs_pl = tink_divs_pl.group_by(tink_divs_pl.drop("correct_divs").columns).agg(pl.col("correct_divs").sum())

    tink_divs_pl.write_parquet(settings.data_dir / "auxiliary" / "tinkoff_divs_filtered.parquet")


def collect_tink_dividends(isin_for_parsing: list):
    """
    Orchestrates the complete process of collecting and processing dividend data from Tinkoff API.
    This function performs three main steps:
    1. Retrieves FIGI information for given ISINs
    2. Downloads dividend history for the instruments
    3. Filters and processes the dividend data

    :param isin_for_parsing: List of ISIN codes to collect dividend information for
    :return: None - Results are saved to parquet files in the data directory
    """
    tink_figi_pl = tink_get_figi(isin_for_parsing)
    tink_divs_pl = tink_download_dividends(tink_figi_pl)
    tink_filtration(tink_figi_pl, tink_divs_pl)
