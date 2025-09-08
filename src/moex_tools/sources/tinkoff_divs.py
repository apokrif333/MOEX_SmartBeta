import datetime
import time

import polars as pl
from tinkoff.invest import Client
from tinkoff.invest.utils import now
from tqdm import tqdm

from moex_tools.config import settings


def tink_get_figi(isin_for_parsing: list) -> pl.DataFrame:
    tink_figi = {"isin": [], "figi": [], "ticker": [], "class_code": []}
    count = 0
    with Client(settings.tinkoff_api_token) as client:
        for isin in tqdm(isin_for_parsing):
            answ = client.instruments.find_instrument(query=isin)
            for i in answ.instruments:
                tink_figi["isin"].append(i.isin)
                tink_figi["figi"].append(i.figi)
                tink_figi["ticker"].append(i.ticker)
                tink_figi["class_code"].append(i.class_code)

            count += 1
            if count == 199:
                time.sleep(60)
                count = 0

    tink_figi_pl = pl.from_dict(tink_figi)
    tink_figi_pl.write_parquet(settings.data_dir / "auxiliary" / "tinkoff_figi.parquet")

    return tink_figi_pl


def tink_download_dividends(tink_figi_pl: pl.DataFrame) -> pl.DataFrame:
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
    tink_divs_pl.write_parquet(settings.data_dir / data / "tinkoff_divs.parquet")

    return tink_divs_pl


def tink_filtration(tink_figi_pl: pl.DataFrame, tink_divs_pl: pl.DataFrame) -> pl.DataFrame:
    tink_divs_pl = tink_divs_pl.filter(tink_divs_pl["currency"] != "usd").join(
        tink_figi_pl, on="figi", how="left"
    )

    test_pl = tink_divs_pl.unique(subset=["figi", "isin"])
    need_figi = test_pl.filter(~test_pl["isin"].is_duplicated())["figi"]

    duplicated_isins = test_pl.filter(test_pl["isin"].is_duplicated()).sort("isin")["figi"]
    test_pl = (
        tink_divs_pl.filter(pl.col("figi").is_in(duplicated_isins))
        .group_by(["figi", "isin", "class_code"])
        .agg(
            pl.col("currency").count(),
            pl.col("div_units").sum(),
            pl.col("div_nano").sum(),
            pl.col("cls_units").sum(),
            pl.col("cls_nano").sum(),
        )
        .sort(["isin", "cls_units", "cls_nano", "class_code"], descending=True)
    )
    need_figs_duplicated = test_pl.unique(subset=["isin", "div_nano"], keep="first")["figi"]
    need_figi = need_figi.append(need_figs_duplicated)

    tink_divs_pl = tink_divs_pl.filter(pl.col("figi").is_in(need_figi)).sort(
        ["ticker", "last_buy_date", "class_code"]
    )
    tink_divs_pl = tink_divs_pl.filter((pl.col("div_units") != 0) | (pl.col("div_nano") != 0))
    tink_divs_pl = tink_divs_pl.with_columns(
        (pl.col("div_units") + pl.col("div_nano") / int("1" + "0" * 9)).alias("correct_divs"),
        (pl.col("cls_units") + pl.col("cls_nano") / int("1" + "0" * 9)).alias("ex_div_cls"),
    )

    need_cols = [
        "figi",
        "currency",
        "last_buy_date",
        "cls_units",
        "cls_nano",
        "isin",
        "ticker",
        "class_code",
        "ex_div_cls",
    ]
    tink_divs_pl = tink_divs_pl.group_by(need_cols).agg(
        pl.col("div_units").sum(), pl.col("div_nano").sum(), pl.col("correct_divs").sum()
    )

    tink_divs_pl.write_parquet(settings.data_dir / "data" / "tinkoff_divs.parquet")

    return tink_divs_pl


def collect_tink_dividends(isin_for_parsing: list):
    tink_figi_pl = tink_get_figi(isin_for_parsing)
    tink_divs_pl = tink_download_dividends(tink_figi_pl)
    tink_filtration(tink_figi_pl, tink_divs_pl)
