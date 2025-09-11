import polars as pl

from ..config import settings
from ..sources.moex import *
from ..sources.divs_finance_marker import collect_fm_dividends
from ..sources.divs_tinkoff import collect_tink_dividends
from ..sources.divs_bcs import collect_bcs_dividends
from ..sources.splits_moex import create_split_base


def isin_stocks_for_parsing() -> dict:
    """
    Extract unique ISIN codes and stock tickers from MOEX stock data.
    Filters out RM and RX class stocks and returns dictionary containing
    lists of unique ISIN codes and stock tickers for further parsing.

    Returns:
        dict: Dictionary containing two keys:
            - 'isin': List of unique ISIN codes
            - 'stocks': List of unique stock tickers (SECID without RM/RX suffix)
    """
    path = settings.data_dir / "raw" / "union_raw_moex_descript.parquet"
    base_pl = pl.read_parquet(path)

    stocks_for_parsing = base_pl.filter(pl.col("class") == "Акции")
    stocks_for_parsing = stocks_for_parsing.with_columns(
        stocks_for_parsing["SECID"]
        .str.split_exact("-", 1)
        .struct.rename_fields(["SECID_splt", "-part"])
        .alias("fields")
        .to_frame()
        .unnest("fields")
    )
    stocks_for_parsing = stocks_for_parsing.with_columns(
        stocks_for_parsing["-part"].fill_null("").alias("-part")
    )
    stocks_for_parsing = stocks_for_parsing.filter(
        (pl.col("-part") != "RM") & (pl.col("-part") != "RX")
    )

    for_parsing = {
        "isin": stocks_for_parsing["isin"].unique(),
        "stocks": stocks_for_parsing["SECID_splt"].unique(),
    }

    return for_parsing


def run() -> None:
    # Raw MOEX data
    # fetch_exchange_range_parallel()
    # check_loaded_moex_data()
    # create_union_raw_moex_data()
    # create_description_json()
    # update_description_in_new_prices()

    # Dividends
    # for_parsing = isin_stocks_for_parsing()
    # collect_fm_dividends(for_parsing["stocks"])
    # collect_tink_dividends(for_parsing["isin"])
    # collect_bcs_dividends(for_parsing["isin"])
    create_split_base()
