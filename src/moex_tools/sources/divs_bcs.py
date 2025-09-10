import numpy as np
import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from moex_tools.config import settings


def bcs_download(isin_for_parsing: list) -> pl.DataFrame:
    """
    Fetches dividend information from the BCS API and constructs a data frame with the results.

    The function iterates over the provided list of ISINs (International Securities
    Identification Numbers), queries the BCS API for dividend information associated
    with each ISIN, and organizes the results into a structured Polars DataFrame.

    :param isin_for_parsing: List of ISIN strings for which dividend information is
        to be retrieved.
    :type isin_for_parsing: list
    :return: Polars DataFrame containing dividend information, including company name,
        dividend value, yield, last buy day, closing date, and more, structured for further
        analysis.
    :rtype: pl.DataFrame
    """
    main_url = "https://api.bcs.ru/divcalendar/v1/dividend/{}"

    div_df = pd.DataFrame(
        columns=['id', 'company_name', 'last_buy_day', 'closing_date', 'dividend_value', 'close_price', 'yield', 'ISIN']
    )
    with requests.Session() as s:
        for t in tqdm(isin_for_parsing, desc="Fetching dividends from BCS"):
            answ = s.get(main_url.format(t))
            if answ.status_code == 404:
                continue

            answ = answ.json()
            table = pd.DataFrame(
                data=np.array([answ['id'], answ['company_name'], answ['last_buy_day'],
                               answ['closing_date'], answ['dividend_value'], answ['close_price'],
                               answ['yield']]
                              ).reshape(1, 7),
                columns=div_df.columns.drop('ISIN')
            )
            additional_table = pd.DataFrame.from_dict(answ['previous_dividends'])
            if len(additional_table) > 0:
                table = pd.concat([table, additional_table])
            table['ISIN'] = t

            div_df = pd.concat([div_df, table])

    div_df.columns = ['id', 'Наименование', 'Последний день для покупки акций', 'Дата закрытия реестра',
                      'Размер дивиденда', 'Цена акции на закрытие', 'Доходность', 'ISIN']

    div_df['Последний день для покупки акций'] = pd.to_datetime(
        div_df['Последний день для покупки акций'].str.split('T', expand=True)[0]
    )
    div_df['Дата закрытия реестра'] = pd.to_datetime(div_df['Дата закрытия реестра'].str.split('T', expand=True)[0])
    div_df.drop('id', axis=1, inplace=True)

    div_df['Наименование'] = div_df['Наименование'].astype(str)
    div_df['Размер дивиденда'] = div_df['Размер дивиденда'].astype(float)
    div_df['Цена акции на закрытие'] = div_df['Цена акции на закрытие'].astype(float)
    div_df['Доходность'] = div_df['Доходность'].astype(float)
    div_df['ISIN'] = div_df['ISIN'].astype(str)
    div_df = pl.from_pandas(div_df)

    return div_df


def bcs_prepare_for_save(divs_pl: pl.DataFrame) -> None:
    """
    Prepares the dividend data for saving by filtering, deduplicating, and grouping specific attributes.
    The final processed data is saved as a parquet file in the defined directory.

    :param divs_pl: A Polars DataFrame containing dividend-related data to process.
    :type divs_pl: pl.DataFrame
    :return: None
    """
    divs_pl = divs_pl.filter(
            (divs_pl['Размер дивиденда'] != 0)
            & (divs_pl['Цена акции на закрытие'] != 0)
            & (~divs_pl['Последний день для покупки акций'].is_null() & ~divs_pl['Дата закрытия реестра'].is_null())
        ).unique(subset=['ISIN', 'Дата закрытия реестра', 'Размер дивиденда']) \
        .drop(['Наименование', 'Доходность'])

    divs_pl = divs_pl.group_by(divs_pl.drop(['Размер дивиденда'])).agg(pl.col('Размер дивиденда').sum())

    divs_pl.write_parquet(settings.data_dir / "auxiliary" / "bcs_divs.parquet")


def collect_bcs_dividends(isin_for_parsing):
    """
    Collects dividend information from the BCS API and prepares it for saving.

    :param isin_for_parsing: List of ISIN strings for which dividend information is to be retrieved.
    :type isin_for_parsing: list
    :return: None
    """
    div_df = bcs_download(isin_for_parsing)
    bcs_prepare_for_save(div_df)
