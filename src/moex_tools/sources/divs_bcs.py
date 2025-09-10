import numpy as np
import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from moex_tools.config import settings


def bcs_download(isin_for_parsing: list) -> pl.DataFrame:
    main_url = "https://api.bcs.ru/divcalendar/v1/dividend/{}"

    div_df = pd.DataFrame(
        columns=['id', 'company_name', 'last_buy_day', 'closing_date', 'dividend_value', 'close_price', 'yield', 'ISIN']
    )
    with requests.Session() as s:
        for t in tqdm(isin_for_parsing):
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

    div_df.write_parquet(settings.data_dir / "auxiliary" / "bcs_divs.parquet")

    return div_df


def bcs_prepare_for_save(div_df: pl.DataFrame, base_pl: pl.DataFrame) -> pl.DataFrame:
    div_df = div_df.join(
        base_pl[['SECID', 'isin']].rename({'SECID': 'SecureCode', 'isin': 'ISIN'}).unique(['SecureCode', 'ISIN']),
        how='left',
        on='ISIN'
    )
    div_df = div_df.filter(~(div_df['Размер дивиденда'] == 0))
    div_df = div_df.filter(~(
            (div_df['Последний день для покупки акций'].is_null()) & (div_df['Дата закрытия реестра'].is_null())
    ))
    div_df = div_df.filter(
        ~(pl.col('Последний день для покупки акций').is_null()) &
        ~(div_df[['ISIN', 'Последний день для покупки акций', 'Размер дивиденда']].is_duplicated())
    )

    need_cols = ['SecureCode', 'Последний день для покупки акций', 'Дата закрытия реестра', 'Цена акции на закрытие',
                 'ISIN']
    div_df = div_df.group_by(need_cols).agg(pl.col('Размер дивиденда').sum())

    div_df.write_parquet(f'data/{today}_bcs_divs.parquet')

    return div_df


def collect_bcs_dividends(isin_for_parsing):
    div_df = bcs_download(isin_for_parsing)
    # div_df = bcs_prepare_for_save(div_df)
