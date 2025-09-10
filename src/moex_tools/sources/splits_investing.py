import pandas as pd
import polars as pl
import re
import datetime
import json

from subprocess import run, PIPE
from bs4 import BeautifulSoup
from tqdm import tqdm

from moex_tools.config import settings

pd.options.display.max_rows = 200
pd.options.display.max_columns = 100
pd.options.display.width = 20_000


def get_exchange_base() -> pd.DataFrame:
    prices_final = pl.read_parquet(f'data/exchange_data.parquet')
    prices_final = prices_final.with_columns(prices_final['DATE'].str.to_date().alias('DATE'))
    prices_final = prices_final.with_columns(
        pl.when(pl.col('CLOSE').is_nan())\
            .then((pl.col('HIGH') + pl.col('LOW')) / 2)\
            .otherwise(pl.col('CLOSE'))\
            .alias('CLOSE'),
        true_split=pl.lit(1),
    )

    tickers_in_base = prices_final \
        .filter(pl.col('class').is_in(['Акции', 'Биржевые фонды']) | pl.col('class').is_null()) \
        .sort(['SECID', 'DATE'], descending=True) \
        .unique(subset=['SECID'], keep='first', maintain_order=True) \
        .filter(pl.col('isin').str.contains('RU'))

    return tickers_in_base.to_pandas()


def curl_get_table(val) -> list:
    url = f'https://www.investing.com/equities/{val}-historical-data-splits'
    print('url:', url)
    cnfg = ['curl.exe', '-s', '-A ' + 'Chrome/91.0.4472.114', '-H Content-Type: application/json', '-d ', url]
    print('config:', cnfg, chr(10))

    htm_doc = run(cnfg, stdout=PIPE).stdout.decode('utf-8')
    soup = BeautifulSoup(htm_doc, 'html.parser')
    table = soup.find("table", {"class": "w-full border-0 text-xs leading-4 text-[#181C21]"})

    if table:
        # headers = [header.text.strip() for header in table.find_all("th")]
        # print("Table Headers:", headers)

        rows = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            row_data = [col.text.strip() for col in cols]
            rows.append(row_data)
        print(f"{val}. Table Data:", rows)
    else:
        print("Table not found or the table is empty.")
        return None

    return rows


def curl_make_search(isin: str):
    url = f'https://www.investing.com/search/?q={isin}'
    cnfg = ['curl.exe', '-s', '-A ' + 'Chrome/91.0.4472.114', '-H Content-Type: application/json', '-d ', url]
    print('config:', cnfg, chr(10))

    htm_doc = run(cnfg, stdout=PIPE).stdout.decode('utf-8')
    soup = BeautifulSoup(htm_doc, 'html.parser')
    search = soup.find(
        "div",
        {"class": "js-inner-all-results-quotes-wrapper newResultsContainer quatesTable"}
    )
    if search == None:
        return None, None, None, None

    need_data = []
    script = search.find_all('script', text=re.compile(r'window\.allResultsQuotesDataArray'))
    for idx, scr in enumerate(script):
        script_text = re.search(r'\[.*\]', scr.string, re.DOTALL).group()
        script_data = json.loads(script_text)
        if script_data[0]['flag'] == 'Russian_Federation':
            need_data.append(script_data[0])

    if len(need_data) != 1:
        raise Exception(f'{isin} returned more that 1 case: {need_data}')

    tag = need_data[0]['link'].split('/')[-1]
    ticker = need_data[0]['symbol']
    name = need_data[0]['name']
    cur_id = need_data[0]['pairId']

    return tag, ticker, name, cur_id


def update_investing_base():
    tickers_in_base = get_exchange_base()
    invesings_ids = pd.read_csv('data/Investing_stocks.csv')

    russia_ids = invesings_ids[invesings_ids['country'] == 'russia']
    total_df = pd.merge(
        tickers_in_base,
        russia_ids[['tag', 'isin', 'id', 'currency', 'symbol']],
        left_on=['isin'],
        right_on=['isin'],
        how='outer'
    )
    out_of_investing = total_df[
        (total_df['DATE'] > (datetime.datetime.now() - pd.DateOffset(days=10))) &
        pd.isna(total_df['tag'])
        ].drop(columns=['tag', 'id', 'currency', 'symbol'])

    dict_search = {}
    for isin in tqdm(out_of_investing['isin']):
        dict_search[isin] = curl_make_search(isin)

    cur_df = pd.DataFrame.from_dict(dict_search).T.reset_index()
    cur_df = cur_df[~pd.isna(cur_df[0])].reset_index(drop=True)
    cur_df.columns = ['isin', 'tag', 'symbol', 'full_name', 'id']
    cur_df['country'] = 'russia'
    cur_df['currency'] = 'RUB'

    if len(cur_df) == 0:
        print("Don't have new tickers for update")
        return invesings_ids.reset_index(drop=True).sort_values(['country', 'id'])

    final_base = pd.concat([invesings_ids, cur_df], axis=0).reset_index(drop=True).sort_values(['country', 'id'])
    final_base.to_csv('data/Investing_stocks.csv', index=False)

    return final_base


def create_splits_df(russia_ids: pd.DataFrame):
    splits_data = {}
    for idx, row in tqdm(russia_ids.iterrows(), total=len(russia_ids)):
        answ = curl_get_table(row['tag'])
        if answ is None:
            continue

        splits_data[row['isin']] = answ

    split_df = pd.DataFrame(
        [[k, cur_v[0], cur_v[1]] for k, v in splits_data.items() for cur_v in v],
        columns=['isin', 'date', 'split']

    )
    split_df['date'] = pd.to_datetime(split_df['date'])
    split_df['split'] = split_df['split'].str.replace(':1 Stock Split', '').str.replace(',', '').astype(float)

    split_df.to_csv('data/investing_splits.csv', index=False)

    return None


if __name__ == "__main__":
    update_investing_base()

    invesings_ids = pd.read_csv('data/Investing_stocks.csv')
    russia_ids = invesings_ids[invesings_ids['country'] == 'russia']
    create_splits_df(russia_ids)
