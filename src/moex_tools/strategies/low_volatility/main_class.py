from __future__ import annotations

import asyncio
import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from pandasgui import show as show_pd
from telegram import Bot
from telegram.constants import ParseMode
from tinkoff.invest import Client, InstrumentIdType, InstrumentType
from tqdm import tqdm

from moex_tools.MOEX_base_functions import get_data_by_ISSClient
from moex_tools.config import settings

# Settings
pio.renderers.default = "browser"


class LowVolatilityStrategy:
    """
    Класс для стратегии низкой волатильности с настраиваемыми параметрами.
    """
    def __init__(
            self,
            period: float = 2,  # 0.25, 0.5, 1, 2, 5, 10

            capital: int = 1_000_000,
            assets: int = 40,  # количество активов в портфеле
            rebalance: str = 'weekly',  # weekly, monthly, quarterly, yearly
    ):
        # External parameters
        self.period = period
        self.capital = capital
        self.assets = assets
        self.rebalance = rebalance

        # Internal parameters
        self._window: str = ''
        self._base_df: pl.DataFrame = pl.DataFrame()
        self.reb_dates: dict

    def load_base_data(self):
        """
        Загружает базовые данные MOEX с учётом сплитов и дивидендов.
        Фильтрует только акции.
        """
        base_pl = pl.read_parquet(settings.data_dir / "moex_splits_divs.parquet")

        last_date = pl.read_parquet(settings.data_dir / 'raw' /'union_raw_moex_data.parquet') \
            .with_columns(pl.col('DATE').cast(pl.Date))['DATE'].max()
        if last_date != base_pl['DATE'].max():
            print(f"Last date in base_pl is {base_pl['DATE'].max()}, last date in union_raw_moex_data is {last_date}")
            raise Exception("Last date in base_pl is not equal to last date in union_raw_moex_data")

        base_pl = base_pl.filter(
            (pl.col('class') == 'Акции') &
            (pl.col('category') != 'Акция привилегированная ') &
            ~pl.col('SECID').str.contains('RM') &
            ~pl.col('SECID').str.contains('RX')
        )

        base_pl = (
            base_pl.with_columns(
                pl.when(pl.col('ISSUESIZE') == 0)
                .then(pl.lit(None))
                .otherwise(pl.col('ISSUESIZE')).alias('ISSUESIZE')
            ).with_columns(pl.col('ISSUESIZE').forward_fill().over('SECID').alias('ISSUESIZE'))
            .with_columns(pl.col('ISSUESIZE').backward_fill().over('SECID').alias('ISSUESIZE'))
        )

        self._base_df  = (
            base_pl.with_columns(
                pl.when(pl.col('ISSUESIZE') == 0)
                .then(pl.lit(None))
                .otherwise(pl.col('ISSUESIZE'))
                .alias('ISSUESIZE')
            ).with_columns(pl.col('ISSUESIZE').forward_fill().over('SECID').alias('ISSUESIZE'))
            .with_columns(pl.col('ISSUESIZE').backward_fill().over('SECID').alias('ISSUESIZE'))
            .with_columns(
                pl.when(pl.col('CLOSE').is_null())
                .then((pl.col('HIGH') + pl.col('LOW')) / 2)
                .otherwise(pl.col('CLOSE'))
                .alias('CLOSE'),
                pl.when(pl.col('OPEN').is_null())
                .then((pl.col('HIGH') + pl.col('LOW')) / 2)
                .otherwise(pl.col('OPEN'))
                .alias('OPEN')
            ).with_columns(
                pl.when(pl.col('DAILYCAPITALIZATION') == 0)
                .then(pl.col('ISSUESIZE') * pl.col('CLOSE'))
                .otherwise(pl.col('DAILYCAPITALIZATION'))
                .alias('DAILYCAPITALIZATION')
            )
        )


    def prepare_data(self):
        """
        Подготавливает данные: вычисляет доходности, применяет фильтры.
        """
        self._base_df = self._base_df.with_columns((pl.col('CLOSE') * pl.col('VOLUME')).alias('RubVol'))

        pl_group = (
            self._base_df.sort(['SECID', 'DATE'])
            .rolling(index_column='DATE', group_by='SECID', period='50d')
            .agg(
                pl.mean('VOLUME').alias('VOLUME_50'),
                pl.mean('RubVol').alias('RubVol_50')
            )
        )
        self._base_df = self._base_df.join(pl_group, how='left', on=['SECID', 'DATE']).sort(['SECID', 'DATE'])

        daily_exchange_data = (
            self._base_df.group_by('DATE').agg([
                pl.median('DAILYCAPITALIZATION').round(0).alias('DayCapMedian'),
                pl.mean('DAILYCAPITALIZATION').round(0).alias('DayCapMean'),
                pl.quantile('DAILYCAPITALIZATION', quantile=0.9).round(0).alias('LargeCap'),
                pl.quantile('DAILYCAPITALIZATION', quantile=0.7).round(0).alias('MidCap'),
                pl.median('DAILYCAPITALIZATION').round(0).alias('MedianCap'),
                pl.median('VOLUME_50').round(0).alias('VolMedian'),
                pl.mean('VOLUME_50').round(0).alias('VolMean'),
                pl.median('RubVol_50').round(0).alias('RubVolMedian'),
                pl.mean('RubVol_50').round(0).alias('RubVolMean')
            ]).sort('DATE')
        )
        need_cols = ['LargeCap', 'MidCap', 'MedianCap', 'RubVolMean', 'RubVolMedian']
        self._base_df = self._base_df.join(
            daily_exchange_data[['DATE'] + need_cols],
            on=['DATE'],
            how='left'
        )

    def calculate_volatility(self):
        """
        Вычисляет волатильность для каждого актива за период.
        """
        self._base_df = (
            self._base_df.sort(['SECID', 'DATE'])
            .with_columns((pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(1) - 1).over('SECID').alias('CLOSE_adj_cng'))
            .with_columns(pl.col('CLOSE_adj_cng').fill_null(0).alias('CLOSE_adj_cng'))
            .with_columns(
                pl.when(pl.col('CLOSE_adj_cng') > 0)
                .then(0)
                .otherwise(pl.col('CLOSE_adj_cng'))
                .alias('CLOSE_adj_cng_semi')
            )
        )

        self._window = f'{round(365 * self.period, 0)}D'
        min_periods: int = 126 if self.period >= 0.5 else 45
        base_pd = self._base_df.to_pandas().sort_values(['SECID', 'DATE']).set_index('DATE')
        total_std = base_pd.groupby('SECID')['CLOSE_adj_cng_semi']\
            .rolling(window=self._window, min_periods=min_periods)\
            .std()
        total_std.name = f'StDev_{self._window}'
        total_std = total_std.reset_index()

        base_pd = pd.merge(base_pd, total_std, on=['SECID', 'DATE'], how='inner')
        self._base_df = pl.from_pandas(base_pd).with_columns(pl.col('DATE').cast(pl.Date))

    def calculate_vol_rank(self):
        """
        Вычисляет веса для выбранных активов.
        """
        self._base_df = self._base_df.with_columns(
            pl.col(f'StDev_{self._window}').rank(method='ordinal').over('DATE').alias('VolRank')
        )

        max_values = self._base_df.group_by('DATE').agg(pl.col('VolRank').max().alias('MaxRank')).sort('DATE')
        self._base_df = (
            self._base_df.join(max_values, how='left', on=['DATE'])
            .with_columns((1 - pl.col('VolRank') / pl.col('MaxRank')).alias('Ranking'))
            .sort(['DATE', 'SECID'])
        )

    def filter_for_daily_check(self) -> pd.DataFrame:
        port = (
            self._base_df.filter(
                (pl.col('DAILYCAPITALIZATION') > pl.col('MedianCap')) &
                ~pl.col('VolRank').is_null()
            ).sort(by=['DATE', 'VolRank'])
            .group_by('DATE').head(self.assets)
            .filter(pl.col('DATE') == pl.col('DATE').max())
            .to_pandas()
        )

        return port

    @staticmethod
    def tink_load_figi(port: pd.DataFrame) -> pd.DataFrame:

        lot_list = []
        figi_list = []
        secid = port['SECID'].to_list()
        with Client(settings.tinkoff_api_token) as client:
            for secid in tqdm(secid, desc="Fetching FIGI from Tinkoff"):
                answ = client.instruments.find_instrument(
                    query=secid,
                    instrument_kind=InstrumentType.INSTRUMENT_TYPE_SHARE
                )
                retrived = False
                for i in answ.instruments:
                    if (i.class_code in ['TQBR', 'TQPI', 'SPBRU']) and (i.ticker == secid):
                        figi_list.append(i.figi)
                        retrived = True
                        break

                if retrived is False:
                    print(secid)
                    print(answ)
                    raise Exception

                data = client.instruments.get_instrument_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                    id=figi_list[-1],
                )
                lot_list.append(data.instrument.lot)

        port['lot'] = lot_list
        port['figi'] = figi_list
        port = port.sort_values('VolRank')

        return port

    @staticmethod
    def tink_load_prices(port) -> pd.DataFrame:

        def make_cur_price(cur_price):
            nano_mult = 10 ** len(str(cur_price.price.nano))
            final_price = cur_price.price.units + (cur_price.price.nano / nano_mult)

            return final_price

        mid_price, lm_order_check, mrk_order_check, api_check = [], [], [], []
        with Client(settings.tinkoff_api_token) as client:
            for idx, row in tqdm(port.iterrows(), desc="Fetching prices from Tinkoff"):
                answ = client.market_data.get_trading_status(figi=row['figi'])

                lm_order_check.append(answ.limit_order_available_flag)
                mrk_order_check.append(answ.market_order_available_flag)
                api_check.append(answ.api_trade_available_flag)
                try:
                    answ = client.market_data.get_order_book(figi=row['figi'], depth=1)
                    ask = make_cur_price(answ.asks[0])
                    bid = make_cur_price(answ.bids[0])
                    m_price = round((ask + bid) / 2, 4)
                    mid_price.append(m_price)
                except Exception as e:
                    print(row)
                    print(answ)
                    raise Exception(e)

        port['lm_order_check'] = lm_order_check
        port['mrk_order_check'] = mrk_order_check
        port['api_check'] = api_check
        port['mid_price'] = mid_price

        return port

    def print_bot_data(self, port: pd.DataFrame, assets: int):
        cur_port = port.sort_values('VolRank')[['SECID', 'VolRank', f'StDev_{self._window}']].iloc[:assets]
        cur_port['Weight'] = (1 / cur_port[f'StDev_{self._window}']) / (1 / cur_port[f'StDev_{self._window}']).sum()

        past_port = pd.read_csv(settings.bot_path / 'Relaible_weights.csv')
        past_port.to_csv(settings.bot_path / 'Relaible_weights_old.csv', index=False)

        both_ports = pd.merge(
            past_port,
            cur_port[['SECID', 'Weight']],
            left_on=['Ticker'],
            right_on=['SECID'],
            suffixes=('_past', '_cur'),
            how='outer'
        )
        remove_tickers = both_ports[pd.isna(both_ports['SECID'])]['Ticker']
        add_tickers = both_ports[pd.isna(both_ports['Ticker'])]['SECID']
        old_tickers = list(both_ports['Ticker'].dropna())

        df_for_save = (
            both_ports[~pd.isna(both_ports['SECID'])]
            .rename(columns={'SECID': 'Ticker', 'Weight_cur': 'Weight'})
            .sort_values(by='Weight', ascending=False)
            [['Ticker', 'Weight']]
        )
        check = 1 - df_for_save['Weight'].sum()
        assert check < 0.01, f'Total assets weight is not equal to 1.0 - {check}'
        df_for_save.to_csv(settings.bot_path / 'Relaible_weights.csv', index=False)

        self.create_bot_messsage(remove_tickers, add_tickers, old_tickers)

    def create_bot_messsage(self, remove_tickers, add_tickers, old_tickers):

        def get_reb_text(tickers, description_dict):
            text = ''
            for t in tickers.values:
                name = description_dict[t]['Полное наименование']
                text += f"<b>{t}</b> ({name}), "
            return text

        def get_gain_text(df, col_name, description_dict):
            cur_df = df.filter(pl.col(col_name) > 1).sort(col_name, descending=True)[:3]['SECID', col_name].to_pandas()
            text = ''
            for idx, row in cur_df.iterrows():
                t = row['SECID']
                name = description_dict[t]['Полное наименование']
                move = (row[col_name] - 1) * 100
                text += f"{t} ({name}) {move:.1f}%\n"
            return text

        path = settings.data_dir / 'auxiliary' / 'all_stocks_description.json'
        with open(path) as f:
            description_dict = json.load(f)

        if len(remove_tickers) == 0 and len(add_tickers) == 0:
            need_reb_template = '\n⚖️ В данный момент ребалансировка не требуется.\n'
        else:
            for_sell = get_reb_text(remove_tickers, description_dict)
            for_buy = get_reb_text(add_tickers, description_dict)
            need_reb_template = f"""\n⚖️ Требуется ребалансировка.
            \nПроданы: {for_sell}
            \nКуплены: {for_buy}\n"""

        secid = old_tickers + ['BOND', 'GOLD']
        temp_base = pl.read_parquet(settings.data_dir / 'moex_splits_divs.parquet')
        gain_df = (
            temp_base.filter(pl.col('SECID').is_in(secid))
            .sort(['SECID', 'DATE'], descending=[False, True])
            .with_columns(
                (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-5)).over('SECID').alias('shift5'),
                (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-21)).over('SECID').alias('shift21'),
                (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-63)).over('SECID').alias('shift63')
            ).group_by('SECID').agg(
                pl.col('DATE').first(),
                pl.col('DATE').shift(-5).first().alias('date5'),
                pl.col('DATE').shift(-21).first().alias('date21'),
                pl.col('DATE').shift(-63).first().alias('date63'),
                pl.col('shift5').first(),
                pl.col('shift21').first(),
                pl.col('shift63').first(),
                pl.col('CLOSE_adj').first()
            )
        )
        for_21 = get_gain_text(gain_df, 'shift21', description_dict)
        for_63 = get_gain_text(gain_df, 'shift63', description_dict)
        template_gain = f'\n📈Лидеры роста за 30 календарных дней (ТОП 3)\n{for_21}'
        template_gain += f'\n💥Лидеры роста за 91 календарный день (ТОП 3)\n{for_63}'

        cur_date = datetime.datetime.now().strftime('%Y-%m-%d')
        hello = 'Доброй новой недели!\n'
        end_text = '\nТекущие доли всегда можно проверить в боте \n@QcmStockRusBot'
        message = f"<b>{cur_date}</b>\n" + hello + need_reb_template + template_gain + end_text

        asyncio.run(self.send_bot_message(message))

    @staticmethod
    async def send_bot_message(message: str):
        chat_id = "-1002471577619"
        async with Bot(token=settings.bot_minvol_token) as bot:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.HTML)

    def run_backtest(self):
        """
        Запускает полный бэктест стратегии.
        """
        self._base_df = self._base_df.with_columns(
            OPEN_adj_gain=(pl.col('OPEN_adj').shift(-1) / pl.col('OPEN_adj')).over('SECID')
        )

        sceid_last_date = self._base_df.group_by('SECID').agg(last_date=pl.col('DATE').max())
        main_rank_df = (
            self._base_df.filter(
                (pl.col('DAILYCAPITALIZATION') > pl.col('MedianCap'))
                # & (pl.col('RubVol_50') > pl.col('RubVolMedian'))
                & ~pl.col('VolRank').is_null()
            ).join(sceid_last_date, on='SECID', how='left')
            .with_columns(day_diff=(pl.col('last_date') - pl.col('DATE')).dt.total_days())
            .filter(
                ~((pl.col('day_diff') < 8) & (pl.col('last_date') != pl.col('DATE').max()))
            ).sort(['DATE', 'VolRank'])
        )

        results_dict = {}
        for assets in tqdm([10, 15, 20, 30], desc="Run backtest"):
            rank_df = (main_rank_df
                .group_by('DATE').head(assets)
                .with_columns(rev_StDev=1 / pl.col(f'StDev_{self._window}'))
                .with_columns(weight=pl.col('rev_StDev') / pl.col('rev_StDev').sum().over('DATE'))
                .to_pandas()
            )
            secid_base = (
                self._base_df.filter(
                    pl.col('SECID').is_in(set(rank_df['SECID']))
                ).to_pandas()
            )

            for rebalance in ['weekly', 'monthly', 'quarterly', 'yearly']:
                print(f"Assets: {assets}. Rebalance: {rebalance}")

                if rebalance == 'weekly':
                    mask = rank_df['DATE'].dt.isocalendar().week != rank_df['DATE'].shift(-1).dt.isocalendar().week
                elif rebalance == 'monthly':
                    mask = rank_df['DATE'].dt.month != rank_df['DATE'].shift(-1).dt.month
                elif rebalance == 'quarterly':
                    mask = rank_df['DATE'].dt.quarter != rank_df['DATE'].shift(-1).dt.quarter
                elif rebalance == 'yearly':
                    mask = rank_df['DATE'].dt.year != rank_df['DATE'].shift(-1).dt.year
                else:
                    print(f'Indefinite rebalancing period - {rebalance}. Will be applied daily')
                    raise Exception("Invalid rebalance period specified")

                unq_dates = np.sort(rank_df['DATE'][mask])
                cur_rank = rank_df[rank_df['DATE'].isin(unq_dates)].copy()

                if rebalance in ['monthly', 'quarterly', 'yearly']:
                    reb_dates = pd.DataFrame({'DATE': unq_dates, 'reb': pd.Series(unq_dates).shift(-1)}).dropna()
                    cur_rank = cur_rank.merge(reb_dates, on='DATE', how='left')[['SECID', 'reb']]
                    cur_rank = pl.from_pandas(cur_rank)
                    add_reb_dates = (
                        self._base_df.filter(pl.col('SECID').is_in(set(cur_rank['SECID'])))
                        .join(cur_rank, on='SECID', how='left')
                        .filter(pl.col('DATE') < pl.col('reb'))
                        .with_columns(
                            day_diff=(pl.col('reb') - pl.col('DATE')).dt.total_days()
                        ).group_by(['SECID', 'reb']).agg(pl.col('day_diff').min(), pl.col('DATE').max())
                        .filter(pl.col('day_diff') > 4)
                        .sort('day_diff', descending=True)
                        .to_pandas()
                    )
                    unq_dates = np.sort(
                        list(set(unq_dates).union(set(add_reb_dates['DATE'])))
                    )
                    cur_rank = rank_df[rank_df['DATE'].isin(unq_dates)].copy()


                port_dict = {}
                prev_i = None
                for i, v in cur_rank.groupby('DATE'):
                    if prev_i:
                        if len(set(v['SECID']) - set(port_dict[prev_i].keys())) == 0:
                            continue

                    i = i.strftime('%Y-%m-%d')
                    port_dict[i] = v[['SECID', 'weight']].set_index('SECID').to_dict()['weight']
                    port_dict[i] = dict(sorted(port_dict[i].items()))
                    prev_i = i

                # Make Trades
                trades_df = pd.DataFrame()
                reb_list = list(port_dict.keys())
                for i in range(len(reb_list[:-1])):
                    cur_port = port_dict[reb_list[i]]

                    cur_base = secid_base[
                            secid_base['SECID'].isin(set(cur_port.keys())) &
                            (secid_base['DATE'] > pd.to_datetime(reb_list[i])) &
                            (secid_base['DATE'] <= pd.to_datetime(reb_list[i + 1]))
                        ]
                    cur_base = cur_base[['DATE', 'SECID', 'OPEN_adj_gain']]
                    cur_base = cur_base.pivot(index='DATE', columns='SECID', values=['OPEN_adj_gain']).fillna(1)
                    cur_base.columns = cur_base.columns.droplevel(0)

                    if i == 0:
                        invest_money = np.array(list(cur_port.values())) * self.capital
                    else:
                        invest_money = np.array(list(cur_port.values())) * trades_df.loc[reb_list[i], 'Port']

                    dont_trades_tickers = set(cur_port.keys()) - set(cur_base.columns)
                    for t in dont_trades_tickers:
                        print(f"Attention {t} don't have data after {reb_list[i]} rebalance date")
                        cur_base[t] = 1

                    cur_base['Port'] = cur_base.cumprod().mul(invest_money).sum(axis=1)
                    trades_df = pd.concat(objs=[trades_df, cur_base], axis=0)

                # Save results
                years = (trades_df['Port'].index[-1] - trades_df['Port'].index[0]).days / 365
                cagr = ((trades_df['Port'].iloc[-1] / self.capital) ** (1 / years)) - 1
                stdev = trades_df['Port'].pct_change().std() * np.sqrt(252)
                dd = (trades_df['Port'] / trades_df['Port'].expanding().max() - 1).min()

                results_dict[f"{assets}_{rebalance}"] = {
                    'CAGR': cagr,
                    'DD': dd,
                    'StDev': stdev,
                    'Sharpe': cagr / stdev,
                    'Data': trades_df
                }

        path = settings.data_dir / 'tests' / f'MinVol_Semi_365x{self.period}D.pickle'
        with open(path, 'wb') as f:
            pickle.dump(results_dict, f)

    @staticmethod
    def analyze_backtests():
        path = settings.data_dir / "tests"

        vol_data = {'index': [], 'CAGR': [], 'DD': [], 'StDev': [], 'Sharpe': [], 'Data': []}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pickle") and ('MinVol_Semi_365x' in file):
                    with open(os.path.join(root, file), 'rb') as f:
                        cur_file = pickle.load(f)

                    name = file.split('x')[1].replace('D.pickle', '')
                    for key, cur_dict in cur_file.items():
                        vol_data['index'].append(f"{name}_{key}")
                        vol_data['CAGR'].append(cur_dict['CAGR'])
                        vol_data['DD'].append(cur_dict['DD'])
                        vol_data['StDev'].append(cur_dict['StDev'])
                        vol_data['Sharpe'].append(cur_dict['Sharpe'])
                        vol_data['Data'].append(cur_dict['Data'])

        results_df = pd.DataFrame.from_dict(vol_data).set_index('index').sort_values('Sharpe', ascending=False)

        return results_df

    @staticmethod
    def download_moex_indexes(name: str = 'MCFTR') -> pd.DataFrame:
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/index/securities/{name}.json"
        cur_index = get_data_by_ISSClient(url, {})
        cur_index['TRADEDATE'] = pd.to_datetime(cur_index['TRADEDATE']).dt.date
        cur_index = cur_index[['TRADEDATE', 'CLOSE']].set_index('TRADEDATE')

        return cur_index

    def visualize_backtested_port(self, results_df: pd.DataFrame, name: str = '2_15_quarterly',
                                  show_table: bool = True):
        if show_table:
            show_pd(results_df[['CAGR', 'DD', 'StDev', 'Sharpe']].sort_values(by='Sharpe', ascending=False))

        cur_port = results_df[results_df.index == name].iloc[0]
        name = (f"{name} CAGR: {cur_port['CAGR'] * 100:.2f}% DD: {cur_port['DD'] * 100:.2f}% "
                f"StDev: {cur_port['StDev'] * 100:.2f}% Sharpe: {cur_port['Sharpe']:.2f}")

        port = cur_port['Data']
        port['DD'] = 1 - port['Port'] / port['Port'].expanding().max()
        port['MCFTR'] = self.download_moex_indexes('MCFTR')['CLOSE']
        port['MCFTR'] = (port['MCFTR'].pct_change().shift(-1) + 1).cumprod() * self.capital

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port.index,
            y=port['Port'],
            mode='lines',
            name='Portfolio',
            line=dict(color='turquoise', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=port.index,
            y=port['MCFTR'],
            mode='lines',
            name='MCFTR',
            line=dict(color='dark red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=port.index,
            y=port['DD'],
            mode='lines',
            name='Drawdown',
            line=dict(color='rgba(255,140,0,0.4)', width=1, dash='dot'),
            yaxis="y2"
        ))
        fig.update_layout(
            template="plotly_dark",
            title=name,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Portfolio (log scale)", type="log"),
            yaxis2=dict(title="Drawdown",overlaying="y",side="right", tickformat=".0%"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
        )
        fig.show()