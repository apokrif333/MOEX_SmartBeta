from __future__ import annotations

from soupsieve.css_types import pickle_register

from moex_tools.strategies.low_volatility.main_class import LowVolatilityStrategy

def run_backtests():
    periods = [0.25, 0.5, 1, 2, 5, 10]
    for p in periods:
        strata = LowVolatilityStrategy(period=p)
        strata.load_base_data()
        strata.prepare_data()
        strata.calculate_volatility()
        strata.calculate_vol_rank()
        strata.run_backtest()


def run_weights_for_bot():
    strata = LowVolatilityStrategy()
    strata.load_base_data()
    strata.prepare_data()
    strata.calculate_volatility()
    strata.calculate_vol_rank()
    port = strata.filter_for_daily_check()
    port = strata.tink_load_figi(port)
    port = strata.tink_load_prices(port)
    strata.print_bot_data(port, assets=15)


def run_analyse_backtest():
    strata = LowVolatilityStrategy()
    results_df = strata.analyze_backtests()
    strata.visualize_backtested_port(results_df, name='2_15_quarterly', show_table=False)


import polars as pl
import pandas as pd
from moex_tools.config import settings

past_port = pd.read_csv(settings.bot_path / 'Relaible_weights.csv')
old_tickers = past_port['Ticker'].to_list()

secid = old_tickers + ['BOND', 'GOLD']

temp_base = pl.read_parquet(settings.data_dir / 'moex_splits_divs.parquet').filter(pl.col('SECID') == 'SBER')
# gain_df = (
#     temp_base.filter(pl.col('SECID').is_in(secid))
#     .sort(['SECID', 'DATE'], descending=[False, True])
#     .with_columns(
#         (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-5)).over('SECID').alias('shift5'),
#         (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-21)).over('SECID').alias('shift21'),
#         (pl.col('CLOSE_adj') / pl.col('CLOSE_adj').shift(-63)).over('SECID').alias('shift63')
#     ).group_by('SECID').agg(
#         pl.col('DATE').first(),
#         pl.col('DATE').shift(-5).first().alias('date5'),
#         pl.col('DATE').shift(-21).first().alias('date21'),
#         pl.col('DATE').shift(-63).first().alias('date63'),
#         pl.col('shift5').first(),
#         pl.col('shift21').first(),
#         pl.col('shift63').first(),
#         pl.col('CLOSE_adj').first()
#     )
# )
print(temp_base)