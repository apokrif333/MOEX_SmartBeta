from __future__ import annotations
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