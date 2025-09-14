import polars as pl

from moex_tools.config import settings


def prepare_files():
    path = settings.data_dir / "auxiliary"

    bcs_pl = (
        pl.read_parquet(path / "bcs_divs.parquet")
        .rename({
            'Дата закрытия реестра': 'record_date', 'Последний день для покупки акций': 'last_buy_date',
            'Размер дивиденда': 'size', 'Цена акции на закрытие': 'close', 'ISIN': 'isin'
        }).with_columns(
            pl.col('record_date').cast(pl.Date), pl.col('last_buy_date').cast(pl.Date),
            pl.col('size').round(4),
            pl.col('close').round(4)
        )
    )
    bcs_pl = bcs_pl.group_by(['last_buy_date', 'isin', 'close']).agg(pl.col('size').sum())

    tink_pl = (
        pl.read_parquet(path / 'tinkoff_divs_filtered.parquet')
        .drop(['class_code', 'currency'])
        .rename({'correct_divs': 'size', 'ex_div_cls': 'close'})
        .with_columns(
            pl.col('payment_date').cast(pl.Date), pl.col('record_date').cast(pl.Date),
            pl.col('last_buy_date').cast(pl.Date), pl.col('declared_date').cast(pl.Date),
            pl.col('size').round(4),
            pl.col('close').round(4)
        )
    )
    tink_pl = tink_pl.group_by(['last_buy_date', 'ticker', 'close']).agg(pl.col('size').sum())

    fm_pl = (
        pl.read_parquet(path / 'financemarker_dividends.parquet')
        .with_columns(
            pl.col('record_date').cast(pl.Date), pl.col('exdiv_date').cast(pl.Date),
            pl.col('registry_date').cast(pl.Date), pl.col('size').round(4),
        )
    )
    fm_pl = fm_pl.group_by(['exdiv_date', 'ticker']).agg(pl.col('size').sum())

    path = r"G:\Market\moex-equity\data\intermediate\union_raw_moex_descript_split.parquet"
    df = pl.read_parquet(path).with_columns(pl.col('DATE').cast(pl.Date))

    dfs = {'base': df, 'bcs': bcs_pl, 'tink': tink_pl, 'fm': fm_pl}

    return dfs


def prep_bcs(dfs):
    bcs_pl = dfs['bcs']
    df = dfs['base']

    bcs_pl = (
        bcs_pl.join(
            df['isin', 'DATE', 'CLOSE', 'CLOSE_adj', 'adj_factor'],
            left_on=['isin', 'last_buy_date'],
            right_on=['isin', 'DATE'],
            how='left'
        ).filter(
            pl.col('CLOSE_adj').is_not_null()
        ).with_columns(
            cls_diff=(pl.col('CLOSE') - pl.col('close')).abs() / pl.col('CLOSE'),
            cls_adj_diff=(pl.col('CLOSE_adj') - pl.col('close')).abs() / pl.col('CLOSE_adj'),
        ).filter(
            (pl.col('cls_diff') < 0.005) | (pl.col('cls_adj_diff') < 0.005)
        ).with_columns(
            pl.when(pl.col('cls_adj_diff') > 0.005)
            .then(pl.col('size') * pl.col('adj_factor'))
            .otherwise(pl.col('size'))
            .alias('bcs_div')
        ).drop('size')
    )
    dfs['bcs'] = bcs_pl['last_buy_date', 'isin', 'bcs_div']

    return dfs


def prep_tink(dfs):
    tink_pl = dfs['tink']
    df = dfs['base']

    tink_pl = (
        tink_pl.join(
            df['SECID', 'DATE', 'CLOSE', 'CLOSE_adj', 'adj_factor'],
            left_on=['ticker', 'last_buy_date'],
            right_on=['SECID', 'DATE'],
            how='left'
        ).filter(
            pl.col('CLOSE_adj').is_not_null()
        )
        .with_columns(
            cls_diff=(pl.col('CLOSE') - pl.col('close')).abs() / pl.col('CLOSE'),
            cls_adj_diff=(pl.col('CLOSE_adj') - pl.col('close')).abs() / pl.col('CLOSE_adj'),
        ).filter(
            (pl.col('cls_diff') < 0.005) | (pl.col('cls_adj_diff') < 0.005)
        ).with_columns(
            pl.when(pl.col('cls_adj_diff') > 0.005)
            .then(pl.col('size') * pl.col('adj_factor'))
            .otherwise(pl.col('size'))
            .alias('tink_div')
        ).drop('size')
    )
    dfs['tink'] = tink_pl['last_buy_date', 'ticker', 'tink_div']

    return dfs


def prep_fm(dfs):
    fm_pl = dfs['fm']
    df = dfs['base']

    fm_pl = (
        fm_pl.join(
            df['SECID', 'DATE', 'CLOSE', 'CLOSE_adj', 'adj_factor'],
            left_on=['ticker', 'exdiv_date'],
            right_on=['SECID', 'DATE'],
            how='left'
        ).filter(
            pl.col('CLOSE_adj').is_not_null()
        ).rename({'size': 'fm_div'})
    )
    dfs['fm'] = fm_pl['exdiv_date', 'ticker', 'fm_div']

    return dfs


def divs_vola(df, first_col: str, second_col: str, threshold: float):
    cur_df = (
        df.with_columns(
            divs_mean=pl.mean_horizontal(first_col, second_col)
        ).with_columns(
            ((
                     (pl.col(first_col) - pl.col("divs_mean")).abs() +
                     (pl.col(second_col) - pl.col("divs_mean")).abs()
             ) / 2).alias('divs_vol')
        ).with_columns(
            (pl.col('divs_vol') / pl.col('divs_mean')).alias('divs_vol_ratio')
        )
    )

    cur_df = (
        cur_df.with_columns(
            pl.when(
                (pl.col(first_col) != 0) & (pl.col(second_col) != 0) &
                (pl.col('divs_vol_ratio') < threshold) & pl.col('true_div').is_null()
            ).then((pl.col(first_col) + pl.col(second_col)) / 2)
            .otherwise(pl.col('true_div'))
            .alias('true_div')
        ).drop(['divs_mean', 'divs_vol', 'divs_vol_ratio'])
    )

    return cur_df


def create_join_divs(df, first_col: str, second_col: str, div_df):
    join_df = (
        df.filter(pl.col(first_col) != 0)
        .join(
            df['SECID', 'DATE', second_col].filter(pl.col(second_col) != 0),
            how='left',
            left_on=['SECID', first_col],
            right_on=['SECID', second_col],
            suffix='_other'
        ).filter(
            pl.col('DATE_other').is_not_null()
        ).with_columns(
            time_diff=(pl.col('DATE') - pl.col('DATE_other')).dt.total_days()
        ).filter(
            (pl.col('time_diff') > 0) & (pl.col('time_diff') < 30)
        )
    )
    check = join_df.filter(join_df['SECID', 'DATE'].is_duplicated())
    assert len(check) == 0, f"The base has duplicates after filtration. Need to reduce time_diff range:\n {check}"

    join_df = join_df['SECID', 'DATE', first_col].rename({first_col: 'join_divs'})
    div_df = (
        div_df.join(join_df, on=['SECID', 'DATE'], how='left')
        .with_columns(
            pl.when(pl.col('true_div').is_null())
            .then(pl.col('join_divs'))
            .otherwise(pl.col('true_div'))
            .alias('true_div')
        ).drop('join_divs')
    )

    return div_df


def connect_divs_to_base(dfs):
    tink_pl = dfs['tink']
    fm_pl = dfs['fm']
    bcs_pl = dfs['bcs']
    df = dfs['base']

    div_df = (
        df.join(bcs_pl, left_on=['isin', 'DATE'], right_on=['isin', 'last_buy_date'], how='left')
        .join(tink_pl, left_on=['SECID', 'DATE'], right_on=['ticker', 'last_buy_date'], how='left')
        .join(fm_pl, left_on=['SECID', 'DATE'], right_on=['ticker', 'exdiv_date'], how='left')
        .with_columns(
            pl.col('bcs_div').shift(1).fill_null(0),
            pl.col('tink_div').shift(1).fill_null(0),
            pl.col('fm_div').fill_null(0),
            pl.lit(None).alias('true_div')
        )
    )

    div_df = divs_vola(div_df, first_col='bcs_div', second_col='tink_div', threshold=0.1)
    div_df = divs_vola(div_df, first_col='bcs_div', second_col='fm_div', threshold=0.1)
    div_df = divs_vola(div_df, first_col='tink_div', second_col='fm_div', threshold=0.1)

    null_divs = div_df.filter(
        pl.col('true_div').is_null() &
        ((pl.col('fm_div') != 0) | (pl.col('tink_div') != 0) | (pl.col('bcs_div') != 0))
    )
    null_divs = null_divs['SECID', 'DATE', 'adj_factor', 'true_div', 'bcs_div', 'tink_div', 'fm_div']

    div_df = create_join_divs(null_divs, first_col='bcs_div', second_col='tink_div', div_df=div_df)
    div_df = create_join_divs(null_divs, first_col='bcs_div', second_col='fm_div', div_df=div_df)
    div_df = create_join_divs(null_divs, first_col='tink_div', second_col='bcs_div', div_df=div_df)
    div_df = create_join_divs(null_divs, first_col='tink_div', second_col='fm_div', div_df=div_df)
    div_df = create_join_divs(null_divs, first_col='fm_div', second_col='bcs_div', div_df=div_df)
    div_df = create_join_divs(null_divs, first_col='fm_div', second_col='tink_div', div_df=div_df)

    div_df = (
        div_df.drop(['bcs_div', 'tink_div', 'fm_div'])
        .sort(['SECID', 'DATE'])
        .with_columns(pl.col('true_div').shift(-1).fill_null(0))
    )
    check = div_df.filter(div_df['SECID', 'DATE'].is_duplicated())
    assert len(check) == 0, f"The base has duplicates after changes. Need to investigate\n {check}"

    return div_df


def make_adj_prices(div_df):
    div_df = (
        div_df.sort(['SECID', 'DATE'], descending=[False, True])
        .with_columns(
            div_adj=((pl.col('CLOSE_adj') - pl.col('true_div')) / pl.col('CLOSE_adj'))
        ).with_columns(
            div_adj_cumprod=pl.col('div_adj').cum_prod().over('SECID')
        ).with_columns([
            pl.col('OPEN_adj') * pl.col('div_adj_cumprod'),
            pl.col('HIGH_adj') * pl.col('div_adj_cumprod'),
            pl.col('LOW_adj') * pl.col('div_adj_cumprod'),
            pl.col('CLOSE_adj') * pl.col('div_adj_cumprod'),
        ]).sort(
            ['SECID', 'DATE']
        ).drop(
            ['adj_factor', 'div_adj', 'div_adj_cumprod']
        )
    )

    return div_df


def run_divs_to_adj():
    dfs = prepare_files()
    dfs = prep_bcs(dfs)
    dfs = prep_tink(dfs)
    dfs = prep_fm(dfs)

    div_df = connect_divs_to_base(dfs)
    div_df = make_adj_prices(div_df)
    div_df.write_parquet(
        settings.data_dir / "moex_splits_divs.parquet", compression="lz4"
    )