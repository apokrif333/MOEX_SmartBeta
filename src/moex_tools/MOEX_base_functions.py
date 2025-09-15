import xml.etree.ElementTree as ET

import apimoex
import pandas as pd
import requests
from numpy.testing.print_coercion_tables import print_new_cast_table

# Settings
pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
pd.options.display.width = 1200
pd.set_option("display.float_format", lambda x: "%.3f" % x)


"""
https://wlm1ke.github.io/apimoex/build/html/api.html#id4
file:///C:/Users/Alex/Desktop/ForDel/iss-api-rus-v14.pdf
https://iss.moex.com/iss/reference/
https://iss.moex.com/iss/index
"""


def get_data_by_ISSClient(url: str, query: dict, key: str = "") -> pd.DataFrame:
    cur_data = []
    with requests.Session() as s:
        client = apimoex.ISSClient(s, url, query)
        data = client.get()  # get(), get_all()

        if key == "":
            keys_len = {}
            for cur_key in data.keys():
                if '.' in cur_key:
                    continue
                keys_len[cur_key] = len(data[cur_key])
            keys_len = dict(sorted(keys_len.items(), key=lambda item: item[1]))
            main_key = list(keys_len)[-1]
        else:
            main_key = key

        cur_data.extend(data[main_key])
        cursor = len(data[main_key])
        idx = len(data[main_key])
        if idx >= 100:
            while True:
                client = apimoex.ISSClient(s, url, query)
                data = client.get(start=idx)
                if not data[main_key]:
                    break

                cur_data.extend(data[main_key])
                idx += len(data[main_key])
                if len(data[main_key]) < cursor:
                    break

        # Old code -----------
        # if len(data[main_key]) == 1000:
        #     for idx in range(1000, 1_000_000, 1000):
        #         print(idx)
        #
        #         client = apimoex.ISSClient(s, url, query)
        #         data = client.get(start=idx)
        #         cur_data.extend(data[main_key])
        #
        #         if len(data[main_key]) < 1000:
        #             break
        #
        # if len(data[main_key]) == 100:
        #     for idx in range(100, 1_000_000, 100):
        #         print(idx)
        #
        #         client = apimoex.ISSClient(s, url, query)
        #         data = client.get(start=idx)
        #         cur_data.extend(data[main_key])
        #
        #         if len(data[main_key]) < 100:
        #             break

        # Old code -----------
        # cur_data.extend(data[main_key])
        # if 'history.cursor' in data.keys():
        #     total = data['history.cursor'][0]['TOTAL']
        #     start = data['history.cursor'][0]['PAGESIZE']
        #     for idx in range(start, total, start):
        #         client = apimoex.ISSClient(s, url, query)
        #         data = client.get(start=idx)
        #         cur_data.extend(data[main_key])

    df = pd.DataFrame(cur_data)
    return df


def get_placeholder_values(s: requests.Session, placeholder="engines") -> pd.DataFrame:
    # 'engines, markets, boards, boardgroups, durations, securitytypes, securitygroups, securitycollection'
    answer = apimoex.get_reference(s, placeholder)
    df = pd.DataFrame(answer)
    print(df)

    return df


def get_all_indexes() -> pd.DataFrame:
    answ = requests.get("https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics")
    root = ET.fromstring(answ.text)

    data = []
    for row in root.findall(".//row"):
        indexid = row.get("indexid")
        shortname = row.get("shortname")
        from_date = row.get("from")
        till_date = row.get("till")

        data.append(
            {
                "indexid": indexid,
                "shortname": shortname,
                "from_date": from_date,
                "till_date": till_date,
            }
        )

    df = pd.DataFrame(data).sort_values("from_date")

    return df


def get_index_components(s, index="IMOEX", data="2001-05-17") -> pd.DataFrame:
    answer = apimoex.get_index_tickers(s, index, date=data)

    return pd.DataFrame(answer)


def get_company_description(s, security="GAZP") -> pd.DataFrame:
    answer = apimoex.find_security_description(s, security)

    return pd.DataFrame(answer)


def get_OHLC(s, start: str, ticker="GAZP") -> pd.DataFrame:
    # Могут быть дубликаты в днях, из-за разного режима торгов - нужен чек
    answer = apimoex.get_market_candles(s, ticker, start=start)
    df = pd.DataFrame(answer)

    return df


def get_current_tickers_list(s, type="securities") -> pd.DataFrame:
    answer = apimoex.get_board_securities(s, type)
    df = pd.DataFrame(answer)

    return df


def get_tickers_list_at_spec_data(date="2001-08-20"):
    main_url = "https://iss.moex.com/iss"
    main_url += r"/history/engines/stock/markets/shares/securities.json"
    query = {"date": date}
    df = get_data_by_ISSClient(main_url, query)

    return df


def all_delisted_tickers() -> pd.DataFrame:
    main_url = "https://iss.moex.com/iss"
    main_url += r"/history/engines/stock/markets/shares/listing.json"
    query = {}
    df = get_data_by_ISSClient(main_url, query).sort_values("history_from")

    return df


def get_tickers_names_cng() -> pd.DataFrame:
    main_url = "https://iss.moex.com/iss"
    main_url += r"/history/engines/stock/markets/shares/securities/changeover" + ".json"
    query = {}
    df = get_data_by_ISSClient(main_url, query)

    return df


def get_historical_stocks_data_links(year="2005", datatype="monthly"):
    """

    :param year:
    :param datatype: yearly, monthly или daily
    :return:
    """

    main_url = "https://iss.moex.com/iss"
    main_url += rf"/archives/engines/stock/markets/shares/securities/{datatype}" + ".json"
    query = {"year": year}
    df = get_data_by_ISSClient(main_url, query)

    return df


def split_history(ticker="SBER") -> pd.DataFrame:
    main_url = "https://iss.moex.com/iss"
    main_url += rf"/statistics/engines/stock/splits/{ticker}" + ".json"
    query = {}
    df = get_data_by_ISSClient(main_url, query)

    return df


def index_weights(index_id="IMOEX", date="2005-02-17") -> pd.DataFrame:
    main_url = "https://iss.moex.com/iss"
    main_url += f"/statistics/engines/stock/markets/index/analytics/{index_id}" + ".json"
    query = {"date": date}
    df = get_data_by_ISSClient(main_url, query)

    return df


def total_index_components(index_id="IMOEX") -> pd.DataFrame:
    main_url = "https://iss.moex.com/iss"
    main_url += rf"/statistics/engines/stock/markets/index/analytics/{index_id}/tickers" + ".json"
    query = {}
    df = get_data_by_ISSClient(main_url, query)

    return df


def get_exchange_total_info(date="2012-02-22") -> pd.DataFrame:
    """
    Returns the total information for the exchange on a specific date.
    :param date: Format 'YYYY-MM-DD'
    :return: DataFrame with exchange information
    """

    main_url = "https://iss.moex.com/iss"
    main_url += r"/history/engines/stock/totals/securities" + ".json"
    query = {"date": date}
    df = get_data_by_ISSClient(main_url, query)

    return df


def get_columns_description() -> pd.DataFrame:
    # https://ftp.moex.com/pub/ClientsAPI/ASTS/Bridge_Interfaces/MarketData/Equities35_Info_Russian.htm

    main_url = "https://iss.moex.com"
    main_url += r"/iss/engines/stock/markets/shares/securities/columns" + ".json"
    query = {}
    df = get_data_by_ISSClient(main_url, query)
    df.to_csv("discription/columns_description.csv", index=False, encoding="cp1251")

    return df


def get_current_stocks_data(sec_list: str):
    # https://capital-gain.ru/posts/portfolio-performance-usage/

    main_url = "https://iss.moex.com/iss"
    main_url += r"/engines/stock/markets/shares/boards/TQBR/securities" + ".json"
    query = {"securities": sec_list}
    df = get_data_by_ISSClient(main_url, query)

    # return df[['SECID', 'MARKETPRICE']]
    return df


if __name__ == "__main__":
    # with requests.Session() as s:
    #     get_placeholder_values(s, 'boardgroups')
    # raise Exception

    # with requests.Session() as s:
    #     answ = get_OHLC(s, 'SNGSP')
    # print(answ)
    # raise Exception

    # with requests.Session() as s:
    #     df = get_tickers_list_at_spec_data('2011-12-19')
    #     print(df['SECID'])
    # raise Exception

    # main_url = 'https://iss.moex.com/iss'
    # main_url += rf"/engines/stock/markets/shares/securities" + '.json'
    # query = {'securities': 'AFLT,BOND,GOLD'}
    # df = get_data_by_ISSClient(main_url, query)
    # df = df[df['BOARDID'].isin(['TQBR', 'TQTF'])]
    # print(df)

    # df = get_current_stocks_data('RU000A104Z22')
    # print(df)

    print(get_exchange_total_info('2025-09-12'))

    # main_url = "https://iss.moex.com/iss"
    # main_url += r"/history/engines/stocks/markets/shares/boards/57/listing.json"
    # query = {}
    # df = get_data_by_ISSClient(main_url, query)
    # print(df)

    # /iss/history/engines/stock/totals/boards/[board]/securities/[security]
    # Нехватает только полной доходности, включая дивы. Возмрожно по этому поводу стоит позвонить на биржу.

    # main_url = 'https://iss.moex.com'
    # main_url += r"/iss/history/engines/stock/totals/securities" + '.json'
    # # query = {'from': '2010-08-20', 'till': '2010-09-20'}
    # query = {'date': '2020-04-10'}
    # # query = {}

    # answer = requests.get(main_url, params=query)
    # pprint(answer.json())
    # raise Exception

    # df = get_data_by_ISSClient(main_url, query)
    # print(df)
    # raise Exception

    # with requests.Session() as s:
    # answ = apimoex.find_securities(s, 'RU000A104Z22')
    # answ = apimoex.find_security_description(s, 'BOND')
    # print(answ)

    # main_url = 'https://iss.moex.com/iss'
    # main_url += rf"/engines" + '.json'
    # query = {}
    # df = get_data_by_ISSClient(main_url, query)
    # print(df)

    # main_url = 'https://iss.moex.com/iss'
    # main_url += rf"/engines/stock/markets" + '.json'
    # query = {}
    # df = get_data_by_ISSClient(main_url, query)
    # print(df)

    # main_url = 'https://iss.moex.com/iss'
    # main_url += rf"/engines/stock/markets/shares/boards" + '.json'
    # query = {}
    # df = get_data_by_ISSClient(main_url, query)
    # print(df)

    # main_url = 'https://iss.moex.com/iss'
    # main_url += rf'/engines/stock/markets/shares/securities' + '.json'
    # query = {'securities': 'BOND', 'BOARDID': 'TQF'}
    # df = get_data_by_ISSClient(main_url, query, 'marketdata')
    # print(df)
