import logging

import apimoex
import pandas as pd
import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

from moex_tools.config import settings

# Settings
pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
pd.options.display.width = 1200

# @QcmStockRusBot, @QcmTestBot
TELEGRAM_TOKEN = settings.bot_calc_test_token

SELECT_PORTFOLIO, CALCULATE = range(2)
SHOW_PORTFOLIO = range(3)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# Calc Func ------------------------------------------------------------------------------------------------------------
def get_ports():
    url = "https://drive.google.com/file/d/1E-5adtWX7uqqvWN3WZgIA9J8BfA76v6H/view"
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    df = pd.read_csv(download_url)

    reliable_port = dict(zip(df["Ticker"], df["Weight"]))
    reliable_hedge_port = {
        **{k: v / 2 for k, v in reliable_port.items()},
        **{"BOND": 0.30, "GOLD": 0.20},
    }

    return reliable_port, reliable_hedge_port


def get_data_by_ISSClient(url: str, query: dict) -> pd.DataFrame:
    cur_data = []
    with requests.Session() as s:
        client = apimoex.ISSClient(s, url, query)
        data = client.get()  # get(), get_all()

        keys_len = {}
        for key in data.keys():
            keys_len[key] = len(data[key])
        keys_len = dict(sorted(keys_len.items(), key=lambda item: item[1]))
        main_key = list(keys_len)[-1]

        cur_data.extend(data[main_key])
        if len(data[main_key]) == 1000:
            for idx in range(1000, 1_000_000, 1000):
                print(idx)

                client = apimoex.ISSClient(s, url, query)
                data = client.get(start=idx)
                cur_data.extend(data[main_key])

                if len(data[main_key]) < 1000:
                    break

    df = pd.DataFrame(cur_data)
    return df


def get_current_stocks_data(sec_list: pd.Series) -> pd.DataFrame:
    # https://capital-gain.ru/posts/portfolio-performance-usage/

    total_prices = pd.DataFrame()
    for i in range(0, len(sec_list), 10):
        cur_secs = sec_list.iloc[i : i + 10]
        cur_secs = ",".join(cur_secs.to_list())

        main_url = "https://iss.moex.com/iss"
        main_url += r"/engines/stock/markets/shares/securities" + ".json"
        query = {
            "securities": cur_secs,
            "marketdata.columns": "UPDATETIME,SECID,BOARDID,LAST,BID,OFFER,MARKETPRICE",
        }
        cur_df = get_data_by_ISSClient(main_url, query)
        total_prices = pd.concat([total_prices, cur_df])

    total_prices = total_prices[total_prices["BOARDID"].isin(["TQBR", "TQTF"])]

    return total_prices


def optimize_weights(cur_port: pd.DataFrame, capital: float):
    offer_na = sum(cur_port["OFFER"].isna())
    last_na = sum(cur_port["LAST"].isna())
    market_na = sum(cur_port["MARKETPRICE"].isna())
    if offer_na == 0:
        price = "OFFER"
    elif last_na == 0:
        price = "LAST"
    elif market_na == 0:
        price = "MARKETPRICE"
    else:
        raise Exception(
            f"OFFER {offer_na} | LAST {last_na} | MARKETPRICE {market_na} have nan values"
        )

    shares_prices = {
        "assets": len(cur_port),
        "assetsPrices": cur_port[price].to_list(),
        "assetsWeights": cur_port["weight"].to_list(),
        "portfolioValue": capital,
    }

    url = "https://api.portfoliooptimizer.io/v1"
    req = "/portfolio/construction/investable"
    answ = requests.post(url + req, json=shares_prices)

    cur_port["shares_quantity"] = answ.json()["assetsPositions"]
    cur_port["true_weight"] = answ.json()["assetsWeights"]

    return cur_port


def calc_port(capital: float, port: dict):
    cur_port = pd.DataFrame.from_dict(port, orient="index").reset_index()
    cur_port.columns = ["ticker", "weight"]

    cur_price = get_current_stocks_data(cur_port["ticker"])
    cur_port = cur_port.merge(cur_price, left_on="ticker", right_on="SECID", how="inner")

    print(cur_port)
    cur_port = optimize_weights(cur_port, capital)

    return cur_port


# Bot Func ------------------------------------------------------------------------------------------------------------
async def check_subscription(user_id: int) -> bool:
    CHAT_ID = "-1002471577619"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getChatMember"
    params = {"chat_id": CHAT_ID, "user_id": user_id}
    response = requests.get(url, params=params).json()
    print(response)
    if response.get("ok"):
        status = response["result"].get("status")
        return status in ["member", "administrator", "creator"]
    else:
        return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Перед началом работы, ознакомьтесь с дисклеймером.")
    return await disclaimer(update, context)


async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    disclaimer_text = (
        "Представленная в данном боте информация не является индивидуальной инвестиционной рекомендацией, "
        "ни при каких условиях, в том числе при внешнем совпадении её содержания с требованиями нормативно-правовых актов, "
        "предъявляемых к индивидуальной инвестиционной рекомендации. Любое сходство представленной информации с "
        "индивидуальной инвестиционной рекомендацией является случайным. Какие-либо из указанных финансовых инструментов "
        "или операций могут не соответствовать вашему инвестиционному профилю. Упомянутые в представленных сообщениях операции "
        "и (или) финансовые инструменты ни при каких обстоятельствах не гарантируют доход, на который вы, возможно, рассчитываете, "
        "при условии использования предоставленной информации для принятия инвестиционных решений. Доходность, полученная в прошлом, "
        "не гарантирует доходность в будущем. Исполнитель не несёт ответственности за возможные убытки инвестора в случае совершения операций, "
        "либо инвестирования в финансовые инструменты, упомянутые в представленной информации. Во всех случаях определение соответствия "
        "финансового инструмента либо операции инвестиционным целям, инвестиционному горизонту и толерантности к риску является задачей инвестора."
    )
    reply_keyboard = [["Я согласен"]]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(disclaimer_text, reply_markup=markup)
    return SHOW_PORTFOLIO


async def show_portfolio_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_keyboard = [
        ["Высоконадёжный портфель", "Высоконад. портфель с хэджем"],
        ["Перезапустить"],
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(
        "Выберите один из портфелей для расчета: \n`предрасчёт занимает 3 секунды`",
        reply_markup=markup,
        parse_mode="MarkdownV2",
    )
    return SELECT_PORTFOLIO


async def handle_portfolio_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await check_subscription(user_id):
        await update.message.reply_text(
            "Вы не подписаны на канал 'Моя стратегия РФ'. " "Просьба связаться с @emfmanager_bot"
        )
        return ConversationHandler.END

    user_choice = update.message.text
    if user_choice == "Перезапустить":
        return await start(update, context)

    reliable_port, reliable_hedge_port = get_ports()

    user_choice = update.message.text
    if user_choice == "Высоконадёжный портфель":
        context.user_data["portfolio"] = reliable_port
        await update.message.reply_text(
            f"Вы выбрали {user_choice}. Отправьте текущую стоимость вашего капитала одной цифрой в рублях:"
        )
        return CALCULATE

    elif user_choice == "Высоконад. портфель с хэджем":
        context.user_data["portfolio"] = reliable_hedge_port
        await update.message.reply_text(
            f"Вы выбрали {user_choice}. Отправьте текущую стоимость вашего капитала одной цифрой в рублях:"
        )
        return CALCULATE

    await update.message.reply_text("Пожалуйста, выберите один из предложенных вариантов.")
    return SELECT_PORTFOLIO


async def start_calc_port(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text
        capital = float(user_input)
        selected_port = context.user_data.get("portfolio")

        cur_port = calc_port(capital, selected_port)

        response = []
        if cur_port["OFFER"].isna().sum() > 0:
            response.append(
                "В данный момент фондовый рынок закрыт. Рекомедуется запускать расчёты в момент работы "
                " Московской Биржи, с 10:15 до 18:54 в рабочие дни.\n"
            )

        response.append(f"Расчет долей сделан на сумму ₽{capital:,.0f} \n")
        for idx, row in cur_port.iterrows():
            t = row["ticker"]
            shares = row["shares_quantity"]
            weigh = row["true_weight"]
            if shares > 0:
                response.append(
                    f"{t} \n Количество акций: {shares:,.0f} \n Вес в портфеле: {weigh:.2%} \n"
                )
            else:
                response.append(
                    f"{t} \n Удалена из портфеля, чтобы купить иные бумаги в соотвествии с их весами. "
                    f"Чтобы купить все ценные бумаги, увеличьте сумму инвестиций.\n"
                )

        await update.message.reply_text("\n".join(response))

        await update.message.reply_text("Хотите рассчитать другой портфель?")

        return await show_portfolio_selection(update, context)

    except ValueError:
        await update.message.reply_text("Введите корректную сумму капитала, типа 300000")
        return CALCULATE


def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SHOW_PORTFOLIO: [
                MessageHandler(filters.Regex("^Я согласен$"), show_portfolio_selection)
            ],
            SELECT_PORTFOLIO: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_portfolio_selection)
            ],
            CALCULATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, start_calc_port)],
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("disclaimer", disclaimer))

    print("Бот запущен")
    application.run_polling()


if __name__ == "__main__":
    main()
