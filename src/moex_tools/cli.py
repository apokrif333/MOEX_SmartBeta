import logging

import typer

from moex_tools.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger("moex_tools.cli")

import datetime as dt
import time
from zoneinfo import ZoneInfo
from moex_tools.pipelines.make_clean import run as make_clean_run
from moex_tools.strategies.low_volatility.pipelines import run_weights_for_bot

app = typer.Typer(help="MOEX Tools CLI")
MSK_ZONE = ZoneInfo("Europe/Moscow")
UTC_ZONE = dt.timezone.utc


def _next_monday_0950_msk(now_utc: dt.datetime) -> dt.datetime:
    now_msk = now_utc.astimezone(MSK_ZONE)
    target_msk = now_msk.replace(hour=9, minute=50, second=0, microsecond=0)
    days_ahead = (0 - now_msk.weekday()) % 7
    target_msk = target_msk + dt.timedelta(days=days_ahead)
    return target_msk.astimezone(UTC_ZONE)


@app.command()
def wait():
    logging.info("WAIT: entering sleep loop (target = Monday 09:50 MSK).")

    is_logged = False
    while True:
        now_utc = dt.datetime.now(tz=UTC_ZONE)
        tgt_utc = _next_monday_0950_msk(now_utc)
        remaining = (tgt_utc - now_utc).total_seconds()

        logging.info(f"{now_utc}, {tgt_utc}, {remaining}")
        if remaining <= 0:
            logging.info("WAIT: it's time — Monday 09:50 MSK reached.")
            return

        if not is_logged:
            tgt_msk_str = tgt_utc.astimezone(MSK_ZONE).strftime("%Y-%m-%d %H:%M MSK")
            mins = int(remaining // 60)
            logging.info(f"WAIT: {tgt_msk_str} (~{mins} min left).")
            is_logged = True

        time.sleep(10)


@app.command()
def make_clean():
    logger.info("make_clean_run: START")
    make_clean_run()
    logger.info("make_clean_run: DONE")


@app.command()
def low_volatility():
    logger.info("run_weights_for_bot: START")
    run_weights_for_bot()
    logger.info("run_weights_for_bot: DONE")


if __name__ == "__main__":
    app()