from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    data_dir: Path = DEFAULT_DATA_DIR
    max_workers: int = 30
    war_days: set = {
        "20220301",
        "20220315",
        "20220316",
        "20220311",
        "20220310",
        "20220318",
        "20220302",
        "20220317",
        "20220309",
        "20220314",
        "20220303",
        "20220304",
        "20220228",
    }
    moex_ticker_for_dates_check: list = ["SBER", "AFLT"]
    moex_data_start: str = "20111219"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
