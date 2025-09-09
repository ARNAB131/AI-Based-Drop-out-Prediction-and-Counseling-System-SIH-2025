from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Rajasthan Dropout Risk API"
    APP_VERSION: str = "0.2.0"
    MODEL_PATH: str = "models/baseline_xgb.pkl"
    DATA_CSV: str = "../../data/demo_students.csv"
    FEATS: list[str] = ["attendance_rate","avg_score","distance_km","transfers","scholarship_flag"]
    DATABASE_URL: str = "sqlite:///./dropout.db"   # set to postgres: postgresql+psycopg2://user:pass@host/db

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache(maxsize=1)
def get_settings():
    return Settings()
