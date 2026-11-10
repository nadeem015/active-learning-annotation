from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./active_learning.db"
    data_path: str = "data/demo_pool.csv"
    test_size: int = 60
    model_backend: str = "sklearn"
    random_seed: int = 42
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
