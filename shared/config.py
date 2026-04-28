"""Configuration settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # DuckDB backend
    duckdb_path: str = "output/eval_cards.duckdb"
    default_aggregate_jsonl_path: str = "data/simulated/public_leaderboards/aggregate_rows_200.jsonl"
    default_instance_jsonl_path: str = "data/simulated/public_leaderboards/instance_rows_200.jsonl"
    ingestion_base_dir: str = "data"

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/every_eval_ever"

    # Cloudflare R2
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "eval-bundles"

    # API
    api_key: str = ""

    # Worker
    worker_poll_interval_seconds: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
