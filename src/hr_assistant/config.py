"""Konfigurasi aplikasi dari environment dan berkas .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768

    typesense_url: str = "http://localhost:8108"
    typesense_api_key: str | None = None
    typesense_collection: str = "hr_docs"

    database_url: str = "postgresql://postgres@localhost:5432/hr_assistant"

    docs_dir: Path = Field(default=PROJECT_ROOT / "data" / "docs", validation_alias="HR_DOCS_DIR")
    top_k: int = 4
    min_score: float = Field(default=0.7, validation_alias="HR_MIN_SCORE")
    max_steps: int = 6


def get_settings() -> Settings:
    return Settings()
