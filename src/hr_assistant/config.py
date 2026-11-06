"""Konfigurasi aplikasi dari environment dan berkas .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    index_dir: Path
    employee_data: Path
    collection: str = "hr_docs"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_key: str | None = None
    top_k: int = 4
    min_score: float = 0.45
    max_steps: int = 6


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        docs_dir=_path("HR_DOCS_DIR", "data/docs"),
        index_dir=_path("HR_INDEX_DIR", ".index"),
        employee_data=_path("HR_EMPLOYEE_DATA", "data/employees.json"),
        gemini_model=os.environ.get("GEMINI_MODEL") or Settings.gemini_model,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        min_score=float(os.environ.get("HR_MIN_SCORE", Settings.min_score)),
    )


def _path(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path
