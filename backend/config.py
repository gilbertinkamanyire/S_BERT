"""Application configuration for VERITAS VeriText."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_PORT = 5050
MAX_UPLOAD_SIZE = 32 * 1024 * 1024


def environment_value(*names: str, default: str = "") -> str:
    """Return the first configured environment value, trimmed of surrounding quotes."""
    for name in names:
        value = os.environ.get(name, "")
        if value:
            value = value.strip().strip('"').strip("'").strip()
            if value:
                return value
    return default


def get_gemma_api_key() -> str:
    return environment_value("GEMMA_API_KEY", "GOOGLE_API_KEY")


def get_gemma_model() -> str:
    return environment_value("GEMMA_MODEL", default=DEFAULT_MODEL)


def get_port() -> int:
    try:
        return max(1, min(65535, int(environment_value("PORT", default=str(DEFAULT_PORT)))))
    except (TypeError, ValueError):
        return DEFAULT_PORT


UPLOAD_DIR = BASE_DIR / "data" / "uploads"
TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"
