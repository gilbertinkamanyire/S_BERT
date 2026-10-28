"""Thin launcher for the packaged VERITAS application."""

from backend.app import main
from backend.veritext_app import app

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
