"""Logging configuration - optional structured JSON output for production."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Settings


class JsonFormatter(logging.Formatter):  # pragma: no cover - simple formatter
    """Minimal JSON log formatter (no extra deps required)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("request_id", "latency_ms", "stage"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    """Apply the application logging configuration to the root logger."""
    root = logging.getLogger()
    root.setLevel(settings.logging.level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if settings.logging.json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(handler)

    # Tame noisy third-party loggers.
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("tabula").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)