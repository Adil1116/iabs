from __future__ import annotations

from datetime import datetime, timezone
import json
import logging


LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = 'INFO', log_format: str = 'json') -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    resolved_format = (log_format or 'json').strip().lower()
    handler = logging.StreamHandler()
    if resolved_format == 'json':
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)
    for logger_name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        logging.getLogger(logger_name).setLevel(resolved_level)
