"""This is a utility module for setting up logging."""

import contextlib
import logging
from logging.handlers import RotatingFileHandler


@contextlib.contextmanager
def setup_logging():
    """Set up logging configuration."""
    log = logging.getLogger()

    try:
        log.setLevel(logging.INFO)
        max_bytes = 5 * 1024 * 1024  # 5 MB
        handler = RotatingFileHandler(
            filename='output.log',
            encoding='utf-8',
            mode='w',
            maxBytes=max_bytes,
            backupCount=3
        )
        dt_fmt = "%d-%m-%Y %H:%M:%S"
        fmt = logging.Formatter(
            '[{asctime}] [{levelname:<7}] {name}: {message}',
            dt_fmt,
            style='{'
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
        # Log to console
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)

        yield
    finally:
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)
