"""This is a utility module for setting up logging."""

import contextlib
import logging
from logging.handlers import RotatingFileHandler


@contextlib.contextmanager
def setup_logging():
    """Set up logging configuration with two handlers."""
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    try:
        # Log errors to file
        max_bytes = 5 * 1024 * 1024  # 5 MB
        file_handler = RotatingFileHandler(
            filename='output.log',
            encoding='utf-8',
            mode='w',
            maxBytes=max_bytes,
            backupCount=3
        )
        dt_fmt = "%d-%m-%Y %H:%M:%S"
        file_fmt = logging.Formatter(
            '[{asctime}] [{levelname:<7}] {name}: {message}',
            dt_fmt,
            style='{'
        )
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(logging.ERROR)
        log.addHandler(file_handler)

        # Log informations to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter(
            '[{asctime}] [{levelname:<7}] {message}',
            dt_fmt,
            style='{'
        )
        console_handler.setFormatter(file_fmt)
        log.addHandler(console_handler)

        yield
    finally:
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)
