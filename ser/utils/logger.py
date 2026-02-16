"""Logging helpers used across the SER package."""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates or retrieves a configured logger.

    Args:
        name: Logger name, usually `__name__`.
        level: Logging level passed to `logging.basicConfig`.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger: logging.Logger = logging.getLogger(name)
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(format=LOG_FORMAT, level=level)
    return logger
