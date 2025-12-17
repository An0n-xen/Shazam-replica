import os
import logging


def setup_logger(name=None, level=logging.INFO):
    """
    Set up logger with file and line number information.

    Args:
        name: Logger name (use __name__ to get module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logger: Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handler:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter with file name, line number, and function
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(filename)s:%(lineno)d  | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
