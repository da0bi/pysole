"""
Logging Module for PySole.
Provides centralized logging to both console (stdout) and a log file (default: pysole.log).
"""

import logging
import os
import sys
from typing import Optional, Union
from tqdm import tqdm

# Global PySole logger
logger = logging.getLogger("pysole")


def setup_logging(
    log_file: Optional[str] = "pysole.log",
    log_level: Union[str, int] = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """Configures the PySole package logger with file and console handlers.

    Parameters
    ----------
    log_file : Optional[str]
        Path to the log file. If set (default: 'pysole.log'), logs are appended to this file.
        If None, file logging is disabled.
    log_level : Union[str, int]
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'.
    console_output : bool
        If True (default), log messages are also output to stdout.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        numeric_level = log_level

    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on re-initialization
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (StreamHandler)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if log_file:
        try:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize log file '{log_file}': {e}")

    return logger


def get_progress_bar(
    iterable=None,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    """Creates a styled tqdm progress bar adhering to PySole's Modern Unicode Block design system.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable to wrap with progress bar.
    total : int, optional
        Total number of iterations.
    desc : str
        Description tag shown in front of the progress bar.
    unit : str
        Unit label for iterations (e.g. 'kc', 'picks', 'chunks', 'bins', 'pixels').
    disable : bool
        If True, disables the progress bar animation entirely.

    Returns
    -------
    tqdm
        Styled tqdm instance.
    """
    bar_format = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        ascii=" ▕█░▏",
        bar_format=bar_format,
        disable=disable,
        leave=True,
    )

