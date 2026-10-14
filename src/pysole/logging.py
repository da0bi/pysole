"""
Logging Module for PySole.
Provides centralized logging to both console (stdout) and a log file (default: pysole.log).
"""

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import sys
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc="", unit="", ascii=None, bar_format=None, disable=False, leave=True, **kwargs):
        if iterable is not None:
            return iterable
        class DummyBar:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def update(self, n=1): pass
            def refresh(self): pass
            def close(self): pass
            def set_description(self, desc=None, refresh=True): pass
            def set_description_str(self, desc=None, refresh=True): pass
        return DummyBar()

# Global PySole logger
logger = logging.getLogger("pysole")


def setup_logging(
    log_file: str | Path | None = "pysole.log",
    log_level: str | int = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """Configures the centralized PySole package logger with file and console handlers.

    Parameters
    ----------
    log_file : str, Path, or None, default 'pysole.log'
        Path to the log file. If specified, logs are appended in UTF-8 format.
        If None, file logging is disabled.
    log_level : str or int, default 'INFO'
        Logging level threshold ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    console_output : bool, default True
        If True, log messages are also formatted and directed to stdout.

    Returns
    -------
    configured_logger : logging.Logger
        Configured logger instance named 'pysole'.
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
            log_path = Path(log_file).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Initialized PySole log file at: {log_path}")
        except Exception as e:
            logger.warning(f"Could not initialize log file '{log_file}': {e}")

    return logger


def get_progress_bar(
    iterable=None,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    """Creates a styled tqdm progress bar adhering to PySole's Modern Unicode Block design system.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable to wrap with progress monitoring.
    total : int, optional
        Total number of iteration steps if wrapping a generator or updating manually.
    desc : str, default ""
        Prefix description label displayed in front of the animated progress bar.
    unit : str, default 'it'
        Unit descriptor for iterations (e.g. 'kc', 'picks', 'chunks', 'bins', 'pixels').
    disable : bool, default False
        If True, suppresses progress bar animation completely (useful in non-interactive / batch modes).

    Returns
    -------
    progress_bar : tqdm.std.tqdm or DummyBar
        Configured progress bar instance.
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


def _format_clock(seconds: float) -> str:
    """Formats a duration as ``MM:SS`` (``H:MM:SS`` from one hour on)."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


@contextmanager
def activity_indicator(desc: str, disable: bool = False, delay: float = 1.0, interval: float = 1.0):
    """Shows an elapsed-time spinner while a long, uninterruptible computation runs (e.g. a dense LAPACK call).

    A daemon thread redraws one line ``<desc> [MM:SS]`` every ``interval`` seconds. It only sleeps and redraws
    (the BLAS/LAPACK calls release the GIL), so the computation is neither slowed nor changed. Nothing is shown
    for computations that finish within ``delay`` seconds. The thread is stopped and joined when the context
    exits, also if the body raises.

    Parameters
    ----------
    desc : str
        Description label shown in front of the elapsed time.
    disable : bool, default False
        If True nothing is shown and no thread is started (non-interactive / batch modes).
    delay : float, default 1.0
        Seconds before the line appears.
    interval : float, default 1.0
        Seconds between redraws.
    """
    if disable:
        yield
        return

    started = time.monotonic()
    stop = threading.Event()
    shown: list = []

    def line() -> str:
        return f"{desc} [{_format_clock(time.monotonic() - started)}]"

    def run() -> None:
        if stop.wait(delay):
            return
        bar = tqdm(total=None, desc=line(), bar_format="{desc}", leave=True)
        shown.append(bar)
        while not stop.wait(interval):
            bar.set_description_str(line())

    worker = threading.Thread(target=run, name="pysole-activity-indicator", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        if shown:
            shown[0].set_description_str(line())
            shown[0].close()

