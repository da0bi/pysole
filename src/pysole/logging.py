"""
Logging Module for PySole.
Provides centralized logging to both console (stdout) and a log file (default: pysole.log).
"""

import logging
import threading
import time
from contextlib import contextmanager
import shutil
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

_current_log_path: Path | None = None
_current_log_level: int | None = None
_current_console_output: bool | None = None


def reset_logging_state() -> None:
    """Resets cached logging configuration state (primarily for test fixtures)."""
    global _current_log_path, _current_log_level, _current_console_output
    _current_log_path = None
    _current_log_level = None
    _current_console_output = None


def setup_logging(
    log_file: str | Path | None = "pysole.log",
    log_level: str | int = "INFO",
    console_output: bool = True,
    log_init_msg: bool = True,
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
    log_init_msg : bool, default True
        If True, logs an informative notice upon file handler initialization.

    Returns
    -------
    configured_logger : logging.Logger
        Configured logger instance named 'pysole'.
    """
    global _current_log_path, _current_log_level, _current_console_output

    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        numeric_level = log_level

    target_log_path = Path(log_file).expanduser().resolve() if log_file else None

    # Idempotent check: if logger already configured with identical target file, level, and console mode, return
    if (
        logger.handlers
        and _current_log_path == target_log_path
        and _current_log_level == numeric_level
        and _current_console_output == console_output
    ):
        return logger

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
    if target_log_path:
        try:
            target_log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target_log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            _current_log_path = target_log_path
            _current_log_level = numeric_level
            _current_console_output = console_output
            if log_init_msg:
                logger.info(f"Initialized PySole log file at: {target_log_path}")
        except Exception as e:
            logger.warning(f"Could not initialize log file '{log_file}': {e}")
    else:
        _current_log_path = None
        _current_log_level = numeric_level
        _current_console_output = console_output

    return logger


def get_welcome_banner(version: str | None = None, term_width: int | None = None) -> str:
    """Renders the PySole ASCII art welcome banner, dynamically centered for the terminal.

    Parameters
    ----------
    version : str, optional
        Version string override. If None, dynamically resolved from `pysole.__version__`.
    term_width : int, optional
        Terminal column width for centering. If None, resolved via `shutil.get_terminal_size()`.

    Returns
    -------
    banner : str
        Centered multi-line banner string.
    """
    if version is None:
        try:
            from . import __version__
            version = __version__
        except Exception:
            version = "0.5.2"

    version_str = version if str(version).startswith("v") else f"v{version}"

    if term_width is None:
        term_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    term_width = max(term_width, 40)

    ascii_art = [
        r"    ____        _____       __    ",
        r"   / __ \__  _ / ___/____  / /__  ",
        r"  / /_/ / / / /\__ \/ __ \/ / _ \ ",
        r" / ____/ /_/ /___/ / /_/ / /  __/ ",
        r"/_/    \__, /_____/\____/_/\___/  ",
        r"      /____/                      ",
    ]

    block_width = max(len(line) for line in ascii_art)
    art_margin = max(0, (term_width - block_width) // 2)
    art_indent = " " * art_margin

    lines = [""]
    for art_line in ascii_art:
        lines.append(f"{art_indent}{art_line}".rstrip())

    lines.append("")
    lines.append("The soul of the sole".center(term_width).rstrip())
    lines.append("")
    lines.append(version_str.center(term_width).rstrip())
    lines.append("")
    lines.append("Released under the MIT License.".center(term_width).rstrip())
    lines.append("")

    return "\n".join(lines)


def print_welcome_banner(version: str | None = None, file=None) -> None:
    """Prints the dynamically centered PySole welcome banner to stdout (or specified stream).

    Parameters
    ----------
    version : str, optional
        Version string override.
    file : TextIO, optional
        Target text stream (defaults to sys.stdout).
    """
    out = file if file is not None else sys.stdout
    out.write(get_welcome_banner(version=version) + "\n")
    out.flush()


def log_section_header(title: str, line_char: str = "=", width: int = 80) -> None:
    """Logs a prominent section headline bounded by decorative divider lines.

    Parameters
    ----------
    title : str
        Headline title to display.
    line_char : str, default '='
        Divider character.
    width : int, default 80
        Total line width for the divider and centering.
    """
    divider = line_char * width
    centered = title.center(width)
    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info(divider)
    logger.info(centered)
    logger.info(divider)
    sys.stdout.write("\n")
    sys.stdout.flush()


_BaseBar = tqdm if isinstance(tqdm, type) else object


class PySoleProgressBar(_BaseBar):
    """Wrapped tqdm progress bar providing single empty line isolation before and after execution."""

    def __init__(self, *args, **kwargs):
        self._is_disabled = kwargs.get("disable", False)
        self._target_file = kwargs.get("file", sys.stdout)
        kwargs.setdefault("file", self._target_file)
        if not self._is_disabled:
            self._target_file.write("\n")
            self._target_file.flush()
        if isinstance(tqdm, type):
            super().__init__(*args, **kwargs)
        self._post_nl_done = False

    def close(self):
        if isinstance(tqdm, type):
            super().close()
        if not getattr(self, "_is_disabled", False) and not getattr(self, "_post_nl_done", False):
            self._post_nl_done = True
            out = getattr(self, "_target_file", sys.stdout)
            out.write("\n")
            out.flush()


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
    progress_bar : PySoleProgressBar or DummyBar
        Configured progress bar instance.
    """
    if not isinstance(tqdm, type):
        return tqdm(iterable=iterable, total=total, desc=desc, unit=unit, disable=disable)

    bar_format = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
    return PySoleProgressBar(
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


__all__ = [
    "logger",
    "setup_logging",
    "reset_logging_state",
    "get_welcome_banner",
    "print_welcome_banner",
    "log_section_header",
    "PySoleProgressBar",
    "get_progress_bar",
    "activity_indicator",
]
