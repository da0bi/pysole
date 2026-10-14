"""Exception types shared by the PySole configuration, drift and solver layers."""


class ConfigError(ValueError):
    """Invalid PySole configuration, raised before any computation starts.

    Raised for malformed configuration values (wrong types, unknown drift terms, contradictory Kriging
    settings, ...) both while loading a configuration file and when a ``Solver`` is constructed. It subclasses
    ``ValueError`` so that generic ``except ValueError`` handlers keep working; the command-line interface
    reports it as a one-line ``pysole: error: ...`` message without a traceback.
    """
