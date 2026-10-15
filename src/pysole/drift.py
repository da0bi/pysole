"""
Drift model rules shared by the configuration loader, the Solver and the Drift Analyzer.

This module is the single place that defines which drift terms exist, how a ``drift_terms`` list is validated,
which candidate suite the Drift Analyzer evaluates and how ``method`` / ``drift_analyzer`` / ``drift_terms``
interact. It is deliberately free of heavy imports (no NumPy, no rasters) so that the configuration can be
validated before any data is read.

Rules (``resolve_drift_plan``)
------------------------------
``drift_terms`` is ``None`` (omitted) or a flat list of the primitives in :data:`DRIFT_PRIMITIVES`. Inside an
analyzer suite exactly one nested ``[]`` adds the stand-alone Ordinary Kriging candidate. A top-level ``[]`` is
rejected (use ``None``).

* ``ordinary`` overrides ``drift_analyzer`` and ``drift_terms``: both are ignored (the terms are not even
  validated) and a single warning is issued if either was set.
* ``universal`` without analyzer: ``drift_terms`` is ONE multi-drift model and must be given.
  If both ``linear_xy`` and ``quadratic_xy`` are specified, ``linear_xy`` is dropped with a warning (fallback to
  ``['quadratic_xy']`` which contains the linear terms [x, y]).
* ``universal`` with analyzer: ``drift_terms`` ``None`` evaluates the default suite (Ordinary + all drift
  models), a list evaluates all non-empty subsets of the listed primitives (never ``linear_xy`` together with
  ``quadratic_xy``) plus Ordinary if ``[]`` is nested. A suite of exactly one candidate overrides the analyzer
  and that model is applied.
* ``regression``: ``drift_terms`` are the Random Forest covariates (default ``None``: all primitives
  ``["sia", "z_dem", "curvature_dem", "quadratic_xy"]``; spatial coordinates only: ``["linear_xy"]``); the analyzer is skipped.
  The method is only available through PyKrige (also with ``engine`` ``native``, which is reported).
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .errors import ConfigError

DRIFT_PRIMITIVES: tuple[str, ...] = ("sia", "z_dem", "curvature_dem", "linear_xy", "quadratic_xy")
"""Supported primitive drift terms in canonical (display and column) order."""

RASTER_DRIFT_TERMS: tuple[str, ...] = ("sia", "z_dem", "curvature_dem")
"""Primitive drift terms that need a raster covariate grid, in canonical column order."""

DEFAULT_REGRESSION_DRIFT_TERMS: tuple[str, ...] = ("sia", "z_dem", "curvature_dem", "quadratic_xy")
"""Default feature covariates for method 'regression' when drift_terms is None (all primitives with quadratic_xy providing coordinates)."""

FALLBACK_QUADRATIC_NOTICE: str = (
    "drift_terms: ['linear_xy', 'quadratic_xy'] is falling back to ['quadratic_xy'] "
    "which contains the linear terms [x, y]."
)


def prune_redundant_drift_terms(terms: list[str]) -> tuple[list[str], bool]:
    """
    Removes 'linear_xy' from a single model term list if 'quadratic_xy' is also present,
    as 'quadratic_xy' already fits the linear plane [x, y].

    Parameters
    ----------
    terms : list of str
        Primitive drift terms for a single model.

    Returns
    -------
    pruned_terms : list of str
        The terms with 'linear_xy' removed if 'quadratic_xy' is present.
    was_pruned : bool
        True if 'linear_xy' was removed because 'quadratic_xy' was present.
    """
    if "linear_xy" in terms and "quadratic_xy" in terms:
        return [t for t in terms if t != "linear_xy"], True
    return list(terms), False

KRIGING_METHODS: tuple[str, ...] = ("ordinary", "universal", "regression")
KRIGING_ENGINES: tuple[str, ...] = ("native", "pykrige")

PYKRIGE_UNIVERSAL_WARNING = (
    "engine 'pykrige' with method 'universal' can be very slow: PyKrige's UniversalKriging computes a "
    "leave-one-out fit statistic when it is created, which grows roughly with N^3 to N^4 in the number of survey "
    "points (minutes above about 1000 points), whatever the drift terms are; the kriging itself is fast. "
    "Use engine 'native' for Universal Kriging with many survey points."
)

_CANONICAL_RANK = {term: rank for rank, term in enumerate(DRIFT_PRIMITIVES)}

ORDINARY_NAME = "ordinary"
"""Display name of the candidate without any drift term (Ordinary Kriging)."""


def format_drift_name(terms: list[str] | tuple[str, ...]) -> str:
    """Formats drift terms in canonical order (``'sia + z_dem'``); no terms is ``'ordinary'``."""
    if not terms:
        return ORDINARY_NAME
    return " + ".join(sorted(terms, key=lambda t: _CANONICAL_RANK.get(t, len(_CANONICAL_RANK))))


def normalize_method(method: Any) -> str:
    """
    Normalizes a Kriging method name (case-insensitive, ``*_kriging`` spelling accepted).

    Raises
    ------
    ConfigError
        If the method is not ``ordinary``, ``universal`` or ``regression`` (the former method ``sia`` is
        explicitly rejected with a pointer to ``universal`` + ``['sia']``).
    """
    clean = str(method).lower().strip().removesuffix("_kriging")
    if clean == "sia":
        raise ConfigError(
            "Kriging method 'sia' was removed. Use method 'universal' with drift_terms ['sia'] instead."
        )
    if clean not in KRIGING_METHODS:
        raise ConfigError(f"Invalid Kriging method {method!r}. Supported methods: {list(KRIGING_METHODS)}.")
    return clean


def normalize_engine(engine: Any) -> str:
    """Normalizes a Kriging engine name (``native`` or ``pykrige``; the ``*_kriging`` spelling is accepted)."""
    clean = str(engine).lower().strip().removesuffix("_kriging")
    if clean not in KRIGING_ENGINES:
        raise ConfigError(f"Invalid Kriging engine {engine!r}. Supported engines: {list(KRIGING_ENGINES)}.")
    return clean


def parse_drift_terms(drift_terms: Any, allow_ordinary_candidate: bool = False) -> tuple[list[str], bool]:
    """
    Validates a non-null ``drift_terms`` value.

    Parameters
    ----------
    drift_terms : list
        Flat list of primitive names; with ``allow_ordinary_candidate`` additionally exactly one nested ``[]``.
    allow_ordinary_candidate : bool, default False
        Accept one nested ``[]`` (the stand-alone Ordinary candidate of an analyzer suite).

    Returns
    -------
    primitives : list of str
        The listed primitives, deduplicated, in canonical order.
    has_ordinary : bool
        True if the nested ``[]`` was given.

    Raises
    ------
    ConfigError
        For anything that is not a list of known primitives, a top-level ``[]``, a nested ``[]`` where it is not
        allowed (or more than once) and any other nested list.
    """
    if not isinstance(drift_terms, (list, tuple)):
        raise ConfigError(
            f"drift_terms must be null or a list of drift terms from {list(DRIFT_PRIMITIVES)}, got {drift_terms!r}."
        )
    if len(drift_terms) == 0:
        raise ConfigError("drift_terms must not be an empty list [] - use null (omit drift_terms) instead.")

    primitives: list[str] = []
    n_ordinary = 0
    for term in drift_terms:
        if isinstance(term, (list, tuple)):
            if len(term) != 0:
                raise ConfigError(f"drift_terms must not contain nested lists except one empty [], got {list(term)!r}.")
            if not allow_ordinary_candidate:
                raise ConfigError(
                    "A nested [] (stand-alone Ordinary candidate) is only allowed in drift_terms when "
                    "method is 'universal' and drift_analyzer is enabled."
                )
            n_ordinary += 1
            continue
        if term not in _CANONICAL_RANK:
            raise ConfigError(
                f"Unknown drift term {term!r}. Shorthand compound names are no longer supported. "
                f"Please specify an explicit list of primitive drift terms from: {list(DRIFT_PRIMITIVES)}."
            )
        if term not in primitives:
            primitives.append(term)
    if n_ordinary > 1:
        raise ConfigError("drift_terms may contain the nested [] (Ordinary candidate) only once.")
    primitives.sort(key=_CANONICAL_RANK.__getitem__)
    return primitives, n_ordinary == 1


def _suite_from_primitives(primitives: list[str], has_ordinary: bool) -> list[list[str]]:
    suite: list[list[str]] = [[]] if has_ordinary else []
    for size in range(1, len(primitives) + 1):
        for subset in combinations(primitives, size):
            if "linear_xy" in subset and "quadratic_xy" in subset:
                continue  # quadratic_xy already contains the linear columns
            suite.append(list(subset))
    return suite


def build_drift_suite(drift_terms: list | None) -> list[list[str]]:
    """
    Builds the candidate suite of the Drift Analyzer.

    Parameters
    ----------
    drift_terms : list or None
        ``None``: default suite - Ordinary (``[]``) plus all 23 non-empty subsets of the five primitives
        that do not contain ``linear_xy`` and ``quadratic_xy`` together. A list: all such subsets of the listed
        primitives, plus Ordinary if the list contains one nested ``[]``.

    Returns
    -------
    suite : list of list of str
        Candidates ordered Ordinary first, then by size and canonical term order. Ordinary is ``[]``.
    """
    if drift_terms is None:
        return _suite_from_primitives(list(DRIFT_PRIMITIVES), True)
    primitives, has_ordinary = parse_drift_terms(drift_terms, allow_ordinary_candidate=True)
    return _suite_from_primitives(primitives, has_ordinary)


def default_model(
    method: str | None,
    drift_terms: list | None,
    interpolation_target: str,
    direct_target: str,
    drift_analyzer: bool,
) -> tuple[str, list | None]:
    """
    Applies the default Kriging method of a stage; a ``method`` that is given is returned unchanged.

    Only if ``method`` is omitted: a direct target (``T`` before, ``D`` after migration) selects ``universal`` and
    - with ``drift_terms`` omitted and no analyzer - the single drift ``['sia']`` (with the analyzer: the default
    suite); any other target (the product ``P``) selects ``ordinary``.

    Parameters
    ----------
    method : str or None
        Configured method, ``None`` if omitted.
    drift_terms : list or None
        Configured drift terms, ``None`` if omitted.
    interpolation_target : str
        Interpolation target of the stage (``P``, ``T`` or ``D``).
    direct_target : str
        The direct target of the stage (``T`` pre-migration, ``D`` post-migration).
    drift_analyzer : bool
        Whether the drift analyzer is enabled for the stage.

    Returns
    -------
    method, drift_terms : tuple
        The method (``None`` input resolved) and the drift terms to use.
    """
    if method is not None:
        return method, drift_terms
    if str(interpolation_target).upper().strip() != str(direct_target).upper():
        return "ordinary", drift_terms
    if drift_terms is None and not drift_analyzer:
        return "universal", ["sia"]
    return "universal", drift_terms


@dataclass
class DriftPlan:
    """
    Resolved drift configuration of one Kriging stage.

    Attributes
    ----------
    method : str
        ``ordinary``, ``universal`` or ``regression`` that is executed (a single-candidate analyzer suite of
        Ordinary resolves to ``ordinary``).
    analyze : bool
        True if the Drift Analyzer ranks ``suite`` and its winner decides the final model.
    suite : list of list of str
        Analyzer candidates (only if ``analyze``).
    terms : list of str
        Drift terms of the single model to apply (``universal``) or the Random Forest covariates
        (``regression``); empty for ``ordinary`` and for an analyzer run.
    notices : list of (str, str)
        ``(level, message)`` pairs, level ``'info'`` or ``'warning'``, to be logged once by the caller.
    """

    method: str
    analyze: bool = False
    suite: list[list[str]] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        """Short human-readable description for log headers."""
        if self.method == "ordinary":
            return "Ordinary Kriging"
        if self.method == "regression":
            return f"Regression Kriging (covariates: {format_drift_name(self.terms)})"
        if self.analyze:
            return f"Universal Kriging (drift analyzer on {len(self.suite)} candidate models)"
        return f"Universal Kriging (drift: {format_drift_name(self.terms)})"


def resolve_drift_plan(method: Any, drift_analyzer: bool, drift_terms: list | None, engine: str = "native") -> DriftPlan:
    """
    Resolves ``method`` / ``drift_analyzer`` / ``drift_terms`` of one stage according to the module rules.

    Pure function: nothing is logged; the returned :attr:`DriftPlan.notices` are logged by the caller.

    Raises
    ------
    ConfigError
        For an invalid method, engine or drift_terms and for ``universal`` without any drift model to apply.
    """
    method_clean = normalize_method(method)
    engine_clean = normalize_engine(engine)
    analyzer = bool(drift_analyzer)

    if method_clean == "ordinary":
        notices = []
        if analyzer or drift_terms is not None:
            ignored = [n for n, given in (("drift_analyzer", analyzer), ("drift_terms", drift_terms is not None)) if given]
            notices.append((
                "warning",
                f"Kriging method 'ordinary' overrides {' and '.join(repr(n) for n in ignored)}: "
                "the drift analyzer is not run and drift_terms are ignored.",
            ))
        return DriftPlan(method="ordinary", notices=notices)

    if method_clean == "regression":
        notices = []
        if drift_terms is None:
            terms = list(DEFAULT_REGRESSION_DRIFT_TERMS)
        else:
            raw_terms = parse_drift_terms(drift_terms)[0]
            terms, pruned = prune_redundant_drift_terms(raw_terms)
            if pruned:
                notices.append(("warning", FALLBACK_QUADRATIC_NOTICE))
        if engine_clean == "native":
            notices.append((
                "info",
                "Method 'regression' is only available through PyKrige: PyKrige is used for this stage although "
                "the engine is 'native'.",
            ))
        if analyzer:
            notices.append((
                "warning",
                "drift_analyzer evaluates Universal Kriging drift models and is skipped for method 'regression'.",
            ))
        return DriftPlan(method="regression", terms=terms, notices=notices)

    # universal
    slow_universal = [("warning", PYKRIGE_UNIVERSAL_WARNING)] if engine_clean == "pykrige" else []
    if not analyzer:
        if drift_terms is None:
            raise ConfigError(
                "Kriging method 'universal' requires drift_terms (a list of drift terms) or an enabled "
                "drift_analyzer - specify drift_terms or use method 'ordinary'."
            )
        raw_terms, _ = parse_drift_terms(drift_terms)
        terms, pruned = prune_redundant_drift_terms(raw_terms)
        notices = list(slow_universal)
        if pruned:
            notices.append(("warning", FALLBACK_QUADRATIC_NOTICE))
        return DriftPlan(method="universal", terms=terms, notices=notices)

    suite = build_drift_suite(drift_terms)
    if len(suite) == 1:
        only = suite[0]
        return DriftPlan(
            method="universal" if only else "ordinary",
            terms=list(only),
            notices=[(
                "info",
                f"drift_analyzer overridden: drift_terms define exactly one candidate model "
                f"('{format_drift_name(only)}'), which is applied directly.",
            )] + (slow_universal if only else []),
        )
    notices = []
    if engine_clean == "pykrige":
        notices.append((
            "info",
            "The drift analyzer ranks the candidate models with the native engine; the winner is then applied "
            "with the PyKrige engine.",
        ))
        notices.extend(slow_universal)
    return DriftPlan(method="universal", analyze=True, suite=suite, notices=notices)
