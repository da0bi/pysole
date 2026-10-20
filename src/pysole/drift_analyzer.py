"""
Drift Analyzer helper tool for Universal Kriging drift model recommendation.
Analyzes input survey datasets and surface DEM covariates to evaluate, rank,
and recommend optimal single and multi-drift models for Universal Kriging.

All candidates are cross-validated exactly and in one pass: the Ordinary kriging system of the survey is factorised
once and every drift candidate is a low-rank update of it (:class:`pysole.kriging_core.LeaveBlockOut`), so every
survey point is evaluated with the fitted variogram and the regularisation of the production engine.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

from .drift import DRIFT_PRIMITIVES, ORDINARY_NAME, format_drift_name
from .interpolation import DriftBasis
from .kriging_core import KrigingSystem, LeaveBlockOut, max_pairwise_distance, resolve_variogram_params
from .logging import activity_indicator, end_terminal_block, get_progress_bar, logger, log_section_header
from .memory import plan_cv_memory
from .raster import GridGeometry
from .variogram import resolve_native_variogram_model

FAILED_RMSE = 9999.0
"""Placeholder RMSE of a candidate whose cross-validation failed numerically."""

FAILED_AICC = float("inf")
"""Placeholder AICc of a candidate whose cross-validation failed numerically (it is ranked last)."""

SIA_PRODUCT_WARNING = "SIA slope drift on Product P target causes 1/sin^2(alpha) double-scaling margin artifacts."


@dataclass
class CandidateDriftResult:
    """
    Diagnostic evaluation metrics for a candidate Universal Kriging drift model.

    Attributes
    ----------
    name : str
        Candidate model identifier (e.g. 'ordinary', 'sia', 'linear_xy', 'sia + z_dem').
    terms : list of str
        List of individual primitive drift term names included in this model (empty for Ordinary Kriging).
    n_terms : int
        Total number of drift feature columns added to the Kriging system (without the constant).
    vif_max : float
        Maximum Variance Inflation Factor across all drift covariates.
        Values exceeding 5.0 to 10.0 indicate high multicollinearity.
    is_multicollinear : bool
        True if `vif_max` exceeds the configured multicollinearity threshold.
    cv_rmse : float
        Cross-validation Root Mean Squared Error [m or ns].
    cv_mae : float
        Cross-validation Mean Absolute Error [m or ns].
    cv_r2 : float
        Cross-validation Coefficient of Determination :math:`R^2` (dimensionless).
    aicc : float
        Corrected Akaike Information Criterion balancing goodness-of-fit against model complexity.
    is_penalized : bool, default False
        True if the candidate is ranked behind all others because of severe multicollinearity or because its
        cross-validation failed numerically. A warning (see ``warning_msg``) never penalizes a model.
    warning_msg : str or None, optional
        Diagnostic warning note (e.g. SIA slope double-scaling warning on product target); informational only.
    delta_aicc : float, default 0.0
        AICc minus the best AICc among the non-penalized models (NaN for failed candidates).
    delta_rmse_pct : float, default 0.0
        Relative RMSE difference to the best-ranked model in percent (NaN for failed candidates).
    """
    name: str
    terms: list[str]
    n_terms: int
    vif_max: float
    is_multicollinear: bool
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    aicc: float
    is_penalized: bool = False
    warning_msg: str | None = None
    delta_aicc: float = 0.0
    delta_rmse_pct: float = 0.0
    rank: int = 0


def farthest_point_indices(x: np.ndarray, y: np.ndarray, n_select: int) -> np.ndarray:
    """
    Deterministic farthest-point subset (max-min distance) of ``n_select`` points, starting at the point closest to
    the centroid; returns sorted indices. Used for the large-N fallback of the cross-validation.
    """
    n = len(x)
    if n_select >= n:
        return np.arange(n)
    xy = np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    first = int(np.argmin(np.hypot(*(xy - xy.mean(axis=0)).T)))
    chosen = [first]
    nearest = np.hypot(xy[:, 0] - xy[first, 0], xy[:, 1] - xy[first, 1])
    for _ in range(n_select - 1):
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, np.hypot(xy[:, 0] - xy[nxt, 0], xy[:, 1] - xy[nxt, 1]))
    return np.sort(np.array(chosen, dtype=np.intp))


def add_ranking_deltas(results: list[CandidateDriftResult]) -> None:
    """
    Fills ``delta_aicc`` and ``delta_rmse_pct`` of an already ranked result list (in place).

    ``delta_aicc`` is the AICc minus the best AICc among the non-penalized models (among all models if every
    model is penalized); ``delta_rmse_pct`` is ``100 * (RMSE / RMSE of the best-ranked model - 1)``. Failed
    candidates get NaN.
    """
    if not results:
        return
    usable = [r for r in results if np.isfinite(r.aicc) and r.aicc < FAILED_AICC]
    pool = [r for r in usable if not r.is_penalized] or usable
    best_aicc = min((r.aicc for r in pool), default=float("nan"))
    best_rmse = results[0].cv_rmse if np.isfinite(results[0].aicc) and results[0].aicc < FAILED_AICC else float("nan")
    for r in results:
        if not np.isfinite(r.aicc) or r.aicc >= FAILED_AICC:
            r.delta_aicc = r.delta_rmse_pct = float("nan")
            continue
        r.delta_aicc = float(r.aicc - best_aicc)
        r.delta_rmse_pct = float(100.0 * (r.cv_rmse / best_rmse - 1.0)) if best_rmse and best_rmse > 0 else 0.0


def resolve_rmse_unit(
    mode: str = "pass_1",
    interpolation_target: str = "P",
    survey_data_type: str | None = None,
    velocity: float | None = None,
) -> str:
    """
    Resolves the user-facing RMSE unit string for Drift Analyzer reporting.

    Parameters
    ----------
    mode : str
        Stage name (e.g. 'pass_1', 'pre_migration', 'pass_2', 'post_migration').
    interpolation_target : str
        Target variable ('P', 'T', or 'D').
    survey_data_type : str, optional
        Canonical survey data type ('one_way_traveltime', 'two_way_traveltime', 'depth').
    velocity : float, optional
        Effective propagation velocity (e.g. 0.160 m/ns for GPR radar, > 100 m/s for seismics).

    Returns
    -------
    unit_str : str
        Descriptive physical unit string (e.g. 'm', 'm x rad', 'ns', 'ns x rad', 'ms', etc.).
    """
    target = str(interpolation_target or "P").upper().strip()
    dtype = str(survey_data_type or "one_way_traveltime").lower().strip()
    m = str(mode).lower().replace("-", "_").replace(" ", "_")
    is_pre = "pre" in m or "pass_1" in m or "pass1" in m or m == "1"

    # If post-migration, or if pre-migration input is already direct depth
    if not is_pre or dtype == "depth":
        return "m x rad" if target == "P" else "m"

    # Pre-migration with traveltime data:
    # Option C: infer time unit from velocity if available
    if velocity is not None and velocity > 0:
        if velocity <= 1.0:  # GPR radar: velocity in m/ns (typically 0.15 - 0.18 m/ns)
            time_unit = "ns"
        elif velocity >= 100.0:  # Reflection seismics: velocity in m/s (typically 3000 - 4000 m/s)
            time_unit = "ms" if velocity < 10000.0 else "s"
        elif 1.0 < velocity < 100.0:  # e.g. velocity in m/ms (typically 3.0 - 4.0 m/ms)
            time_unit = "ms"
        else:
            time_unit = "traveltime units"
    else:
        # Default radar traveltime in glaciology is nanoseconds (ns)
        time_unit = "ns"

    return f"{time_unit} x rad" if target == "P" else time_unit


def format_ranking_table(
    results: list[CandidateDriftResult],
    title: str,
    rmse_unit: str = "m",
) -> str:
    """Formats the ranked candidate list (RMSE, dRMSE %, AICc, dAICc, VIF, status) as console text."""
    rmse_hdr = f"RMSE ({rmse_unit})"
    rmse_col_w = max(10, len(rmse_hdr) + 2)
    width = max(112, 102 + rmse_col_w)
    lines = ["=" * width, f"       {title}", "=" * width]
    lines.append(
        f"{'#':<3} {'Drift Model':<44} {rmse_hdr:<{rmse_col_w}} {'dRMSE %':<9} {'AICc':<10} {'dAICc':<9} {'VIF':<8} {'Status'}"
    )
    lines.append("-" * width)

    def _fmt(value: float, spec: str) -> str:
        return "n/a" if not np.isfinite(value) else format(value, spec)

    for idx, res in enumerate(results, 1):
        status = "PENALIZED" if res.is_penalized else ("WARNING" if res.warning_msg else "VALID")
        rmse_val_str = _fmt(res.cv_rmse, f"<{rmse_col_w - 2}.2f") if np.isfinite(res.cv_rmse) and res.cv_rmse < FAILED_RMSE else "n/a"
        aicc_val_str = _fmt(res.aicc, "<10.1f") if np.isfinite(res.aicc) and res.aicc < FAILED_AICC else "n/a"
        lines.append(
            f"{idx:<3} {res.name:<44} {rmse_val_str:<{rmse_col_w}} "
            f"{_fmt(res.delta_rmse_pct, '<9.2f'):<9} {aicc_val_str:<10} "
            f"{_fmt(res.delta_aicc, '<9.1f'):<9} {res.vif_max:<8.1f} {status}"
        )
    lines.append("-" * width)
    for note in sorted({r.warning_msg for r in results if r.warning_msg}):
        lines.append(f"WARNING: {note}")
    return "\n".join(lines)


@dataclass
class DriftComparisonRecord:
    """Evaluation and distribution summary record for one candidate drift model."""

    pass_label: str
    rank: int
    model_name: str
    model_slug: str
    terms: list[str]
    cv_rmse: float
    delta_rmse_pct: float | None
    aicc: float
    delta_aicc: float | None
    vif_max: float
    status: str
    rmse_unit: str = "m"
    target_mean: float = 0.0
    target_median: float = 0.0
    target_std: float = 0.0
    target_max: float = 0.0
    target_unit: str = "ns"
    thick_mean: float = 0.0
    thick_median: float = 0.0
    thick_std: float = 0.0
    thick_max: float = 0.0
    bss_mean: float = 0.0
    bss_median: float = 0.0
    bss_std: float = 0.0
    bss_max: float = 0.0
    unc_mean: float | None = None
    is_failed: bool = False
    failure_reason: str | None = None


def parse_model_selection(
    text: str, results: list[CandidateDriftResult]
) -> tuple[list[CandidateDriftResult] | None, str | None]:
    """
    Parses user input from interactive drift model selection.

    Accepted:
    - Empty input or '#' -> [results[0]] (recommended model)
    - Single row number ('3' or '#3') -> [results[2]]
    - Range of rows ('1-5' or '#1-#5') -> results[0:5]
    - Comma-separated rows and ranges ('1,2,5-8')
    - Model name as printed ('sia + z_dem'), or primitive terms
    - 'ordinary' or '[]' for Ordinary Kriging

    Parameters
    ----------
    text : str
        The user input.
    results : list of CandidateDriftResult
        The ranked candidates that were evaluated.

    Returns
    -------
    chosen : list of CandidateDriftResult or None
        The selected candidates (at least one), or None if invalid.
    error : str or None
        Error message if input was rejected, or None if valid.
    """
    clean = text.strip()
    if not clean or clean == "#":
        if not results:
            return None, "No evaluated candidates available."
        return [results[0]], None

    cand_chars = set("0123456789,- #")
    if all(c in cand_chars for c in clean):
        segments = [s.strip() for s in clean.split(",") if s.strip()]
        indices: list[int] = []
        for seg in segments:
            seg_clean = seg.lstrip("#").strip()
            if "-" in seg_clean:
                parts = seg_clean.split("-", 1)
                p0, p1 = parts[0].strip().lstrip("#").strip(), parts[1].strip().lstrip("#").strip()
                if not (p0.isdigit() and p1.isdigit()):
                    return None, f"Invalid range '{seg}'; expected format 'start-end' (e.g. '1-5')."
                start, end = int(p0), int(p1)
                if start < 1 or end > len(results):
                    return None, f"Range '{seg}' out of bounds; choose row numbers from 1 to {len(results)}."
                if start > end:
                    return None, f"Invalid range '{seg}': start row ({start}) must be <= end row ({end})."
                indices.extend(range(start - 1, end))
            elif seg_clean.isdigit():
                val = int(seg_clean)
                if 1 <= val <= len(results):
                    indices.append(val - 1)
                else:
                    return None, f"Row {seg_clean} does not exist; choose a number from 1 to {len(results)}."
            else:
                return None, f"Invalid row selector '{seg}'."
        if indices:
            unique_indices = list(dict.fromkeys(indices))
            return [results[i] for i in unique_indices], None

    stripped = clean.strip("[]()").replace("'", "").replace('"', "")
    tokens = [t.strip() for t in stripped.replace("+", ",").split(",") if t.strip()]
    if not tokens or [t.lower() for t in tokens] == [ORDINARY_NAME]:
        wanted: set[str] = set()
    else:
        unknown = [t for t in tokens if t not in DRIFT_PRIMITIVES]
        if unknown:
            return None, (
                f"Unknown drift term(s) {unknown}; choose row numbers (e.g. '1-5', '1,2,5-8'), "
                f"'ordinary', '[]' or terms from {list(DRIFT_PRIMITIVES)}."
            )
        wanted = set(tokens)
    for res in results:
        if set(res.terms) == wanted:
            return [res], None
    return None, f"The drift model '{format_drift_name(sorted(wanted)) if wanted else ORDINARY_NAME}' was not evaluated."


def format_drift_comparison_table(
    records: list[DriftComparisonRecord],
    stage_label: str = "pass2",
) -> str:
    """Formats comparison table of evaluated drift models sorted by canonical Drift Analyzer rank."""
    sorted_records = sorted(records, key=lambda r: r.rank)
    stage_norm = stage_label.lower().replace(" ", "").replace("_", "")
    is_p1 = "pass1" in stage_norm or "pre" in stage_norm or stage_norm == "1"
    rmse_unit = sorted_records[0].rmse_unit if sorted_records else ("ns" if is_p1 else "m")
    target_unit = sorted_records[0].target_unit if sorted_records else "ns"
    has_unc = any(r.unc_mean is not None for r in sorted_records)

    header_title = (
        "DRIFT MODEL INTERPOLATION COMPARISON TABLE (PASS 1)"
        if is_p1
        else "DRIFT MODEL INTERPOLATION COMPARISON TABLE (PASS 2)"
    )

    if is_p1:
        if has_unc:
            width = 133
            lines = [
                "=" * width,
                f"{header_title:^{width}}",
                "=" * width,
                f"{'#':<3} {'Drift Model':<32} {f'RMSE ({rmse_unit})':<11} {'dRMSE %':<9} {'AICc':<10} {'dAICc':<8} {'VIF':<6} {'Status':<10} | {f'Traveltime [{target_unit}]':<41}",
                f"{'':<82} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8} {'Mean Unc':<8}",
                "-" * width,
            ]
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    t_stats = f"{'FAILED':<41}"
                else:
                    unc_str = f"{r.unc_mean:<8.2f}" if r.unc_mean is not None else f"{'n/a':<8}"
                    t_stats = f"{r.target_mean:<8.2f} {r.target_median:<8.2f} {r.target_std:<8.2f} {r.target_max:<8.2f} {unc_str}"
                lines.append(
                    f"{r.rank:<3} {r.model_name:<32} {r.cv_rmse:<11.2f} {d_rmse:<9} {aicc_str:<10} {d_aicc:<8} {r.vif_max:<6.1f} {r.status:<10} | {t_stats}"
                )
            lines.append("-" * width)
        else:
            width = 124
            lines = [
                "=" * width,
                f"{header_title:^{width}}",
                "=" * width,
                f"{'#':<3} {'Drift Model':<32} {f'RMSE ({rmse_unit})':<11} {'dRMSE %':<9} {'AICc':<10} {'dAICc':<8} {'VIF':<6} {'Status':<10} | {f'Traveltime [{target_unit}]':<32}",
                f"{'':<82} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8}",
                "-" * width,
            ]
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    t_stats = f"{'FAILED':<32}"
                else:
                    t_stats = f"{r.target_mean:<8.2f} {r.target_median:<8.2f} {r.target_std:<8.2f} {r.target_max:<8.2f}"
                lines.append(
                    f"{r.rank:<3} {r.model_name:<32} {r.cv_rmse:<11.2f} {d_rmse:<9} {aicc_str:<10} {d_aicc:<8} {r.vif_max:<6.1f} {r.status:<10} | {t_stats}"
                )
            lines.append("-" * width)
    else:
        if has_unc:
            width = 165
            lines = [
                "=" * width,
                f"{header_title:^{width}}",
                "=" * width,
                f"{'#':<3} {'Drift Model':<32} {f'RMSE ({rmse_unit})':<11} {'dRMSE %':<9} {'AICc':<10} {'dAICc':<8} {'VIF':<6} {'Status':<10} | {'Ice Thickness [m]':<41} | {'Basal Shear Stress [kPa]':<32}",
                f"{'':<82} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8} {'Mean Unc':<8} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8}",
                "-" * width,
            ]
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    th_stats = f"{'FAILED':<41}"
                    bss_stats = f"{'FAILED':<32}"
                else:
                    unc_str = f"{r.unc_mean:<8.2f}" if r.unc_mean is not None else f"{'n/a':<8}"
                    th_stats = f"{r.thick_mean:<8.2f} {r.thick_median:<8.2f} {r.thick_std:<8.2f} {r.thick_max:<8.2f} {unc_str}"
                    bss_stats = f"{r.bss_mean:<8.2f} {r.bss_median:<8.2f} {r.bss_std:<8.2f} {r.bss_max:<8.2f}"
                lines.append(
                    f"{r.rank:<3} {r.model_name:<32} {r.cv_rmse:<11.2f} {d_rmse:<9} {aicc_str:<10} {d_aicc:<8} {r.vif_max:<6.1f} {r.status:<10} | {th_stats} | {bss_stats}"
                )
            lines.append("-" * width)
        else:
            width = 156
            lines = [
                "=" * width,
                f"{header_title:^{width}}",
                "=" * width,
                f"{'#':<3} {'Drift Model':<32} {f'RMSE ({rmse_unit})':<11} {'dRMSE %':<9} {'AICc':<10} {'dAICc':<8} {'VIF':<6} {'Status':<10} | {'Ice Thickness [m]':<32} | {'Basal Shear Stress [kPa]':<32}",
                f"{'':<82} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8} | {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8}",
                "-" * width,
            ]
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    th_stats = f"{'FAILED':<32}"
                    bss_stats = f"{'FAILED':<32}"
                else:
                    th_stats = f"{r.thick_mean:<8.2f} {r.thick_median:<8.2f} {r.thick_std:<8.2f} {r.thick_max:<8.2f}"
                    bss_stats = f"{r.bss_mean:<8.2f} {r.bss_median:<8.2f} {r.bss_std:<8.2f} {r.bss_max:<8.2f}"
                lines.append(
                    f"{r.rank:<3} {r.model_name:<32} {r.cv_rmse:<11.2f} {d_rmse:<9} {aicc_str:<10} {d_aicc:<8} {r.vif_max:<6.1f} {r.status:<10} | {th_stats} | {bss_stats}"
                )
            lines.append("-" * width)

    return "\n".join(lines)


def save_drift_comparison_files(
    records: list[DriftComparisonRecord],
    plots_dir: str | Path,
    stage_label: str = "pass2",
) -> list[Path]:
    """Saves comparison tables in .txt, .csv, and .md formats into plots_dir."""
    p_dir = Path(plots_dir).expanduser().resolve()
    stage_norm = stage_label.lower().replace(" ", "").replace("_", "")
    is_p1 = "pass1" in stage_norm or "pre" in stage_norm or stage_norm == "1"
    base_name = "drift_model_comparison_pass1" if is_p1 else "drift_model_comparison_pass2"
    sorted_records = sorted(records, key=lambda r: r.rank)
    has_unc = any(r.unc_mean is not None for r in sorted_records)

    # 1. Plaintext ASCII table (.txt)
    txt_path = p_dir / f"{base_name}.txt"
    txt_content = format_drift_comparison_table(sorted_records, stage_label)
    txt_path.write_text(txt_content + "\n", encoding="utf-8")

    # 2. Machine-readable CSV (.csv)
    csv_path = p_dir / f"{base_name}.csv"
    csv_lines = []
    if is_p1:
        if has_unc:
            csv_lines.append("rank,drift_model,slug,rmse,delta_rmse_pct,aicc,delta_aicc,vif,status,traveltime_mean,traveltime_median,traveltime_std,traveltime_max,traveltime_unc_mean,unit,is_failed")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:.2f}" if r.delta_rmse_pct is not None else ""
                d_aicc = f"{r.delta_aicc:.2f}" if r.delta_aicc is not None else ""
                aicc_val = f"{r.aicc:.2f}" if np.isfinite(r.aicc) else ""
                unc_val = f"{r.unc_mean:.4f}" if r.unc_mean is not None else ""
                csv_lines.append(
                    f'{r.rank},"{r.model_name}",{r.model_slug},{r.cv_rmse:.4f},{d_rmse},{aicc_val},{d_aicc},{r.vif_max:.2f},{r.status},'
                    f'{r.target_mean:.4f},{r.target_median:.4f},{r.target_std:.4f},{r.target_max:.4f},{unc_val},{r.target_unit},{r.is_failed}'
                )
        else:
            csv_lines.append("rank,drift_model,slug,rmse,delta_rmse_pct,aicc,delta_aicc,vif,status,traveltime_mean,traveltime_median,traveltime_std,traveltime_max,unit,is_failed")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:.2f}" if r.delta_rmse_pct is not None else ""
                d_aicc = f"{r.delta_aicc:.2f}" if r.delta_aicc is not None else ""
                aicc_val = f"{r.aicc:.2f}" if np.isfinite(r.aicc) else ""
                csv_lines.append(
                    f'{r.rank},"{r.model_name}",{r.model_slug},{r.cv_rmse:.4f},{d_rmse},{aicc_val},{d_aicc},{r.vif_max:.2f},{r.status},'
                    f'{r.target_mean:.4f},{r.target_median:.4f},{r.target_std:.4f},{r.target_max:.4f},{r.target_unit},{r.is_failed}'
                )
    else:
        if has_unc:
            csv_lines.append("rank,drift_model,slug,rmse,delta_rmse_pct,aicc,delta_aicc,vif,status,thickness_mean_m,thickness_median_m,thickness_std_m,thickness_max_m,thickness_unc_mean_m,bss_mean_kpa,bss_median_kpa,bss_std_kpa,bss_max_kpa,is_failed")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:.2f}" if r.delta_rmse_pct is not None else ""
                d_aicc = f"{r.delta_aicc:.2f}" if r.delta_aicc is not None else ""
                aicc_val = f"{r.aicc:.2f}" if np.isfinite(r.aicc) else ""
                unc_val = f"{r.unc_mean:.4f}" if r.unc_mean is not None else ""
                csv_lines.append(
                    f'{r.rank},"{r.model_name}",{r.model_slug},{r.cv_rmse:.4f},{d_rmse},{aicc_val},{d_aicc},{r.vif_max:.2f},{r.status},'
                    f'{r.thick_mean:.4f},{r.thick_median:.4f},{r.thick_std:.4f},{r.thick_max:.4f},{unc_val},'
                    f'{r.bss_mean:.4f},{r.bss_median:.4f},{r.bss_std:.4f},{r.bss_max:.4f},{r.is_failed}'
                )
        else:
            csv_lines.append("rank,drift_model,slug,rmse,delta_rmse_pct,aicc,delta_aicc,vif,status,thickness_mean_m,thickness_median_m,thickness_std_m,thickness_max_m,bss_mean_kpa,bss_median_kpa,bss_std_kpa,bss_max_kpa,is_failed")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:.2f}" if r.delta_rmse_pct is not None else ""
                d_aicc = f"{r.delta_aicc:.2f}" if r.delta_aicc is not None else ""
                aicc_val = f"{r.aicc:.2f}" if np.isfinite(r.aicc) else ""
                csv_lines.append(
                    f'{r.rank},"{r.model_name}",{r.model_slug},{r.cv_rmse:.4f},{d_rmse},{aicc_val},{d_aicc},{r.vif_max:.2f},{r.status},'
                    f'{r.thick_mean:.4f},{r.thick_median:.4f},{r.thick_std:.4f},{r.thick_max:.4f},'
                    f'{r.bss_mean:.4f},{r.bss_median:.4f},{r.bss_std:.4f},{r.bss_max:.4f},{r.is_failed}'
                )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # 3. Markdown table (.md)
    md_path = p_dir / f"{base_name}.md"
    md_lines = []
    if is_p1:
        md_lines.append("# Drift Model Interpolation Comparison Table (Pass 1)")
        md_lines.append("")
        if has_unc:
            md_lines.append("| # | Drift Model | RMSE | ΔRMSE % | AICc | ΔAICc | VIF | Status | Mean Traveltime | Median Traveltime | Std Dev | Max Traveltime | Mean Uncertainty |")
            md_lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    stats = "FAILED | FAILED | FAILED | FAILED | FAILED"
                else:
                    unc_md = f"{r.unc_mean:.2f} {r.target_unit}" if r.unc_mean is not None else "n/a"
                    stats = f"{r.target_mean:.2f} {r.target_unit} | {r.target_median:.2f} {r.target_unit} | {r.target_std:.2f} {r.target_unit} | {r.target_max:.2f} {r.target_unit} | {unc_md}"
                md_lines.append(f"| {r.rank} | `{r.model_name}` | {r.cv_rmse:.2f} {r.rmse_unit} | {d_rmse} | {aicc_str} | {d_aicc} | {r.vif_max:.1f} | {r.status} | {stats} |")
        else:
            md_lines.append("| # | Drift Model | RMSE | ΔRMSE % | AICc | ΔAICc | VIF | Status | Mean Traveltime | Median Traveltime | Std Dev | Max Traveltime |")
            md_lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    stats = "FAILED | FAILED | FAILED | FAILED"
                else:
                    stats = f"{r.target_mean:.2f} {r.target_unit} | {r.target_median:.2f} {r.target_unit} | {r.target_std:.2f} {r.target_unit} | {r.target_max:.2f} {r.target_unit}"
                md_lines.append(f"| {r.rank} | `{r.model_name}` | {r.cv_rmse:.2f} {r.rmse_unit} | {d_rmse} | {aicc_str} | {d_aicc} | {r.vif_max:.1f} | {r.status} | {stats} |")
    else:
        md_lines.append("# Drift Model Interpolation Comparison Table (Pass 2)")
        md_lines.append("")
        if has_unc:
            md_lines.append("| # | Drift Model | RMSE | ΔRMSE % | AICc | ΔAICc | VIF | Status | Mean Thick (m) | Median Thick (m) | Max Thick (m) | Mean Unc (m) | Mean BSS (kPa) | Median BSS (kPa) | Max BSS (kPa) |")
            md_lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    stats = "FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED"
                else:
                    unc_md = f"{r.unc_mean:.2f}" if r.unc_mean is not None else "n/a"
                    stats = f"{r.thick_mean:.2f} | {r.thick_median:.2f} | {r.thick_max:.2f} | {unc_md} | {r.bss_mean:.2f} | {r.bss_median:.2f} | {r.bss_max:.2f}"
                md_lines.append(f"| {r.rank} | `{r.model_name}` | {r.cv_rmse:.2f} {r.rmse_unit} | {d_rmse} | {aicc_str} | {d_aicc} | {r.vif_max:.1f} | {r.status} | {stats} |")
        else:
            md_lines.append("| # | Drift Model | RMSE | ΔRMSE % | AICc | ΔAICc | VIF | Status | Mean Thick (m) | Median Thick (m) | Max Thick (m) | Mean BSS (kPa) | Median BSS (kPa) | Max BSS (kPa) |")
            md_lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in sorted_records:
                d_rmse = f"{r.delta_rmse_pct:+.1f}%" if r.delta_rmse_pct is not None else "n/a"
                d_aicc = f"{r.delta_aicc:+.1f}" if r.delta_aicc is not None else "n/a"
                aicc_str = f"{r.aicc:.1f}" if np.isfinite(r.aicc) else "inf"
                if r.is_failed:
                    stats = "FAILED | FAILED | FAILED | FAILED | FAILED | FAILED"
                else:
                    stats = f"{r.thick_mean:.2f} | {r.thick_median:.2f} | {r.thick_max:.2f} | {r.bss_mean:.2f} | {r.bss_median:.2f} | {r.bss_max:.2f}"
                md_lines.append(f"| {r.rank} | `{r.model_name}` | {r.cv_rmse:.2f} {r.rmse_unit} | {d_rmse} | {aicc_str} | {d_aicc} | {r.vif_max:.1f} | {r.status} | {stats} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    logger.info(f"   Saved comparison tables to: {txt_path.name}, {csv_path.name}, {md_path.name}")
    return [txt_path, csv_path, md_path]


class DriftAnalyzer:
    """
    Automated diagnostic recommendation engine for Universal Kriging drift models.

    Assists glaciologists and geophysicists in selecting the physically most sound and
    statistically robust drift model for Universal Kriging. It evaluates:
    1. **Multicollinearity via Variance Inflation Factors (VIF)**: Identifies redundant
       covariates that cause ill-conditioned Kriging covariance matrices.
    2. **Spatial Cross-Validation (LOPO-CV & Buffer LOOCV)**: Computes out-of-sample
       prediction accuracy (RMSE, MAE, R²) by leaving out whole profiles or radius neighbourhoods, exactly
       and for every survey point.
    3. **Model Parsimony via AICc**: Balances reconstruction fidelity against parameter
       count to prevent overfitting to sparse radar profiles.

    Parameters
    ----------
    mode : str, default "pre_migration"
        Processing stage: 'pre_migration' (traveltimes) or 'post_migration' (bedrock depths).
    target_name : str, default "traveltime"
        Physical target quantity name ('traveltime' or 'depth').
    interpolation_target : str, default "P"
        Interpolation target variable: 'P' (product :math:`P = D \\cdot \\sin\\alpha` or :math:`T \\cdot \\sin\\alpha`)
        or direct measurement ('T' or 'D').
    include_zero_boundary : bool, default True
        Whether zero-value boundary constraints on the glacier outline are active.
    survey_profile_column : str, optional
        CSV column name identifying individual survey profiles for Leave-One-Profile-Out CV.
    vif_threshold : float, default 10.0
        Variance Inflation Factor threshold above which models are flagged as multicollinear.
    slope_floor_deg : float, default 5.0
        Minimum slope angle in degrees [°] for stabilizing the SIA drift denominator.
    max_memory_fraction : float, default 0.5
        Fraction of the available memory the cross-validation may use (see :func:`pysole.memory.plan_cv_memory`).
    show_progress : bool, default True
        Show an elapsed-time spinner while the base kriging system is factorised and a progress bar over the
        leave-block-out evaluation. The display only reports; it never changes or slows the calculation.
    """

    def __init__(
        self,
        mode: str = "pre_migration",
        target_name: str = "traveltime",
        interpolation_target: str = "P",
        include_zero_boundary: bool = True,
        survey_profile_column: str | None = None,
        vif_threshold: float = 10.0,
        slope_floor_deg: float = 5.0,
        max_memory_fraction: float = 0.5,
        show_progress: bool = True,
        survey_data_type: str = "one_way_traveltime",
        velocity: float | None = None,
    ):
        self.mode = mode
        self.target_name = target_name
        self.interpolation_target = interpolation_target
        self.include_zero_boundary = include_zero_boundary
        self.survey_profile_column = survey_profile_column
        self.vif_threshold = vif_threshold
        self.slope_floor_deg = slope_floor_deg
        self.max_memory_fraction = max_memory_fraction
        self.show_progress = show_progress
        self.survey_data_type = survey_data_type
        self.velocity = velocity
        self.selected_models: list[CandidateDriftResult] = []

    @property
    def rmse_unit(self) -> str:
        """Returns the physical unit string for RMSE reporting."""
        return resolve_rmse_unit(
            mode=self.mode,
            interpolation_target=self.interpolation_target,
            survey_data_type=self.survey_data_type,
            velocity=self.velocity,
        )

    @property
    def pass_label(self) -> str:
        """Returns clean user-facing pass label ('Pass 1' or 'Pass 2') derived from mode."""
        m = str(self.mode).lower().replace("-", "_").replace(" ", "_")
        if "pre" in m or "pass_1" in m or "pass1" in m or m == "1":
            return "Pass 1"
        if "post" in m or "pass_2" in m or "pass2" in m or m == "2":
            return "Pass 2"
        return self.mode.replace("_", " ").title()

    def calculate_vif(self, X_drift: np.ndarray) -> tuple[float, list[float]]:
        """
        Calculates Variance Inflation Factor (VIF) for each column in a drift basis matrix.

        The Variance Inflation Factor for covariate :math:`i` is:
        .. math::
            \\text{VIF}_i = \\frac{1}{1 - R_i^2}

        where :math:`R_i^2` is the coefficient of determination from regressing column :math:`i`
        against all remaining drift columns. A VIF exceeding 5 to 10 indicates that the
        covariate is largely collinear with other terms, which can cause Kriging systems
        to become singular or produce wild extrapolation artifacts.

        Parameters
        ----------
        X_drift : np.ndarray
            Evaluated drift covariate matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        max_vif : float
            Maximum VIF value across all evaluated covariates.
        vif_list : list of float
            Individual VIF values for each covariate column.
        """
        n_samples, n_features = X_drift.shape
        if n_features <= 1:
            return 1.0, [1.0]

        vifs: list[float] = []
        for i in range(n_features):
            y_i = X_drift[:, i]
            if np.std(y_i) <= 1e-12:
                vifs.append(1.0)
                continue
            X_others = np.delete(X_drift, i, axis=1)
            A = np.column_stack([np.ones(n_samples), X_others])
            try:
                coef = np.linalg.lstsq(A, y_i, rcond=None)[0]
                y_pred = A @ coef
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                if ss_tot <= 1e-12:
                    r_sq = 0.0
                else:
                    ss_res = np.sum((y_i - y_pred) ** 2)
                    r_sq = max(0.0, 1.0 - (ss_res / ss_tot))
            except Exception:
                r_sq = 0.0

            if r_sq >= 0.9999:
                vif = 999.0
            else:
                vif = 1.0 / (1.0 - r_sq)
            vifs.append(float(vif))

        return float(np.max(vifs)), vifs

    def _plan_blocks(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        cv_mode: str,
        buffer_radius_m: float,
        profile_ids: np.ndarray | None,
    ):
        """
        Builds the removed blocks and the evaluated targets of the cross-validation and the memory plan.

        Returns ``(blocks, targets, plan)``: one block per profile (``lopo``; all members are targets) or per
        evaluated point (``buffer``; the block is every point within ``buffer_radius_m``, the target is the point).
        If the memory plan asks for a subset, a deterministic farthest-point subset of points / profiles is used.
        """
        n_points = len(x_pts)
        xy = np.column_stack((x_pts, y_pts))
        if cv_mode == "lopo" and profile_ids is not None:
            groups = [np.flatnonzero(profile_ids == p) for p in np.unique(profile_ids)]
            mean_block = n_points / len(groups)
            plan = plan_cv_memory(n_points, mean_block, self.max_memory_fraction)
            if plan.n_eval is not None:
                keep = max(3, int(round(plan.n_eval / mean_block)))
                centroids = np.array([xy[g].mean(axis=0) for g in groups])
                groups = [groups[i] for i in farthest_point_indices(centroids[:, 0], centroids[:, 1], keep)]
            return groups, groups, plan

        tree = cKDTree(xy)
        mean_block = float(np.mean(tree.query_ball_point(xy, buffer_radius_m, return_length=True)))
        plan = plan_cv_memory(n_points, mean_block, self.max_memory_fraction)
        evaluated = np.arange(n_points) if plan.n_eval is None else farthest_point_indices(x_pts, y_pts, plan.n_eval)
        blocks = [np.asarray(b, dtype=np.intp) for b in tree.query_ball_point(xy[evaluated], buffer_radius_m, return_sorted=True)]
        return blocks, [np.array([i], dtype=np.intp) for i in evaluated], plan

    def evaluate_spatial_cv(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        z_values: np.ndarray,
        geometry: GridGeometry,
        drift_grids: dict[str, np.ndarray],
        candidates: list[list[str]],
        variogram_model: str,
        variogram_params: dict[str, float] | None,
        cv_mode: str = "buffer",
        buffer_radius_m: float = 50.0,
        profile_ids: np.ndarray | None = None,
    ) -> list[CandidateDriftResult]:
        """
        Executes exact Spatial Cross-Validation across the candidate drift models.

        Supports two spatial validation strategies:
        1. **Leave-One-Profile-Out (LOPO-CV)**: Systematically leaves out entire radar
           profiles to measure how well the drift model predicts unprobed glacier sectors.
        2. **Spatial Buffer LOOCV**: Leaves out individual points plus all neighbors
           within a spatial buffer radius :math:`r_{\\text{buffer}} \\approx a_0 / 2`
           (half the variogram range) to eliminate spatial autocorrelation leakage.

        The residuals of all candidates come from one factorisation of the Ordinary kriging system and low-rank
        updates for the drift columns (:class:`pysole.kriging_core.LeaveBlockOut`); they equal the residuals of a
        fresh kriging fit per fold up to rounding error. Every survey point is evaluated (n = number of points)
        unless the memory plan (:func:`pysole.memory.plan_cv_memory`) selects a farthest-point subset for very
        large surveys. The AICc of all candidates is computed on the same set of points.

        Parameters
        ----------
        x_pts : np.ndarray
            X coordinates of survey observations [m].
        y_pts : np.ndarray
            Y coordinates of survey observations [m].
        z_values : np.ndarray
            Target values (traveltimes [ns] or depths [m]).
        geometry : GridGeometry
            Grid definition shared with the production run.
        drift_grids : dict of str to np.ndarray
            Raster covariates ``sia`` / ``z_dem`` / ``curvature_dem`` of the production run (see
            :func:`pysole.interpolation.build_drift_grids`); the analyzer samples them exactly like the
            production engine.
        candidates : list of list of str
            Candidate drift models as lists of primitive drift terms; the empty list is Ordinary Kriging (see
            :func:`pysole.drift.build_drift_suite`).
        variogram_model : str
            Variogram model name (e.g. 'spherical').
        variogram_params : dict of str to float, optional
            Variogram parameters ('range', 'sill', 'nugget') of the production run; missing entries (or ``None``)
            fall back to the engine defaults (:func:`pysole.kriging_core.resolve_variogram_params`).
        cv_mode : str, default "buffer"
            Cross-validation mode: 'lopo' or 'buffer'.
        buffer_radius_m : float, default 50.0
            Spatial exclusion buffer radius [m] for buffer LOOCV.
        profile_ids : np.ndarray, optional
            Profile ID labels for each survey point.

        Returns
        -------
        ranked_candidates : list of CandidateDriftResult
            Sorted list of candidate drift models, ranked from best to worst (not penalized first, then by AICc,
            then by RMSE), including the ΔAICc / ΔRMSE % columns.
        """
        x_pts = np.asarray(x_pts, dtype=np.float64)
        y_pts = np.asarray(y_pts, dtype=np.float64)
        z_values = np.asarray(z_values, dtype=np.float64)
        n_points = len(z_values)
        if not candidates or n_points == 0:
            return []

        model = resolve_native_variogram_model(variogram_model, warn=False)
        params = resolve_variogram_params(
            variogram_params, z_values, max_pairwise_distance(np.column_stack((x_pts, y_pts)))
        )
        if profile_ids is not None:
            profile_ids = np.asarray(profile_ids)

        blocks, targets, plan = self._plan_blocks(x_pts, y_pts, cv_mode, buffer_radius_m, profile_ids)
        if plan.message:
            logger.warning(plan.message)
        evaluated = np.zeros(n_points, dtype=bool)
        for target in targets:
            evaluated[target] = True
        logger.info(
            f"   [Drift Analyzer] Exact block leave-out cross-validation: {len(blocks)} blocks, "
            f"n = {int(evaluated.sum())} evaluated points, {len(candidates)} candidate models."
        )

        columns_list: list[np.ndarray | None] = []
        meta = []
        for drift_terms in candidates:
            name = format_drift_name(drift_terms)
            warning_msg = None
            if self.interpolation_target == "P" and "sia" in drift_terms:
                warning_msg = SIA_PRODUCT_WARNING
                logger.warning(
                    f"   [Drift Analyzer] Candidate '{name}' contains the 'sia' drift on the BSS product P target: "
                    f"{SIA_PRODUCT_WARNING} The model is evaluated and ranked normally."
                )
            basis = DriftBasis(
                drift_terms=drift_terms,
                x_ref=x_pts,
                y_ref=y_pts,
                geometry=geometry,
                drift_grids=drift_grids,
            )
            x_drift = basis.evaluate(x_pts, y_pts)
            vif_max, _ = self.calculate_vif(x_drift)
            columns_list.append(x_drift if basis.n_columns else None)
            meta.append((name, drift_terms, basis.n_columns, vif_max, vif_max > self.vif_threshold, warning_msg))

        label = self.pass_label
        with activity_indicator(
            f"   [Drift Analyzer] {label}: factorising kriging system (N = {n_points})",
            disable=not self.show_progress,
        ):
            system = KrigingSystem(x_pts, y_pts, z_values, None, model, params)
            leave_block_out = LeaveBlockOut(system, blocks, targets, full_inverse=plan.full_inverse)
        with get_progress_bar(
            total=leave_block_out.n_blocks,
            desc=f"   [Drift Analyzer] {label}: leave-block-out CV",
            unit="profiles" if cv_mode == "lopo" and profile_ids is not None else "points",
            disable=not self.show_progress,
        ) as bar:
            residuals = leave_block_out.residuals(columns_list, progress=bar.update)

        valid = ~np.isnan(residuals) & evaluated
        failed = [valid[k].sum() < meta[k][2] + 3 for k in range(len(meta))]
        working = [k for k in range(len(meta)) if not failed[k]]
        if not working:
            logger.warning(
                "   [Drift Analyzer] No candidate could be cross-validated: every removed block leaves too few "
                "observations. Use a smaller buffer radius or more survey points."
            )
        common = np.all(valid[working], axis=0) if working else np.zeros(n_points, dtype=bool)
        if common.sum() < 5:
            common = None  # too few shared points: every candidate uses its own valid points

        results: list[CandidateDriftResult] = []
        for k, (name, drift_terms, n_drift, vif_max, is_multicollinear, warning_msg) in enumerate(meta):
            if failed[k]:
                results.append(CandidateDriftResult(
                    name=name, terms=drift_terms, n_terms=n_drift, vif_max=vif_max,
                    is_multicollinear=is_multicollinear, cv_rmse=FAILED_RMSE, cv_mae=FAILED_RMSE, cv_r2=-1.0,
                    aicc=FAILED_AICC, is_penalized=True,
                    warning_msg=warning_msg or "Numerical singularity during CV.",
                ))
                continue

            use = (common & valid[k]) if common is not None else valid[k]
            n_valid = int(use.sum())
            residual = residuals[k, use]
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            mae = float(np.mean(np.abs(residual)))
            ss_res = float(np.sum(residual ** 2))
            ss_tot = float(np.sum((z_values[use] - np.mean(z_values[use])) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Akaike Information Criterion corrected (AICc)
            k_param = n_drift + 1
            if n_valid - k_param - 1 > 0 and ss_res > 0:
                aicc = float(n_valid * np.log(ss_res / n_valid) + 2 * k_param + (2 * k_param * (k_param + 1)) / (n_valid - k_param - 1))
            else:
                aicc = FAILED_AICC

            results.append(CandidateDriftResult(
                name=name, terms=drift_terms, n_terms=n_drift, vif_max=vif_max,
                is_multicollinear=is_multicollinear, cv_rmse=rmse, cv_mae=mae, cv_r2=r2, aicc=aicc,
                is_penalized=is_multicollinear or (not np.isfinite(aicc)) or aicc >= FAILED_AICC, warning_msg=warning_msg,
            ))

        results.sort(key=lambda r: (r.is_penalized, r.aicc, r.cv_rmse))
        add_ranking_deltas(results)
        return results

    @staticmethod
    def _prompt_buffer_radius(a_0: float, default_radius: float) -> float:
        """Asks for the buffer radius on the terminal; invalid input is reported and the default is kept."""
        print(f"[DriftAnalyzer] Fitted spatial range: a_0 = {a_0:.1f} meters.")
        try:
            answer = input(f"[DriftAnalyzer] Enter spatial exclusion buffer radius in meters [default: {default_radius:.1f} m]: ").strip()
        except EOFError:
            end_terminal_block()
            return default_radius
        if not answer:
            end_terminal_block()
            return default_radius
        try:
            radius = float(answer)
        except ValueError:
            print(f"[DriftAnalyzer] '{answer}' is not a number; keeping the default radius of {default_radius:.1f} m.")
            end_terminal_block()
            return default_radius
        if not np.isfinite(radius) or radius <= 0:
            print(f"[DriftAnalyzer] The radius must be a positive number; keeping the default of {default_radius:.1f} m.")
            end_terminal_block()
            return default_radius
        end_terminal_block()
        return radius

    @staticmethod
    def _prompt_model_selection(results: list[CandidateDriftResult]) -> list[CandidateDriftResult]:
        """Asks which model(s) to apply until the answer is valid; returns list of chosen results."""
        prompt = (
            f"[DriftAnalyzer] Select drift model(s) to apply [1-{len(results)}, range e.g. '1-5', '1,2,5-8', "
            f"name or terms, default: 1 ('{results[0].name}')]: "
        )
        while True:
            try:
                answer = input(prompt)
            except EOFError:
                end_terminal_block()
                return [results[0]]
            chosen, error = parse_model_selection(answer, results)
            if error is None and chosen:
                end_terminal_block()
                return chosen
            if error is not None:
                print(f"[DriftAnalyzer] {error}")

    def run_diagnostics(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        z_values: np.ndarray,
        geometry: GridGeometry,
        drift_grids: dict[str, np.ndarray],
        candidates: list[list[str]],
        variogram_model: str,
        variogram_params: dict[str, float] | None,
        profile_data: np.ndarray | None = None,
        interactive: bool = True,
        is_predefined_suite: bool = False,
    ) -> list[CandidateDriftResult]:
        """
        Executes the full diagnostic workflow and presents candidate drift model rankings.

        Extracts spatial covariates, performs multicollinearity checks, conducts spatial
        cross-validation (LOPO-CV or buffer LOOCV), computes AICc parsimony scores, and
        provides an interactive command-line selection interface.

        Parameters
        ----------
        x_pts : np.ndarray
            X coordinates of survey soundings [m].
        y_pts : np.ndarray
            Y coordinates of survey soundings [m].
        z_values : np.ndarray
            Observed target values (traveltime [ns] or depth [m]).
        geometry : GridGeometry
            Grid definition shared with the production run.
        drift_grids : dict of str to np.ndarray
            Raster covariates ``sia`` / ``z_dem`` / ``curvature_dem`` of the production run (see
            :func:`pysole.interpolation.build_drift_grids`).
        candidates : list of list of str
            Candidate drift models as lists of primitive drift terms; the empty list is Ordinary Kriging (see
            :func:`pysole.drift.build_drift_suite`).
        variogram_model : str
            Fitted variogram model name (e.g. 'spherical').
        variogram_params : dict of str to float, optional
            Fitted variogram parameters dictionary containing 'range', 'sill', 'nugget' (the production run's
            parameters; ``None`` uses the engine defaults).
        profile_data : np.ndarray, optional
            Profile ID labels for each sounding point.
        interactive : bool, default True
            If True and in an interactive terminal, displays interactive model selection prompt.
        is_predefined_suite : bool, default False
            If True, candidate suite was predefined in JSON configuration: suppresses interactive prompt
            and selects all evaluated candidates for comparison.

        Returns
        -------
        recommendations : list of CandidateDriftResult
            Ranked list of evaluated candidate models, with the recommended model at index 0.
        """
        t_char = "T" if "travel" in str(self.target_name).lower() else "D"
        target_descr = f"Product P(x,y) = {t_char}*sin(alpha)" if self.interpolation_target == "P" else f"Direct Field {self.interpolation_target}(x,y)"
        logger.info(f"   [Drift Analyzer] Initialized (Mode: {self.pass_label}, Target: {self.target_name} [{target_descr}], Boundary Conditions: {self.include_zero_boundary}).")
        logger.info(f"   [Drift Analyzer] Extracting spatial covariates (Z_dem, C_kc, sin(alpha_opt)^-1, X, Y) at N = {len(z_values)} sample locations...")

        x_pts = np.asarray(x_pts, dtype=np.float64)
        y_pts = np.asarray(y_pts, dtype=np.float64)
        z_values = np.asarray(z_values, dtype=np.float64)
        range_a, sill, nugget = resolve_variogram_params(
            variogram_params, z_values, max_pairwise_distance(np.column_stack((x_pts, y_pts)))
        )
        params = {"range": range_a, "sill": sill, "nugget": nugget}
        a_0 = float(range_a)
        cv_mode = "buffer"
        buffer_radius_m = a_0 / 2.0
        profile_ids = None
        show_prompts = interactive and sys.stdin.isatty() and not is_predefined_suite
        banner_shown = False

        if self.survey_profile_column and profile_data is not None:
            profile_ids = np.asarray(profile_data)
            n_profs = len(np.unique(profile_ids))
            if n_profs >= 3:
                cv_mode = "lopo"
                logger.info(f"   [Drift Analyzer] Cross-Validation Mode: Leave-One-Profile-Out (LOPO-CV) using column '{self.survey_profile_column}' (N_profiles = {n_profs}).")
            else:
                logger.info(f"   [Drift Analyzer] Fewer than 3 unique profile IDs found (N_profiles = {n_profs}). Falling back to Spatial Buffer LOOCV.")

        if cv_mode == "buffer":
            if show_prompts:
                log_section_header("INTERACTIVE DRIFT MODEL SELECTION")
                banner_shown = True
                buffer_radius_m = self._prompt_buffer_radius(a_0, buffer_radius_m)
            logger.info(f"   [Drift Analyzer] Fitted spatial range: a_0 = {a_0:.1f} meters. Spatial exclusion buffer radius set to r_buffer = {buffer_radius_m:.1f} meters.")

        logger.info(f"   [Drift Analyzer] Evaluating {len(candidates)} candidate drift models via block leave-out {cv_mode.upper()}...")
        eval_results = self.evaluate_spatial_cv(
            x_pts, y_pts, z_values, geometry, drift_grids, candidates,
            variogram_model, params, cv_mode=cv_mode,
            buffer_radius_m=buffer_radius_m, profile_ids=profile_ids,
        )
        for rank_idx, res in enumerate(eval_results, 1):
            res.rank = rank_idx

        top_model = eval_results[0] if eval_results else None
        if top_model:
            logger.info(f"   [Drift Analyzer] Ranking complete. Top model: '{top_model.name}' (RMSE = {top_model.cv_rmse:.2f} {self.rmse_unit}, AICc = {top_model.aicc:.1f}).")

        if is_predefined_suite:
            self.selected_models = list(eval_results)
            logger.info(f"   [Drift Analyzer] Pre-defined suite in configuration detected ({len(eval_results)} models). Bypassing interactive selection prompt.")
        elif show_prompts and eval_results:
            if not banner_shown:
                log_section_header("INTERACTIVE DRIFT MODEL SELECTION")
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
            print(format_ranking_table(eval_results, f"DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS ({self.pass_label.upper()})", rmse_unit=self.rmse_unit))
            print(f"   [Drift Analyzer] Recommended model: #1 '{eval_results[0].name}'")
            chosen = self._prompt_model_selection(eval_results)
            self.selected_models = chosen if chosen else [eval_results[0]]
        else:
            self.selected_models = [eval_results[0]] if eval_results else []

        if self.selected_models:
            primary = self.selected_models[0]
            if primary in eval_results:
                eval_results.remove(primary)
                eval_results.insert(0, primary)
            logger.info(f"   [Drift Analyzer] Drift model '{self.selected_models[0].name}' selected for the production Kriging run.")
            if len(self.selected_models) > 1:
                names = [m.name for m in self.selected_models]
                logger.info(f"   [Drift Comparison] {len(self.selected_models)} models queued for comparison: {names}")

        return eval_results
