"""
Robustness plots for latent geometry, decoding, and information-theoretic metrics.

This module visualises the parameter sweeps implemented in robustness.py, corresponding
to the robustness checks discussed in Section 5.4 of the preprint: latent dimensionality,
peri‑stimulus window, logistic regularisation C, and class weighting.

Inputs
------
All plotting functions operate on a pandas.DataFrame built from a list of RobustnessRow
objects (see robustness.py). A minimal pattern is:

    from fep_geo.robustness import robustness_for_all_sessions
    from fep_geo.plots_robustness import robustness_rows_to_dataframe

    rows = robustness_for_all_sessions()
    df = robustness_rows_to_dataframe(rows)

You can then call the plotting functions on `df` directly, or on filtered subsets
(e.g. a single session, or a single class_weight) for more targeted visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .robustness import RobustnessRow


# ---------------------------------------------------------------------------
# Helper: convert RobustnessRow list → DataFrame
# ---------------------------------------------------------------------------


def robustness_rows_to_dataframe(rows: Sequence[RobustnessRow]) -> pd.DataFrame:
    """
    Convert a sequence of RobustnessRow objects into a pandas DataFrame.

    Each field of RobustnessRow becomes a column; this is the expected input
    format for all plotting functions in this module.
    """
    return pd.DataFrame([r.to_dict() for r in rows])


def _pretty_metric_label(metric: str) -> str:
    """Human-readable y-axis label for common metrics."""
    mapping = {
        "ideal_accuracy": "Ideal listener accuracy (fraction correct)",
        "actual_test_accuracy": "Actual decoding accuracy (fraction correct)",
        "IG_mean": "Mean information gain IG (nats)",
        "IG_uniform_mean": "Mean information gain IG (uniform prior, nats)",
        "loss_mean": "Mean pragmatic loss ℓ* (nats)",
        "loss_prior_mean": "Mean pragmatic loss ℓ* (prior-aware actual, nats)",
        "loss_uniform_mean": "Mean pragmatic loss ℓ* (uniform prior ideal, nats)",
        "loss_uniform_prior_mean": "Mean pragmatic loss ℓ* (uniform prior + prior-aware actual, nats)",
    }
    return mapping.get(metric, metric)


def _title_prefix(metric: str) -> str:
    """Short title prefix for standard metrics."""
    mapping = {
        "ideal_accuracy": "Ideal accuracy",
        "actual_test_accuracy": "Actual accuracy",
        "IG_mean": "Mean IG",
        "IG_uniform_mean": "Mean IG (uniform prior)",
        "loss_mean": "Mean ℓ*",
        "loss_prior_mean": "Mean ℓ* (prior-aware actual)",
        "loss_uniform_mean": "Mean ℓ* (uniform prior ideal)",
        "loss_uniform_prior_mean": "Mean ℓ* (uniform prior + prior-aware actual)",
    }
    return mapping.get(metric, metric)


def _filter_session(df: pd.DataFrame, session_tag: Optional[str]) -> pd.DataFrame:
    """Optionally restrict to a single session."""
    if session_tag is None:
        return df
    return df[df["session_tag"] == session_tag].copy()


def _compute_group_stats(values: np.ndarray) -> tuple[float, float]:
    """
    Compute (mean, std) with Bessel correction, returning std=0.0 for singletons.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


# ---------------------------------------------------------------------------
# Metric vs latent dimensionality d
# ---------------------------------------------------------------------------


def plot_metric_vs_latent_dim(
    df: pd.DataFrame,
    metric: str = "actual_test_accuracy",
    session_tag: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a metric as a function of latent dimensionality d.

    This addresses robustness to the choice of FA dimensionality (e.g. d ∈ {5,10,20}),
    as requested in the robustness checks (Section 5.4).

    Parameters
    ----------
    df :
        DataFrame built from RobustnessRow objects.
    metric :
        Column name to plot on the y-axis (e.g. 'actual_test_accuracy',
        'ideal_accuracy', 'IG_mean', 'loss_mean').
    session_tag :
        If provided, restricts the plot to a single session; otherwise aggregates
        across all sessions and hyperparameter settings.
    ax :
        Optional matplotlib Axes; if None, a new figure and axes are created.

    Returns
    -------
    ax :
        Axes with an errorbar plot of mean ± SD over all rows with a given d.
    """
    df_plot = _filter_session(df, session_tag)

    if "n_latent" not in df_plot.columns:
        raise KeyError("DataFrame must contain a 'n_latent' column.")

    unique_d = np.sort(df_plot["n_latent"].unique())
    means = []
    stds = []

    for d in unique_d:
        vals = df_plot.loc[df_plot["n_latent"] == d, metric].to_numpy(dtype=float)
        m, s = _compute_group_stats(vals)
        means.append(m)
        stds.append(s)

    x = unique_d.astype(float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, means, yerr=stds, fmt="-o", capsize=4)

    ax.set_xlabel("Latent dimensionality d")
    ax.set_ylabel(_pretty_metric_label(metric))

    if session_tag is None:
        ax.set_title(f"{_title_prefix(metric)} vs latent dimensionality (all sessions)")
    else:
        ax.set_title(
            f"{_title_prefix(metric)} vs latent dimensionality\n"
            f"session: {session_tag}"
        )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# Metric vs peri-stimulus window
# ---------------------------------------------------------------------------


def plot_metric_vs_time_window(
    df: pd.DataFrame,
    metric: str = "actual_test_accuracy",
    session_tag: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a metric as a function of the peri-stimulus analysis window.

    Windows are parameterised by (time_window_start, time_window_end) in seconds,
    as in SessionConfig.time_window (Section 3.3).

    Parameters
    ----------
    df :
        DataFrame built from RobustnessRow objects.
    metric :
        Column name to plot on the y-axis.
    session_tag :
        Optional session filter.
    ax :
        Optional matplotlib Axes.

    Returns
    -------
    ax :
        Axes with an errorbar plot over discrete time-window conditions.
    """
    df_plot = _filter_session(df, session_tag)

    for col in ("time_window_start", "time_window_end"):
        if col not in df_plot.columns:
            raise KeyError(f"DataFrame must contain '{col}' column.")

    # Collect unique (t0, t1) pairs and sort by (t0, t1)
    windows = sorted(
        {
            (float(t0), float(t1))
            for t0, t1 in zip(
                df_plot["time_window_start"], df_plot["time_window_end"]
            )
        }
    )

    means = []
    stds = []
    labels = []

    for (t0, t1) in windows:
        mask = (df_plot["time_window_start"] == t0) & (
            df_plot["time_window_end"] == t1
        )
        vals = df_plot.loc[mask, metric].to_numpy(dtype=float)
        m, s = _compute_group_stats(vals)
        means.append(m)
        stds.append(s)
        labels.append(f"{t0:.2f}–{t1:.2f} s")

    x = np.arange(len(windows), dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, means, yerr=stds, fmt="-o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)

    ax.set_xlabel("Peri-stimulus window [t0, t1]")
    ax.set_ylabel(_pretty_metric_label(metric))

    if session_tag is None:
        ax.set_title(f"{_title_prefix(metric)} vs time window (all sessions)")
    else:
        ax.set_title(
            f"{_title_prefix(metric)} vs time window\nsession: {session_tag}"
        )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# Metric vs logistic regularisation C
# ---------------------------------------------------------------------------


def plot_metric_vs_logistic_C(
    df: pd.DataFrame,
    metric: str = "actual_test_accuracy",
    session_tag: Optional[str] = None,
    class_weight_filter: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a metric as a function of logistic regression regularisation C.

    This visualises robustness of decoding and ℓ* to the choice of C in the
    logistic decoder (Section 3.8 / robustness checks).

    Parameters
    ----------
    df :
        DataFrame built from RobustnessRow objects.
    metric :
        Column name to plot on the y-axis.
    session_tag :
        Optional session filter.
    class_weight_filter :
        If not None, restrict to a single class_weight setting. Use the literal
        string 'balanced' or 'None'; the latter selects rows with no class
        weighting (logistic_class_weight is None).
    ax :
        Optional matplotlib Axes.

    Returns
    -------
    ax :
        Axes with an errorbar plot over C values (log-scale x-axis).
    """
    df_plot = _filter_session(df, session_tag)

    if "logistic_C" not in df_plot.columns:
        raise KeyError("DataFrame must contain 'logistic_C' column.")

    if class_weight_filter is not None:
        if class_weight_filter.lower() in {"none", "null"}:
            df_plot = df_plot[df_plot["logistic_class_weight"].isna()]
        else:
            df_plot = df_plot[df_plot["logistic_class_weight"] == class_weight_filter]

    unique_C = np.sort(df_plot["logistic_C"].unique())
    means = []
    stds = []

    for C in unique_C:
        vals = df_plot.loc[df_plot["logistic_C"] == C, metric].to_numpy(dtype=float)
        m, s = _compute_group_stats(vals)
        means.append(m)
        stds.append(s)

    x = unique_C.astype(float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, means, yerr=stds, fmt="-o", capsize=4)
    ax.set_xscale("log")

    ax.set_xlabel("Logistic regularisation C (inverse L2 strength)")
    ax.set_ylabel(_pretty_metric_label(metric))

    title = _title_prefix(metric) + " vs logistic C"
    if class_weight_filter is not None:
        title += f" (class_weight={class_weight_filter})"
    if session_tag is not None:
        title += f"\nsession: {session_tag}"
    ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# Metric vs logistic class_weight
# ---------------------------------------------------------------------------


def plot_metric_vs_class_weight(
    df: pd.DataFrame,
    metric: str = "actual_test_accuracy",
    session_tag: Optional[str] = None,
    C_filter: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a metric as a function of logistic class_weight.

    This compares, for example, unweighted vs. 'balanced' logistic regression
    across robustness runs (Section 5.4).

    Parameters
    ----------
    df :
        DataFrame built from RobustnessRow objects.
    metric :
        Column name to plot on the y-axis.
    session_tag :
        Optional session filter.
    C_filter :
        If not None, restrict to a single logistic_C value (exact match).
    ax :
        Optional matplotlib Axes.

    Returns
    -------
    ax :
        Axes with an errorbar plot over discrete class_weight conditions.
    """
    df_plot = _filter_session(df, session_tag)

    if C_filter is not None:
        df_plot = df_plot[df_plot["logistic_C"] == C_filter]

    if "logistic_class_weight" not in df_plot.columns:
        raise KeyError("DataFrame must contain 'logistic_class_weight' column.")

    unique_weights = df_plot["logistic_class_weight"].unique()
    # Sort with None (no weighting) first
    unique_weights = sorted(
        unique_weights, key=lambda v: (v is not None, str(v) if v is not None else "")
    )

    labels = []
    means = []
    stds = []

    for w in unique_weights:
        if w is None:
            mask = df_plot["logistic_class_weight"].isna()
            label = "None"
        else:
            mask = df_plot["logistic_class_weight"] == w
            label = str(w)

        vals = df_plot.loc[mask, metric].to_numpy(dtype=float)
        m, s = _compute_group_stats(vals)
        labels.append(label)
        means.append(m)
        stds.append(s)

    x = np.arange(len(labels), dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, means, yerr=stds, fmt="-o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Logistic class_weight")
    ax.set_ylabel(_pretty_metric_label(metric))

    title = _title_prefix(metric) + " vs class_weight"
    if C_filter is not None:
        title += f" (C={C_filter:g})"
    if session_tag is not None:
        title += f"\nsession: {session_tag}"
    ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# High-level helper: save a standard set of robustness figures
# ---------------------------------------------------------------------------


def save_standard_robustness_figures(
    df: pd.DataFrame,
    output_dir: str | Path,
    session_tag: Optional[str] = None,
    close: bool = True,
) -> Dict[str, Path]:
    """
    Save a standard battery of robustness plots to disk.

    For the given DataFrame of robustness runs, this function creates PNG files
    for the following combinations:

      - metric ∈ {ideal_accuracy, actual_test_accuracy, IG_mean, loss_mean}
      - parameter ∈ {latent_dim, time_window, logistic_C, class_weight}

    For logistic_C and class_weight we only use the main metric
    'actual_test_accuracy' and 'loss_mean', which are the most relevant for
    decoder robustness and pragmatic loss, respectively. This yields a compact
    but informative set of figures that directly address the robustness points
    raised in Section 5.4.

    Parameters
    ----------
    df :
        DataFrame built from RobustnessRow objects.
    output_dir :
        Directory in which to save PNG files. Created if necessary.
    session_tag :
        Optional session filter; if provided, only rows for that session are
        used and the filenames are prefixed accordingly.
    close :
        If True, each figure is closed after saving (recommended for batch use).

    Returns
    -------
    paths :
        Dict mapping a short key (e.g. 'latent_actual_test_accuracy') to
        the corresponding file Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base = "all_sessions" if session_tag is None else session_tag

    paths: Dict[str, Path] = {}

    metrics_latent = [
        "ideal_accuracy",
        "actual_test_accuracy",
        "IG_mean",
        "IG_uniform_mean",
        "loss_mean",
        "loss_prior_mean",
        "loss_uniform_mean",
        "loss_uniform_prior_mean",
    ]

    metrics_decoder = [
        "actual_test_accuracy",
        "loss_mean",
        "loss_prior_mean",
        "loss_uniform_mean",
        "loss_uniform_prior_mean",
    ]

    # 1. Metrics vs latent dimension d
    for metric in metrics_latent:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        plot_metric_vs_latent_dim(df, metric=metric, session_tag=session_tag, ax=ax)
        key = f"latent_{metric}"
        fname = f"{base}_latent_{metric}.png"
        path = out_path / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[key] = path

    # 2. Metrics vs time window
    for metric in metrics_latent:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        plot_metric_vs_time_window(df, metric=metric, session_tag=session_tag, ax=ax)
        key = f"time_window_{metric}"
        fname = f"{base}_time_window_{metric}.png"
        path = out_path / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[key] = path

    # 3. Actual accuracy and loss vs logistic C
    for metric in metrics_decoder:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        plot_metric_vs_logistic_C(
            df, metric=metric, session_tag=session_tag, class_weight_filter=None, ax=ax
        )
        key = f"logistic_C_{metric}"
        fname = f"{base}_logistic_C_{metric}.png"
        path = out_path / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[key] = path

    # 4. Actual accuracy and loss vs class_weight
    for metric in metrics_decoder:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        plot_metric_vs_class_weight(
            df, metric=metric, session_tag=session_tag, C_filter=None, ax=ax
        )
        key = f"class_weight_{metric}"
        fname = f"{base}_class_weight_{metric}.png"
        path = out_path / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[key] = path

    return paths


__all__ = [
    "robustness_rows_to_dataframe",
    "plot_metric_vs_latent_dim",
    "plot_metric_vs_time_window",
    "plot_metric_vs_logistic_C",
    "plot_metric_vs_class_weight",
    "save_standard_robustness_figures",
]
