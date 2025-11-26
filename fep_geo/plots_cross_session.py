"""
Cross-session figures: decoding accuracy, information gain, pragmatic loss, and behaviour.

This module implements *cross-session* plots based on the aggregated tables described
in Sections 3.10–3.11 and used in Results Section 4 of the preprint:

- Figure 3a: ideal vs. actual decoding accuracy across sessions, with binomial
  standard errors (`plot_sessions_accuracy_scatter`);
- Figure 3b: session-wise mean information gain vs. mean pragmatic loss,
  with across-trial standard deviations (`plot_sessions_IG_vs_loss`);
- block-wise comparison of information gain and pragmatic loss as a function
  of block prior P(Left | c) (`plot_blockwise_IG_loss`), summarising Table 4;
- optional behaviour–neural scatter plots linking behavioural accuracy to
  neural metrics (`plot_behaviour_vs_metric`), used in Section 4.4.3.

All plotting functions operate directly on pandas DataFrames such as those returned in
the `tables` dict by `aggregate.run_multi_session`:

- tables["summary_all"]
- tables["by_context_all"]
- tables["test_only_summary"]
- tables["behaviour_link"]

A high-level helper `save_cross_session_figures` saves a standard set of PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_PATTERN_SPECS = {
    "l1": dict(
        suffix="_l1",
        IG_col="IG_mean",
        IG_std_col="IG_std",
        loss_col="loss_mean",
        loss_std_col="loss_std",
    ),
    "l2": dict(
        suffix="_l2",
        IG_col="IG_mean",
        IG_std_col="IG_std",
        loss_col="loss_prior_mean",
        loss_std_col="loss_prior_std",
    ),
    "l3": dict(
        suffix="_l3",
        IG_col="IG_uniform_mean",
        IG_std_col="IG_uniform_std",
        loss_col="loss_uniform_mean",
        loss_std_col="loss_uniform_std",
    ),
    "l4": dict(
        suffix="_l4",
        IG_col="IG_uniform_mean",
        IG_std_col="IG_uniform_std",
        loss_col="loss_uniform_prior_mean",
        loss_std_col="loss_uniform_prior_std",
    ),
}

CONTEXT_PATTERN_SPECS = {
    "l1": dict(
        suffix="_l1",
        IG_col="IG_mean",
        IG_std_col="IG_std",
        loss_col="loss_mean",
        loss_std_col="loss_std",
    ),
    "l2": dict(
        suffix="_l2",
        IG_col="IG_mean",
        IG_std_col="IG_std",
        loss_col="loss_prior_mean",
        loss_std_col="loss_prior_std",
    ),
    "l3": dict(
        suffix="_l3",
        IG_col="IG_uniform_mean",
        IG_std_col="IG_uniform_std",
        loss_col="loss_uniform_mean",
        loss_std_col="loss_uniform_std",
    ),
    "l4": dict(
        suffix="_l4",
        IG_col="IG_uniform_mean",
        IG_std_col="IG_uniform_std",
        loss_col="loss_uniform_prior_mean",
        loss_std_col="loss_uniform_prior_std",
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_session_tag_column(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'session_tag' column of the form '{session_id}_{probe}' to summary_all.

    This reconstructs the PID-like session_tag strings used in the other tables.
    """
    df = summary_df.copy()
    if "session_tag" not in df.columns:
        df["session_tag"] = df["session_id"].astype(str) + "_" + df["probe"].astype(str)
    return df


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Simple Pearson correlation; returns NaN if variance is zero or N<2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = float(np.sum(x_c * y_c))
    den = float(np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2)))
    if den == 0.0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# Figure 3a: ideal vs. actual decoding accuracy across sessions
# ---------------------------------------------------------------------------


def plot_sessions_accuracy_scatter(
    summary_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = False,
) -> plt.Axes:
    """
    Cross-session scatter of ideal vs. actual decoding accuracy (Figure 3a).

    Parameters
    ----------
    summary_df :
        DataFrame like tables["summary_all"], with columns
        ['session_id', 'probe', 'n_trials', 'ideal_accuracy', 'actual_accuracy', ...].
    test_df :
        Optional DataFrame like tables["test_only_summary"], providing
        'session_tag' and 'n_test' to compute binomial SE for actual accuracy.
        If None, `n_trials` is used as a proxy for n_test.
    ax :
        Optional matplotlib Axes; if None, a new figure and axes are created.
    annotate :
        If True, annotate each point with a short session label (probe ID).

    Returns
    -------
    ax :
        Axes with the scatter plot and error bars.
    """
    df = _add_session_tag_column(summary_df)

    if test_df is not None:
        # Keep only the columns we need
        test_cols = test_df[["session_tag", "n_test"]].copy()
        df = df.merge(test_cols, on="session_tag", how="left")
    else:
        df["n_test"] = np.nan

    # Fallback: if n_test missing, use n_trials
    df["n_test"].fillna(df["n_trials"], inplace=True)

    x = df["ideal_accuracy"].to_numpy(dtype=float)
    y = df["actual_accuracy"].to_numpy(dtype=float)

    n_ideal = df["n_trials"].to_numpy(dtype=float)
    n_actual = df["n_test"].to_numpy(dtype=float)

    se_ideal = np.sqrt(x * (1.0 - x) / np.maximum(n_ideal, 1.0))
    se_actual = np.sqrt(y * (1.0 - y) / np.maximum(n_actual, 1.0))

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(
        x,
        y,
        xerr=se_ideal,
        yerr=se_actual,
        fmt="o",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
        alpha=0.9,
    )

    # Diagonal reference line (ideal = actual)
    x_min = max(0.5, float(np.min(x) - 0.05))
    x_max = min(1.0, float(np.max(x) + 0.05))
    ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", linewidth=0.8)

    ax.set_xlabel("Ideal listener accuracy (fraction correct)")
    ax.set_ylabel("Actual listener accuracy (fraction correct)")
    ax.set_title("Ideal vs. actual decoding accuracy across sessions")

    # Optional annotations (shortened tags)
    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                row["probe"],
                (row["ideal_accuracy"], row["actual_accuracy"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    return ax


# ---------------------------------------------------------------------------
# Figure 3b: session-wise mean IG vs. mean pragmatic loss
# ---------------------------------------------------------------------------

def _plot_sessions_IG_vs_loss_from_cols(
    summary_df: pd.DataFrame,
    IG_col: str,
    IG_std_col: str,
    loss_col: str,
    loss_std_col: str,
    ax: Optional[plt.Axes] = None,
    annotate: bool = False,
    title_suffix: str = "",
) -> plt.Axes:
    df = _add_session_tag_column(summary_df)

    IG_mean = df[IG_col].to_numpy(dtype=float)
    IG_std = df[IG_std_col].to_numpy(dtype=float)
    loss_mean = df[loss_col].to_numpy(dtype=float)
    loss_std = df[loss_std_col].to_numpy(dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(
        IG_mean,
        loss_mean,
        xerr=IG_std,
        yerr=loss_std,
        fmt="o",
        capsize=3,
        alpha=0.8,
    )

    r = _pearson_corr(IG_mean, loss_mean)
    ax.set_xlabel("Mean information gain IG (nats)")
    ax.set_ylabel("Mean pragmatic loss ℓ* (nats)")
    ax.set_title(f"Mean IG vs. mean ℓ* across sessions{title_suffix}\n"
                 f"Pearson r ≈ {r:.3f}")

    if annotate:
        for tag, x, y in zip(df["session_tag"], IG_mean, loss_mean):
            ax.annotate(str(tag), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


def plot_sessions_IG_vs_loss(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    annotate: bool = False,
) -> plt.Axes:
    spec = SUMMARY_PATTERN_SPECS["l1"]
    return _plot_sessions_IG_vs_loss_from_cols(
        summary_df,
        IG_col=spec["IG_col"],
        IG_std_col=spec["IG_std_col"],
        loss_col=spec["loss_col"],
        loss_std_col=spec["loss_std_col"],
        ax=ax,
        annotate=annotate,
        title_suffix=" (ℓ1)",
    )


def plot_sessions_IG_vs_loss_pattern(
    summary_df: pd.DataFrame,
    pattern: str = "l1",
    ax: Optional[plt.Axes] = None,
    annotate: bool = False,
) -> plt.Axes:

    if pattern not in SUMMARY_PATTERN_SPECS:
        raise ValueError(f"Unknown pattern '{pattern}'. Expected {list(SUMMARY_PATTERN_SPECS)}")

    spec = SUMMARY_PATTERN_SPECS[pattern]
    required = [spec["IG_col"], spec["IG_std_col"], spec["loss_col"], spec["loss_std_col"]]
    for col in required:
        if col not in summary_df.columns:
            raise KeyError(f"Column '{col}' not found in summary_df for pattern '{pattern}'")

    return _plot_sessions_IG_vs_loss_from_cols(
        summary_df,
        IG_col=spec["IG_col"],
        IG_std_col=spec["IG_std_col"],
        loss_col=spec["loss_col"],
        loss_std_col=spec["loss_std_col"],
        ax=ax,
        annotate=annotate,
        title_suffix=f" ({pattern})",
    )


# ---------------------------------------------------------------------------
# Block-wise IG / loss by block prior (Table 4)
# ---------------------------------------------------------------------------

def _block_weighted_stats_generic(
    by_context_df: pd.DataFrame,
    IG_col: str,
    IG_std_col: str,
    loss_col: str,
    loss_std_col: str,
) -> pd.DataFrame:

    records = []
    for p, grp in by_context_df.groupby("P_left_prior"):
        n = grp["n_trials"].to_numpy(dtype=float)

        IG_m = grp[IG_col].to_numpy(dtype=float)
        IG_s = grp[IG_std_col].to_numpy(dtype=float)

        L_m = grp[loss_col].to_numpy(dtype=float)
        L_s = grp[loss_std_col].to_numpy(dtype=float)

        N_total = float(n.sum())
        if N_total <= 0:
            continue

        IG_mean = float(np.average(IG_m, weights=n))
        loss_mean = float(np.average(L_m, weights=n))

        IG_var = float(np.average(IG_s ** 2 + (IG_m - IG_mean) ** 2, weights=n))
        loss_var = float(np.average(L_s ** 2 + (L_m - loss_mean) ** 2, weights=n))

        IG_std = float(np.sqrt(max(IG_var, 0.0)))
        loss_std = float(np.sqrt(max(loss_var, 0.0)))

        records.append(
            {
                "P_left_prior": float(p),
                "n_trials_total": int(N_total),
                "IG_mean": IG_mean,
                "IG_std": IG_std,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
            }
        )

    return pd.DataFrame.from_records(records)


def _block_weighted_stats(by_context_df: pd.DataFrame) -> pd.DataFrame:
    return _block_weighted_stats_generic(
        by_context_df,
        IG_col="IG_mean",
        IG_std_col="IG_std",
        loss_col="loss_mean",
        loss_std_col="loss_std",
    )


def _block_weighted_stats_pattern(
    by_context_df: pd.DataFrame,
    pattern: str = "l1",
) -> pd.DataFrame:
    if pattern not in CONTEXT_PATTERN_SPECS:
        raise ValueError(f"Unknown pattern '{pattern}'. Expected {list(CONTEXT_PATTERN_SPECS)}")

    spec = CONTEXT_PATTERN_SPECS[pattern]
    required = [spec["IG_col"], spec["IG_std_col"], spec["loss_col"], spec["loss_std_col"]]
    for col in required:
        if col not in by_context_df.columns:
            raise KeyError(f"Column '{col}' not found in by_context_df for pattern '{pattern}'")

    return _block_weighted_stats_generic(
        by_context_df,
        IG_col=spec["IG_col"],
        IG_std_col=spec["IG_std_col"],
        loss_col=spec["loss_col"],
        loss_std_col=spec["loss_std_col"],
    )


def plot_blockwise_IG_loss(
    by_context_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot trial-weighted mean IG and ℓ* by block prior, summarising Table 4.

    Parameters
    ----------
    by_context_df :
        DataFrame like tables["by_context_all"], with columns
        ['P_left_prior', 'n_trials', 'IG_mean', 'IG_std', 'loss_mean', 'loss_std', ...].
    ax :
        Optional matplotlib Axes; if None, a new figure and axes are created.

    Returns
    -------
    ax :
        Axes with grouped bars for IG and ℓ* at each block prior.
    """
    stats = _block_weighted_stats(by_context_df).sort_values("P_left_prior")
    if stats.empty:
        raise ValueError("by_context_df produced no block-wise statistics.")

    priors = stats["P_left_prior"].to_numpy(dtype=float)
    IG_mean = stats["IG_mean"].to_numpy(dtype=float)
    IG_std = stats["IG_std"].to_numpy(dtype=float)
    loss_mean = stats["loss_mean"].to_numpy(dtype=float)
    loss_std = stats["loss_std"].to_numpy(dtype=float)

    x = np.arange(len(priors))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots()

    # Bars for IG and loss side by side
    ax.bar(
        x - width / 2,
        IG_mean,
        width,
        yerr=IG_std,
        capsize=4,
        label="Information gain IG",
    )
    ax.bar(
        x + width / 2,
        loss_mean,
        width,
        yerr=loss_std,
        capsize=4,
        label="Pragmatic loss ℓ*",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in priors])
    ax.set_xlabel("Block prior P(Left | c)")
    ax.set_ylabel("Mean IG / ℓ* (nats)")
    ax.set_title("Information gain and pragmatic loss by block prior")
    ax.legend(loc="best", frameon=False)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    return ax


def plot_blockwise_IG_loss_pattern(
    by_context_df: pd.DataFrame,
    pattern: str = "l1",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:

    stats = _block_weighted_stats_pattern(by_context_df, pattern=pattern).sort_values(
        "P_left_prior"
    )
    if stats.empty:
        raise ValueError("by_context_df produced no block-wise statistics.")

    priors = stats["P_left_prior"].to_numpy(dtype=float)
    IG_mean = stats["IG_mean"].to_numpy(dtype=float)
    IG_std = stats["IG_std"].to_numpy(dtype=float)
    loss_mean = stats["loss_mean"].to_numpy(dtype=float)
    loss_std = stats["loss_std"].to_numpy(dtype=float)

    x = np.arange(len(priors))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(
        x - width / 2,
        IG_mean,
        width,
        yerr=IG_std,
        capsize=4,
        label="Information gain IG",
    )
    ax.bar(
        x + width / 2,
        loss_mean,
        width,
        yerr=loss_std,
        capsize=4,
        label="Pragmatic loss ℓ*",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in priors])
    ax.set_xlabel("Block prior P(Left | c)")
    ax.set_ylabel("Mean IG / ℓ* (nats)")
    ax.set_title(f"Information gain and pragmatic loss by block prior ({pattern})")
    ax.legend(loc="best", frameon=False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    return ax


# ---------------------------------------------------------------------------
# Behaviour vs. neural metrics (Section 4.4.3)
# ---------------------------------------------------------------------------


def plot_behaviour_vs_metric(
    behaviour_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric: str = "IG_mean",
    use_error_rate: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter of behavioural performance vs. a neural metric across sessions.

    This corresponds to the exploratory analysis in Section 4.4.3, where behavioural
    error rates are related to decoding accuracies, mean IG, and mean ℓ*.

    Parameters
    ----------
    behaviour_df :
        DataFrame like tables["behaviour_link"], with columns
        ['session_tag', 'behav_accuracy', 'behav_error', ...].
    summary_df :
        DataFrame like tables["summary_all"], providing session-wise neural metrics.
    metric :
        Name of the neural metric column in summary_df to use as x-axis. Common
        choices are 'ideal_accuracy', 'actual_accuracy', 'IG_mean', 'loss_mean'.
    use_error_rate :
        If True, plot behavioural error rate (1 − accuracy); otherwise plot
        behavioural accuracy.
    ax :
        Optional matplotlib Axes; if None, a new figure and axes are created.

    Returns
    -------
    ax :
        Axes with the scatter plot and fitted regression line.
    """
    # Add session_tag and merge with behaviour table
    summ = _add_session_tag_column(summary_df)
    merged = behaviour_df.merge(summ, on="session_tag", how="inner")

    if merged.empty:
        raise ValueError("No overlapping sessions between behaviour_df and summary_df.")

    x = merged[metric].to_numpy(dtype=float)
    if use_error_rate:
        y = merged["behav_error"].to_numpy(dtype=float)
        y_label = "Behavioural error rate"
    else:
        y = merged["behav_accuracy"].to_numpy(dtype=float)
        y_label = "Behavioural accuracy"

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x, y, s=30, alpha=0.8)

    # Simple linear fit y = a x + b
    if x.size >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        a, b = coeffs[0], coeffs[1]
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linestyle="--", linewidth=0.8)

        r = _pearson_corr(x, y)
        title_suffix = f" (r ≈ {r:.2f})" if not np.isnan(r) else ""
    else:
        title_suffix = ""

    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel(y_label)
    ax.set_title(f"Behaviour vs. {metric}{title_suffix}")

    return ax


# ---------------------------------------------------------------------------
# High-level helper: save standard cross-session figures
# ---------------------------------------------------------------------------


def save_cross_session_figures(
    tables: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    close: bool = True,
) -> Dict[str, Path]:
    """
    Save a standard set of cross-session figures to disk.

    Expected entries in `tables` (as returned by `aggregate.run_multi_session`):

        - "summary_all"
        - "by_context_all"
        - "test_only_summary"   (optional but recommended)
        - "behaviour_link"      (optional)

    Parameters
    ----------
    tables :
        Dict of table name → DataFrame.
    output_dir :
        Directory in which to save PNG files. Created if necessary.
    close :
        If True, close each figure after saving.

    Returns
    -------
    paths :
        Dict mapping a short key to the corresponding file Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_df = tables["summary_all"]
    by_context_df = tables["by_context_all"]
    test_df = tables.get("test_only_summary", None)
    behaviour_df = tables.get("behaviour_link", None)

    paths: Dict[str, Path] = {}

    # Figure 3a: accuracy scatter
    fig, ax = plt.subplots()
    plot_sessions_accuracy_scatter(summary_df, test_df=test_df, ax=ax)
    p = out_path / "sessions_accuracy_scatter.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["sessions_accuracy_scatter"] = p

    # Figure 3b: IG vs loss
    fig, ax = plt.subplots()
    plot_sessions_IG_vs_loss(summary_df, ax=ax)
    p = out_path / "sessions_IG_vs_loss.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["sessions_IG_vs_loss"] = p

    for pattern in ("l2", "l3", "l4"):
        spec = SUMMARY_PATTERN_SPECS.get(pattern)
        if spec is None:
            continue
        required = [spec["IG_col"], spec["IG_std_col"], spec["loss_col"], spec["loss_std_col"]]
        if not all(col in summary_df.columns for col in required):
            print(f"[save_cross_session_figures] Skipping sessions_IG_vs_loss{spec['suffix']} "
                  f"(missing columns {required})")
            continue

        fig, ax = plt.subplots()
        plot_sessions_IG_vs_loss_pattern(summary_df, pattern=pattern, ax=ax)
        p = out_path / f"sessions_IG_vs_loss{spec['suffix']}.png"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[f"sessions_IG_vs_loss{spec['suffix']}"] = p

    # Block-wise IG / loss bar plot
    fig, ax = plt.subplots()
    plot_blockwise_IG_loss(by_context_df, ax=ax)
    p = out_path / "contexts_IG_loss_bar.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["contexts_IG_loss_bar"] = p

    for pattern in ("l2", "l3", "l4"):
        spec = CONTEXT_PATTERN_SPECS.get(pattern)
        if spec is None:
            continue
        required = [spec["IG_col"], spec["IG_std_col"], spec["loss_col"], spec["loss_std_col"]]
        if not all(col in by_context_df.columns for col in required):
            print(f"[save_cross_session_figures] Skipping contexts_IG_loss_bar{spec['suffix']} "
                  f"(missing columns {required})")
            continue

        fig, ax = plt.subplots()
        plot_blockwise_IG_loss_pattern(by_context_df, pattern=pattern, ax=ax)
        p = out_path / f"contexts_IG_loss_bar{spec['suffix']}.png"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths[f"contexts_IG_loss_bar{spec['suffix']}"] = p

    # Optional behaviour vs IG (or other metric)
    if behaviour_df is not None:
        fig, ax = plt.subplots()
        plot_behaviour_vs_metric(
            behaviour_df, summary_df, metric="IG_mean", use_error_rate=False, ax=ax
        )
        p = out_path / "behaviour_vs_IG_mean.png"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        if close:
            plt.close(fig)
        paths["behaviour_vs_IG_mean"] = p

        if "IG_uniform_mean" in summary_df.columns:
            fig, ax = plt.subplots()
            plot_behaviour_vs_metric(
                behaviour_df, summary_df, metric="IG_uniform_mean", use_error_rate=False, ax=ax
            )
            p = out_path / "behaviour_vs_IG_uniform_mean.png"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            if close:
                plt.close(fig)
            paths["behaviour_vs_IG_uniform_mean"] = p

        loss_metrics = [
            ("loss_mean", "behaviour_vs_loss_mean"),
            ("loss_prior_mean", "behaviour_vs_loss_prior_mean"),
            ("loss_uniform_mean", "behaviour_vs_loss_uniform_mean"),
            ("loss_uniform_prior_mean", "behaviour_vs_loss_uniform_prior_mean"),
        ]
        for metric_name, key in loss_metrics:
            if metric_name not in summary_df.columns:
                continue
            fig, ax = plt.subplots()
            plot_behaviour_vs_metric(
                behaviour_df, summary_df, metric=metric_name, use_error_rate=False, ax=ax
            )
            p = out_path / f"{key}.png"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            if close:
                plt.close(fig)
            paths[key] = p

    return paths


__all__ = [
    "plot_sessions_accuracy_scatter",
    "plot_sessions_IG_vs_loss",
    "plot_sessions_IG_vs_loss_pattern",
    "plot_blockwise_IG_loss",
    "plot_blockwise_IG_loss_pattern",
    "plot_behaviour_vs_metric",
    "save_cross_session_figures",
]
