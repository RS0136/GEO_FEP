"""
Per-session figures: latent scatter, decoding accuracy, IG / loss histograms, and IG–loss scatter.

This module implements the per-session plots described in Section 3.10 and illustrated in
Figures 1–2 of the preprint:​

- a 2D latent scatter plot coloured by world state;
- a bar plot of ideal vs. actual decoding accuracy with binomial standard errors;
- histograms of information gain IG and pragmatic loss ℓ*;
- a scatter plot of IG versus ℓ*.

All plotting functions operate on a SessionResult (see aggregate.py) and either draw into
a provided matplotlib Axes or create/save standalone PNGs via `save_session_figures`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .aggregate import SessionResult

PATTERN_SPECS = {
    "l1": dict(
        suffix="_l1",
        IG_attr="IG",
        loss_attr="loss",
        ig_label="IG (block prior)",
        loss_label="ℓ1 = KL(L_ideal^prior || L0^agnostic)",
    ),
    "l2": dict(
        suffix="_l2",
        IG_attr="IG",
        loss_attr="loss_prior",
        ig_label="IG (block prior)",
        loss_label="ℓ2 = KL(L_ideal^prior || L0^prior)",
    ),
    "l3": dict(
        suffix="_l3",
        IG_attr="IG_uniform",
        loss_attr="loss_uniform",
        ig_label="IG (uniform prior)",
        loss_label="ℓ3 = KL(L_ideal^unif || L0^agnostic)",
    ),
    "l4": dict(
        suffix="_l4",
        IG_attr="IG_uniform",
        loss_attr="loss_uniform_prior",
        ig_label="IG (uniform prior)",
        loss_label="ℓ4 = KL(L_ideal^unif || L0^prior)",
    ),
}

# ---------------------------------------------------------------------------
# Low-level plotting helpers (one Axes each)
# ---------------------------------------------------------------------------


def _extract_latent_codes(result):
    """
    Extract latent codes Z (n_trials × d) from SessionResult as flexibly as possible.

    - First, try result.Z / result.latent_codes / result.latent / result.codes
    - Next, search within result.geometry.* for:
      *elements named Z / latent_codes / latent / codes*
      or a 2D array with the same number of rows as the trial count in TrialData
    - If still not found, return None
    """

    td = getattr(result, "trials", None)
    n_trials = None
    if td is not None:
        n_trials = getattr(td, "n_trials", None)
        if n_trials is None and hasattr(td, "X"):
            n_trials = int(td.X.shape[0])

    for name in ("Z", "latent_codes", "latent", "codes"):
        if hasattr(result, name):
            arr = np.asarray(getattr(result, name))
            if arr.ndim >= 1:
                return arr

    geom = getattr(result, "geometry", None)
    if geom is not None:
        for name in ("Z", "latent_codes", "latent", "codes"):
            if hasattr(geom, name):
                arr = np.asarray(getattr(geom, name))
                if arr.ndim >= 1:
                    return arr

        if n_trials is not None:
            candidates = []
            for attr_name in dir(geom):
                if attr_name.startswith("_"):
                    continue
                val = getattr(geom, attr_name)
                if isinstance(val, np.ndarray) and val.ndim == 2:
                    if val.shape[0] == n_trials:
                        candidates.append(val)

            if candidates:
                candidates.sort(key=lambda a: a.shape[1])
                return candidates[0]

    return None


def plot_latent_scatter(result, ax=None):
    """
    Plot a scatter plot for latent code Z.

    - When d >= 2: 2D scatter plot using the first and second components
    - When d == 1: 1D scatter plot with x = first component and small jitter added to the y-axis
    - If Z is not found: Issue a warning and return without plotting anything
    """

    if ax is None:
        _, ax = plt.subplots()

    Z = _extract_latent_codes(result)

    if Z is None:
        print(
            "[plots_session] WARNING: Could not find latent codes on "
            "SessionResult; skipping latent scatter."
        )
        return ax

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]

    if Z.ndim != 2:
        print(
            f"[plots_session] WARNING: Latent code Z has unexpected shape "
            f"{Z.shape}; expected (n_trials, d). Skipping latent scatter."
        )
        return ax

    n_trials, d = Z.shape
    if d == 0:
        print("[plots_session] WARNING: Latent code dimension d=0; skipping.")
        return ax

    w = None
    td = getattr(result, "trials", None)
    if td is not None and hasattr(td, "w"):
        w_candidate = np.asarray(td.w)
        if w_candidate.shape[0] == n_trials:
            w = w_candidate

    if w is not None:
        colors = np.where(w == 0, "C0", "C1")
    else:
        colors = "C0"

    if d >= 2:
        x = Z[:, 0]
        y = Z[:, 1]
        ax.scatter(x, y, c=colors, s=8, alpha=0.6, linewidths=0)
        ax.set_xlabel("Latent dim 1")
        ax.set_ylabel("Latent dim 2")
        ax.set_title("Latent geometry (first two dimensions)")
    else:
        x = Z[:, 0]
        rng = np.random.default_rng(0)
        y = 0.02 * rng.standard_normal(size=n_trials)

        ax.scatter(x, y, c=colors, s=8, alpha=0.6, linewidths=0)
        ax.set_xlabel("Latent dim 1")
        ax.set_ylabel("jitter")
        ax.set_title("Latent geometry (1D latent code; jittered)")

    if w is not None:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="C0", markeredgecolor="none", label="w=0 (Left)"),
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="C1", markeredgecolor="none", label="w=1 (Right)"),
        ]
        ax.legend(handles=handles, frameon=False)

    return ax


def plot_decoding_accuracy(
    result: SessionResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot ideal vs. actual decoding accuracy with binomial standard errors (Figure 1b).

    Ideal accuracy is computed on all valid trials; actual accuracy is the test-set accuracy
    of the logistic decoder. Standard errors are sqrt(p (1-p) / N) for the relevant N.
    """
    if ax is None:
        fig, ax = plt.subplots()

    p_ideal = float(result.ideal.accuracy)
    p_actual = float(result.actual.accuracy_test)

    n_ideal = int(result.trials.n_trials)
    n_actual = int(result.geometry.idx_test.size)

    se_ideal = np.sqrt(p_ideal * (1.0 - p_ideal) / max(1, n_ideal))
    se_actual = np.sqrt(p_actual * (1.0 - p_actual) / max(1, n_actual))

    x = np.array([0, 1], dtype=float)
    heights = np.array([p_ideal, p_actual], dtype=float)
    errors = np.array([se_ideal, se_actual], dtype=float)

    labels = ["Ideal listener", "Actual listener"]

    ax.bar(x, heights, yerr=errors, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Decoding accuracy ({result.session_tag})")

    # Optional: light grid for readability
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    return ax


def plot_IG_hist(
    result: SessionResult,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
) -> plt.Axes:
    """
    Plot a histogram of single-trial information gain IG in nats (Figure 2a).
    """
    IG = np.asarray(result.metrics.IG, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(IG, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Information gain IG (nats)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Information gain ({result.session_tag})")

    return ax


def plot_loss_hist(
    result: SessionResult,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
) -> plt.Axes:
    """
    Plot a histogram of single-trial pragmatic loss ℓ* in nats (Figure 2b).
    """
    loss = np.asarray(result.metrics.loss, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(loss, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pragmatic loss ℓ* (nats)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Pragmatic loss ({result.session_tag})")

    return ax


def plot_IG_vs_loss(
    result: SessionResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a joint scatter of IG vs. pragmatic loss ℓ* (Figure 2c).
    """
    IG = np.asarray(result.metrics.IG, dtype=float)
    loss = np.asarray(result.metrics.loss, dtype=float)

    if IG.shape != loss.shape:
        raise ValueError(
            f"IG and loss must have the same shape; got {IG.shape} vs {loss.shape}."
        )

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(IG, loss, s=10, alpha=0.5)
    ax.set_xlabel("Information gain IG (nats)")
    ax.set_ylabel("Pragmatic loss ℓ* (nats)")
    ax.set_title(f"IG vs. pragmatic loss ({result.session_tag})")

    return ax

def plot_loss_hist_pattern(
    result: SessionResult,
    pattern: str = "l1",
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
) -> plt.Axes:

    if pattern not in PATTERN_SPECS:
        raise ValueError(f"Unknown pattern '{pattern}'. Expected one of {list(PATTERN_SPECS)}")

    spec = PATTERN_SPECS[pattern]
    loss = getattr(result.metrics, spec["loss_attr"], None)
    if loss is None:
        raise ValueError(
            f"result.metrics.{spec['loss_attr']} is None; "
            f"cannot plot loss histogram for pattern '{pattern}'."
        )

    loss = np.asarray(loss, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(loss, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"Pragmatic loss {spec['loss_label']} (nats)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Pragmatic loss ({pattern}, {result.session_tag})")

    return ax


def plot_IG_vs_loss_pattern(
    result: SessionResult,
    pattern: str = "l1",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:

    if pattern not in PATTERN_SPECS:
        raise ValueError(f"Unknown pattern '{pattern}'. Expected one of {list(PATTERN_SPECS)}")

    spec = PATTERN_SPECS[pattern]
    IG = getattr(result.metrics, spec["IG_attr"], None)
    loss = getattr(result.metrics, spec["loss_attr"], None)

    if IG is None or loss is None:
        raise ValueError(
            f"metrics for pattern '{pattern}' are missing "
            f"(IG_attr={spec['IG_attr']}, loss_attr={spec['loss_attr']})."
        )

    IG = np.asarray(IG, dtype=float)
    loss = np.asarray(loss, dtype=float)
    if IG.shape != loss.shape:
        raise ValueError(f"IG and loss must have the same shape; got {IG.shape} vs {loss.shape}.")

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(IG, loss, s=10, alpha=0.5)
    ax.set_xlabel(f"Information gain {spec['ig_label']} (nats)")
    ax.set_ylabel(f"Pragmatic loss {spec['loss_label']} (nats)")
    ax.set_title(f"IG vs pragmatic loss ({pattern}, {result.session_tag})")

    return ax

# ---------------------------------------------------------------------------
# High-level helper: save all figures for a session
# ---------------------------------------------------------------------------


def save_session_figures(
    result: SessionResult,
    output_dir: str | Path,
    prefix: Optional[str] = None,
    close: bool = True,
) -> Dict[str, Path]:
    """
    Create and save all per-session figures for a SessionResult.

    This mirrors the behaviour described in Section 3.10: one latent scatter, one decoding
    accuracy bar plot, two histograms (IG and ℓ*), and one IG–loss scatter, saved under
    systematic filenames such as

        {prefix}_latent_scatter.png
        {prefix}_decoding_accuracy.png
        {prefix}_IG_hist.png
        {prefix}_loss_hist.png
        {prefix}_IG_vs_loss.png.

    Parameters
    ----------
    result :
        SessionResult object for a single probe insertion.
    output_dir :
        Directory in which to save the PNG files. Created if necessary.
    prefix :
        Filename prefix; if None, `result.session_tag` is used.
    close :
        If True (default), each figure is closed after saving to free memory.

    Returns
    -------
    paths :
        Dict mapping a short key to the corresponding file Path, e.g.

            {
              "latent_scatter": Path(...),
              "decoding_accuracy": Path(...),
              "IG_hist": Path(...),
              "loss_hist": Path(...),
              "IG_vs_loss": Path(...),
            }
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = result.session_tag

    paths: Dict[str, Path] = {}

    # Latent scatter
    fig, ax = plt.subplots()
    plot_latent_scatter(result, ax=ax)
    fname = f"{prefix}_latent_scatter.png"
    fpath = out_path / fname
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["latent_scatter"] = fpath

    # Decoding accuracy
    fig, ax = plt.subplots()
    plot_decoding_accuracy(result, ax=ax)
    fname = f"{prefix}_decoding_accuracy.png"
    fpath = out_path / fname
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["decoding_accuracy"] = fpath

    # IG histogram
    fig, ax = plt.subplots()
    plot_IG_hist(result, ax=ax)
    fname = f"{prefix}_IG_hist.png"
    fpath = out_path / fname
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["IG_hist"] = fpath

    # Loss histogram
    fig, ax = plt.subplots()
    plot_loss_hist(result, ax=ax)
    fname = f"{prefix}_loss_hist.png"
    fpath = out_path / fname
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["loss_hist"] = fpath

    # IG vs. loss scatter
    fig, ax = plt.subplots()
    plot_IG_vs_loss(result, ax=ax)
    fname = f"{prefix}_IG_vs_loss.png"
    fpath = out_path / fname
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["IG_vs_loss"] = fpath


    for pattern in ("l2", "l3", "l4"):
        spec = PATTERN_SPECS[pattern]

        # Loss histogram
        fig, ax = plt.subplots()
        try:
            plot_loss_hist_pattern(result, pattern=pattern, ax=ax)
        except ValueError as exc:
            print(f"[save_session_figures] Skip loss histogram for pattern {pattern}: {exc}")
            plt.close(fig)
        else:
            fname = f"{prefix}_loss_hist{spec['suffix']}.png"
            path = out_path / fname
            fig.savefig(path, bbox_inches="tight", dpi=150)
            if close:
                plt.close(fig)
            paths[f"loss_hist{spec['suffix']}"] = path

        # IG vs loss
        fig, ax = plt.subplots()
        try:
            plot_IG_vs_loss_pattern(result, pattern=pattern, ax=ax)
        except ValueError as exc:
            print(f"[save_session_figures] Skip IG vs loss for pattern {pattern}: {exc}")
            plt.close(fig)
        else:
            fname = f"{prefix}_IG_vs_loss{spec['suffix']}.png"
            path = out_path / fname
            fig.savefig(path, bbox_inches="tight", dpi=150)
            if close:
                plt.close(fig)
            paths[f"IG_vs_loss{spec['suffix']}"] = path


    return paths


__all__ = [
    "plot_latent_scatter",
    "plot_decoding_accuracy",
    "plot_IG_hist",
    "plot_loss_hist",
    "plot_IG_vs_loss",
    "plot_loss_hist_pattern",
    "plot_IG_vs_loss_pattern",
    "save_session_figures",
]
