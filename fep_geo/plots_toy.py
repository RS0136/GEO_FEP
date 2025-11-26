"""
Toy plots to illustrate the scale of information gain and pragmatic loss.

This module provides *didactic* figures to accompany the explanations of information
gain IG and pragmatic loss ℓ* in the text (Section 2.6 and the Results discussion).

Goals
-----
- Show, in a binary world-state example, what an IG of ~0.1 nats means in terms of
  prior/posterior probabilities.
- Show, again in a binary case, what a pragmatic loss ℓ* of ~0.1 nats corresponds to
  in terms of a mismatch between an ideal and an actual listener posterior.

The functions below work purely with simple Bernoulli distributions:
    P(W = 1 | c) = p
and visualise:
    IG = H(prior) − H(posterior)
    ℓ* = KL(ideal || actual)

measured in nats, as in the main theory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Basic binary information-theoretic helpers
# ---------------------------------------------------------------------------


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    """
    Binary entropy H(p) in nats for Bernoulli(p).

    H(p) = − [p log p + (1 − p) log(1 − p)].

    Parameters
    ----------
    p :
        Probability of W = 1. Can be scalar or NumPy array.

    Returns
    -------
    H :
        Entropy in nats, same shape as p.
    """
    p_arr = np.asarray(p, dtype=float)
    eps = 1e-12
    p_clip = np.clip(p_arr, eps, 1.0 - eps)

    H = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))
    if np.isscalar(p):
        return float(H)
    return H


def information_gain_binary(p_prior: float, p_post: np.ndarray | float) -> np.ndarray | float:
    """
    Information gain IG = H(prior) − H(posterior) for a binary world state.

    Parameters
    ----------
    p_prior :
        Prior probability P(W = 1 | c).
    p_post :
        Posterior probability P(W = 1 | u, c); scalar or array.

    Returns
    -------
    IG :
        Information gain in nats, same shape as p_post.
    """
    H_prior = binary_entropy(p_prior)
    H_post = binary_entropy(p_post)
    return H_prior - H_post


def kl_binary(p_ideal: float, p_actual: np.ndarray | float) -> np.ndarray | float:
    """
    Binary KL divergence KL(ideal || actual) in nats.

    Parameters
    ----------
    p_ideal :
        Ideal posterior P_ideal(W = 1 | u, c).
    p_actual :
        Actual posterior P_0(W = 1 | u, c); scalar or array.

    Returns
    -------
    KL :
        Kullback–Leibler divergence in nats, same shape as p_actual.
    """
    eps = 1e-12
    p = np.clip(np.asarray(p_ideal, dtype=float), eps, 1.0 - eps)
    q = np.clip(np.asarray(p_actual, dtype=float), eps, 1.0 - eps)

    p0 = 1.0 - p
    q0 = 1.0 - q

    KL = p * np.log(p / q) + p0 * np.log(p0 / q0)
    if np.isscalar(p_actual):
        return float(KL)
    return KL


# ---------------------------------------------------------------------------
# Toy plots for information gain
# ---------------------------------------------------------------------------


def plot_IG_curve_binary(
    p_prior: float = 0.5,
    ax: Optional[plt.Axes] = None,
    highlight_posterior: float = 0.75,
) -> plt.Axes:
    """
    Plot IG as a function of posterior P(W = 1 | u, c) for a fixed prior.

    This figure is useful for text that explains, e.g.,

        "For a binary prior of 0.5, an IG of ~0.1 nats corresponds to shifting the
         ideal posterior from 0.5 towards ~0.7–0.8."

    Parameters
    ----------
    p_prior :
        Prior probability P(W = 1 | c). Default 0.5 (maximally uncertain).
    ax :
        Optional matplotlib Axes; if None, a new figure and axes are created.
    highlight_posterior :
        Posterior value to highlight with a marker and annotation.

    Returns
    -------
    ax :
        Axes with IG(p_post) curve and annotation.
    """
    p_post = np.linspace(0.01, 0.99, 400)
    IG = information_gain_binary(p_prior, p_post)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(p_post, IG)
    ax.set_xlabel("Posterior P(W = 1 | u, c)")
    ax.set_ylabel("Information gain IG (nats)")
    ax.set_title(f"Binary information gain for prior P(W = 1 | c) = {p_prior:.2f}")

    # Highlight a specific posterior value
    if 0.0 < highlight_posterior < 1.0:
        IG_h = float(information_gain_binary(p_prior, highlight_posterior))
        ax.scatter([highlight_posterior], [IG_h], zorder=3)
        ax.axhline(IG_h, linestyle=":", linewidth=0.8)
        ax.annotate(
            f"IG ≈ {IG_h:.3f} nats\n(posterior ≈ {highlight_posterior:.2f})",
            xy=(highlight_posterior, IG_h),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


def plot_IG_prior_posterior_bars(
    p_prior: float = 0.5,
    p_post: float = 0.75,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Bar plot of prior vs posterior probabilities for a binary world state,
    with IG annotated.

    This figure makes the *shape* of the belief update concrete: a prior
    (p_prior, 1 − p_prior) vs a posterior (p_post, 1 − p_post).
    """
    if ax is None:
        fig, ax = plt.subplots()

    IG_val = float(information_gain_binary(p_prior, p_post))

    # x positions: two groups (prior, posterior), each with two bars (W=0, W=1)
    x0 = np.array([0.0, 1.0])
    width = 0.35

    probs_prior = np.array([1.0 - p_prior, p_prior])
    probs_post = np.array([1.0 - p_post, p_post])

    ax.bar(x0 - width / 2, probs_prior, width=width, label="Prior")
    ax.bar(x0 + width / 2, probs_post, width=width, label="Posterior")

    ax.set_xticks(x0)
    ax.set_xticklabels(["W = 0", "W = 1"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Prior vs posterior (IG ≈ {IG_val:.3f} nats)\n"
        f"prior={p_prior:.2f}, posterior={p_post:.2f}"
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    return ax


# ---------------------------------------------------------------------------
# Toy plots for pragmatic loss
# ---------------------------------------------------------------------------


def plot_loss_curve_binary(
    p_ideal: float = 0.85,
    ax: Optional[plt.Axes] = None,
    highlight_actual: float = 0.65,
) -> plt.Axes:
    """
    Plot pragmatic loss ℓ* = KL(ideal || actual) as a function of the actual
    posterior P_0(W = 1 | u, c), for a fixed ideal posterior.

    A useful choice is p_ideal ≈ 0.85 and highlight_actual ≈ 0.65; this yields
    ℓ* ≈ 0.1 nats and can be used in the text to say:

        "A pragmatic loss of roughly 0.1 nats corresponds, in a binary case,
         to an actual listener that assigns about 20 percentage points less
         confidence to the true state than the ideal listener does."
    """
    p_actual = np.linspace(0.01, 0.99, 400)
    KL = kl_binary(p_ideal, p_actual)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(p_actual, KL)
    ax.set_xlabel("Actual posterior P₀(W = 1 | u, c)")
    ax.set_ylabel("Pragmatic loss ℓ* (nats)")
    ax.set_title(f"Binary pragmatic loss for ideal P(W = 1 | u, c) = {p_ideal:.2f}")

    # Highlight a particular actual value
    if 0.0 < highlight_actual < 1.0:
        KL_h = float(kl_binary(p_ideal, highlight_actual))
        ax.scatter([highlight_actual], [KL_h], zorder=3)
        ax.axhline(KL_h, linestyle=":", linewidth=0.8)
        ax.annotate(
            f"ℓ* ≈ {KL_h:.3f} nats\n(actual ≈ {highlight_actual:.2f})",
            xy=(highlight_actual, KL_h),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    return ax


def plot_loss_ideal_actual_bars(
    p_ideal: float = 0.85,
    p_actual: float = 0.65,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Bar plot comparing ideal and actual posteriors for a binary world state,
    with ℓ* annotated.

    This directly visualises the mismatch that yields a given pragmatic loss.
    """
    if ax is None:
        fig, ax = plt.subplots()

    KL_val = float(kl_binary(p_ideal, p_actual))

    x = np.array([0.0, 1.0])
    width = 0.35

    probs_ideal = np.array([1.0 - p_ideal, p_ideal])
    probs_actual = np.array([1.0 - p_actual, p_actual])

    ax.bar(x - width / 2, probs_ideal, width=width, label="Ideal listener")
    ax.bar(x + width / 2, probs_actual, width=width, label="Actual listener")

    ax.set_xticks(x)
    ax.set_xticklabels(["W = 0", "W = 1"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Ideal vs actual posterior (ℓ* ≈ {KL_val:.3f} nats)\n"
        f"ideal={p_ideal:.2f}, actual={p_actual:.2f}"
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    return ax


# ---------------------------------------------------------------------------
# High-level helper: save standard toy figures
# ---------------------------------------------------------------------------


def save_toy_figures(
    output_dir: str | Path,
    close: bool = True,
) -> Dict[str, Path]:
    """
    Save a standard pair of toy figures illustrating IG and ℓ*.

    The function writes:

        toy_IG_curve.png       – IG vs posterior for prior 0.5,
        toy_IG_bars.png        – prior vs posterior bars with IG annotation,
        toy_loss_curve.png     – ℓ* vs actual posterior for ideal 0.85,
        toy_loss_bars.png      – ideal vs actual bars with ℓ* annotation.

    These are intended for use in methods/appendix sections that give an intuitive
    sense of what values like IG ≈ 0.1 nats or ℓ* ≈ 0.1 nats *mean* in simple
    binary examples.

    Parameters
    ----------
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

    paths: Dict[str, Path] = {}

    # IG curve (prior 0.5)
    fig, ax = plt.subplots()
    plot_IG_curve_binary(p_prior=0.5, highlight_posterior=0.75, ax=ax)
    p = out_path / "toy_IG_curve.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["IG_curve"] = p

    # IG bars
    fig, ax = plt.subplots()
    plot_IG_prior_posterior_bars(p_prior=0.5, p_post=0.75, ax=ax)
    p = out_path / "toy_IG_bars.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["IG_bars"] = p

    # ℓ* curve (ideal 0.85)
    fig, ax = plt.subplots()
    plot_loss_curve_binary(p_ideal=0.85, highlight_actual=0.65, ax=ax)
    p = out_path / "toy_loss_curve.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["loss_curve"] = p

    # ℓ* bars
    fig, ax = plt.subplots()
    plot_loss_ideal_actual_bars(p_ideal=0.85, p_actual=0.65, ax=ax)
    p = out_path / "toy_loss_bars.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    if close:
        plt.close(fig)
    paths["loss_bars"] = p

    return paths


__all__ = [
    "binary_entropy",
    "information_gain_binary",
    "kl_binary",
    "plot_IG_curve_binary",
    "plot_IG_prior_posterior_bars",
    "plot_loss_curve_binary",
    "plot_loss_ideal_actual_bars",
    "save_toy_figures",
]
