"""
Top-level script: generate toy binary IG / ℓ* examples and figures.

This module is a small, self-contained entry point for the *didactic* examples
of information gain IG and pragmatic loss ℓ* described around Section 2.6 of
the preprint.

It does two things:

1. **Print numeric toy examples** for a binary world state:
   - IG(p_prior → p_post) for a grid of posterior probabilities, in nats and bits;
   - ℓ*(p_ideal ∥ p_actual) for a grid of actual posteriors, again in nats and bits.
   These are useful for text like “IG ≈ 0.1 nats corresponds to shifting a
   binary posterior from 0.5 towards ~0.7–0.8” or “ℓ* ≈ 0.1 nats corresponds
   to an actual listener being ~20 percentage points less confident than the
   ideal listener.”

2. **Generate the toy figures** implemented in `plots_toy.py`:
   - `toy_IG_curve.png`
   - `toy_IG_bars.png`
   - `toy_loss_curve.png`
   - `toy_loss_bars.png`

Usage (from the project root, where `fep_geo` is importable):

    python -m fep_geo.run_toy_ig_examples --output-root results

See `main()` for optional CLI arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .plots_toy import (
    information_gain_binary,
    kl_binary,
    save_toy_figures,
)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_ig_examples(
    p_prior: float,
    posterior_grid: Sequence[float],
) -> None:
    """
    Print a small table of IG values for a fixed prior and a grid of posteriors.
    """
    print(
        "[run_toy_ig_examples] Binary information gain examples "
        f"(prior p = {p_prior:.3f})"
    )
    print("  posterior    IG [nats]    IG [bits]")
    for p_post in posterior_grid:
        ig = float(information_gain_binary(p_prior, p_post))
        ig_bits = ig / float(np.log(2.0))
        print(f"    {p_post:8.3f}   {ig:9.3f}   {ig_bits:10.3f}")
    print()


def _print_loss_examples(
    p_ideal: float,
    actual_grid: Sequence[float],
) -> None:
    """
    Print a small table of ℓ* values for a fixed ideal posterior and a grid of
    actual posteriors.
    """
    print(
        "[run_toy_ig_examples] Binary pragmatic loss examples "
        f"(ideal p = {p_ideal:.3f})"
    )
    print("   actual    ℓ* [nats]   ℓ* [bits]")
    for p_act in actual_grid:
        loss = float(kl_binary(p_ideal, p_act))
        loss_bits = loss / float(np.log(2.0))
        print(f"    {p_act:8.3f}   {loss:9.3f}   {loss_bits:9.3f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    """
    Command-line entry point for generating toy IG / ℓ* examples.

    Steps:
      1. Generate and print numeric examples for:
           - IG(p_prior → p_post) on a grid of posteriors;
           - ℓ*(p_ideal ∥ p_actual) on a grid of actual posteriors.
      2. Optionally (default: yes) write the four toy figures defined in
         `plots_toy.save_toy_figures` under `{output_root}/toy/`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate toy binary examples for information gain IG and "
            "pragmatic loss ℓ*, and save illustrative figures."
        )
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=str,
        default="results",
        help=(
            "Root directory for outputs (toy figures will be written under "
            "'{output_root}/toy'). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="Binary prior p = P(W = 1 | c) used in IG examples (default: 0.5).",
    )
    parser.add_argument(
        "--posterior-grid",
        type=float,
        nargs="+",
        default=[0.60, 0.70, 0.75, 0.80, 0.90],
        help=(
            "List of posterior probabilities P(W = 1 | u, c) for which to "
            "compute IG (default: 0.60 0.70 0.75 0.80 0.90)."
        ),
    )
    parser.add_argument(
        "--ideal",
        type=float,
        default=0.85,
        help=(
            "Ideal posterior p_ideal = P_ideal(W = 1 | u, c) used in ℓ* "
            "examples (default: 0.85)."
        ),
    )
    parser.add_argument(
        "--actual-grid",
        type=float,
        nargs="+",
        default=[0.55, 0.65, 0.70, 0.75, 0.80],
        help=(
            "List of actual posteriors P₀(W = 1 | u, c) for which to compute "
            "ℓ* (default: 0.55 0.65 0.70 0.75 0.80)."
        ),
    )
    parser.add_argument(
        "--no-figs",
        action="store_true",
        help="If set, skip saving PNG toy figures and only print numeric tables.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # 1. Numeric tables
    _print_ig_examples(args.prior, args.posterior_grid)
    _print_loss_examples(args.ideal, args.actual_grid)

    # 2. Figures
    if not args.no_figs:
        output_root = Path(args.output_root)
        toy_dir = output_root / "toy"
        toy_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[run_toy_ig_examples] Saving toy IG / ℓ* figures under: "
            f"{toy_dir.resolve()}"
        )
        paths = save_toy_figures(toy_dir, close=True)
        for key, path in paths.items():
            print(f"[run_toy_ig_examples]  {key}: {path}")
    else:
        print("[run_toy_ig_examples] Skipping figure generation (--no-figs).")

    print("[run_toy_ig_examples] Done.")


if __name__ == "__main__":
    main()


__all__ = ["main"]
