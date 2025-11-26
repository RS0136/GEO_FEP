"""
Top-level script: compute “extras” summaries (test-only metrics, prior-aware control,
and behaviour link) across all sessions.

This module corresponds to the additional analyses described in Section 3.11 of the
preprint: test-only information gain and pragmatic loss, a prior-aware control decoder,
and session-wise links between neural communication metrics and behavioural accuracy.

It is intended as a lighter-weight entry point than `run_all_sessions.py` when you only
need the extra summary tables, not per-session figures.

Usage
-----
From the package root (where this module is importable as `fep_geo.run_extras`), run:

    python -m fep_geo.run_extras --output-root results

This will:

1. run the full multi-session analysis (ideal & actual listeners, IG, ℓ*) with
   a prior-aware control decoder,
2. write the following cross-session tables under
       {output_root}/aggregate/tables/
       - test_only_summary.csv       (Table 5 in the preprint)
       - prior_control_summary.csv   (Table 6)
       - behaviour_link.csv          (Table 2)
3. optionally write matching LaTeX tables (.tex) for inclusion in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import MULTI_SESSION_TAGS
from .aggregate import run_multi_session


# ---------------------------------------------------------------------------
# Helpers for writing tables
# ---------------------------------------------------------------------------


def _write_extras_tables(
    tables: Dict[str, pd.DataFrame],
    output_root: Path,
    write_tex: bool = False,
    float_format: str = "%.3f",
) -> Dict[str, Path]:
    """
    Write the three “extras” tables to CSV (and optionally LaTeX).

    Files are written under:

        {output_root}/aggregate/tables/
            test_only_summary.csv
            prior_control_summary.csv  (if available)
            behaviour_link.csv

    Parameters
    ----------
    tables :
        Dict of table name → DataFrame as returned by `run_multi_session`.
    output_root :
        Root directory for outputs.
    write_tex :
        If True, also write `.tex` versions of each table.
    float_format :
        Format string for floats in LaTeX output.

    Returns
    -------
    paths :
        Dict mapping table name → path to the CSV file.
    """
    tables_dir = output_root / "aggregate" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    to_write: Dict[str, pd.DataFrame] = {}

    # Always expected
    if "test_only_summary" in tables:
        to_write["test_only_summary"] = tables["test_only_summary"]
    else:
        raise KeyError(
            "tables dict does not contain 'test_only_summary'; "
            "did run_multi_session run with test metrics enabled?"
        )

    if "behaviour_link" in tables:
        to_write["behaviour_link"] = tables["behaviour_link"]
    else:
        raise KeyError(
            "tables dict does not contain 'behaviour_link'; "
            "behaviour summary appears to be missing."
        )

    # Prior-aware summary is present only if prior-aware decoder was run
    if "prior_control_summary" in tables:
        to_write["prior_control_summary"] = tables["prior_control_summary"]

    paths: Dict[str, Path] = {}

    for name, df in to_write.items():
        csv_path = tables_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        paths[name] = csv_path

        if write_tex:
            tex_path = tables_dir / f"{name}.tex"
            df.to_latex(tex_path, index=False, float_format=float_format)

    return paths


def _print_global_extras_stats(tables: Dict[str, pd.DataFrame]) -> None:
    """
    Print a brief numeric summary of the extras, mirroring Eqs. (107)–(110).

    This is purely informational and does not affect saved outputs.
    """
    test_df = tables.get("test_only_summary", None)
    prior_df = tables.get("prior_control_summary", None)

    if test_df is not None:
        # Trial-weighted means across sessions
        n_test = test_df["n_test"].to_numpy(dtype=float)
        IG_t = test_df["IG_test_mean"].to_numpy(dtype=float)
        loss_t = test_df["loss_test_mean"].to_numpy(dtype=float)

        IG_test = float(np.sum(n_test * IG_t) / np.sum(n_test))
        loss_test = float(np.sum(n_test * loss_t) / np.sum(n_test))

        print(
            f"[run_extras] Trial-weighted test-only means: "
            f"IG_test ≈ {IG_test:.3f} nats, "
            f"loss_test ≈ {loss_test:.3f} nats"
        )

    if prior_df is not None:
        loss_p = prior_df["loss_prior_mean"].to_numpy(dtype=float)
        loss_prior_mean = float(loss_p.mean())
        loss_prior_std = float(loss_p.std(ddof=1)) if loss_p.size > 1 else 0.0

        print(
            "[run_extras] Prior-aware pragmatic loss: "
            f"mean ≈ {loss_prior_mean:.3f} ± {loss_prior_std:.3f} nats"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """
    Command-line entry point for running the “extras” analyses.

    Steps:

      1. Run `run_multi_session` over `config.MULTI_SESSION_TAGS` with
         `do_prior_control=True`, so that both the main and prior-aware
         actual listeners are fitted.
      2. Extract the extras tables from the returned `tables` dict:
           - test_only_summary
           - prior_control_summary (if present)
           - behaviour_link
      3. Write them to CSV (and optionally LaTeX) under
         `{output_root}/aggregate/tables`.
      4. Print brief global statistics (trial-weighted means) as a sanity check.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute extras: test-only IG/ℓ*, prior-aware control decoder, "
            "and behaviour–neural link tables."
        )
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=str,
        default="results",
        help="Root directory for outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--no-prior-control",
        action="store_true",
        help=(
            "Skip fitting the prior-aware control decoder. "
            "In this case, 'prior_control_summary' will not be written."
        ),
    )
    parser.add_argument(
        "--tex",
        action="store_true",
        help="Also write LaTeX (.tex) versions of the extras tables.",
    )

    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Run multi-session analysis (ideal + actual + prior-aware)
    # ------------------------------------------------------------------
    print("[run_extras] Running multi-session analysis to compute extras...")
    results, tables = run_multi_session(
        session_tags=MULTI_SESSION_TAGS,
        base_cfg=None,
        do_prior_control=not args.no_prior_control,
    )

    # ------------------------------------------------------------------
    # 2. Write extras tables
    # ------------------------------------------------------------------
    print("[run_extras] Writing extras tables (test-only, prior-control, behaviour)...")
    paths = _write_extras_tables(
        tables=tables,
        output_root=output_root,
        write_tex=args.tex,
        float_format="%.3f",
    )

    for name, path in paths.items():
        print(f"[run_extras] Wrote {name}: {path}")

    # ------------------------------------------------------------------
    # 3. Print brief global stats
    # ------------------------------------------------------------------
    _print_global_extras_stats(tables)

    print(
        f"[run_extras] Done. Extras written under: "
        f"{(output_root / 'aggregate' / 'tables').resolve()}"
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
