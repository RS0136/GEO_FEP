"""
Top-level script: run the full geometric FEP–pragmatics pipeline on all sessions.

This module provides a simple command-line entry point that:

1. runs the full analysis for all Neuropixels probe insertions listed in
   `config.MULTI_SESSION_TAGS` (Section 3.1);
2. saves per-session tables and figures (Section 3.10);
3. builds cross-session aggregate tables and figures (Sections 3.10–3.11);
4. optionally runs robustness sweeps and plots (Section 5.4);
5. generates toy figures that explain the scale of IG and ℓ* (Section 2.6).

Usage
-----
From the package root (where this module is importable as `fep_geo.run_all_sessions`),
you can run:

    python -m fep_geo.run_all_sessions --output-root results

Command-line options are documented in `main()` below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import MULTI_SESSION_TAGS, SessionConfig, default_session_config
from .aggregate import (
    SessionResult,
    make_context_summary_rows,
    make_session_summary_row,
    run_multi_session,
    write_tables,
)
from .plots_session import save_session_figures
from .plots_cross_session import save_cross_session_figures
from .plots_toy import save_toy_figures


# ---------------------------------------------------------------------------
# Helpers for per-session outputs
# ---------------------------------------------------------------------------


def _write_session_tables(
    result: SessionResult,
    tables_dir: Path,
) -> Dict[str, Path]:
    """
    Write per-session summary and context tables for a single SessionResult.

    This mirrors the per-session CSV tables described in Section 3.10: one summary
    row (summary_{session_tag}.csv) and one context table
    (by_context_{session_tag}.csv).
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # One-row summary table
    summary_row = make_session_summary_row(result)
    df_summary = pd.DataFrame([summary_row.to_dict()])
    p_summary = tables_dir / f"summary_{result.session_tag}.csv"
    df_summary.to_csv(p_summary, index=False)
    paths["summary"] = p_summary

    # By-context table (one row per block prior in this session)
    ctx_rows = make_context_summary_rows(result)
    df_ctx = pd.DataFrame([r.to_dict() for r in ctx_rows])
    p_ctx = tables_dir / f"by_context_{result.session_tag}.csv"
    df_ctx.to_csv(p_ctx, index=False)
    paths["by_context"] = p_ctx

    return paths


def _write_all_session_outputs(
    results: List[SessionResult],
    sessions_root: Path,
    make_figures: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    For each SessionResult, create a subdirectory and write tables and figures.

    Directory layout:

        sessions_root/
            {session_tag}/
                tables/
                    summary_{session_tag}.csv
                    by_context_{session_tag}.csv
                figures/
                    {session_tag}_latent_scatter.png
                    {session_tag}_decoding_accuracy.png
                    {session_tag}_IG_hist.png
                    {session_tag}_loss_hist.png
                    {session_tag}_IG_vs_loss.png
    """
    outputs: Dict[str, Dict[str, Path]] = {}

    for res in results:
        sess_dir = sessions_root / res.session_tag
        tables_dir = sess_dir / "tables"
        figures_dir = sess_dir / "figures"

        paths: Dict[str, Path] = {}
        paths.update(_write_session_tables(res, tables_dir))

        if make_figures:
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig_paths = save_session_figures(
                res, output_dir=figures_dir, prefix=res.session_tag
            )
            # Prefix keys with 'fig_' to avoid collisions
            for k, v in fig_paths.items():
                paths[f"fig_{k}"] = v

        outputs[res.session_tag] = paths

    return outputs


# ---------------------------------------------------------------------------
# Optional robustness sweeps
# ---------------------------------------------------------------------------


def _run_and_save_robustness(
    output_root: Path,
    latent_dims: Optional[list[int]] = None,
    time_windows: Optional[list[tuple[float, float]]] = None,
    logistic_C_values: Optional[list[float]] = None,
    class_weight_options: Optional[list[Optional[str]]] = None,
) -> Path:
    """
    Run robustness sweeps over all sessions and save a global grid + per-session plots.

    This function assumes that `robustness.py` exposes
    `run_robustness_for_all_sessions` and that `plots_robustness.py` exposes
    `robustness_rows_to_dataframe` and `save_standard_robustness_figures`,
    as implemented earlier. It is a thin convenience wrapper.

    Returns
    -------
    grid_csv_path :
        Path to the global robustness CSV table.
    """
    from .robustness import robustness_for_all_sessions
    from .plots_robustness import (
        robustness_rows_to_dataframe,
        save_standard_robustness_figures,
    )

    robust_root = output_root / "robustness"
    robust_root.mkdir(parents=True, exist_ok=True)

    # Defaults chosen to match the robustness checks requested in the comments:
    #   latent_dims  ∈ {5, 10, 20}
    #   time_windows ∈ {(0.0, 0.2), (0.1, 0.3)}
    #   logistic_C   ∈ {0.1, 1.0, 10.0}
    #   class_weight ∈ {None, "balanced"}
    if latent_dims is None:
        latent_dims = [5, 10, 20]
    if time_windows is None:
        time_windows = [(0.0, 0.2), (0.1, 0.3)]
    if logistic_C_values is None:
        logistic_C_values = [0.1, 1.0, 10.0]
    if class_weight_options is None:
        class_weight_options = [None, "balanced"]

    rows = robustness_for_all_sessions(
        session_tags=MULTI_SESSION_TAGS,
        base_cfg=None,
        latent_dims=latent_dims,
        time_windows=time_windows,
        logistic_C_values=logistic_C_values,
        class_weight_options=class_weight_options,
    )

    df_robust = robustness_rows_to_dataframe(rows)
    grid_csv = robust_root / "robustness_grid.csv"
    df_robust.to_csv(grid_csv, index=False)

    # Per-session robustness figures
    for session_tag in df_robust["session_tag"].unique():
        sess_df = df_robust[df_robust["session_tag"] == session_tag]
        base_cfg: SessionConfig = default_session_config(session_tag=session_tag)
        save_standard_robustness_figures(
            sess_df,
            output_dir=robust_root / session_tag,
            session_tag=session_tag,
            close=True,
        )

    return grid_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """
    Command-line entry point.

    Parameters
    ----------
    argv :
        Optional list of command-line arguments (excluding the program name).
        If None, `sys.argv[1:]` is used.

    The script performs the following steps:

      1. runs the full multi-session analysis (ideal/actual listeners, IG, ℓ*);
      2. writes per-session tables and figures;
      3. writes cross-session tables and figures;
      4. generates toy figures explaining the scale of IG/ℓ*;
      5. optionally runs robustness sweeps and associated plots.
    """
    parser = argparse.ArgumentParser(
        description="Run the geometric FEP–pragmatics pipeline on all sessions."
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=str,
        default="results",
        help="Root directory for all outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--no-session-figs",
        action="store_true",
        help="Skip per-session figures (latent scatter, accuracy, IG/loss histograms).",
    )
    parser.add_argument(
        "--no-cross-figs",
        action="store_true",
        help="Skip cross-session figures (accuracy/IG/loss summary plots).",
    )
    parser.add_argument(
        "--no-toy-figs",
        action="store_true",
        help="Skip toy IG/ℓ* figures.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness sweeps over latent dimension, time window, "
        "logistic C, and class_weight.",
    )

    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    sessions_root = output_root / "sessions"
    aggregate_root = output_root / "aggregate"

    sessions_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Multi-session analysis (ideal + actual listeners, IG, ℓ*)
    # ------------------------------------------------------------------
    print("[run_all_sessions] Running multi-session analysis...")
    results, tables = run_multi_session(
        session_tags=MULTI_SESSION_TAGS,
        base_cfg=None,
        do_prior_control=True,
    )

    # ------------------------------------------------------------------
    # 2. Per-session tables and figures
    # ------------------------------------------------------------------
    print("[run_all_sessions] Writing per-session tables and figures...")
    _write_all_session_outputs(
        results,
        sessions_root=sessions_root,
        make_figures=not args.no_session_figs,
    )

    # ------------------------------------------------------------------
    # 3. Cross-session tables and figures
    # ------------------------------------------------------------------
    print("[run_all_sessions] Writing cross-session tables...")
    aggregate_tables_dir = aggregate_root / "tables"
    write_tables(
        tables=tables,
        output_dir=aggregate_tables_dir,
        write_tex=False,  # user can enable LaTeX externally if desired
    )

    if not args.no_cross_figs:
        print("[run_all_sessions] Creating cross-session figures...")
        aggregate_figs_dir = aggregate_root / "figures"
        aggregate_figs_dir.mkdir(parents=True, exist_ok=True)
        save_cross_session_figures(tables, output_dir=aggregate_figs_dir, close=True)

    # ------------------------------------------------------------------
    # 4. Toy IG / ℓ* figures
    # ------------------------------------------------------------------
    if not args.no-toy_figs if hasattr(args, "no-toy_figs") else not args.no_toy_figs:
        # The conditional above is purely defensive; argparse guarantees the attribute.
        print("[run_all_sessions] Generating toy IG / ℓ* figures...")
        toy_dir = output_root / "toy"
        toy_dir.mkdir(parents=True, exist_ok=True)
        save_toy_figures(toy_dir, close=True)

    # ------------------------------------------------------------------
    # 5. Optional robustness sweeps
    # ------------------------------------------------------------------
    if args.robustness:
        print("[run_all_sessions] Running robustness sweeps (this may take a while)...")
        _run_and_save_robustness(output_root=output_root)
        print("[run_all_sessions] Robustness results saved under 'robustness/'.")

    print(f"[run_all_sessions] Done. Outputs written under: {output_root.resolve()}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
