"""
Top-level script: robustness sweeps over latent geometry and decoder parameters.

This module provides a command-line entry point for the robustness analyses described in
Section 5.4 of the preprint: varying latent dimensionality, peri-stimulus time window, and
logistic regression hyperparameters (C and class_weight), and examining their impact on
decoding accuracy, information gain, and pragmatic loss.

Concretely, it:

1. runs `run_robustness_for_all_sessions` on a grid of analysis parameters over all (or a
   specified subset of) Neuropixels probe insertions;
2. writes a global robustness grid `robustness_grid.csv` with one row per configuration;
3. generates a standard battery of robustness figures, both aggregated across sessions and
   separately for each session, using `plots_robustness.save_standard_robustness_figures`.

Typical usage from the package root:

    python -m fep_geo.run_robustness --output-root results

This will create a directory

    results/robustness/

containing the global CSV and one subdirectory per session with PNG plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import MULTI_SESSION_TAGS
from .robustness import RobustnessRow, run_robustness_for_all_sessions
from .plots_robustness import (
    robustness_rows_to_dataframe,
    save_standard_robustness_figures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_time_windows(
    args_time_window: Optional[Sequence[Sequence[float]]],
) -> List[Tuple[float, float]]:
    """
    Parse --time-window arguments into a list of (t0, t1) tuples.

    If `args_time_window` is None, returns the default windows used in the robustness
    checks discussed in the text: (0.0, 0.2) and (0.1, 0.3) seconds.
    """
    if args_time_window is None:
        return [(0.0, 0.2), (0.1, 0.3)]

    windows: List[Tuple[float, float]] = []
    for pair in args_time_window:
        if len(pair) != 2:
            raise ValueError(
                f"Each --time-window must have exactly two floats (t0 t1); got {pair!r}."
            )
        t0, t1 = float(pair[0]), float(pair[1])
        if not t1 > t0:
            raise ValueError(
                f"Invalid time window [t0, t1] = [{t0}, {t1}] (need t1 > t0)."
            )
        windows.append((t0, t1))
    return windows


def _parse_class_weights(
    args_class_weight: Optional[Sequence[str]],
) -> List[Optional[str]]:
    """
    Parse --class-weight arguments into a list of options for logistic_class_weight.

    Recognised values (case-insensitive):
        - 'none' / 'null'   → None
        - 'balanced'        → 'balanced'
        - any other string  → passed through as-is
    """
    if args_class_weight is None:
        # Defaults used in Section 5.4 robustness checks.
        return [None, "balanced"]

    options: List[Optional[str]] = []
    for s in args_class_weight:
        s_lower = s.lower()
        if s_lower in {"none", "null"}:
            options.append(None)
        else:
            options.append(s)
    return options


def _write_global_grid(
    rows: List[RobustnessRow],
    robust_root: Path,
) -> Path:
    """
    Convert a list of RobustnessRow objects into a DataFrame and write robustness_grid.csv.
    """
    df = robustness_rows_to_dataframe(rows)
    csv_path = robust_root / "robustness_grid.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_figures_for_all(
    df: pd.DataFrame,
    robust_root: Path,
    make_per_session: bool = True,
) -> Dict[str, Path]:
    """
    Generate robustness figures aggregated across all sessions and, optionally, per session.

    Aggregated figures (using all rows) are saved under:

        robust_root / "all_sessions"

    Per-session figures are saved under:

        robust_root / {session_tag} /
    """
    paths: Dict[str, Path] = {}

    # Aggregated over all sessions
    all_dir = robust_root / "all_sessions"
    all_dir.mkdir(parents=True, exist_ok=True)
    agg_paths = save_standard_robustness_figures(
        df, output_dir=all_dir, session_tag=None, close=True
    )
    for key, path in agg_paths.items():
        paths[f"all_{key}"] = path

    if make_per_session:
        for session_tag in df["session_tag"].unique():
            sess_df = df[df["session_tag"] == session_tag]
            sess_dir = robust_root / session_tag
            sess_dir.mkdir(parents=True, exist_ok=True)
            sess_paths = save_standard_robustness_figures(
                sess_df,
                output_dir=sess_dir,
                session_tag=session_tag,
                close=True,
            )
            for key, path in sess_paths.items():
                paths[f"{session_tag}_{key}"] = path

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """
    Command-line entry point for robustness sweeps.

    Steps:

      1. Build a grid over latent dimensionality, time window, logistic C, and class_weight.
      2. Run `run_robustness_for_all_sessions` over all (or selected) sessions.
      3. Write the combined robustness grid as `robustness_grid.csv`.
      4. Generate standard robustness plots, both aggregated and per session.

    This directly targets the robustness checks requested in the manuscript and reviews:
    stability of IG and decoding with respect to latent dimension and time window, and
    sensitivity of logistic decoder performance to C and class_weight.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run robustness sweeps over latent geometry and logistic-decoder parameters."
        )
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=str,
        default="results",
        help=(
            "Root directory for outputs. Robustness results will be written under "
            "'{output_root}/robustness/'. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--session-tags",
        nargs="+",
        default=None,
        help=(
            "Optional subset of session tags (PIDs) to analyse. "
            "If omitted, all entries in config.MULTI_SESSION_TAGS are used."
        ),
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help=(
            "Latent dimensionalities d to sweep over in factor analysis "
            "(default: 5 10 20)."
        ),
    )
    parser.add_argument(
        "--time-window",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        action="append",
        help=(
            "Peri-stimulus time window [t0, t1] in seconds. "
            "Can be supplied multiple times to define a set of windows, "
            "e.g. --time-window 0.0 0.2 --time-window 0.1 0.3. "
            "If omitted, defaults to 0.0–0.2 and 0.1–0.3."
        ),
    )
    parser.add_argument(
        "--logistic-C",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help=(
            "Values of the logistic regularisation parameter C (inverse L2 strength) "
            "to sweep over (default: 0.1 1.0 10.0)."
        ),
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        nargs="+",
        default=["none", "balanced"],
        help=(
            "Options for logistic class_weight. Recognised values: 'none', 'balanced', "
            "or any string accepted by scikit-learn. "
            "Multiple values can be supplied (default: none balanced)."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, skip generating robustness figures and only write the CSV grid.",
    )
    parser.add_argument(
        "--no-per-session-plots",
        action="store_true",
        help=(
            "If set, generate only aggregated 'all_sessions' robustness figures and skip "
            "per-session plots."
        ),
    )

    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    robust_root = output_root / "robustness"
    robust_root.mkdir(parents=True, exist_ok=True)

    if args.session_tags is None:
        session_tags = list(MULTI_SESSION_TAGS)
        print(
            f"[run_robustness] No session tags specified; using all "
            f"{len(session_tags)} entries from MULTI_SESSION_TAGS."
        )
    else:
        session_tags = list(args.session_tags)
        print(
            f"[run_robustness] Restricting analysis to {len(session_tags)} "
            f"user-specified session tags."
        )

    latent_dims = sorted(set(int(d) for d in args.latent_dims))
    time_windows = _parse_time_windows(args.time_window)
    logistic_C_values = sorted(set(float(c) for c in args.logistic_C))
    class_weight_options = _parse_class_weights(args.class_weight)

    print("[run_robustness] Configuration grid:")
    print(f"  sessions          : {len(session_tags)}")
    print(f"  latent_dims       : {latent_dims}")
    print(f"  time_windows      : {time_windows}")
    print(f"  logistic_C_values : {logistic_C_values}")
    print(f"  class_weight_opts : {class_weight_options}")

    # ------------------------------------------------------------------
    # 1–2. Run robustness sweeps for all sessions
    # ------------------------------------------------------------------
    print("[run_robustness] Running robustness sweeps (this may take a while)...")
    rows = run_robustness_for_all_sessions(
        session_tags=session_tags,
        base_cfg=None,
        latent_dims=latent_dims,
        time_windows=time_windows,
        logistic_C_values=logistic_C_values,
        class_weight_options=class_weight_options,
    )
    print(f"[run_robustness] Completed {len(rows)} configurations.")

    # ------------------------------------------------------------------
    # 3. Global CSV grid
    # ------------------------------------------------------------------
    print("[run_robustness] Writing robustness_grid.csv...")
    grid_csv_path = _write_global_grid(rows, robust_root)
    print(f"[run_robustness] Wrote global grid: {grid_csv_path}")

    if args.no_plots:
        print("[run_robustness] Skipping figure generation (--no-plots set).")
        print(f"[run_robustness] Done. Outputs under: {robust_root.resolve()}")
        return

    # ------------------------------------------------------------------
    # 4. Robustness figures (aggregated + per session)
    # ------------------------------------------------------------------
    print("[run_robustness] Generating robustness figures...")
    df = robustness_rows_to_dataframe(rows)
    make_per_session = not args.no_per_session_plots

    paths = _make_figures_for_all(
        df=df,
        robust_root=robust_root,
        make_per_session=make_per_session,
    )

    print(f"[run_robustness] Generated {len(paths)} figure files.")
    print(f"[run_robustness] Done. Outputs under: {robust_root.resolve()}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
