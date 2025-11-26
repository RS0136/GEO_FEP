"""
Robustness analyses for latent geometry, decoding, and information-theoretic metrics.

This module provides *systematic* parameter sweeps around the baseline analysis,
as suggested in your comments and summarised in Section 5.4 of the preprint:

- latent dimensionality d (e.g. 5, 10, 20);
- peri‑stimulus window (e.g. 0–0.2 s vs. 0.1–0.3 s);
- logistic regression regularisation C;
- logistic class weighting (None vs. "balanced").

For each Neuropixels probe insertion and each point in the robustness grid, we:

  1. build TrialData from RawSessionData under a given SessionConfig;
  2. construct latent geometry (FA + Gaussian prototypes);
  3. construct an ideal listener and an actual logistic listener;
  4. compute information gain IG and pragmatic loss ℓ⋆ per trial;
  5. summarise these quantities and decoding accuracies in a table.

The emphasis is on *statistical transparency*:

- every hyperparameter change is explicit;
- splits, priors, and metrics are held fixed except for the parameter under test;
- results are returned as plain NumPy arrays / dicts that can be written to CSV.

No plotting routines are included here; a separate figure module can consume
the summary tables this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, List, Tuple

import logging
import numpy as np

from .config import (
    SessionConfig,
    RobustnessGrid,
    MULTI_SESSION_TAGS,
    default_session_config,
)
from .data_io import RawSessionData, load_raw_session_from_pid
from .trials import construct_trial_data
from .listeners import construct_latent_geometry
from .metrics import (  # ideal + logistic listeners
    ideal_listener_gaussian,
    logistic_actual_listener,
)
from .metrics import compute_trial_metrics  # IG + pragmatic loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for robustness summaries
# ---------------------------------------------------------------------------


@dataclass
class RobustnessRow:
    """
    Single row of a robustness summary table for *one* session and parameter
    setting.

    Each row records the hyperparameters that differ from the baseline, plus
    decoding accuracies and information-theoretic metrics.

    Attributes
    ----------
    session_tag :
        Neuropixels probe identifier (PID string).

    n_trials :
        Number of valid trials analysed under this configuration.

    n_neurons :
        Number of units retained after quality and activity filtering.

    n_latent :
        Effective latent dimension d used by FA under this configuration.

    time_window_start, time_window_end :
        Peri‑stimulus window [t0, t1] in seconds relative to stimulus onset.

    logistic_C :
        Inverse L2 regularisation strength of the logistic decoder.

    logistic_class_weight :
        Class weighting scheme (None or "balanced").

    ideal_accuracy :
        Fraction of trials on which the ideal listener’s MAP estimate matches
        the true world label w.

    actual_test_accuracy :
        Test‑set decoding accuracy of the logistic decoder (actual listener),
        using the same train–test split as in the main analysis.

    IG_mean, IG_std :
        Mean and standard deviation of single‑trial information gains IG(u, c)
        across *all* valid trials under this configuration (nats).

    loss_mean, loss_std :
        Mean and standard deviation of pragmatic loss ℓ⋆(u, c) across all
        trials (nats).

    notes :
        Optional free‑form string (e.g. "baseline", "sweep_d", "sweep_window").
    """

    session_tag: str

    n_trials: int
    n_neurons: int
    n_latent: int

    time_window_start: float
    time_window_end: float

    logistic_C: float
    logistic_class_weight: str | None

    ideal_accuracy: float
    actual_test_accuracy: float

    IG_mean: float
    IG_std: float

    loss_mean: float
    loss_std: float

    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return a plain-Python dict suitable for building a DataFrame/CSV."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core per‑session robustness computation
# ---------------------------------------------------------------------------


def _run_single_config_on_session(
    raw: RawSessionData,
    cfg: SessionConfig,
) -> RobustnessRow:
    """
    Run the *entire* geometric FEP–pragmatics pipeline for one session and a
    given SessionConfig, returning summary statistics.

    Steps (mirroring Section 3 of the preprint):

      1. construct TrialData (world labels, priors, spike matrix X);
      2. construct LatentGeometry (FA latent codes, prototypes, Σ, Σ⁻¹);
      3. build ideal listener posteriors from geometry and priors;
      4. fit logistic decoder (actual listener) on latent codes;
      5. compute IG and ℓ⋆ for all trials.

    Parameters
    ----------
    raw :
        RawSessionData containing spikes, clusters, and trials for one probe.
    cfg :
        SessionConfig specifying analysis hyperparameters.

    Returns
    -------
    row : RobustnessRow
        Summary of decoding accuracies and IG / ℓ⋆ for this configuration.
    """
    # 1. Trial-wise data
    trial_data = construct_trial_data(raw, cfg)

    # 2. Latent geometry (FA + prototypes + Σ⁻¹)
    geom = construct_latent_geometry(trial_data, cfg)

    # 3. Ideal listener from geometry + block priors
    ideal = ideal_listener_gaussian(
        Z=geom.Z,
        w_true=trial_data.w,
        priors_left=trial_data.priors_left,
        mu_left=geom.mu_left,
        mu_right=geom.mu_right,
        Sigma_inv=geom.Sigma_inv,
        cfg=cfg,
    )

    # 4. Actual listener: logistic regression on latent codes
    actual = logistic_actual_listener(
        Z=geom.Z,
        w_true=trial_data.w,
        idx_train=geom.idx_train,
        idx_test=geom.idx_test,
        cfg=cfg,
    )

    # 5. Information gain and pragmatic loss (all trials)
    metrics = compute_trial_metrics(
        priors_left=trial_data.priors_left,
        L_ideal=ideal.posterior,
        L_actual=actual.posterior,
        cfg=cfg,
    )

    IG_mean = float(metrics.IG.mean())
    IG_std = float(metrics.IG.std(ddof=1)) if metrics.IG.size > 1 else 0.0

    loss_mean = float(metrics.loss.mean())
    loss_std = float(metrics.loss.std(ddof=1)) if metrics.loss.size > 1 else 0.0

    t0, t1 = cfg.time_window

    row = RobustnessRow(
        session_tag=trial_data.pid,
        n_trials=trial_data.n_trials,
        n_neurons=trial_data.n_neurons,
        n_latent=geom.d,
        time_window_start=float(t0),
        time_window_end=float(t1),
        logistic_C=cfg.logistic_C,
        logistic_class_weight=cfg.logistic_class_weight,
        ideal_accuracy=float(ideal.accuracy),
        actual_test_accuracy=float(actual.accuracy_test),
        IG_mean=IG_mean,
        IG_std=IG_std,
        loss_mean=loss_mean,
        loss_std=loss_std,
        notes="",
    )
    return row


def _iter_config_variants(
    base_cfg: SessionConfig,
    grid: RobustnessGrid,
) -> Iterable[SessionConfig]:
    """
    Generate SessionConfig variants by overriding selected hyperparameters.

    The resulting configs *only* differ from `base_cfg` in:

      - n_latent  ∈ grid.latent_dims
      - time_window ∈ grid.time_windows
      - logistic_C ∈ grid.logistic_C_values
      - logistic_class_weight ∈ grid.logistic_class_weights

    For each combination we call `cfg.validate()` to ensure theoretical and
    numerical sanity (e.g. valid window, positive C).
    """
    base_cfg.validate()
    grid.validate()

    for d, win, C, cw in product(
        grid.latent_dims,
        grid.time_windows,
        grid.logistic_C_values,
        grid.logistic_class_weights,
    ):
        cfg = SessionConfig(**asdict(base_cfg))
        cfg.n_latent = int(d)
        cfg.time_window = (float(win[0]), float(win[1]))
        cfg.logistic_C = float(C)
        cfg.logistic_class_weight = cw  # None or "balanced"
        cfg.validate()
        yield cfg


# ---------------------------------------------------------------------------
# Public API: robustness for a single session
# ---------------------------------------------------------------------------


def robustness_for_session(
    session_tag: str,
    base_cfg: SessionConfig | None = None,
    grid: RobustnessGrid | None = None,
) -> List[RobustnessRow]:
    """
    Run robustness sweeps for a *single* probe insertion.

    Parameters
    ----------
    session_tag :
        Probe UUID (PID), as listed in config.MULTI_SESSION_TAGS and the
        preprint’s tables (e.g. "fece187f-b47f-4870-a1d6-619afe942a7d_probe01").
    base_cfg :
        Baseline SessionConfig. If None, uses default_session_config with
        this session_tag attached.
    grid :
        RobustnessGrid describing the hyperparameter values to sweep.
        If None, uses default values suggested in the text and in your
        comments (d ∈ {5,10,20}, windows (0–0.2, 0.1–0.3, 0–0.3), C ∈
        {0.1, 1, 10}, class_weight ∈ {None, "balanced"}).

    Returns
    -------
    rows : list of RobustnessRow
        One row per parameter combination, in the order generated by
        `_iter_config_variants`.
    """
    if base_cfg is None:
        base_cfg = default_session_config(session_tag=session_tag)
    if grid is None:
        from .config import default_robustness_grid

        grid = default_robustness_grid()

    # Load raw data only once per session
    raw = load_raw_session_from_pid(session_tag)

    rows: List[RobustnessRow] = []

    for cfg in _iter_config_variants(base_cfg, grid):
        log.info(
            "Running robustness config for %s: n_latent=%d, "
            "time_window=%s, C=%.3g, class_weight=%r",
            session_tag,
            cfg.n_latent,
            cfg.time_window,
            cfg.logistic_C,
            cfg.logistic_class_weight,
        )
        row = _run_single_config_on_session(raw, cfg)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Public API: robustness across all 13 sessions
# ---------------------------------------------------------------------------


def robustness_for_all_sessions(
    session_tags: Iterable[str] | None = None,
    base_cfg: SessionConfig | None = None,
    grid: RobustnessGrid | None = None,
    *,
    latent_dims: Iterable[int] | None = None,
    time_windows: Iterable[Tuple[float, float]] | None = None,
    logistic_C_values: Iterable[float] | None = None,
    class_weight_options: Iterable[str | None] | None = None,
) -> List[RobustnessRow]:
    if session_tags is None:
        session_tags = MULTI_SESSION_TAGS

    if grid is None:
        from .config import default_robustness_grid
        grid = default_robustness_grid()

    if any(v is not None for v in (latent_dims, time_windows,
                                   logistic_C_values, class_weight_options)):
        grid = RobustnessGrid(
            latent_dims=list(latent_dims) if latent_dims is not None else list(grid.latent_dims),
            time_windows=list(time_windows) if time_windows is not None else list(grid.time_windows),
            logistic_C_values=list(logistic_C_values) if logistic_C_values is not None
            else list(grid.logistic_C_values),
            logistic_class_weights=list(class_weight_options) if class_weight_options is not None
            else list(grid.logistic_class_weights),
        )

    grid.validate()

    all_rows: List[RobustnessRow] = []

    for tag in session_tags:
        if base_cfg is not None:
            sess_cfg = SessionConfig(**asdict(base_cfg))
            sess_cfg.session_tag = tag
        else:
            sess_cfg = default_session_config(session_tag=tag)

        rows = robustness_for_session(
            session_tag=tag,
            base_cfg=sess_cfg,
            grid=grid,
        )
        all_rows.extend(rows)

    return all_rows


__all__ = [
    "RobustnessRow",
    "robustness_for_session",
    "robustness_for_all_sessions",
]
