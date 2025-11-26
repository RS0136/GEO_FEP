"""
Behavioural summaries and neural–behaviour link analyses.

This module implements the "behaviour link" part of the pipeline:

- per-session behavioural summaries (behavioural accuracy and
  IG / pragmatic loss split by correct vs. error trials), as reported
  in Table 2;
- cross-session summaries of these quantities (session-wise means and
  standard deviations of the differences IG_correct − IG_error and
  loss_correct − loss_error);
- cross-session correlations and simple linear regressions that relate
  behavioural accuracy to neural metrics (ideal / actual decoding
  accuracy and session-wise mean IG / loss).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .metrics import TrialMetrics


@dataclass
class BehaviourSummary:

    session_tag: str

    behav_accuracy: float
    behav_error_rate: float
    n_trials: int

    IG_correct: float
    IG_error: float

    IG_uniform_correct: float
    IG_uniform_error: float

    loss_correct: float
    loss_error: float

    loss_prior_correct: float
    loss_prior_error: float

    loss_uniform_correct: float
    loss_uniform_error: float

    loss_uniform_prior_correct: float
    loss_uniform_prior_error: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _mean_on_mask(
    arr: Optional[np.ndarray],
    mask: np.ndarray,
    n_trials_ref: int,
) -> float:

    if arr is None:
        return float("nan")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n_trials_ref:
        return float("nan")

    vals = arr[mask]
    if vals.size == 0:
        return float("nan")

    return float(vals.mean())


def summarise_behaviour_for_session(
    is_correct: np.ndarray,
    trial_metrics: TrialMetrics,
    session_tag: str,
) -> Optional[BehaviourSummary]:

    is_correct = np.asarray(is_correct, dtype=bool)
    n_trials = int(is_correct.size)

    if n_trials == 0:
        return None

    if trial_metrics.IG.shape[0] != n_trials:
        raise ValueError(
            f"is_correct (len={n_trials}) and trial_metrics.IG "
            f"(len={trial_metrics.IG.shape[0]}) does not match."
        )

    behav_accuracy = float(is_correct.mean())
    behav_error_rate = 1.0 - behav_accuracy

    mask_correct = is_correct
    mask_error = ~is_correct

    # IG (block-prior ideal)
    IG_correct = _mean_on_mask(trial_metrics.IG, mask_correct, n_trials)
    IG_error = _mean_on_mask(trial_metrics.IG, mask_error, n_trials)

    # IG for uniform ideal
    IG_uniform_correct = _mean_on_mask(
        trial_metrics.IG_uniform, mask_correct, n_trials
    )
    IG_uniform_error = _mean_on_mask(
        trial_metrics.IG_uniform, mask_error, n_trials
    )

    # ℓ1 = KL(L_ideal^prior || L_0^agnostic)
    loss_correct = _mean_on_mask(trial_metrics.loss, mask_correct, n_trials)
    loss_error = _mean_on_mask(trial_metrics.loss, mask_error, n_trials)

    # ℓ2 = KL(L_ideal^prior || L_0^prior)
    loss_prior_correct = _mean_on_mask(
        trial_metrics.loss_prior, mask_correct, n_trials
    )
    loss_prior_error = _mean_on_mask(
        trial_metrics.loss_prior, mask_error, n_trials
    )

    # ℓ3 = KL(L_ideal^unif || L_0^agnostic)
    loss_uniform_correct = _mean_on_mask(
        trial_metrics.loss_uniform, mask_correct, n_trials
    )
    loss_uniform_error = _mean_on_mask(
        trial_metrics.loss_uniform, mask_error, n_trials
    )

    # ℓ4 = KL(L_ideal^unif || L_0^prior)
    loss_uniform_prior_correct = _mean_on_mask(
        trial_metrics.loss_uniform_prior, mask_correct, n_trials
    )
    loss_uniform_prior_error = _mean_on_mask(
        trial_metrics.loss_uniform_prior, mask_error, n_trials
    )

    return BehaviourSummary(
        session_tag=session_tag,
        behav_accuracy=behav_accuracy,
        behav_error_rate=behav_error_rate,
        n_trials=n_trials,
        IG_correct=IG_correct,
        IG_error=IG_error,
        IG_uniform_correct=IG_uniform_correct,
        IG_uniform_error=IG_uniform_error,
        loss_correct=loss_correct,
        loss_error=loss_error,
        loss_prior_correct=loss_prior_correct,
        loss_prior_error=loss_prior_error,
        loss_uniform_correct=loss_uniform_correct,
        loss_uniform_error=loss_uniform_error,
        loss_uniform_prior_correct=loss_uniform_prior_correct,
        loss_uniform_prior_error=loss_uniform_prior_error,
    )


__all__ = ["BehaviourSummary", "summarise_behaviour_for_session"]
