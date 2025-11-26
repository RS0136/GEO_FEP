"""
Session-level pipeline and cross-session aggregation.

This module implements the high-level pieces described in Sections 3.1–3.11 of the
preprint: for each Neuropixels probe insertion (PID) we run the full geometric
FEP–pragmatics pipeline and then aggregate the resulting metrics across sessions
and block priors.

Per-session pipeline
--------------------
For a single PID we:

1. load spike sorting and trials via `load_raw_session_from_pid` (Section 3.1);
2. construct TrialData (world labels, priors, correctness, spike matrix X; Sections 3.2–3.3);
3. construct LatentGeometry (FA latent codes, Gaussian prototypes, Σ, Σ⁻¹; Sections 3.5–3.6);
4. build an ideal listener from geometry and block priors (Section 3.7);
5. fit an actual listener via logistic regression on latent codes, with a fixed train–test split
   (Section 3.8);
6. compute trial-wise information gain IG and pragmatic loss ℓ⋆ (Section 3.9).

Cross-session aggregation
-------------------------
From these per-session objects we build the tables described in Sections 3.10–3.11:

- `summary_all.csv`: session-wise decoding accuracy, IG, and ℓ⋆ (Table 3);
- `by_context_all.csv`: block-prior-wise IG and ℓ⋆ (Table 4);
- `test_only_summary.csv`: test-only IG and ℓ⋆ (Table 5);
- `prior_control_summary.csv`: prior-aware decoder performance (Table 6);
- `behavior_link.csv`: IG / ℓ⋆ split by correct vs. error trials (Table 2).

This module focuses on *numerical* aggregation; plotting and CLI wrappers live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SessionConfig, MULTI_SESSION_TAGS, default_session_config
from .data_io import load_raw_session_from_pid, RawSessionData
from .trials import TrialData, construct_trial_data
from .listeners import LatentGeometry, construct_latent_geometry
from .metrics import (
    IdealListenerResult,
    ActualListenerResult,
    ideal_listener_gaussian,
    logistic_actual_listener,
    prior_aware_logistic_listener,
)
from .metrics import TrialMetrics, compute_trial_metrics
from .behaviour import BehaviourSummary, summarise_behaviour_for_session


# ---------------------------------------------------------------------------
# Core per-session result container
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """
    Full result of running the geometric FEP–pragmatics pipeline on one probe.

    Attributes
    ----------
    session_tag :
        Probe UUID (PID) string, e.g. 'fece187f-b47f-4870-a1d6-619afe942a7d_probe01'.
    config :
        SessionConfig actually used for this session.
    raw :
        RawSessionData (spikes, clusters, channels, trials).
    trials :
        TrialData with world labels, priors, correctness, and spike matrix X.
    geometry :
        LatentGeometry (FA codes, Gaussian prototypes, Σ, Σ⁻¹, train–test indices).
    ideal :
        IdealListenerResult for all valid trials.
    actual :
        ActualListenerResult for the main prior-agnostic logistic decoder.
    metrics :
        TrialMetrics with IG and pragmatic loss for all trials.
    prior_actual :
        Optional ActualListenerResult for a prior-aware logistic decoder.
    metrics_prior :
        Optional TrialMetrics with pragmatic loss for the prior-aware decoder.
    behaviour :
        BehaviourSummary with IG / loss split by correct vs. error trials.
    """

    session_tag: str
    config: SessionConfig

    raw: RawSessionData
    trials: TrialData
    geometry: LatentGeometry

    ideal: IdealListenerResult
    actual: ActualListenerResult
    metrics: TrialMetrics

    prior_actual: Optional[ActualListenerResult] = None
    metrics_prior: Optional[TrialMetrics] = None

    behaviour: Optional[BehaviourSummary] = None


# ---------------------------------------------------------------------------
# Per-session pipeline
# ---------------------------------------------------------------------------


def run_session(
    session_tag: str,
    cfg: Optional[SessionConfig] = None,
    do_prior_control: bool = True,
) -> SessionResult:
    """
    Run the full pipeline for a single probe insertion (PID).
    """
    if cfg is None:
        cfg = default_session_config(session_tag=session_tag)
    else:
        cfg = SessionConfig(**asdict(cfg))
        cfg.session_tag = session_tag
    cfg.validate()

    # 1. Read raw data
    raw = load_raw_session_from_pid(session_tag)

    # 2. TrialData
    trial_data = construct_trial_data(raw, cfg)
    
    # 3. Latent geometry
    geom = construct_latent_geometry(trial_data, cfg)

    # 4. Ideal listeners
    priors_left_block = trial_data.priors_left
    priors_left_uniform = np.full_like(priors_left_block, 0.5, dtype=float)

    # (a) L_ideal^prior
    ideal_prior = ideal_listener_gaussian(
        Z=geom.Z,
        w_true=trial_data.w,
        priors_left=priors_left_block,
        mu_left=geom.mu_left,
        mu_right=geom.mu_right,
        Sigma_inv=geom.Sigma_inv,
        cfg=cfg,
    )

    # (b) L_ideal^unif
    ideal_uniform = ideal_listener_gaussian(
        Z=geom.Z,
        w_true=trial_data.w,
        priors_left=priors_left_uniform,
        mu_left=geom.mu_left,
        mu_right=geom.mu_right,
        Sigma_inv=geom.Sigma_inv,
        cfg=cfg,
    )

    # 5. Actual listener: prior-agnostic
    actual = logistic_actual_listener(
        Z=geom.Z,
        w_true=trial_data.w,
        idx_train=geom.idx_train,
        idx_test=geom.idx_test,
        cfg=cfg,
    )

    # 5'. Prior-aware control decoder L_0^prior
    prior_actual = None
    if do_prior_control and cfg.prior_feature != "none":
        prior_actual = prior_aware_logistic_listener(
            Z=geom.Z,
            w_true=trial_data.w,
            priors_left=priors_left_block,
            idx_train=geom.idx_train,
            idx_test=geom.idx_test,
            cfg=cfg,
        )

    metrics = compute_trial_metrics(
        priors_left=priors_left_block,                   # P_left(c_i)
        L_ideal=ideal_prior.posterior,                  # L_ideal^prior
        L_actual=actual.posterior,                      # L_0^agnostic
        cfg=cfg,
        L_prior=prior_actual.posterior
            if prior_actual is not None else None,      # L_0^prior
        L_ideal_uniform=ideal_uniform.posterior,        # L_ideal^unif
    )

    behaviour = summarise_behaviour_for_session(
        is_correct=trial_data.is_correct,
        trial_metrics=metrics,
        session_tag=session_tag,
    )

    return SessionResult(
        session_tag=session_tag,
        config=cfg,
        raw=raw,
        trials=trial_data,
        geometry=geom,
        ideal=ideal_prior,
        actual=actual,
        metrics=metrics,
        prior_actual=prior_actual,
        metrics_prior=None,
        behaviour=behaviour,
    )

# -----------------------------------------------------
# Session-wise summary rows (Table 3, 4, 5, 6)
# -----------------------------------------------------


@dataclass
class SessionSummaryRow:
    """Row for summary_all.csv / Table 3."""

    session_id: str
    probe: str
    region: Optional[str]

    n_trials: int
    n_neurons: int
    latent_dim: int

    ideal_accuracy: float
    ideal_accuracy_test: float
    actual_accuracy: float

    loss_mean: float
    loss_std: float

    # IG (L_ideal^prior)
    IG_mean: float
    IG_std: float

    # IG for L_ideal^unif
    IG_uniform_mean: float
    IG_uniform_std: float

    # ℓ2 = KL(L_ideal^prior || L_0^prior)
    loss_prior_mean: float
    loss_prior_std: float

    # ℓ3 = KL(L_ideal^unif || L_0^agnostic)
    loss_uniform_mean: float
    loss_uniform_std: float

    # ℓ4 = KL(L_ideal^unif || L_0^prior)
    loss_uniform_prior_mean: float
    loss_uniform_prior_std: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ContextSummaryRow:
    """Row for by_context_all.csv / Table 4."""

    P_left_prior: float
    n_trials: int

    IG_mean: float
    IG_std: float

    loss_mean: float
    loss_std: float

    IG_uniform_mean: float
    IG_uniform_std: float

    loss_prior_mean: float
    loss_prior_std: float

    loss_uniform_mean: float
    loss_uniform_std: float

    loss_uniform_prior_mean: float
    loss_uniform_prior_std: float

    session_tag: str

    session_tag: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TestOnlySummaryRow:
    """
    Row for test_only_summary.csv
    - IG_test_*                : IG(L_ideal^prior)
    - IG_uniform_test_*        : IG(L_ideal^unif)
    - loss_test_*              : ℓ1 = KL(L_ideal^prior || L_0^agnostic)
    - loss_prior_test_*        : ℓ2 = KL(L_ideal^prior || L_0^prior)
    - loss_uniform_test_*      : ℓ3 = KL(L_ideal^unif  || L_0^agnostic)
    - loss_uniform_prior_test_*: ℓ4 = KL(L_ideal^unif  || L_0^prior)
    """

    session_tag: str
    acc_test: float

    IG_test_mean: float
    IG_test_std: float

    IG_uniform_test_mean: float
    IG_uniform_test_std: float

    loss_test_mean: float
    loss_test_std: float

    loss_prior_test_mean: float
    loss_prior_test_std: float

    loss_uniform_test_mean: float
    loss_uniform_test_std: float

    loss_uniform_prior_test_mean: float
    loss_uniform_prior_test_std: float

    n_test: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PriorControlSummaryRow:
    """Row for prior_control_summary.csv / Table 6."""

    session_tag: str
    acc_test_prior: float

    # ℓ2 = KL(L_ideal^prior || L0^prior)
    loss_prior_mean: float
    loss_prior_std: float

    # ℓ4 = KL(L_ideal^unif || L0^prior)
    loss_uniform_prior_mean: float
    loss_uniform_prior_std: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _split_session_tag(session_tag: str) -> Tuple[str, str]:
    """
    Split a PID string of the form '{session_id}_probeXX' into (session_id, probe).

    If no '_probe' substring is found, returns (session_tag, "").
    """
    marker = "_probe"
    if marker in session_tag:
        idx = session_tag.index(marker)
        return session_tag[:idx], session_tag[idx + 1 :]
    return session_tag, ""


def make_session_summary_row(result: SessionResult) -> SessionSummaryRow:
    """Create a SessionSummaryRow from a SessionResult (Table 3)."""
    session_id, probe = _split_session_tag(result.session_tag)

    trial_data = result.trials
    geom = result.geometry

    w_true = trial_data.w
    idx_test = np.asarray(geom.idx_test, dtype=int)

    ideal_acc_all = float(result.ideal.accuracy)
    ideal_acc_test = float(
        (result.ideal.predicted_w[idx_test] == w_true[idx_test]).mean()
    )

    n_trials = result.trials.n_trials
    n_neurons = result.trials.n_neurons
    latent_dim = result.geometry.d

    m = result.metrics

    # ℓ1, IG (block prior ideal)
    IG = np.asarray(m.IG)
    loss = np.asarray(m.loss)

    IG_mean = float(IG.mean())
    IG_std = float(IG.std(ddof=1)) if IG.size > 1 else 0.0

    loss_mean = float(loss.mean())
    loss_std = float(loss.std(ddof=1)) if loss.size > 1 else 0.0

    IG_uniform_mean = IG_uniform_std = 0.0
    loss_prior_mean = loss_prior_std = 0.0
    loss_uniform_mean = loss_uniform_std = 0.0
    loss_uniform_prior_mean = loss_uniform_prior_std = 0.0

    # IG_uniform
    if m.IG_uniform is not None:
        IG_u = np.asarray(m.IG_uniform)
        if IG_u.size > 0:
            IG_uniform_mean = float(IG_u.mean())
            IG_uniform_std = float(IG_u.std(ddof=1)) if IG_u.size > 1 else 0.0

    # ℓ2
    if m.loss_prior is not None:
        lp = np.asarray(m.loss_prior)
        if lp.size > 0:
            loss_prior_mean = float(lp.mean())
            loss_prior_std = float(lp.std(ddof=1)) if lp.size > 1 else 0.0

    # ℓ3
    if m.loss_uniform is not None:
        lu = np.asarray(m.loss_uniform)
        if lu.size > 0:
            loss_uniform_mean = float(lu.mean())
            loss_uniform_std = float(lu.std(ddof=1)) if lu.size > 1 else 0.0

    # ℓ4
    if m.loss_uniform_prior is not None:
        lup = np.asarray(m.loss_uniform_prior)
        if lup.size > 0:
            loss_uniform_prior_mean = float(lup.mean())
            loss_uniform_prior_std = float(
                lup.std(ddof=1)
            ) if lup.size > 1 else 0.0

    return SessionSummaryRow(
        session_id=session_id,
        probe=probe,
        region=result.config.region,
        n_trials=n_trials,
        n_neurons=n_neurons,
        latent_dim=latent_dim,
        ideal_accuracy=ideal_acc_all,
        ideal_accuracy_test=ideal_acc_test,
        actual_accuracy=float(result.actual.accuracy_test),
        IG_mean=IG_mean,
        IG_std=IG_std,
        loss_mean=loss_mean,
        loss_std=loss_std,
        IG_uniform_mean=IG_uniform_mean,
        IG_uniform_std=IG_uniform_std,
        loss_prior_mean=loss_prior_mean,
        loss_prior_std=loss_prior_std,
        loss_uniform_mean=loss_uniform_mean,
        loss_uniform_std=loss_uniform_std,
        loss_uniform_prior_mean=loss_uniform_prior_mean,
        loss_uniform_prior_std=loss_uniform_prior_std,
    )


def make_context_summary_rows(result: SessionResult) -> List[ContextSummaryRow]:
    """
    Create ContextSummaryRow objects for each distinct block prior.
    """
    priors = np.asarray(result.trials.priors_left, dtype=float)
    m = result.metrics

    IG = np.asarray(m.IG, dtype=float)
    loss = np.asarray(m.loss, dtype=float)

    IG_u = np.asarray(m.IG_uniform, dtype=float) if m.IG_uniform is not None else None
    loss_prior = (
        np.asarray(m.loss_prior, dtype=float) if m.loss_prior is not None else None
    )
    loss_uniform = (
        np.asarray(m.loss_uniform, dtype=float) if m.loss_uniform is not None else None
    )
    loss_uniform_prior = (
        np.asarray(m.loss_uniform_prior, dtype=float)
        if m.loss_uniform_prior is not None
        else None
    )

    rows = []
    for p in np.unique(priors):
        mask = priors == p
        n = int(mask.sum())
        if n == 0:
            continue

        IG_mean = float(IG[mask].mean())
        IG_std = float(IG[mask].std(ddof=1)) if n > 1 else 0.0
        loss_mean = float(loss[mask].mean())
        loss_std = float(loss[mask].std(ddof=1)) if n > 1 else 0.0

        IG_uniform_mean = IG_uniform_std = 0.0
        loss_prior_mean = loss_prior_std = 0.0
        loss_uniform_mean = loss_uniform_std = 0.0
        loss_uniform_prior_mean = loss_uniform_prior_std = 0.0

        if IG_u is not None:
            IG_uniform_mean = float(IG_u[mask].mean())
            IG_uniform_std = float(IG_u[mask].std(ddof=1)) if n > 1 else 0.0

        if loss_prior is not None:
            loss_prior_mean = float(loss_prior[mask].mean())
            loss_prior_std = float(loss_prior[mask].std(ddof=1)) if n > 1 else 0.0

        if loss_uniform is not None:
            loss_uniform_mean = float(loss_uniform[mask].mean())
            loss_uniform_std = float(loss_uniform[mask].std(ddof=1)) if n > 1 else 0.0

        if loss_uniform_prior is not None:
            loss_uniform_prior_mean = float(loss_uniform_prior[mask].mean())
            loss_uniform_prior_std = (
                float(loss_uniform_prior[mask].std(ddof=1)) if n > 1 else 0.0
            )

        rows.append(
            ContextSummaryRow(
                P_left_prior=float(p),
                n_trials=n,
                IG_mean=IG_mean,
                IG_std=IG_std,
                loss_mean=loss_mean,
                loss_std=loss_std,
                IG_uniform_mean=IG_uniform_mean,
                IG_uniform_std=IG_uniform_std,
                loss_prior_mean=loss_prior_mean,
                loss_prior_std=loss_prior_std,
                loss_uniform_mean=loss_uniform_mean,
                loss_uniform_std=loss_uniform_std,
                loss_uniform_prior_mean=loss_uniform_prior_mean,
                loss_uniform_prior_std=loss_uniform_prior_std,
                session_tag=result.session_tag,
            )
        )

    return rows


def make_test_only_summary_row(result: SessionResult) -> TestOnlySummaryRow:
    """
    Create a TestOnlySummaryRow by restricting IG / ℓ⋆ to held-out test trials.
    """
    import numpy as np

    idx_test = np.asarray(result.geometry.idx_test, dtype=int)
    n_test = int(idx_test.size)

    if n_test == 0:
        nan = float("nan")
        return TestOnlySummaryRow(
            session_tag=result.session_tag,
            acc_test=float("nan"),
            IG_test_mean=nan,
            IG_test_std=nan,
            IG_uniform_test_mean=nan,
            IG_uniform_test_std=nan,
            loss_test_mean=nan,
            loss_test_std=nan,
            loss_prior_test_mean=nan,
            loss_prior_test_std=nan,
            loss_uniform_test_mean=nan,
            loss_uniform_test_std=nan,
            loss_uniform_prior_test_mean=nan,
            loss_uniform_prior_test_std=nan,
            n_test=0,
        )

    def mean_std_on_idx(arr: Optional[np.ndarray]) -> Tuple[float, float]:
        if arr is None:
            return float("nan"), float("nan")
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != result.metrics.n_trials:
            return float("nan"), float("nan")

        vals = arr[idx_test]
        if vals.size == 0:
            return float("nan"), float("nan")

        mean = float(vals.mean())
        if vals.size > 1:
            std = float(vals.std(ddof=1))
        else:
            std = 0.0
        return mean, std

    # IG (block-prior ideal)
    IG_mean, IG_std = mean_std_on_idx(result.metrics.IG)

    # IG for ideal listener with uniform prior
    IG_uniform_mean, IG_uniform_std = mean_std_on_idx(result.metrics.IG_uniform)

    # ℓ1 = KL(L_ideal^prior || L_0^agnostic)
    loss_mean, loss_std = mean_std_on_idx(result.metrics.loss)

    # ℓ2 = KL(L_ideal^prior || L_0^prior)
    loss_prior_mean, loss_prior_std = mean_std_on_idx(result.metrics.loss_prior)

    # ℓ3 = KL(L_ideal^unif || L_0^agnostic)
    loss_uniform_mean, loss_uniform_std = mean_std_on_idx(result.metrics.loss_uniform)

    # ℓ4 = KL(L_ideal^unif || L_0^prior)
    loss_uniform_prior_mean, loss_uniform_prior_std = mean_std_on_idx(
        result.metrics.loss_uniform_prior
    )

    return TestOnlySummaryRow(
        session_tag=result.session_tag,
        acc_test=float(result.actual.accuracy_test),
        IG_test_mean=IG_mean,
        IG_test_std=IG_std,
        IG_uniform_test_mean=IG_uniform_mean,
        IG_uniform_test_std=IG_uniform_std,
        loss_test_mean=loss_mean,
        loss_test_std=loss_std,
        loss_prior_test_mean=loss_prior_mean,
        loss_prior_test_std=loss_prior_std,
        loss_uniform_test_mean=loss_uniform_mean,
        loss_uniform_test_std=loss_uniform_std,
        loss_uniform_prior_test_mean=loss_uniform_prior_mean,
        loss_uniform_prior_test_std=loss_uniform_prior_std,
        n_test=n_test,
    )



def make_prior_control_summary_row(
    result: SessionResult,
) -> Optional[PriorControlSummaryRow]:
    """
    Create a PriorControlSummaryRow, if a prior-aware decoder is present.
    """
    # prior-aware decoder
    if result.prior_actual is None:
        return None

    m = result.metrics

    # ℓ2(u,c) = KL(L_ideal^prior ∥ L_0^prior)
    if m.loss_prior is None:
        return None

    loss_prior = np.asarray(m.loss_prior, dtype=float)
    if loss_prior.ndim != 1:
        raise ValueError("metrics.loss_prior must be a 1D array.")

    n = int(loss_prior.size)
    if n == 0:
        return None

    loss_prior_mean = float(loss_prior.mean())
    loss_prior_std = float(loss_prior.std(ddof=1)) if n > 1 else 0.0

    # ℓ4(u,c) = KL(L_ideal^unif ∥ L_0^prior)
    if m.loss_uniform_prior is not None and m.loss_uniform_prior.size > 0:
        l4 = np.asarray(m.loss_uniform_prior, dtype=float)
        loss_uniform_prior_mean = float(l4.mean())
        loss_uniform_prior_std = float(l4.std(ddof=1)) if l4.size > 1 else 0.0
    else:
        loss_uniform_prior_mean = float("nan")
        loss_uniform_prior_std = 0.0

    return PriorControlSummaryRow(
        session_tag=result.session_tag,
        acc_test_prior=float(result.prior_actual.accuracy_test),
        loss_prior_mean=loss_prior_mean,
        loss_prior_std=loss_prior_std,
        loss_uniform_prior_mean=loss_uniform_prior_mean,
        loss_uniform_prior_std=loss_uniform_prior_std,
    )


# ---------------------------------------------------------------------------
# Multi-session aggregation
# ---------------------------------------------------------------------------


def run_multi_session(
    session_tags: Optional[Iterable[str]] = None,
    base_cfg: Optional[SessionConfig] = None,
    do_prior_control: bool = True,
) -> Tuple[List[SessionResult], Dict[str, pd.DataFrame]]:
    """
    Run the full pipeline for multiple sessions and build aggregate tables.

    Parameters
    ----------
    session_tags :
        Iterable of PIDs to analyse. If None, uses `MULTI_SESSION_TAGS`, the
        canonical list of 13 IBL benchmark insertions.
    base_cfg :
        Optional SessionConfig template. For each session a copy is made and
        its `session_tag` is overwritten. If None, `default_session_config`
        is called separately for each session.
    do_prior_control :
        Whether to run the prior-aware decoder and include the corresponding
        aggregation tables.

    Returns
    -------
    results :
        List of SessionResult objects, one per session.
    tables :
        Dict mapping table name → pandas.DataFrame:

        - "summary_all"          : session-wise summary (Table 3),
        - "by_context_all"       : block-prior-wise summary (Table 4),
        - "test_only_summary"    : test-only metrics (Table 5),
        - "prior_control_summary": prior-aware decoder summary (Table 6, if run),
        - "behaviour_link"       : behaviour–neural link (Table 2).
    """
    if session_tags is None:
        session_tags = MULTI_SESSION_TAGS

    results: List[SessionResult] = []

    for tag in session_tags:
        if base_cfg is None:
            cfg = default_session_config(session_tag=tag)
        else:
            cfg = SessionConfig(**asdict(base_cfg))
            cfg.session_tag = tag
        result = run_session(tag, cfg, do_prior_control=do_prior_control)
        results.append(result)

    # Build per-table row lists
    summary_rows: List[SessionSummaryRow] = []
    context_rows: List[ContextSummaryRow] = []
    test_rows: List[TestOnlySummaryRow] = []
    prior_rows: List[PriorControlSummaryRow] = []
    behaviour_rows: List[Dict[str, object]] = []

    for res in results:
        summary_rows.append(make_session_summary_row(res))
        context_rows.extend(make_context_summary_rows(res))
        test_rows.append(make_test_only_summary_row(res))

        prior_row = make_prior_control_summary_row(res)
        if prior_row is not None:
            prior_rows.append(prior_row)

        if res.behaviour is not None:
            behaviour_rows.append(res.behaviour.to_dict())

    tables: Dict[str, pd.DataFrame] = {
        "summary_all": pd.DataFrame([r.to_dict() for r in summary_rows]),
        "by_context_all": pd.DataFrame([r.to_dict() for r in context_rows]),
        "test_only_summary": pd.DataFrame([r.to_dict() for r in test_rows]),
        "behaviour_link": pd.DataFrame(behaviour_rows),
    }

    if prior_rows:
        tables["prior_control_summary"] = pd.DataFrame(
            [r.to_dict() for r in prior_rows]
        )

    return results, tables


# ---------------------------------------------------------------------------
# I/O helpers for writing aggregate tables
# ---------------------------------------------------------------------------


def write_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: str | Path,
    write_tex: bool = False,
    float_format: str = "%.3f",
) -> None:
    """
    Write aggregate tables to CSV (and optionally LaTeX) in `output_dir`.

    This mirrors the behaviour of the aggregation step described at the end
    of Section 3.10, which writes summary_all.csv/.tex and
    by_context_all.csv/.tex, as well as the extras in Section 3.11.

    Parameters
    ----------
    tables :
        Dict from table name to pandas.DataFrame, as returned by
        `run_multi_session`.
    output_dir :
        Directory in which to write files. Created if needed.
    write_tex :
        If True, also write `<name>.tex` via `DataFrame.to_latex`.
    float_format :
        Format string for floating-point numbers in LaTeX output.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        csv_path = out_path / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        if write_tex:
            tex_path = out_path / f"{name}.tex"
            df.to_latex(tex_path, index=False, float_format=float_format)


__all__ = [
    "SessionResult",
    "SessionSummaryRow",
    "ContextSummaryRow",
    "TestOnlySummaryRow",
    "PriorControlSummaryRow",
    "run_session",
    "run_multi_session",
    "write_tables",
    "make_session_summary_row",
    "make_context_summary_rows",
    "make_test_only_summary_row",
    "make_prior_control_summary_row",
]
