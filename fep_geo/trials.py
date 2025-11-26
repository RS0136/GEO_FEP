"""
Trial selection and spike-count matrix construction.

This module implements the parts of the analysis pipeline described in
Section 3.2–3.3 of the preprint:

- definition of binary world states w and contexts c from the IBL `trials`
  object (Eqs. (73)–(75));
- construction of a peri-stimulus spike-count matrix X for all valid trials
  in a given probe insertion (Eqs. (76)–(78));
- defensive handling of the choice sign convention when computing
  behavioural correctness.

The public entry point is :func:`construct_trial_data`, which takes a
RawSessionData (spike sorting + trials) and a SessionConfig and returns a
TrialData dataclass containing all trial-wise variables and the spike
matrix ready for latent-geometry analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins

from .config import SessionConfig
from .data_io import RawSessionData


# ---------------------------------------------------------------------------
# Dataclass representing per-trial data for a single probe insertion
# ---------------------------------------------------------------------------

@dataclass
class TrialData:
    """
    Trial-wise variables and spike-count matrix for a single probe insertion.

    This dataclass is the bridge between raw IBL objects (spikes, clusters,
    trials) and the latent-geometry part of the pipeline. All arrays are
    aligned so that row i corresponds to the same valid trial across:

      - world state w[i] ∈ {0, 1},
      - block prior P(Left | c_i) = priors_left[i],
      - behavioural choice choice[i] ∈ {−1, +1},
      - correctness is_correct[i] ∈ {0, 1},
      - stimulus onset stim_on_times[i],
      - spike counts X[i, :].

    Attributes
    ----------
    pid:
        Probe UUID (PID) for this insertion, as in the config and tables.

    w:
        Array of shape (N_valid,) with binary world labels:
        0 for Left, 1 for Right, as defined in Eq. (73).

    priors_left:
        Array of shape (N_valid,) with block priors P(Left | c_i)
        = trials['probabilityLeft'][idx_valid_i].

    choice:
        Array of shape (N_valid,) with behavioural choices restricted to
        valid trials, taking values in {−1, +1} (no-go trials have been
        removed by the valid-trial mask).

    is_correct:
        Array of shape (N_valid,) with 0/1 indicators of behavioural
        correctness, defined by Eqs. (74)–(75), with the sign convention
        chosen to maximise the fraction of correct trials.

    stim_on_times:
        Array of shape (N_valid,) with stimulus onset times for the valid
        trials (trials['stimOn_times'][idx_valid_i]).

    valid_mask:
        Boolean array of shape (N_raw_trials,) indicating which trials in
        the original `trials` object are included as valid trials, according
        to the criteria in Section 3.2.

    original_trial_indices:
        Integer array of shape (N_valid,) giving the indices into the raw
        `trials` object corresponding to each row of X and each entry of w.

    X:
        Spike-count matrix of shape (N_valid, N_units), with X[i, j] equal
        to the number of spikes of unit j in the peri-stimulus window for
        trial i, as in Eq. (77). Counts are stored as floating-point numbers
        for compatibility with downstream linear algebra.

    unit_ids:
        Integer array of shape (N_units,) containing cluster indices (IDs)
        of the units included in X, after filtering by quality, region, and
        minimum total spike count.

    region:
        Anatomical region filter used (from SessionConfig.region); None
        means all good units on the probe were used.

    time_window:
        Tuple (t0, t1) in seconds relative to stimulus onset defining the
        peri-stimulus window used to compute X.

    min_total_spikes:
        Threshold on Σ_i X[i, j] used to decide whether to keep a unit
        (cf. Eq. (78)).
    """

    pid: str

    w: np.ndarray
    priors_left: np.ndarray
    choice: np.ndarray
    is_correct: np.ndarray
    stim_on_times: np.ndarray

    valid_mask: np.ndarray
    original_trial_indices: np.ndarray

    X: np.ndarray
    unit_ids: np.ndarray

    region: str | None
    time_window: Tuple[float, float]
    min_total_spikes: int

    @property
    def n_trials(self) -> int:
        """Number of valid trials (rows of X)."""
        return int(self.X.shape[0])

    @property
    def n_neurons(self) -> int:
        """Number of analysed units (columns of X)."""
        return int(self.X.shape[1])


# ---------------------------------------------------------------------------
# World states, contexts, and valid trials (Section 3.2)
# ---------------------------------------------------------------------------

def _compute_world_labels(
    contrast_left: np.ndarray,
    contrast_right: np.ndarray,
) -> np.ndarray:
    """
    Compute binary world labels w ∈ {0, 1} from signed Michelson contrasts.

    Implements Eq. (73):

        c_L_tilde = |c_L|, c_R_tilde = |c_R|,
        world_raw := sign(c_R_tilde − c_L_tilde),
        w = 1 if world_raw > 0 (Right), 0 otherwise (Left).

    Trials in which both sides are zero or both sides have nonzero contrast
    are excluded later by the valid-trial mask; here we simply encode them
    as w = 0 (world_raw ≤ 0) for completeness.

    Parameters
    ----------
    contrast_left, contrast_right:
        Arrays of shape (N_raw_trials,) with signed Michelson contrasts
        from `trials['contrastLeft']` and `trials['contrastRight']`.

    Returns
    -------
    w_all : np.ndarray
        Integer array of shape (N_raw_trials,) with entries 0 (Left) or
        1 (Right).
    """
    cL = np.nan_to_num(np.asarray(contrast_left, dtype=float), nan=0.0)
    cR = np.nan_to_num(np.asarray(contrast_right, dtype=float), nan=0.0)

    cL_abs = np.abs(cL)
    cR_abs = np.abs(cR)

    world_raw = np.sign(cR_abs - cL_abs)  # Eq. (73)

    w_all = np.zeros_like(world_raw, dtype=np.int64)
    w_all[world_raw > 0] = 1  # Right
    # world_raw <= 0 is treated as Left (0)
    return w_all


def select_valid_trials(trials) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the valid-trial mask and extract world labels, priors, and
    choices for valid trials.

    This implements the inclusion criteria and definitions in Section 3.2:​

    1. Exactly one side has a non-zero stimulus:
       (|c_L| > 0) XOR (|c_R| > 0);
    2. A left or right choice was made: choice ∈ {−1, +1};
       no-go trials (choice = 0) are excluded;
    3. Stimulus onset time is finite (stimOn_times is not NaN or ±∞).

    World labels w ∈ {0, 1} are defined by Eq. (73), block priors are taken
    from trials['probabilityLeft'], and choices are taken from
    trials['choice'].

    Parameters
    ----------
    trials:
        The ALF `trials` object loaded via ONE, assumed to support
        item access by field name.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean array of shape (N_raw_trials,) indicating valid trials.

    w_valid : np.ndarray
        Integer array of shape (N_valid,) with world labels (0=Left, 1=Right)
        for valid trials.

    stim_on_valid : np.ndarray
        Array of shape (N_valid,) with stimulus onset times for valid trials.

    priors_left_valid : np.ndarray
        Array of shape (N_valid,) with P(Left | c_i) for valid trials.

    choice_valid : np.ndarray
        Integer array of shape (N_valid,) with choices ∈ {−1, +1} for
        valid trials.
    """
    # Extract raw fields
    cL_raw = np.asarray(trials["contrastLeft"])
    cR_raw = np.asarray(trials["contrastRight"])
    choice_raw = np.asarray(trials["choice"])
    stim_on_raw = np.asarray(trials["stimOn_times"])
    prob_left_raw = np.asarray(trials["probabilityLeft"])

    # Treat missing contrasts as zero (no stimulus on that side)
    cL_abs = np.abs(np.nan_to_num(cL_raw, nan=0.0))
    cR_abs = np.abs(np.nan_to_num(cR_raw, nan=0.0))

    # (1) exactly one side has a non-zero stimulus
    stim_one_side = (cL_abs > 0) ^ (cR_abs > 0)

    # (2) left or right choice (no-go removed)
    choice_valid_mask = np.isin(choice_raw, (-1, +1))

    # (3) finite, non-NaN stimulus onset
    onset_valid_mask = np.isfinite(stim_on_raw)

    valid_mask = stim_one_side & choice_valid_mask & onset_valid_mask

    if not np.any(valid_mask):
        raise ValueError("No valid trials remain after applying inclusion criteria.")

    # World labels for all trials, then restrict to valid ones
    w_all = _compute_world_labels(cL_raw, cR_raw)
    w_valid = w_all[valid_mask]

    stim_on_valid = stim_on_raw[valid_mask]
    priors_left_valid = prob_left_raw[valid_mask]
    choice_valid = choice_raw[valid_mask]

    return valid_mask, w_valid, stim_on_valid, priors_left_valid, choice_valid


def compute_correct_choices(
    w_valid: np.ndarray,
    choice_valid: np.ndarray,
) -> np.ndarray:
    """
    Compute behavioural correctness for valid trials.

    Conceptually, a trial is correct if the choice matches the true world
    side: Left choice when w = 0, Right choice when w = 1, as in
    Eqs. (74)–(75).

    In practice, to be robust to potential inversions of the sign coding of
    `trials['choice']` in the source dataset, we follow the defensive
    strategy described in Section 3.2: we evaluate correctness under both
    possible sign conventions

      (−1, +1) = (Left, Right)  and
      (−1, +1) = (Right, Left),

    and adopt the convention that yields the higher fraction of correct
    trials.

    Parameters
    ----------
    w_valid:
        Array of shape (N_valid,) with world labels 0 (Left) or 1 (Right).
    choice_valid:
        Array of shape (N_valid,) with choices in {−1, +1}.

    Returns
    -------
    is_correct : np.ndarray
        Array of shape (N_valid,) with entries 0 or 1 indicating correctness
        under the chosen sign convention.
    """
    w_valid = np.asarray(w_valid, dtype=np.int64)
    choice_valid = np.asarray(choice_valid, dtype=int)

    if w_valid.shape != choice_valid.shape:
        raise ValueError(
            f"w_valid and choice_valid must have the same shape; "
            f"got {w_valid.shape} and {choice_valid.shape}"
        )

    def correctness_for_mapping(left_code: int) -> np.ndarray:
        right_code = -left_code
        is_left = choice_valid == left_code
        is_right = choice_valid == right_code
        # Eq. (74)–(75) under a given mapping
        correct = ((is_left & (w_valid == 0)) | (is_right & (w_valid == 1)))
        return correct.astype(np.int8)

    # Try both possible assignments for "left" choice
    correct_left_minus = correctness_for_mapping(left_code=-1)
    correct_left_plus = correctness_for_mapping(left_code=+1)

    acc_minus = correct_left_minus.mean()
    acc_plus = correct_left_plus.mean()

    # Adopt the convention with higher accuracy; tie-breaker defaults to left=-1
    if acc_plus > acc_minus:
        is_correct = correct_left_plus
    else:
        is_correct = correct_left_minus

    return is_correct


# ---------------------------------------------------------------------------
# Peri-stimulus spike-count matrix (Section 3.3)
# ---------------------------------------------------------------------------

def build_spike_matrix(
    spikes,
    clusters,
    stim_on_valid: np.ndarray,
    cfg: SessionConfig,
) -> Tuple[np.ndarray, np.ndarray]:

    cfg.validate()

    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=int)

    if hasattr(clusters, "columns"):
        colnames = set(clusters.columns)
    elif hasattr(clusters, "keys"):
        colnames = set(clusters.keys())
    else:
        names = getattr(getattr(clusters, "dtype", None), "names", None)
        colnames = set(names or [])

    if "cluster_id" in colnames:
        cluster_ids_all = np.asarray(clusters["cluster_id"], dtype=int)
    else:
        n_clusters = len(clusters)
        cluster_ids_all = np.arange(n_clusters, dtype=int)

    if "label" in colnames:
        labels = np.asarray(clusters["label"])
        good_mask = labels == 1
    elif "ks_label" in colnames:
        ks_label = np.asarray(clusters["ks_label"]).astype(str)
        good_mask = ks_label == "good"
    else:
        print(
            "[trials.build_spike_matrix] WARNING: clusters has no 'label' or "
            "'ks_label' column; treating all clusters as good."
        )
        good_mask = np.ones_like(cluster_ids_all, dtype=bool)

    if cfg.region is not None and "acronym" in colnames:
        acronyms = np.asarray(clusters["acronym"])
        region_mask = acronyms == cfg.region
        unit_mask = good_mask & region_mask
    else:
        unit_mask = good_mask

    candidate_unit_ids = cluster_ids_all[unit_mask]

    if candidate_unit_ids.size == 0:
        raise ValueError(
            "No units remain after applying quality and region filters: "
            f"region={cfg.region!r}"
        )

    t0, t1 = cfg.time_window
    stim_on_valid = np.asarray(stim_on_valid, dtype=float)
    t_start = stim_on_valid + t0
    t_end = stim_on_valid + t1

    if np.any(t_end <= t_start):
        raise ValueError(
            "Invalid time_window resulted in non-positive-length bins; "
            f"got time_window={cfg.time_window}"
        )

    bins = np.c_[t_start, t_end]

    result = get_spike_counts_in_bins(spike_times, spike_clusters, bins)
    if isinstance(result, tuple) and len(result) == 3:
        counts_all, cluster_ids, _ = result
    else:
        counts_all, cluster_ids = result

    cluster_ids = np.asarray(cluster_ids, dtype=int)

    keep_mask = np.isin(cluster_ids, candidate_unit_ids)
    counts = counts_all[keep_mask]
    cluster_ids_kept = cluster_ids[keep_mask]

    if counts.size == 0:
        raise ValueError(
            "No spike counts remain after applying quality/region filters; "
            "this usually indicates a mismatch between cluster_id fields."
        )

    total_spikes_per_unit = counts.sum(axis=1)
    keep_units_mask = total_spikes_per_unit >= cfg.min_total_spikes

    if not np.any(keep_units_mask):
        raise ValueError(
            "All units were discarded by the min_total_spikes threshold "
            f"{cfg.min_total_spikes}; consider lowering this threshold."
        )

    counts = counts[keep_units_mask]
    cluster_ids_kept = cluster_ids_kept[keep_units_mask]

    # (n_units, n_trials) → (n_trials, n_units)
    X = counts.T.astype(float)

    if X.shape[0] != stim_on_valid.shape[0]:
        raise RuntimeError(
            "Mismatch between number of trials in stim_on_valid and spike "
            f"count matrix: {stim_on_valid.shape[0]} vs {X.shape[0]}"
        )

    print(
        f"[trials.build_spike_matrix] Built X with shape "
        f"{X.shape} (n_trials={X.shape[0]}, n_units={X.shape[1]}), "
        f"time_window={cfg.time_window}, min_total_spikes={cfg.min_total_spikes}"
    )

    return X, cluster_ids_kept


# ---------------------------------------------------------------------------
# High-level helper: from RawSessionData to TrialData
# ---------------------------------------------------------------------------

def construct_trial_data(
    raw: RawSessionData,
    cfg: SessionConfig,
) -> TrialData:
    """
    End-to-end construction of TrialData from RawSessionData.

    Given spike-sorting output and the trials object for a single probe
    insertion, this function:

      1. defines world states w and contexts from the IBL `trials` object,
         and applies the valid-trial mask (Section 3.2);
      2. computes behavioural correctness using the defensive procedure
         described in that section;
      3. constructs a peri-stimulus spike-count matrix X for all valid
         trials using the window specified in cfg.time_window and the
         unit filters in cfg (Section 3.3).

    The resulting TrialData bundles all of these trial-wise variables and
    is the natural input to the latent-geometry module.

    Parameters
    ----------
    raw:
        RawSessionData containing spikes, clusters, channels, trials,
        and metadata for a single probe insertion.
    cfg:
        SessionConfig specifying analysis hyperparameters.

    Returns
    -------
    trial_data : TrialData
        Dataclass instance with aligned trial-wise variables and spike
        counts.
    """
    cfg.validate()

    trials = raw.trials

    # Step 1: valid-trial mask, world labels, priors, and choices
    valid_mask, w_valid, stim_on_valid, priors_left_valid, choice_valid = select_valid_trials(
        trials
    )
    original_trial_indices = np.nonzero(valid_mask)[0]

    # Step 2: behavioural correctness with defensive sign convention
    is_correct = compute_correct_choices(w_valid, choice_valid)

    # Step 3: spike-count matrix X for valid trials
    X, unit_ids = build_spike_matrix(
        spikes=raw.spikes,
        clusters=raw.clusters,
        stim_on_valid=stim_on_valid,
        cfg=cfg,
    )

    if X.shape[0] != w_valid.shape[0]:
        raise RuntimeError(
            "Internal inconsistency: number of rows in X does not match "
            f"number of valid trials: {X.shape[0]} vs {w_valid.shape[0]}"
        )

    return TrialData(
        pid=raw.pid,
        w=w_valid,
        priors_left=priors_left_valid,
        choice=choice_valid,
        is_correct=is_correct,
        stim_on_times=stim_on_valid,
        valid_mask=valid_mask,
        original_trial_indices=original_trial_indices,
        X=X,
        unit_ids=unit_ids,
        region=cfg.region,
        time_window=cfg.time_window,
        min_total_spikes=cfg.min_total_spikes,
    )


__all__ = [
    "TrialData",
    "select_valid_trials",
    "compute_correct_choices",
    "build_spike_matrix",
    "construct_trial_data",
]
