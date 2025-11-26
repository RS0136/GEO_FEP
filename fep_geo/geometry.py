"""
Trial selection, world labels, and spike–count matrices.

This module implements the parts of the analysis pipeline described in
Sections 3.2–3.4 of the preprint: construction of a valid–trial mask,
definition of binary world states and contexts, behavioural correctness,
and peri‑stimulus spike–count matrices.

The functions here are intentionally *deterministic* and *pure*:

- they take raw `trials`, `spikes`, and `clusters` objects (as returned by
  the ONE / IBL interfaces),
- apply inclusion criteria and unit filters exactly as specified in the
  methods,
- and return NumPy arrays that downstream modules (geometry, listeners,
  metrics) can use without further assumptions.

No random choices are made in this module; all stochasticity (train–test
splits, FA initialisation, logistic regression) is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import logging
import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins

from .config import SessionConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidTrialSelection:
    """
    Container for the outputs of `select_valid_trials`.

    Attributes
    ----------
    mask :
        Boolean array of shape (N_total_trials,) indicating which trials
        passed all inclusion criteria. This indexes the *original* trials
        object.
    w :
        Array of shape (N_valid,) with binary world labels for valid trials,
        encoded as 0 for Left and 1 for Right (Eqs. (73)–(75)).
    t_stim :
        Array of shape (N_valid,) with stimulus onset times (in seconds)
        for valid trials.
    priors_left :
        Array of shape (N_valid,) with block priors P(Left | c_i) for each
        valid trial (the `probabilityLeft` field in IBL trials).
    choice :
        Array of shape (N_valid,) with behavioural choices for valid trials,
        restricted to ±1 (left / right) as in Eqs. (74)–(75).
    """

    mask: np.ndarray
    w: np.ndarray
    t_stim: np.ndarray
    priors_left: np.ndarray
    choice: np.ndarray

    def __iter__(self) -> Iterable[np.ndarray]:
        """
        Allow tuple-like unpacking:

        >>> mask, w, t_stim, priors, choice = select_valid_trials(trials)
        """
        yield from (self.mask, self.w, self.t_stim, self.priors_left, self.choice)


# ---------------------------------------------------------------------------
# World states, contexts, and valid trials (Section 3.2)
# ---------------------------------------------------------------------------


def select_valid_trials(trials) -> ValidTrialSelection:
    """
    Apply the valid–trial mask and construct world labels and contexts.

    This follows Section 3.2 verbatim: we define a binary world state
    from left and right contrasts, treat missing contrasts as zero, use
    block priors as contexts, and apply three inclusion criteria.

    Parameters
    ----------
    trials :
        The IBL `trials` ALF object as returned by ONE.load_object(eid, "trials").
        Must contain at least the fields:

        - 'contrastLeft', 'contrastRight'
        - 'stimOn_times'
        - 'probabilityLeft'
        - 'choice'

    Returns
    -------
    selection : ValidTrialSelection
        Dataclass bundling the valid–trial mask, world labels w ∈ {0,1},
        stimulus onset times, block priors, and choices.

    Notes
    -----
    World state
        Let c_L and c_R denote the signed Michelson contrasts on the left and
        right. Missing contrasts are treated as 0, and we work with absolute
        values c̃_L = |c_L|, c̃_R = |c_R|. We define

            world_raw := sign(c̃_R − c̃_L),                         (Eq. 73)

        and encode Left as w = 0 (world_raw < 0) and Right as w = 1
        (world_raw > 0). Trials with world_raw = 0 are excluded by the
        inclusion criteria below.

    Valid–trial mask
        A trial is considered valid if and only if:

        1. exactly one side has a nonzero stimulus:
               (|c_L| > 0) XOR (|c_R| > 0)
        2. a left or right choice was made:
               choice ∈ {−1, +1}
        3. the stimulus onset time is finite (not NaN or ±inf).

        The conjunction of these criteria yields `mask`.

    Behavioural correctness
        Behavioural correctness is *not* computed here; see
        `compute_correct_choices`, which implements Eqs. (74)–(75) with an
        additional defensive check on the sign convention for `choice`.
    """
    # Extract and sanitise contrasts; missing values are treated as 0
    cL = np.asarray(trials["contrastLeft"], dtype=float)
    cR = np.asarray(trials["contrastRight"], dtype=float)
    cL = np.where(np.isfinite(cL), cL, 0.0)
    cR = np.where(np.isfinite(cR), cR, 0.0)

    cL_abs = np.abs(cL)
    cR_abs = np.abs(cR)

    # World state before masking: sign(cR_abs - cL_abs)
    world_raw = np.sign(cR_abs - cL_abs).astype(int)

    # Context: block prior over left
    priors_left_full = np.asarray(trials["probabilityLeft"], dtype=float)

    # Behavioural choice and stimulus onset
    choice_full = np.asarray(trials["choice"], dtype=float)
    t_stim_full = np.asarray(trials["stimOn_times"], dtype=float)

    # 1) Exactly one side has a nonzero stimulus
    stim_left = cL_abs > 0
    stim_right = cR_abs > 0
    stim_one_side = np.logical_xor(stim_left, stim_right)

    # 2) Left or right choice (exclude no-go = 0)
    choice_valid = np.isin(choice_full, (-1.0, 1.0))

    # 3) Finite stimulus onset time
    onset_valid = np.isfinite(t_stim_full)

    mask = stim_one_side & choice_valid & onset_valid

    if not np.any(mask):
        raise ValueError("No trials passed the valid–trial mask.")

    # Apply mask
    w_sub = world_raw[mask]
    t_stim_sub = t_stim_full[mask]
    priors_left_sub = priors_left_full[mask]
    choice_sub = choice_full[mask]

    # Map world_raw ∈ {−1, +1} to {0, 1}; world_raw should never be 0 under the mask
    if np.any(w_sub == 0):
        # This would indicate a violation of the “exactly one side nonzero”
        # assumption; we flag it explicitly.
        raise RuntimeError(
            "Encountered world_raw = 0 for a trial that passed the valid–trial "
            "mask. Check contrast fields and mask construction."
        )

    w = np.zeros_like(w_sub, dtype=int)
    w[w_sub > 0] = 1  # Right
    w[w_sub < 0] = 0  # Left

    return ValidTrialSelection(
        mask=mask,
        w=w,
        t_stim=t_stim_sub,
        priors_left=priors_left_sub,
        choice=choice_sub,
    )


def compute_correct_choices(w: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """
    Compute behavioural correctness for each trial, robust to sign conventions.

    Conceptually, correctness is defined as in Eqs. (74)–(75): a trial is
    correct if the choice points to the side with the higher contrast, i.e.

        correct_i = 1  if  (choice_i = Left  and  w_i = 0)
                        or (choice_i = Right and  w_i = 1),
                    0  otherwise.

    In the IBL dataset, choices are coded as −1 (left), 0 (no-go), +1 (right),
    but to guard against possible inversions in the source data, the
    implementation evaluates correctness under *both* sign conventions and
    adopts the one with higher accuracy.

    Parameters
    ----------
    w :
        Array of shape (N,) with binary world labels (0=Left, 1=Right) for
        valid trials.
    choice :
        Array of shape (N,) with choices restricted to ±1 for the same trials.

    Returns
    -------
    is_correct : np.ndarray
        Boolean array of shape (N,) where True indicates a correct choice.

    Raises
    ------
    ValueError
        If `choice` takes values outside {−1, +1}.
    """
    w = np.asarray(w, dtype=int)
    choice = np.asarray(choice, dtype=float)

    unique_choices = np.unique(choice)
    if not np.all(np.isin(unique_choices, (-1.0, 1.0))):
        raise ValueError(
            f"compute_correct_choices expects choices in {{-1, +1}}; "
            f"got unique values {unique_choices!r}"
        )

    def correctness_for_sign(left_code: float) -> np.ndarray:
        """Return correctness under a given mapping of `left_code` to Left."""
        right_code = -left_code
        is_left_choice = choice == left_code
        is_right_choice = choice == right_code
        # Left is w = 0, Right is w = 1
        correct = (is_left_choice & (w == 0)) | (is_right_choice & (w == 1))
        return correct

    correct_left_minus = correctness_for_sign(-1.0)
    correct_left_plus = correctness_for_sign(+1.0)

    acc_minus = correct_left_minus.mean()
    acc_plus = correct_left_plus.mean()

    if acc_minus >= acc_plus:
        is_correct = correct_left_minus
        chosen_mapping = "(-1→Left, +1→Right)"
    else:
        is_correct = correct_left_plus
        chosen_mapping = "(-1→Right, +1→Left)"

    log.info(
        "Behavioural correctness computed using choice mapping %s "
        "(accuracy %.3f vs %.3f for the alternative).",
        chosen_mapping,
        max(acc_minus, acc_plus),
        min(acc_minus, acc_plus),
    )

    return is_correct


# ---------------------------------------------------------------------------
# Spike–count matrices (Section 3.3)
# ---------------------------------------------------------------------------


def build_spike_matrix(
    spikes,
    clusters,
    t_stim: np.ndarray,
    cfg: SessionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a trial × neuron spike–count matrix in a peri‑stimulus window.

    This follows Section 3.3: for each valid trial i we define an analysis
    window aligned to stimulus onset,

        [t_start_i, t_end_i] = [t_stim_i + t0, t_stim_i + t1],      (Eq. 76)

    with (t0, t1) taken from `SessionConfig.time_window`. Spikes of each
    unit falling into this window are counted to yield X_ij, the number of
    spikes of unit j on trial i (Eq. 77).

    Unit selection mirrors the methods exactly:

    - we start from all clusters present in the spike-sorting output,
    - restrict to “good” units according to IBL quality labels
      (`clusters['label'] == 1`),
    - optionally restrict to a particular anatomical region via
      `clusters['acronym'] == cfg.region`,
    - and finally discard units whose total spike count across valid trials
      is below `cfg.min_total_spikes` (Eq. 78).

    Parameters
    ----------
    spikes :
        ALF `spikes` object for this probe, containing at least
        `spikes['times']` (seconds) and `spikes['clusters']` (cluster IDs).
    clusters :
        ALF `clusters` object for this probe, as returned by
        `SpikeSortingLoader.merge_clusters`. Must contain at least the
        fields `label` (quality) and, if `cfg.region` is not None,
        `acronym` (anatomical region).
    t_stim :
        Array of shape (N_trials,) with stimulus onset times (seconds) for
        the *valid* trials, in the same order as the world labels etc.
    cfg :
        SessionConfig instance specifying the peri‑stimulus window
        (time_window), minimum spike count (min_total_spikes), and optional
        region filter.

    Returns
    -------
    X : np.ndarray
        Array of shape (N_trials, N_units) with spike counts X_ij for each
        trial i and retained unit j.
    unit_ids : np.ndarray
        Array of shape (N_units,) with the corresponding cluster IDs in the
        original `clusters` object.

    Raises
    ------
    ValueError
        If no units remain after applying quality, region, and activity
        filters.
    """
    cfg.validate()  # basic sanity checks (time window, thresholds, etc.)

    t_stim = np.asarray(t_stim, dtype=float)
    if t_stim.ndim != 1:
        raise ValueError(
            f"t_stim must be a 1D array of onset times; got shape {t_stim.shape}"
        )

    n_trials = t_stim.shape[0]
    if n_trials == 0:
        raise ValueError("t_stim is empty: there are no valid trials.")

    # Construct per-trial intervals [t_stim + t0, t_stim + t1]
    t0, t1 = cfg.time_window
    intervals = np.column_stack((t_stim + t0, t_stim + t1))

    # Compute spike counts for *all* clusters using brainbox; this returns
    # counts of shape (n_neurons_all, n_trials) and the corresponding
    # cluster_ids array.
    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=int)

    counts_all, cluster_ids_all = get_spike_counts_in_bins(
        spike_times, spike_clusters, intervals
    )  # counts_all: (n_neurons_all, n_trials)

    # --- Quality filter: good units only (clusters['label'] == 1) ---
    labels = np.asarray(clusters["label"], dtype=int)
    # cluster_ids_all indexes into clusters, so we can look up labels
    good_mask = labels[cluster_ids_all] == 1

    if cfg.region is not None:
        # Optional region filter via clusters['acronym']
        if "acronym" not in clusters:
            raise KeyError(
                "cfg.region is not None, but clusters['acronym'] is missing."
            )
        acronyms = np.asarray(clusters["acronym"])
        region_mask = acronyms[cluster_ids_all] == cfg.region
        unit_mask = good_mask & region_mask
    else:
        unit_mask = good_mask

    if not np.any(unit_mask):
        raise ValueError(
            "No units remain after applying quality and region filters. "
            "Consider relaxing cfg.region or inspecting cluster labels."
        )

    counts_sel = counts_all[unit_mask, :]  # (n_units_sel, n_trials)
    unit_ids_sel = cluster_ids_all[unit_mask]

    # --- Activity filter: minimum total spike count across trials (Eq. 78) ---
    total_spikes = counts_sel.sum(axis=1)  # S_j = Σ_i X_ij

    active_mask = total_spikes >= cfg.min_total_spikes
    if not np.any(active_mask):
        raise ValueError(
            "All units were discarded by the min_total_spikes filter "
            f"(threshold {cfg.min_total_spikes})."
        )

    counts_final = counts_sel[active_mask, :]
    unit_ids_final = unit_ids_sel[active_mask]

    # Re-orient to trials × units and cast to float for downstream FA / GLM
    X = counts_final.T.astype(float)  # shape: (n_trials, n_units)

    log.info(
        "Constructed spike matrix with shape %s from %d candidate clusters "
        "(%d after quality/region filtering, %d after activity filter).",
        X.shape,
        counts_all.shape[0],
        counts_sel.shape[0],
        counts_final.shape[0],
    )

    return X, unit_ids_final


# ---------------------------------------------------------------------------
# Train–test split (Section 3.4)
# ---------------------------------------------------------------------------


def train_test_split_indices(
    n_trials: int,
    w: np.ndarray,
    cfg: SessionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a stratified train–test split of trials.

    This is a thin wrapper around `sklearn.model_selection.train_test_split`
    that reproduces the setting described in Section 3.4: a single 80/20
    split stratified by world state w, with a fixed random seed.

    The indices returned by this function are used *consistently* for:

    - fitting the logistic decoder (actual listener),
    - computing test-set accuracy,
    - and, in the “test-only” analyses, recomputing IG and ℓ⋆ on held-out
      trials only.

    Parameters
    ----------
    n_trials :
        Total number of valid trials in the session.
    w :
        Array of shape (n_trials,) with world labels (0/1) for these trials.
    cfg :
        SessionConfig instance whose `test_size`, `random_state`, and
        `stratify_by_world` fields control the split.

    Returns
    -------
    idx_train : np.ndarray
        1D integer array of indices for training trials.
    idx_test : np.ndarray
        1D integer array of indices for test trials.

    Raises
    ------
    ValueError
        If `n_trials` and `w` are inconsistent.
    """
    from sklearn.model_selection import train_test_split

    if n_trials <= 1:
        raise ValueError(
            f"Need at least 2 trials for a train–test split; got n_trials={n_trials}."
        )

    w = np.asarray(w, dtype=int)
    if w.shape[0] != n_trials:
        raise ValueError(
            f"Length of w ({w.shape[0]}) does not match n_trials ({n_trials})."
        )

    indices = np.arange(n_trials)

    if cfg.stratify_by_world:
        stratify = w
    else:
        stratify = None

    idx_train, idx_test = train_test_split(
        indices,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    # Sanity: ensure disjointness and coverage
    if np.intersect1d(idx_train, idx_test).size != 0:
        raise RuntimeError("Train–test indices are not disjoint.")
    if np.union1d(idx_train, idx_test).size != n_trials:
        raise RuntimeError(
            "Train–test indices do not cover all trials; "
            "check n_trials and stratification."
        )

    log.info(
        "Train–test split: %d training trials, %d test trials (test_size=%.2f).",
        idx_train.size,
        idx_test.size,
        cfg.test_size,
    )

    return idx_train, idx_test


__all__ = [
    "ValidTrialSelection",
    "select_valid_trials",
    "compute_correct_choices",
    "build_spike_matrix",
    "train_test_split_indices",
]
