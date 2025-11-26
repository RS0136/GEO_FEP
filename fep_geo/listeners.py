"""
Latent geometry via factor analysis and Gaussian prototypes.

This module implements the parts of the pipeline described in Sections 3.4–3.7
of the preprint:

- a stratified train–test split over trials (Eq. (79));
- a factor-analysis (FA) latent representation of trial-by-neuron spike counts
  (Eqs. (80)–(83));
- Gaussian state-dependent prototypes with a shared covariance in latent space
  (Eqs. (84)–(85));
- squared Mahalanobis distances to these prototypes, which are later used to
  construct the ideal listener (Eqs. (86), (87)–(88)).

The central object is a LatentGeometry dataclass that bundles the latent code
matrix Z, Gaussian prototype parameters, and the train–test split indices for
a single probe insertion. All operations here are purely geometric/statistical;
no decoding or information-theoretic metrics are computed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split

from .config import SessionConfig
from .trials import TrialData


# ---------------------------------------------------------------------------
# Dataclass: latent geometry for a single probe insertion
# ---------------------------------------------------------------------------

@dataclass
class LatentGeometry:
    """
    Factor-analysis latent codes and Gaussian prototypes for a single session.

    Attributes
    ----------
    pid:
        Probe UUID (PID) for this insertion, copied from TrialData.pid.

    Z:
        Latent code matrix of shape (N_trials, d), with one d-dimensional
        vector z_i per valid trial, obtained as the posterior mean E[z_i | x_i]
        under a factor-analysis model (Eqs. (80)–(83)).

    d:
        Effective latent dimensionality used for FA. This is always
            d = min(cfg.n_latent, max(1, D − 1))
        where D is the number of retained units (Eq. (82)).

    idx_train:
        Integer array of shape (N_train,) containing trial indices used for
        fitting the logistic decoder and (optionally) FA when
        cfg.fa_fit_mode == "train". Derived via a stratified train–test
        split as in Eq. (79).

    idx_test:
        Integer array of shape (N_test,) containing trial indices reserved for
        evaluation of the logistic decoder and test-only metrics.

    mu_left:
        Latent-space prototype mean μ̂_0 for world state w = 0 (Left), of
        shape (d,), computed as in Eq. (84).

    mu_right:
        Latent-space prototype mean μ̂_1 for world state w = 1 (Right),
        also of shape (d,).

    Sigma:
        Shared latent covariance Σ̂ estimated by pooling all latent codes,
        shape (d, d), as in Eq. (85). A small ridge term λ I_d with
        λ = cfg.gaussian_reg is added before inversion.

    Sigma_inv:
        Inverse of the regularised covariance, Σ̂^{-1}, shape (d, d).
        This is the matrix used in Mahalanobis distances (Eq. (86)).

    fa_model:
        The fitted sklearn.decomposition.FactorAnalysis instance. Stored
        for reproducibility and potential downstream diagnostics.
    """

    pid: str

    Z: np.ndarray
    d: int
    idx_train: np.ndarray
    idx_test: np.ndarray

    mu_left: np.ndarray
    mu_right: np.ndarray
    Sigma: np.ndarray
    Sigma_inv: np.ndarray

    fa_model: FactorAnalysis

    @property
    def n_trials(self) -> int:
        """Number of valid trials (rows of Z)."""
        return int(self.Z.shape[0])

    @property
    def n_latent(self) -> int:
        """Latent dimensionality d (columns of Z)."""
        return int(self.Z.shape[1])


# ---------------------------------------------------------------------------
# Train–test split (Section 3.4)
# ---------------------------------------------------------------------------

def make_train_test_split(
    trial_data: TrialData,
    cfg: SessionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a stratified train–test split over valid trials.

    This reproduces the procedure described in Section 3.4 (Eq. (79)):

        (idx_train, idx_test, w_train, w_test) =
            train_test_split(
                i, w,
                test_size   = 0.2,
                random_state= 0,
                stratify    = w
            )

    where i = [0, 1, ..., N_trials − 1] indexes valid trials and w is the
    world label (0 for Left, 1 for Right).

    Parameters
    ----------
    trial_data:
        TrialData containing world labels w for all valid trials.
    cfg:
        SessionConfig specifying test_size, random_state, and whether to
        stratify by world label.

    Returns
    -------
    idx_train, idx_test : np.ndarray
        Integer index arrays partitioning the valid trials into training and
        test sets, with no overlap and covering all indices.
    """
    cfg.validate()

    n_trials = trial_data.n_trials
    indices = np.arange(n_trials, dtype=int)
    w = np.asarray(trial_data.w, dtype=int)

    if w.shape[0] != n_trials:
        raise RuntimeError(
            f"Mismatch between number of trials in TrialData.w "
            f"({w.shape[0]}) and X ({n_trials})."
        )

    stratify = w if cfg.stratify_by_world else None

    idx_train, idx_test, _, _ = train_test_split(
        indices,
        w,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    if idx_train.size == 0 or idx_test.size == 0:
        raise RuntimeError(
            f"Empty train or test split (train={idx_train.size}, "
            f"test={idx_test.size}); check cfg.test_size and the number "
            "of valid trials."
        )

    return np.asarray(idx_train, dtype=int), np.asarray(idx_test, dtype=int)


# ---------------------------------------------------------------------------
# Factor analysis latent codes (Section 3.5)
# ---------------------------------------------------------------------------

def _compute_effective_latent_dim(D: int, cfg: SessionConfig) -> int:
    """
    Compute the effective latent dimensionality d as in Eq. (82).

        d = min( n_latent, max(1, D − 1) )

    where D is the number of retained units (columns of X).
    """
    if D < 1:
        raise ValueError("Number of units D must be at least 1.")
    d = min(cfg.n_latent, max(1, D - 1))
    return int(d)


def fit_factor_analysis(
    trial_data: TrialData,
    idx_train: np.ndarray,
    cfg: SessionConfig,
) -> Tuple[np.ndarray, FactorAnalysis]:
    """
    Fit a factor-analysis model and obtain latent codes for all valid trials.

    Model specification (Eqs. (80)–(81)):

        z_i ~ N(0, I_d)
        x_i | z_i ~ N(W z_i + μ, Ψ),

    where x_i is the i-th row of the spike-count matrix X, W is a loading
    matrix, μ is a mean vector, and Ψ is a diagonal noise covariance.

    The effective latent dimensionality is

        d = min(cfg.n_latent, max(1, D − 1)),  (Eq. (82))

    where D is the number of retained units (columns of X).

    Fitting modes
    -------------
    - If cfg.fa_fit_mode == "train":
        Fit FA on training trials X[idx_train] and then call fa.transform(X)
        to obtain posterior means z_i = E[z_i | x_i] for all valid trials.
        This is the main mode used in the preprint and avoids leakage of
        test information into the geometry.

    - If cfg.fa_fit_mode == "all":
        Fit FA on all valid trials X and obtain latent codes via
        fa.fit_transform(X). This is useful for robustness checks but was
        not used for the main reported results.

    Parameters
    ----------
    trial_data:
        TrialData containing the spike-count matrix X.
    idx_train:
        Training-trial indices for fitting FA when fa_fit_mode == "train".
    cfg:
        SessionConfig with FA-related hyperparameters.

    Returns
    -------
    Z : np.ndarray
        Latent code matrix of shape (N_trials, d).
    fa : FactorAnalysis
        Fitted FactorAnalysis object.
    """
    cfg.validate()

    X = np.asarray(trial_data.X, dtype=float)
    N, D = X.shape

    if idx_train.ndim != 1:
        raise ValueError("idx_train must be a 1D array of indices.")
    if np.any(idx_train < 0) or np.any(idx_train >= N):
        raise IndexError("idx_train contains indices outside [0, N_trials).")

    d = _compute_effective_latent_dim(D, cfg)

    fa = FactorAnalysis(
        n_components=d,
        random_state=cfg.random_state,
    )

    if cfg.fa_fit_mode == "train":
        # Fit only on training trials, transform all trials
        fa.fit(X[idx_train])
        Z = fa.transform(X)
    elif cfg.fa_fit_mode == "all":
        Z = fa.fit_transform(X)
    else:
        raise ValueError(
            f"Unknown fa_fit_mode={cfg.fa_fit_mode!r}; "
            "expected 'train' or 'all'."
        )

    if Z.shape != (N, d):
        raise RuntimeError(
            "Unexpected latent shape: got "
            f"{Z.shape}, expected ({N}, {d})."
        )

    return Z, fa


# ---------------------------------------------------------------------------
# Gaussian prototypes and Mahalanobis geometry (Section 3.6)
# ---------------------------------------------------------------------------

def estimate_gaussian_prototypes(
    Z: np.ndarray,
    w: np.ndarray,
    cfg: SessionConfig,
    idx_fit: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Gaussian prototypes and shared covariance in latent space.

    If idx_fit is not None, prototypes and covariance are estimated
    **only** from Z[idx_fit], w[idx_fit]; otherwise all trials are used.
    """
    cfg.validate()

    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=int)

    if idx_fit is not None:
        idx_fit = np.asarray(idx_fit, dtype=int)
        Z_fit = Z[idx_fit]
        w_fit = w[idx_fit]
    else:
        Z_fit = Z
        w_fit = w

    if Z_fit.ndim != 2:
        raise ValueError(f"Z_fit must be 2D, got shape {Z_fit.shape}.")
    if w_fit.shape[0] != Z_fit.shape[0]:
        raise ValueError(
            f"w_fit must have the same length as Z_fit has rows; "
            f"got {w_fit.shape[0]} vs {Z_fit.shape[0]}."
        )

    N, d = Z_fit.shape

    mask_left = w_fit == 0
    mask_right = w_fit == 1
    if not np.any(mask_left) or not np.any(mask_right):
        raise ValueError(
            "Both world states (0 and 1) must be present to estimate "
            "state-dependent prototypes."
        )

    mu_left = Z_fit[mask_left].mean(axis=0)
    mu_right = Z_fit[mask_right].mean(axis=0)

    z_bar = Z_fit.mean(axis=0)
    diff = Z_fit - z_bar
    Sigma = (diff.T @ diff) / max(1, N - 1)

    if cfg.gaussian_reg < 0:
        raise ValueError("gaussian_reg must be ≥ 0.")
    if cfg.gaussian_reg > 0:
        Sigma = Sigma + cfg.gaussian_reg * np.eye(d)

    Sigma_inv = np.linalg.inv(Sigma)

    return mu_left, mu_right, Sigma, Sigma_inv

def mahalanobis_sq(
    z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
) -> float:
    """
    Compute squared Mahalanobis distance d_w^2(z) as in Eq. (86).

        d_w^2(z) := (z − μ_w)^T Σ^{-1} (z − μ_w).

    Parameters
    ----------
    z:
        Latent code vector of shape (d,).
    mu:
        Prototype mean of shape (d,).
    Sigma_inv:
        Inverse covariance Σ^{-1} of shape (d, d).

    Returns
    -------
    d2 : float
        Squared Mahalanobis distance.
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma_inv = np.asarray(Sigma_inv, dtype=float)

    diff = z - mu
    # (1 x d) @ (d x d) @ (d x 1) -> scalar
    return float(diff.T @ Sigma_inv @ diff)


def mahalanobis_sq_all(
    Z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
) -> np.ndarray:
    """
    Vectorised squared Mahalanobis distances for all trials.

    Parameters
    ----------
    Z:
        Latent code matrix of shape (N_trials, d).
    mu:
        Prototype mean of shape (d,).
    Sigma_inv:
        Inverse covariance Σ^{-1} of shape (d, d).

    Returns
    -------
    d2 : np.ndarray
        Array of shape (N_trials,) with d_w^2(z_i) for each row z_i of Z.
    """
    Z = np.asarray(Z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma_inv = np.asarray(Sigma_inv, dtype=float)

    diff = Z - mu  # (N, d)
    # Each row: diff_i @ Sigma_inv @ diff_i^T
    # Compute via (diff @ Sigma_inv) * diff, then sum over columns
    tmp = diff @ Sigma_inv  # (N, d)
    d2 = np.einsum("ij,ij->i", tmp, diff)
    return d2


# ---------------------------------------------------------------------------
# High-level helper: from TrialData to LatentGeometry
# ---------------------------------------------------------------------------

def construct_latent_geometry(
    trial_data: TrialData,
    cfg: SessionConfig,
) -> LatentGeometry:
    """
    End-to-end construction of latent geometry from TrialData.

    This function encapsulates all latent-geometry steps described in
    Sections 3.4–3.7:

      1. Construct a stratified train–test split over trials (Eq. (79));
      2. Fit a factor-analysis model and obtain latent codes Z (Eqs. (80)–(83));
      3. Estimate state-dependent Gaussian prototypes and shared covariance
         (Eqs. (84)–(85));
      4. Store these components, together with the FA model and split indices,
         in a LatentGeometry dataclass.

    The resulting LatentGeometry is the natural input to the listener
    module, which uses Z, μ̂_w, and Σ̂^{-1} to construct ideal and actual
    listeners and to compute information-theoretic metrics.

    Parameters
    ----------
    trial_data:
        TrialData for a single probe insertion.
    cfg:
        SessionConfig specifying latent-geometry hyperparameters.

    Returns
    -------
    geom : LatentGeometry
        Dataclass containing latent codes, Gaussian parameters, and
        train–test indices.
    """
    cfg.validate()

    # Step 1: stratified train–test split (Section 3.4)
    idx_train, idx_test = make_train_test_split(trial_data, cfg)
    Z, fa = fit_factor_analysis(trial_data, idx_train, cfg)

    if cfg.ideal_fit_subset == "train":
        idx_proto = idx_train
    elif cfg.ideal_fit_subset == "all":
        idx_proto = None
    else:
        raise ValueError(f"Unknown ideal_fit_subset={cfg.ideal_fit_subset!r}")

    mu_left, mu_right, Sigma, Sigma_inv = estimate_gaussian_prototypes(
        Z=Z,
        w=trial_data.w,
        cfg=cfg,
        idx_fit=idx_proto,
    )

    geom = LatentGeometry(
        pid=trial_data.pid,
        Z=Z,
        d=Z.shape[1],
        idx_train=idx_train,
        idx_test=idx_test,
        mu_left=mu_left,
        mu_right=mu_right,
        Sigma=Sigma,
        Sigma_inv=Sigma_inv,
        fa_model=fa,
    )

    return geom


__all__ = [
    "LatentGeometry",
    "make_train_test_split",
    "fit_factor_analysis",
    "estimate_gaussian_prototypes",
    "mahalanobis_sq",
    "mahalanobis_sq_all",
    "construct_latent_geometry",
]
