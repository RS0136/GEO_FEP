from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import SessionConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for listener outputs
# ---------------------------------------------------------------------------


@dataclass
class IdealListenerResult:
    """
    Result of applying the ideal geometric listener to all trials.

    posterior :
        (N_trials, 2) array with ideal posteriors
        [P(w=0 | z_i, c_i), P(w=1 | z_i, c_i)].
    log_posterior :
        Same shape as `posterior`, natural logarithms.
    predicted_w :
        (N_trials,) MAP labels (0=Left, 1=Right).
    accuracy :
        Fraction of correctly decoded trials.
    d2_left, d2_right :
        Squared Mahalanobis distances to Left / Right prototypes.
    """

    posterior: np.ndarray
    log_posterior: np.ndarray
    predicted_w: np.ndarray
    accuracy: float
    d2_left: np.ndarray
    d2_right: np.ndarray


@dataclass
class ActualListenerResult:
    """
    Result of fitting a logistic regression decoder.

    posterior :
        (N_trials, 2) array with predicted probabilities
        [P(w=0 | features_i), P(w=1 | features_i)].
    predicted_w :
        (N_trials,) MAP labels.
    accuracy_test :
        Test-set accuracy on held-out trials.
    model :
        Fitted sklearn LogisticRegression object.
    features :
        Feature matrix used to fit the model.
    kind :
        String label, e.g. "actual" or "prior-aware".
    """

    posterior: np.ndarray
    predicted_w: np.ndarray
    accuracy_test: float
    model: LogisticRegression
    features: np.ndarray
    kind: str = "actual"


# ---------------------------------------------------------------------------
# Ideal listener from Gaussian latent geometry
# ---------------------------------------------------------------------------


def _mahalanobis_sq(
    Z: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
) -> np.ndarray:
    """
    Squared Mahalanobis distances d^2(z_i; mu, Sigma_inv) for all trials.
    """
    Z = np.asarray(Z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma_inv = np.asarray(Sigma_inv, dtype=float)

    diff = Z - mu  # broadcasts over trials
    # einsum 'ij,jk,ik->i' gives Σ_{j,k} diff[i,j] * Sigma_inv[j,k] * diff[i,k]
    d2 = np.einsum("ij,jk,ik->i", diff, Sigma_inv, diff)
    return d2


def ideal_listener_gaussian(
    Z: np.ndarray,
    w_true: np.ndarray,
    priors_left: np.ndarray,
    mu_left: np.ndarray,
    mu_right: np.ndarray,
    Sigma_inv: np.ndarray,
    cfg: SessionConfig,
) -> IdealListenerResult:
    """
    Ideal listener L_ideal(w | z_i, c_i) from latent geometry and block priors.
    """
    cfg.validate()

    Z = np.asarray(Z, dtype=float)
    w_true = np.asarray(w_true, dtype=int)
    priors_left = np.asarray(priors_left, dtype=float)

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (N_trials × d); got shape {Z.shape}.")
    n_trials = Z.shape[0]
    if w_true.shape[0] != n_trials:
        raise ValueError(
            f"Length of w_true ({w_true.shape[0]}) does not match "
            f"number of rows in Z ({n_trials})."
        )
    if priors_left.shape[0] != n_trials:
        raise ValueError(
            f"Length of priors_left ({priors_left.shape[0]}) does not match "
            f"number of rows in Z ({n_trials})."
        )

    # Mahalanobis distances for Left (w=0) and Right (w=1)
    d2_left = _mahalanobis_sq(Z, mu_left, Sigma_inv)
    d2_right = _mahalanobis_sq(Z, mu_right, Sigma_inv)

    # Prior probabilities P(w | c_i), clipped away from 0 and 1
    eps = cfg.metrics_eps
    p_left = np.clip(priors_left, eps, 1.0 - eps)
    p_right = np.clip(1.0 - priors_left, eps, 1.0 - eps)

    log_p_left = -0.5 * d2_left + np.log(p_left)
    log_p_right = -0.5 * d2_right + np.log(p_right)

    # Normalisation via log-sum-exp
    log_Z = np.logaddexp(log_p_left, log_p_right)

    log_post_left = log_p_left - log_Z
    log_post_right = log_p_right - log_Z

    log_posterior = np.stack([log_post_left, log_post_right], axis=1)
    posterior = np.exp(log_posterior)

    row_sums = posterior.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        log.warning(
            "Ideal listener posterior rows do not sum exactly to 1 "
            "(min=%.6f, max=%.6f).",
            float(row_sums.min()),
            float(row_sums.max()),
        )

    predicted_w = posterior.argmax(axis=1)
    accuracy = (predicted_w == w_true).mean()

    return IdealListenerResult(
        posterior=posterior,
        log_posterior=log_posterior,
        predicted_w=predicted_w,
        accuracy=float(accuracy),
        d2_left=d2_left,
        d2_right=d2_right,
    )


# ---------------------------------------------------------------------------
# Logistic decoders as actual listeners
# ---------------------------------------------------------------------------


def _fit_logistic_regression(
    X: np.ndarray,
    w_true: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    cfg: SessionConfig,
    kind: str,
) -> ActualListenerResult:
    """
    Fit a logistic regression decoder and return posteriors.
    """
    X = np.asarray(X, dtype=float)
    w_true = np.asarray(w_true, dtype=int)
    idx_train = np.asarray(idx_train, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)

    n_trials, d_feat = X.shape
    if w_true.shape[0] != n_trials:
        raise ValueError(
            f"Length of w_true ({w_true.shape[0]}) does not match "
            f"number of rows in X ({n_trials})."
        )

    # Binary {0,1} world state
    unique_w = np.unique(w_true)
    if not np.array_equal(unique_w, np.array([0, 1], dtype=int)):
        raise ValueError(
            f"w_true must contain exactly the labels {{0,1}}; got {unique_w}."
        )

    clf = LogisticRegression(
        penalty=cfg.logistic_penalty,
        C=cfg.logistic_C,
        class_weight=cfg.logistic_class_weight,
        solver=cfg.logistic_solver,
        max_iter=cfg.logistic_max_iter,
        random_state=cfg.random_state,
    )

    clf.fit(X[idx_train], w_true[idx_train])

    # Test accuracy on held-out trials
    w_pred_test = clf.predict(X[idx_test])
    accuracy_test = (w_pred_test == w_true[idx_test]).mean()

    # Class probabilities for all trials; reorder columns to [P(w=0), P(w=1)]
    proba_all = clf.predict_proba(X)  # (N_trials, n_classes)
    class_order = np.array([0, 1], dtype=int)
    reorder_idx = np.searchsorted(clf.classes_, class_order)
    posterior = proba_all[:, reorder_idx]

    row_sums = posterior.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        log.warning(
            "Actual listener posterior rows do not sum exactly to 1 "
            "(min=%.6f, max=%.6f).",
            float(row_sums.min()),
            float(row_sums.max()),
        )

    predicted_w = posterior.argmax(axis=1)

    return ActualListenerResult(
        posterior=posterior,
        predicted_w=predicted_w,
        accuracy_test=float(accuracy_test),
        model=clf,
        features=X,
        kind=kind,
    )


def logistic_actual_listener(
    Z: np.ndarray,
    w_true: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    cfg: SessionConfig,
) -> ActualListenerResult:
    """
    Prior-agnostic actual listener: logistic regression on latent codes Z.
    """
    return _fit_logistic_regression(
        X=np.asarray(Z, dtype=float),
        w_true=w_true,
        idx_train=idx_train,
        idx_test=idx_test,
        cfg=cfg,
        kind="actual",
    )


def _encode_prior_feature(
    priors_left: np.ndarray,
    cfg: SessionConfig,
) -> np.ndarray:
    """
    Encode block priors as a scalar feature according to cfg.prior_feature.
    """
    p = np.asarray(priors_left, dtype=float)
    if cfg.prior_feature == "probability_left":
        feat = p
    elif cfg.prior_feature == "logit_probability_left":
        eps = cfg.metrics_eps
        p_clipped = np.clip(p, eps, 1.0 - eps)
        feat = np.log(p_clipped / (1.0 - p_clipped))
    elif cfg.prior_feature == "none":
        raise ValueError(
            "cfg.prior_feature='none' is incompatible with prior-aware "
            "decoder; choose 'probability_left' or 'logit_probability_left'."
        )
    else:
        raise ValueError(f"Unknown prior_feature: {cfg.prior_feature!r}")

    return feat.reshape(-1, 1)


def prior_aware_logistic_listener(
    Z: np.ndarray,
    w_true: np.ndarray,
    priors_left: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    cfg: SessionConfig,
) -> ActualListenerResult:
    """
    Prior-aware control listener: logistic regression on [Z, prior feature].
    """
    Z = np.asarray(Z, dtype=float)
    priors_left = np.asarray(priors_left, dtype=float)

    if Z.shape[0] != priors_left.shape[0]:
        raise ValueError(
            f"Z and priors_left must have the same number of trials; "
            f"got {Z.shape[0]} and {priors_left.shape[0]}."
        )

    prior_feat = _encode_prior_feature(priors_left, cfg)
    X_aug = np.concatenate([Z, prior_feat], axis=1)

    return _fit_logistic_regression(
        X=X_aug,
        w_true=w_true,
        idx_train=idx_train,
        idx_test=idx_test,
        cfg=cfg,
        kind="prior-aware",
    )


# ---------------------------------------------------------------------------
# Information-theoretic metrics
# ---------------------------------------------------------------------------


@dataclass
class TrialMetrics:
    """
    Trial-wise information-theoretic metrics for a single session.

    IG      : IG for each trial (ideal listener with block priors)
    loss    : KL(L_ideal^prior || L_0^agnostic)
    loss_prior :
              KL(L_ideal^prior || L_0^prior)

    IG_uniform :
              IG for L_ideal^unif (uniform prior)
    loss_uniform :
              KL(L_ideal^unif || L_0^agnostic)
    loss_uniform_prior :
              KL(L_ideal^unif || L_0^prior)
    """

    IG: np.ndarray
    loss: np.ndarray
    loss_prior: Optional[np.ndarray] = None

    IG_uniform: Optional[np.ndarray] = None
    loss_uniform: Optional[np.ndarray] = None
    loss_uniform_prior: Optional[np.ndarray] = None

    @property
    def n_trials(self) -> int:
        return int(self.IG.shape[0])


# ---------------------------------------------------------------------------
# Low-level helpers: entropy and KL on discrete distributions
# ---------------------------------------------------------------------------


def _clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    """
    Clip probabilities to [eps, 1 − eps] and renormalise along the last axis.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}.")
    p = np.asarray(p, dtype=float)
    p_clipped = np.clip(p, eps, 1.0 - eps)
    Z = p_clipped.sum(axis=-1, keepdims=True)
    Z = np.where(Z <= 0.0, 1.0, Z)
    return p_clipped / Z


def discrete_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy H(p) for discrete distributions (in nats).

    p : (..., K) array, last axis are probabilities.
    """
    p_clipped = _clip_probs(p, eps)
    H = -np.sum(p_clipped * np.log(p_clipped), axis=-1)
    return H


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Kullback–Leibler divergence KL(p ∥ q) for discrete laws, in nats.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape; got {p.shape} vs {q.shape}.")

    p_clipped = _clip_probs(p, eps)
    q_clipped = _clip_probs(q, eps)

    log_ratio = np.log(p_clipped) - np.log(q_clipped)
    kl = np.sum(p_clipped * log_ratio, axis=-1)
    return kl


def compute_information_gain(
    prior: np.ndarray,
    posterior_ideal: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Single-trial information gain IG(u, c) in nats.

    prior :
        (N,) or (N,K). If 1D, binary case P(Left|c_i); P(Right|c_i)=1−prior.
    posterior_ideal :
        (N,K) ideal listener posterior.
    """
    posterior_ideal = np.asarray(posterior_ideal, dtype=float)
    if posterior_ideal.ndim != 2:
        raise ValueError(
            f"posterior_ideal must have shape (N, K); got {posterior_ideal.shape}"
        )
    N, K = posterior_ideal.shape

    prior = np.asarray(prior, dtype=float)
    if prior.ndim == 1:
        if K != 2:
            raise ValueError(
                "1D prior given but posterior_ideal has K != 2; "
                f"K = {K}. For non-binary, pass prior with shape (N, K)."
            )
        prior_probs = np.stack([prior, 1.0 - prior], axis=-1)
    elif prior.ndim == 2:
        if prior.shape != posterior_ideal.shape:
            raise ValueError(
                f"prior and posterior_ideal must have the same shape; "
                f"got {prior.shape} vs {posterior_ideal.shape}."
            )
        prior_probs = prior
    else:
        raise ValueError(
            f"prior must be 1D or 2D array; got shape {prior.shape}."
        )

    if prior_probs.shape[0] != N:
        raise ValueError(
            f"Number of trials in prior ({prior_probs.shape[0]}) "
            f"does not match posterior_ideal ({N})."
        )

    H_prior = discrete_entropy(prior_probs, eps=eps)
    H_post = discrete_entropy(posterior_ideal, eps=eps)
    IG = H_prior - H_post
    return IG


def compute_pragmatic_loss(
    posterior_ideal: np.ndarray,
    posterior_actual: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Trial-wise pragmatic loss ℓ⋆(u, c) = KL(L_ideal(· | u, c) ∥ L_0(· | u, c)).
    """
    posterior_ideal = np.asarray(posterior_ideal, dtype=float)
    posterior_actual = np.asarray(posterior_actual, dtype=float)

    if posterior_ideal.shape != posterior_actual.shape:
        raise ValueError(
            f"posterior_ideal and posterior_actual must have the same shape; "
            f"got {posterior_ideal.shape} vs {posterior_actual.shape}."
        )

    loss = kl_divergence(posterior_ideal, posterior_actual, eps=eps)
    return loss


# ---------------------------------------------------------------------------
# High-level helper: from listeners to TrialMetrics
# ---------------------------------------------------------------------------


def compute_trial_metrics(
    priors_left: np.ndarray,
    L_ideal: np.ndarray,
    L_actual: Optional[np.ndarray],
    cfg: SessionConfig,
    L_prior: Optional[np.ndarray] = None,
    L_ideal_uniform: Optional[np.ndarray] = None,
) -> TrialMetrics:
    """
    Compute IG and pragmatic losses for all trials.

    Parameters
    ----------
    priors_left :
        (N,) array of block priors P(Left | c_i).
    L_ideal :
        (N,2) ideal listener posteriors.
    L_actual :
        (N,2) actual listener posteriors, or None.
    cfg :
        SessionConfig (uses cfg.metrics_eps).
    L_prior :
        Optional (N,2) prior-aware listener posteriors.
    L_ideal_uniform :
        Optional (N,2) ideal listener with uniform prior (for controls).
    """
    cfg.validate()

    priors_left = np.asarray(priors_left, dtype=float).ravel()
    L_ideal = np.asarray(L_ideal, dtype=float)

    if L_ideal.ndim != 2 or L_ideal.shape[1] != 2:
        raise ValueError(
            "L_ideal must have shape (N, 2) for the binary world state case; "
            f"got {L_ideal.shape}."
        )
    if priors_left.shape[0] != L_ideal.shape[0]:
        raise ValueError(
            f"priors_left has length {priors_left.shape[0]}, "
            f"but L_ideal has {L_ideal.shape[0]} rows."
        )

    # 1. Information gain IG(u, c)
    IG = compute_information_gain(
        prior=priors_left,
        posterior_ideal=L_ideal,
        eps=cfg.metrics_eps,
    )

    # 2. Pragmatic loss vs. main actual listener
    if L_actual is not None:
        L_actual = np.asarray(L_actual, dtype=float)
        if L_actual.shape != L_ideal.shape:
            raise ValueError(
                f"L_actual and L_ideal must have the same shape; "
                f"got {L_actual.shape} vs {L_ideal.shape}."
            )
        loss = compute_pragmatic_loss(
            posterior_ideal=L_ideal,
            posterior_actual=L_actual,
            eps=cfg.metrics_eps,
        )
    else:
        loss = np.empty(0, dtype=float)

    # 3. Optional prior-aware pragmatic loss
    loss_prior = None
    if L_prior is not None:
        L_prior = np.asarray(L_prior, dtype=float)
        if L_prior.shape != L_ideal.shape:
            raise ValueError(
                f"L_prior and L_ideal must have the same shape; "
                f"got {L_prior.shape} vs {L_ideal.shape}."
            )
        loss_prior = compute_pragmatic_loss(
            posterior_ideal=L_ideal,
            posterior_actual=L_prior,
            eps=cfg.metrics_eps,
        )

    # 4. Optional uniform-prior ideal listener
    IG_uniform = None
    loss_uniform = None
    loss_uniform_prior = None

    if L_ideal_uniform is not None:
        L_ideal_uniform = np.asarray(L_ideal_uniform, dtype=float)
        if L_ideal_uniform.shape != L_ideal.shape:
            raise ValueError(
                f"L_ideal_uniform and L_ideal must have the same shape; "
                f"got {L_ideal_uniform.shape} vs {L_ideal.shape}."
            )

        n_trials = L_ideal_uniform.shape[0]
        priors_left_uniform = np.full(n_trials, 0.5, dtype=float)

        IG_uniform = compute_information_gain(
            prior=priors_left_uniform,
            posterior_ideal=L_ideal_uniform,
            eps=cfg.metrics_eps,
        )

        if L_actual is not None:
            loss_uniform = compute_pragmatic_loss(
                posterior_ideal=L_ideal_uniform,
                posterior_actual=L_actual,
                eps=cfg.metrics_eps,
            )

        if L_prior is not None:
            loss_uniform_prior = compute_pragmatic_loss(
                posterior_ideal=L_ideal_uniform,
                posterior_actual=L_prior,
                eps=cfg.metrics_eps,
            )

    return TrialMetrics(
        IG=IG,
        loss=loss,
        loss_prior=loss_prior,
        IG_uniform=IG_uniform,
        loss_uniform=loss_uniform,
        loss_uniform_prior=loss_uniform_prior,
    )


__all__ = [
    "IdealListenerResult",
    "ActualListenerResult",
    "ideal_listener_gaussian",
    "logistic_actual_listener",
    "prior_aware_logistic_listener",
    "TrialMetrics",
    "discrete_entropy",
    "kl_divergence",
    "compute_information_gain",
    "compute_pragmatic_loss",
    "compute_trial_metrics",
]
