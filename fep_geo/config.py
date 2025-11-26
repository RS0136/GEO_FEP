"""
Configuration dataclasses for the geometric FEP–pragmatics pipeline.

This module collects *all* user‑visible hyperparameters in one place so that:

- the main analysis reproduces the settings described in Section 3
  of the preprint (time window, latent dimensionality, regularisation, etc.),
- robustness analyses can systematically sweep over parameter grids,
- statistical assumptions (e.g. train–test split, regularisation strength)
  are explicit and testable rather than hard‑coded.

The design mirrors the original fep_geo package structure described in
Section 3.1, but is written to be self‑contained and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Sequence


# ---------------------------------------------------------------------------
# Session‑level configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """
    Configuration for running the full pipeline on a *single* probe insertion.

    Each field here corresponds either to:
      - a modelling assumption made explicit in the theoretical framework
        (e.g. Gaussian latent geometry, ideal vs. actual listener), or
      - an analysis choice documented in Section 3 of the preprint
        (e.g. peri‑stimulus window, train–test split).

    All numeric hyperparameters are given explicit units and constraints.

    Attributes
    ----------
    time_window:
        (t0, t1) in seconds, relative to stimulus onset, defining the
        peri‑stimulus window [t_stim + t0, t_stim + t1] used to count spikes
        on each trial; cf. Eq. (76).

        The main analysis in the preprint uses (0.0, 0.2), i.e. 0–200 ms
        after onset. Values MUST satisfy t0 < t1.

    min_total_spikes:
        Minimum total spike count per unit across all valid trials required
        for inclusion; see Eq. (78) and the accompanying text.
        Units with Σ_i X_ij < min_total_spikes are discarded.

        This threshold trades off variance (too few spikes) against bias
        (removing weak but informative units). The default 5 matches the
        preprint.

    region:
        Optional anatomical acronym used to restrict the analysis to a
        particular brain area (e.g. "VISp"). If None, all “good” units on
        the probe are included, as in the main analysis.

    n_latent:
        Target dimensionality d of the latent space for factor analysis (FA);
        cf. Eqs. (80)–(83).

        The actual dimension used will be
            d_eff = min(n_latent, max(1, n_units - 1)),
        to avoid asking FA for more dimensions than D−1.

        The preprint fixes n_latent = 10 for all sessions.

    fa_fit_mode:
        How FA is fitted:

        - "train" : fit on training trials only and transform all trials.
                    This is the *main* mode used in the preprint to avoid
                    leakage of test information into the geometry.
        - "all"   : fit on all valid trials; used only for robustness checks.

    gaussian_reg:
        Non‑negative ridge term λ added to the latent covariance Σ before
        inversion (Σ ← Σ + λ I_d); cf. Eq. (85).

        This guarantees Σ is positive definite and controls numerical
        stability of Mahalanobis distances. The default 1e‑6 reproduces
        the original code.

    test_size:
        Fraction of valid trials reserved for the test set in the
        stratified train–test split used to fit the logistic decoder;
        see Eq. (79).

        Must lie strictly between 0 and 0.5 for a meaningful split.
        The preprint uses 0.2.

    random_state:
        Seed for all pseudo‑random operations (train–test split, FA
        initialisation, logistic regression). Fixing this value ensures
        exact reproducibility of all reported numbers.

    stratify_by_world:
        If True, the train–test split is stratified by the world label w
        so that both training and test sets preserve the class balance.
        This matches the implementation in Section 3.4.

    logistic_C:
        Inverse ℓ2‑penalty strength for the logistic regression decoder
        (scikit‑learn’s `C` parameter). Larger values = weaker regularisation.

        From a Bayesian perspective this corresponds to a Gaussian prior
        on the decoder weights with variance proportional to C. Choosing
        C = 1.0 matches the default used in the preprint.

    logistic_penalty:
        Penalty type for the logistic decoder. Only "l2" is supported in
        the main analysis to mirror the preprint and to ensure convexity.

    logistic_max_iter:
        Maximum number of iterations for the logistic solver. Must be
        sufficiently large that the optimiser reliably converges on all
        sessions; 1000 is the setting used in Section 3.8.

    logistic_class_weight:
        Either None (no class weighting, as in the main analysis) or
        "balanced" (inverse‑frequency weighting). This affects the *actual*
        listener only and has no influence on the ideal listener or the
        scale of information‑theoretic metrics.

    logistic_solver:
        Name of the scikit‑learn solver used by LogisticRegression.
        For binary problems with ℓ2 penalty, "lbfgs" is a robust default.

    prior_feature:
        How, if at all, the block prior P(Left | c) is encoded as a feature
        for the *prior‑aware* control decoder (Section 3.8, Eq. (89)).

        - "none"                 : no prior feature (main actual listener).
        - "probability_left"     : P(Left | c_i) as a raw scalar feature.
        - "logit_probability_left": logit P(Left | c_i) after clipping, as in
                                    Eq. (89).

        The main prior‑aware analysis uses "logit_probability_left".

    metrics_eps:
        Small positive constant ε used to clip probabilities away from 0
        and 1 before taking logarithms when computing entropies and KL
        divergences; cf. Eqs. (90)–(93).

        This has *no* substantive effect if chosen sufficiently small; it
        only prevents numerical overflow in log(0).

    session_tag:
        Optional human‑readable identifier for the probe (e.g.
        "fece187f-b47f-4870-a1d6-619afe942a7d_probe01" as in Tables 1–4).
        This is not used in the computations but is propagated to
        filenames and summary tables.
    """

    # --- spike counting / unit selection ---
    time_window: Tuple[float, float] = (0.0, 0.2)
    min_total_spikes: int = 5
    region: Optional[str] = None

    # --- latent geometry (factor analysis) ---
    n_latent: int = 10
    fa_fit_mode: Literal["train", "all"] = "train"
    gaussian_reg: float = 1e-6

    ideal_fit_subset: Literal["all", "train"] = "all"

    # --- train–test split for the discriminative listener ---
    test_size: float = 0.2
    random_state: int = 0
    stratify_by_world: bool = True

    # --- logistic regression (actual / prior‑aware listeners) ---
    logistic_C: float = 1.0
    logistic_penalty: Literal["l2"] = "l2"
    logistic_max_iter: int = 1000
    logistic_class_weight: Optional[Literal["balanced"]] = None
    logistic_solver: str = "lbfgs"

    # --- prior feature encoding for prior‑aware control decoder ---
    prior_feature: Literal[
        "none",
        "probability_left",
        "logit_probability_left",
    ] = "logit_probability_left"

    # --- numerical safeguards for information‑theoretic metrics ---
    metrics_eps: float = 1e-12

    # --- optional label for reporting ---
    session_tag: Optional[str] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Perform basic sanity checks on the configuration.

        This method is intentionally conservative: it enforces constraints
        that are required for the theoretical derivations in Sections 2–3
        to make sense (e.g. positive covariance regularisation, non‑empty
        train–test splits) but does *not* attempt to validate dataset‑
        specific assumptions (such as the existence of a given probe ID).
        """
        t0, t1 = self.time_window
        if not (t0 < t1):
            raise ValueError(
                f"time_window must satisfy t0 < t1; got ({t0}, {t1})"
            )

        if self.min_total_spikes < 0:
            raise ValueError(
                f"min_total_spikes must be non‑negative; "
                f"got {self.min_total_spikes}"
            )

        if self.n_latent < 1:
            raise ValueError(
                f"n_latent must be at least 1; got {self.n_latent}"
            )

        if self.gaussian_reg < 0.0:
            raise ValueError(
                f"gaussian_reg must be ≥ 0; got {self.gaussian_reg}"
            )

        if not (0.0 < self.test_size < 0.5):
            raise ValueError(
                f"test_size must lie in (0, 0.5) for a meaningful split; "
                f"got {self.test_size}"
            )

        if self.logistic_C <= 0.0:
            raise ValueError(
                f"logistic_C must be strictly positive; got {self.logistic_C}"
            )

        if self.logistic_max_iter <= 0:
            raise ValueError(
                "logistic_max_iter must be a positive integer; "
                f"got {self.logistic_max_iter}"
            )

        if self.metrics_eps <= 0.0:
            raise ValueError(
                f"metrics_eps must be strictly positive; got {self.metrics_eps}"
            )


# ---------------------------------------------------------------------------
# Parameter grids for robustness analyses
# ---------------------------------------------------------------------------

@dataclass
class RobustnessGrid:
    """
    Parameter grids for systematic robustness checks.

    This dataclass collects the alternative values of key hyperparameters
    to be explored in the robustness analyses you requested:

      - latent dimensionality d,
      - peri‑stimulus time window,
      - logistic regression regularisation strength C,
      - optional class weighting.

    The idea is to *start* from a baseline SessionConfig that reproduces
    the main analysis and then override selected fields with each element
    of these grids, running the full pipeline for every combination.
    """

    latent_dims: Sequence[int] = field(
        default_factory=lambda: (5, 10, 20)
    )
    """
    Alternative latent dimensionalities to explore.

    The preprint uses d = 10 throughout; suggested robustness values are
    d ∈ {5, 10, 20}, probing under‑ and over‑parameterised geometries while
    staying safely below typical neuron counts. IG and ℓ⋆ should be
    qualitatively stable across this range if the results are robust.
    """

    time_windows: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: ((0.0, 0.2), (0.1, 0.3), (0.0, 0.3))
    )
    """
    Alternative peri‑stimulus windows in seconds.

    - (0.0, 0.2): main analysis (0–200 ms after onset),
    - (0.1, 0.3): shifted later by 100 ms,
    - (0.0, 0.3): extended to 300 ms.

    These variants test whether IG and pragmatic loss are sensitive to the
    exact timing of the window over which spikes are counted.
    """

    logistic_C_values: Sequence[float] = field(
        default_factory=lambda: (0.1, 1.0, 10.0)
    )
    """
    Alternative inverse regularisation strengths for the logistic decoder.

    - 0.1 : relatively strong shrinkage of decoder weights,
    - 1.0 : baseline used in the main analysis,
    - 10.0: relatively weak regularisation.

    Robustness requires that qualitative patterns in IG / ℓ⋆ and decoding
    accuracy do not depend critically on this choice.
    """

    logistic_class_weights: Sequence[Optional[str]] = field(
        default_factory=lambda: (None, "balanced")
    )
    """
    Alternative class_weight settings for LogisticRegression.

    Switching between None and "balanced" tests whether moderate class
    imbalance in w affects the estimated pragmatic loss and the relation
    between IG and behaviour. In the preprint, the main analysis uses
    class_weight = None.
    """

    def validate(self) -> None:
        """
        Sanity‑check the robustness grids.

        Ensures that all prospective parameter values are admissible at
        the level of theoretical assumptions (e.g. positive C, valid time
        windows). Individual combinations are further checked when they
        are merged with a SessionConfig and its validate() method.
        """
        if not self.latent_dims:
            raise ValueError("latent_dims must contain at least one value.")
        for d in self.latent_dims:
            if d < 1:
                raise ValueError(
                    f"All latent_dims must be ≥ 1; got {d}"
                )

        if not self.time_windows:
            raise ValueError("time_windows must contain at least one window.")
        for (t0, t1) in self.time_windows:
            if not (t0 < t1):
                raise ValueError(
                    f"All time windows must satisfy t0 < t1; "
                    f"got ({t0}, {t1})"
                )

        if not self.logistic_C_values:
            raise ValueError(
                "logistic_C_values must contain at least one value."
            )
        for C in self.logistic_C_values:
            if C <= 0.0:
                raise ValueError(
                    f"All logistic_C_values must be > 0; got {C}"
                )

        if not self.logistic_class_weights:
            raise ValueError(
                "logistic_class_weights must contain at least one value."
            )
        for cw in self.logistic_class_weights:
            if cw not in (None, "balanced"):
                raise ValueError(
                    f"class_weight must be None or 'balanced'; got {cw!r}"
                )


# ---------------------------------------------------------------------------
# Canonical list of sessions (13 Neuropixels insertions)
# ---------------------------------------------------------------------------

#: Session tags for the 13 Neuropixels probe insertions used in the preprint.
#: These correspond to the `session_tag` column in Tables 1–4.
MULTI_SESSION_TAGS: List[str] = [
    "69c9a415-f7fa-4208-887b-1417c1479b48_probe00",
    "c6db3304-c906-400c-aa0f-45dd3945b2ea_probe00",
    "fece187f-b47f-4870-a1d6-619afe942a7d_probe01",
    "71855308-7e54-41d7-a7a4-b042e78e3b4f_probe01",
    "4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a_probe01",
    "4d8c7767-981c-4347-8e5e-5d5fffe38534_probe01",
    "8928f98a-b411-497e-aa4b-aa752434686d_probe00",
    "4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a_probe00",
    "d23a44ef-1402-4ed7-97f5-47e9a7a504d9_probe00",
    "ff4187b5-4176-4e39-8894-53a24b7cf36b_probe01",
    "21e16736-fd59-44c7-b938-9b1333d25da8_probe00",
    "111c1762-7908-47e0-9f40-2f2ee55b6505_probe01",
    "ff48aa1d-ef30-4903-ac34-8c41b738c1b9_probe01",
]


def default_session_config(session_tag: Optional[str] = None) -> SessionConfig:
    """
    Construct a SessionConfig with the *exact* hyperparameters used for the
    main analysis in the preprint (Section 3), optionally attaching a
    session_tag for reporting.
    """
    cfg = SessionConfig(
        time_window=(0.0, 0.2),
        min_total_spikes=5,
        region=None,
        n_latent=10,
        fa_fit_mode="train",
        gaussian_reg=1e-6,
        test_size=0.2,
        random_state=0,
        stratify_by_world=True,
        logistic_C=1.0,
        logistic_penalty="l2",
        logistic_max_iter=1000,
        logistic_class_weight=None,
        logistic_solver="lbfgs",
        prior_feature="logit_probability_left",
        metrics_eps=1e-12,
        session_tag=session_tag,
    )
    cfg.validate()
    return cfg


def default_robustness_grid() -> RobustnessGrid:
    """
    Construct a RobustnessGrid with the parameter ranges suggested in
    your comments and in the revised Discussion (Section 5.4).
    """
    grid = RobustnessGrid()
    grid.validate()
    return grid
