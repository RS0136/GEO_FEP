"""
Data loading utilities for the geometric FEP–pragmatics pipeline.

This module implements the `load_session_from_pid` helper described in
Section 3.1 of the preprint: for a given probe UUID (PID) it retrieves

- spike–sorting output: spike times and cluster IDs (spikes),
  cluster metadata including quality labels and anatomical acronyms (clusters),
  and channel information (channels), and
- trial-wise task variables from the `trials` object, including (at least)
  `contrastLeft`, `contrastRight`, `stimOn_times`, `probabilityLeft`, and
  `choice`.

All loading is done via the ONE API on top of the public IBL OpenAlyx server
and brainbox’s SpikeSortingLoader. The functions here are intentionally
“thin”: they never subsample, shuffle, or transform the data; they only
retrieve and minimally validate it. All statistical processing happens in
downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import logging

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

log = logging.getLogger(__name__)

def _split_session_tag(session_tag: str) -> tuple[str, str]:
    """
    Split a session_tag like
        '69c9a415-f7fa-4208-887b-1417c1479b48_probe00'
    into (eid, probe_label).

    If there is no '_probe' suffix, the whole string is treated as
    an eid and 'probe00' is assumed.
    """
    if "_probe" in session_tag:
        eid, probe_suffix = session_tag.split("_probe", 1)
        probe_label = "probe" + probe_suffix
    else:
        eid = session_tag
        probe_label = "probe00"
    return eid, probe_label

# ---------------------------------------------------------------------------
# ONE construction
# ---------------------------------------------------------------------------

def make_one(
    one: Optional[ONE] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> ONE:
    """
    Return a ONE instance configured for the IBL public data server.

    Parameters
    ----------
    one:
        Existing ONE instance. If not None, it is returned unchanged.
    base_url:
        Optional override for the ONE server URL. If None, the default
        configuration is used (whatever has been set up via ONE.setup).
        For the IBL OpenAlyx server one typically calls::

            from one.api import ONE
            ONE.setup(base_url='https://openalyx.internationalbrainlab.org',
                      silent=True)
            one = ONE(password='international')

    **kwargs:
        Additional keyword arguments passed to the ONE constructor
        when `one` is None.

    Returns
    -------
    one : ONE
        A ready-to-use ONE client.

    Notes
    -----
    This function does *not* call ONE.setup() automatically, so that the
    user retains full control over credentials, cache location, etc. It
    simply provides a convenient, explicit place where a default ONE
    instance is created if needed.
    """
    if one is not None:
        return one

    if base_url is not None:
        log.info("Creating ONE instance for base_url=%s", base_url)
        return ONE(base_url=base_url, **kwargs)

    log.info("Creating ONE instance using existing ONE.setup configuration")
    return ONE(**kwargs)


# ---------------------------------------------------------------------------
# Raw session container
# ---------------------------------------------------------------------------

@dataclass
class RawSessionData:
    """
    Container for the raw objects needed by the analysis pipeline.

    Attributes
    ----------
    pid:
        Probe UUID (PID) for this insertion, e.g.
        'fece187f-b47f-4870-a1d6-619afe942a7d_probe01'.
    eid:
        Session (experiment) UUID to which this probe belongs.
    probe:
        Probe label within the session, e.g. 'probe00' or 'probe01'.
    spikes:
        Spike-sorting output for this probe (ALF 'spikes' object).
        Must at least contain 'times' and 'clusters'.
    clusters:
        Cluster metadata for this probe. After merging with `channels` via
        SpikeSortingLoader.merge_clusters, this includes quality labels
        (e.g. `label`) and anatomical acronyms (`acronym`) used later for
        filtering and region selection.
    channels:
        Channel information for this probe (ALF 'channels' object),
        including anatomical locations.
    trials:
        Trial-wise behavioural and task variables (ALF 'trials' object),
        including contrasts, block priors, stimulus onset times, and choices.
    one:
        The ONE client used to load these data. Retained so that downstream
        code can, if needed, load additional objects from the same session.
    """

    pid: str
    eid: str
    probe: str
    spikes: Any
    clusters: Any
    channels: Any
    trials: Any
    one: ONE


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_spike_sorting_from_pid(
    session_tag: str,
    one: ONE,
    spike_sorting_collection: str = "pykilosort",
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load spike sorting for one probe using the IBL ONE API.

    Parameters
    ----------
    session_tag :
        String like '{eid}_probe00' or '{eid}_probe01' (the same strings as in
        config.MULTI_SESSION_TAGS and the session_tag column in the tables).
    one :
        Connected ONE instance.
    spike_sorting_collection :
        Name of the spikesorting collection to use (e.g. 'pykilosort').

    Returns
    -------
    spikes, clusters, channels, eid, probe_label
    """

    eid, probe_label = _split_session_tag(session_tag)

    print(f"[data_io] Loading spike sorting for eid={eid}, probe={probe_label}")

    sl = SpikeSortingLoader(eid=eid, pname=probe_label, one=one)
    sl.collection = spike_sorting_collection

    spikes, clusters, channels = sl.load_spike_sorting()

    clusters = sl.merge_clusters(spikes, clusters, channels)

    return spikes, clusters, channels, eid, probe_label


def _load_trials_from_eid(eid: str, one: Optional[ONE] = None) -> Any:
    """
    Load the `trials` ALF object for a given session (eid).

    Parameters
    ----------
    eid:
        Experiment (session) UUID.
    one:
        Optional ONE client. If None, `make_one()` is used to construct one.

    Returns
    -------
    trials:
        The ALF `trials` object as returned by ONE.load_object(eid, 'trials').

    Raises
    ------
    KeyError
        If one of the fields required by the analysis pipeline is missing.

    Notes
    -----
    The analysis in Sections 3.2–3.3 relies on the following fields:

    - contrastLeft, contrastRight  (signed Michelson contrasts)
    - stimOn_times                  (stimulus onset)
    - probabilityLeft              (block prior P(Left | c))
    - choice                       (-1 left, 0 no-go, +1 right)

    This function checks for their presence but does not otherwise alter
    the object.
    """
    one = make_one(one)

    log.info("Loading trials object for eid=%s", eid)
    trials = one.load_object(eid, "trials")

    required_keys = (
        "contrastLeft",
        "contrastRight",
        "stimOn_times",
        "probabilityLeft",
        "choice",
    )
    missing = [k for k in required_keys if k not in trials]
    if missing:
        raise KeyError(
            f"Trials object for eid={eid} is missing required fields: {missing}"
        )

    return trials


# ---------------------------------------------------------------------------
# High-level helpers (public API)
# ---------------------------------------------------------------------------

def load_raw_session_from_pid(
    pid: str,
    one: Optional[ONE] = None,
    spike_sorting_collection: str = "pykilosort",
) -> RawSessionData:
    """
    Load all raw objects required for a single probe insertion.

    This is the main high-level data access function: given a probe UUID
    (PID) from the IBL spike-sorting benchmark, it retrieves spike sorting
    output for that probe and the corresponding behavioural trials from
    the parent session.

    Parameters
    ----------
    pid:
        Probe UUID (PID) string, e.g.
        'fece187f-b47f-4870-a1d6-619afe942a7d_probe01'.
    one:
        Optional ONE client. If None, a default client is constructed
        via `make_one()`.

    Returns
    -------
    data : RawSessionData
        Dataclass bundling pid, eid, probe label, spikes, clusters,
        channels, trials, and the ONE client.

    Examples
    --------
    >>> from fep_geo_ext.core.config import MULTI_SESSION_TAGS
    >>> from fep_geo_ext.core.data_io import load_raw_session_from_pid
    >>> data = load_raw_session_from_pid(MULTI_SESSION_TAGS[0])
    >>> data.spikes['times'].shape
    (n_spikes, )
    """
    one = make_one(one)

    session_tag = pid

    spikes, clusters, channels, eid, probe_label = _load_spike_sorting_from_pid(
        session_tag=session_tag,
        one=one,
        spike_sorting_collection=spike_sorting_collection,
    )

    trials = _load_trials_from_eid(eid=eid, one=one)

    return RawSessionData(
        pid=pid,
        eid=eid,
        probe=probe_label,
        spikes=spikes,
        clusters=clusters,
        channels=channels,
        trials=trials,
        one=one,
    )


def load_session_from_pid(
    pid: str,
    one: Optional[ONE] = None,
) -> Tuple[Any, Any, Any]:
    """
    Convenience wrapper returning only (spikes, clusters, trials).

    This mirrors the simpler helper described informally in Section 3.1
    of the preprint and is sufficient for the core geometric analysis,
    which uses channel information only via fields merged into `clusters`
    (e.g. `clusters['acronym']`).

    Parameters
    ----------
    pid:
        Probe UUID (PID).
    one:
        Optional ONE client.

    Returns
    -------
    spikes, clusters, trials

    Notes
    -----
    If you also need the full channels object, use
    :func:`load_raw_session_from_pid` instead.
    """
    data = load_raw_session_from_pid(pid=pid, one=one)
    return data.spikes, data.clusters, data.trials


__all__ = [
    "RawSessionData",
    "make_one",
    "load_raw_session_from_pid",
    "load_session_from_pid",
]
