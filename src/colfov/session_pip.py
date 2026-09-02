"""Session-level PiP qualification: the causal state-transition contract.

This module is the offline sequential reference implementation of the PiP session
controller -- qualification, transition, validation, stability, TTL and epoch rotation.
`tests/test_parity.py` replays a deterministic contract fixture and asserts identical
behaviour tick by tick.

It is NOT a live-stream runtime. There is no threading, no asynchronous monitor, no
frame-drop accounting and no capture-card timing. Monitoring is fixed-cadence and
causal: each observation is folded in at its own timestamp, and the box reported for an
instant is the box that was live at that instant -- never one derived from later
evidence.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field

import numpy as np

BBoxXYXY = tuple[int, int, int, int]

# --- frozen split family, from crop_policy/stability.py --------------------------
SPLIT_RNG_SEED = 20260809
N_RANDOM_SPLITS = 6
CONTENT_BASIS = "non_padding_frame_area"


def calibration_splits(count: int) -> list[tuple[str, list[int], list[int]]]:
    """The eight disjoint half-pairs every caller must use, in a fixed order."""
    indices = list(range(count))
    out = [("even_odd", indices[::2], indices[1::2]),
           ("front_back", indices[:count // 2], indices[count // 2:])]
    rng = np.random.default_rng(SPLIT_RNG_SEED)
    for k in range(N_RANDOM_SPLITS):
        shuffled = rng.permutation(count)
        out.append((f"random_{k}", sorted(shuffled[:count // 2].tolist()),
                    sorted(shuffled[count // 2:].tolist())))
    return out


@dataclass(frozen=True)
class StabilityThresholds:
    """Frozen 2026-08-07, hole denominator corrected 2026-08-09."""

    warn_box_iou: float = 0.98
    fail_box_iou: float = 0.95
    fail_area_ratio: float = 0.97
    hole_area_frac: float = 0.002
    hole_persistence: float = 0.60


DEFAULT_THRESHOLDS = StabilityThresholds()

# ========================================================================
# from colfov/pip/types.py
# ========================================================================






BBoxXYXY = tuple[int, int, int, int]


# Every reason a frame- or session-level decision can end without emitting a box.
# Kept as a closed vocabulary so the abstention taxonomy stays stable across
# step 14 ("모든 후보 결과 공개 기록") can be tabulated without free text.
REASON_OK = "ok"
REASON_NO_DETECTION = "no_class3_detection"
REASON_NOT_ROUTED = "overlay_without_endoscopic_content_support"
REASON_NO_FOOTPRINT = "no_reliable_popup_footprint"
REASON_INSUFFICIENT_TEMPORAL = "insufficient_temporal_evidence"
REASON_INSUFFICIENT_SPLIT = "insufficient_split_evidence"
REASON_AMBIGUOUS_POSITION = "ambiguous_pip_position"
REASON_POPUP_BLIND = "model_has_no_class3_channel"
REASON_UNSTABLE = "session_box_failed_stability"

# Diagnostic label, NOT a gate. Protocol section 12.1: the only cluster gate is
# `cluster_dominance_frac`. `top_two_cluster_margin` was withdrawn (B17) because
# p1 + p2 <= 1 forces margin >= 2*d - 1, which is >= 0.20 across the whole approved
# dominance grid, so a 0.10 margin gate can never fire. The label is still attached
# INSIDE a dominance failure so post-hoc analysis can tell "two positions genuinely
# competed" from "evidence was scattered and nothing dominated".
DIAGNOSTIC_COMPETING_CLUSTERS = "competing_pip_clusters"
COMPETING_CLUSTER_DIAGNOSTIC_MARGIN = 0.10


# Observation kinds. Protocol section 6.2 requires five drop categories to be recorded
# separately; collapsing them makes a 30-minute run uninterpretable.
OBS_SAMPLED = "sampled"
OBS_INFERRED_TTL = "inferred_from_ttl"
OBS_EXPIRED = "expired"
OBS_PLANNED_SKIP = "planned_cadence_skip"
OBS_INVALID_TIMESTAMP = "missing_or_invalid_timestamp"
OBS_DECODER_DROP = "decoder_drop"
OBS_OVERLOAD_DROP = "overload_induced_monitor_drop"
OBS_FOREGROUND_DROP = "foreground_output_frame_drop"

OBSERVATION_KINDS = (
    OBS_SAMPLED, OBS_INFERRED_TTL, OBS_EXPIRED, OBS_PLANNED_SKIP,
    OBS_INVALID_TIMESTAMP, OBS_DECODER_DROP, OBS_OVERLOAD_DROP, OBS_FOREGROUND_DROP,
)


@dataclass(frozen=True)
class PipStabilityVerdict:
    """Split-half / temporal verdict for one session PiP box.

    Computed over TEMPORALLY THINNED routed-positive bins, not raw frames
    `n_bins` is therefore a bin count; a session with 300
    consecutive routed frames inside one second contributes one bin, not 300.
    """

    stable: bool
    n_bins: int
    worst_split_iou: float | None
    worst_folded_area_ratio: float | None
    n_splits_evaluated: int
    n_splits_insufficient: int
    reason_code: str

    def as_manifest(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SessionPipGeometry:
    """The conditionally session-locked third product.

    `available=False` is a NORMAL outcome, not an error: the protocol's crop
    contract (section 3) is fail-closed, so a session with no routable PiP simply
    yields inner + full. Callers must not treat `box is None` as a failure.
    """

    available: bool
    box: BBoxXYXY | None
    qualifying_frames: int
    qualifying_time_bins: int
    evidence_frame_ids: tuple[str, ...]
    method: str
    stability: PipStabilityVerdict | None
    reason_code: str
    # Populated only when the dominance gate failed; see DIAGNOSTIC_COMPETING_CLUSTERS.
    diagnostic: str = ""
    cluster_summary: dict = field(default_factory=dict)

    def as_manifest(self) -> dict:
        out = asdict(self)
        out["stability"] = self.stability.as_manifest() if self.stability else None
        out["box"] = list(self.box) if self.box else None
        out["evidence_frame_ids"] = list(self.evidence_frame_ids)
        return out


@dataclass(frozen=True)
class FramePipResult:
    """Per-frame record. Compact by construction -- no RGB, no mask.

    Protocol section 2 forbids retaining pixels in the evidence stream: only a
    timestamp-keyed record per observation. `observation_kind` distinguishes a frame
    the monitor actually looked at from one it skipped on schedule, dropped under
    load, or inferred from a still-valid TTL.
    """

    frame_id: str
    t: float
    detected: bool
    candidate_box: BBoxXYXY | None
    routed: bool
    routing_score: float | None
    frame_box: BBoxXYXY | None
    session_box_applied: bool
    output_box: BBoxXYXY | None
    reason_code: str
    observation_kind: str

    def __post_init__(self) -> None:
        if self.observation_kind not in OBSERVATION_KINDS:
            raise ValueError(
                f"unknown observation_kind {self.observation_kind!r}; "
                f"expected one of {OBSERVATION_KINDS}")

    def as_manifest(self) -> dict:
        out = asdict(self)
        for k in ("candidate_box", "frame_box", "output_box"):
            out[k] = list(out[k]) if out[k] else None
        return out


def unavailable(reason_code: str, *, method: str = "", diagnostic: str = "",
                qualifying_frames: int = 0, qualifying_time_bins: int = 0,
                cluster_summary: dict | None = None) -> SessionPipGeometry:
    """The single constructor for 'no session PiP box'.

    Funnelled through one function so every abstention carries a reason code and no
    caller can invent a silent empty geometry.
    """
    return SessionPipGeometry(
        available=False, box=None, qualifying_frames=qualifying_frames,
        qualifying_time_bins=qualifying_time_bins, evidence_frame_ids=(),
        method=method, stability=None, reason_code=reason_code,
        diagnostic=diagnostic, cluster_summary=cluster_summary or {})


# ========================================================================
# from colfov/pip/validation.py
# ========================================================================









REASON_VALID = "valid"
REASON_NOT_FOUR = "box_is_not_four_coordinates"
REASON_NON_FINITE = "box_has_non_finite_coordinate"
REASON_NON_INTEGER = "box_has_non_integer_coordinate"
REASON_INVERTED = "box_is_inverted_or_empty"
REASON_OUT_OF_FRAME = "box_outside_frame_bounds"
REASON_DEGENERATE = "box_below_minimum_size"
REASON_RESOLUTION_MISMATCH = "box_frame_resolution_mismatch"

# A box thinner than this in either axis cannot carry a usable secondary view, and
# downstream resize of a 1-2 px strip produces garbage rather than an error.
MIN_BOX_SIDE_PX = 8


@dataclass(frozen=True)
class BoxValidation:
    valid: bool
    reason_code: str
    detail: str = ""

    def as_manifest(self) -> dict:
        return {"valid": self.valid, "reason_code": self.reason_code,
                "detail": self.detail}


VALID = BoxValidation(True, REASON_VALID)


def validate_box(box, frame_width: int, frame_height: int, *,
                 min_side: int = MIN_BOX_SIDE_PX,
                 expected_resolution: tuple[int, int] | None = None) -> BoxValidation:
    """Validate one box against the frame it claims to belong to.

    `expected_resolution` guards the case that motivated this function: a session box
    derived at one resolution being applied to a stream that changed resolution
    mid-recording. The coordinates can still be individually in-bounds and completely
    wrong.
    """
    if frame_width <= 0 or frame_height <= 0:
        return BoxValidation(False, REASON_RESOLUTION_MISMATCH,
                             f"frame dimensions must be positive, got {frame_width}x{frame_height}")

    if expected_resolution is not None:
        ew, eh = expected_resolution
        if (ew, eh) != (frame_width, frame_height):
            return BoxValidation(
                False, REASON_RESOLUTION_MISMATCH,
                f"box was derived at {ew}x{eh} but the frame is {frame_width}x{frame_height}")

    if box is None:
        return BoxValidation(False, REASON_NOT_FOUR, "box is None")
    try:
        coords = list(box)
    except TypeError:
        return BoxValidation(False, REASON_NOT_FOUR, f"box is not iterable: {box!r}")
    if len(coords) != 4:
        return BoxValidation(False, REASON_NOT_FOUR, f"expected 4 coordinates, got {len(coords)}")

    for i, c in enumerate(coords):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return BoxValidation(False, REASON_NON_INTEGER,
                                 f"coordinate {i} is {type(c).__name__}: {c!r}")
        if isinstance(c, float):
            if not math.isfinite(c):
                return BoxValidation(False, REASON_NON_FINITE,
                                     f"coordinate {i} is {c!r}")
            if not float(c).is_integer():
                return BoxValidation(False, REASON_NON_INTEGER,
                                     f"coordinate {i} is fractional: {c!r}")

    x1, y1, x2, y2 = (int(c) for c in coords)

    if x2 <= x1 or y2 <= y1:
        return BoxValidation(False, REASON_INVERTED,
                             f"({x1},{y1},{x2},{y2}) has non-positive width or height")
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return BoxValidation(
            False, REASON_OUT_OF_FRAME,
            f"({x1},{y1},{x2},{y2}) exceeds frame {frame_width}x{frame_height}")
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return BoxValidation(
            False, REASON_DEGENERATE,
            f"({x1},{y1},{x2},{y2}) is {x2 - x1}x{y2 - y1}, below the {min_side}px floor")
    return VALID


def clamp_for_diagnostic(box: BBoxXYXY, frame_width: int,
                         frame_height: int) -> BBoxXYXY | None:
    """Clamp a box into frame bounds. DIAGNOSTIC / RENDERING ONLY.

    Never call this on the primary path. Clamping an invalid primary box converts a
    detectable fault into a plausible-looking wrong answer.
    """
    x1, y1, x2, y2 = (int(c) for c in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ========================================================================
# from colfov/pip/session_geometry.py
# ========================================================================











# ---------------------------------------------------------------- preregistered defaults
TEMPORAL_BIN_SEC = 1.0                 # primary; 0.5 / 2.0 are the sensitivity arms
MIN_ROUTED_FRAMES = 3                  # grid 2/3/4/6
MIN_TEMPORAL_BINS = 2                  # grid 1/2/3
MIN_POSITIVE_TIME_SPAN_SEC = 3.0       # grid 2/3/5
OCCUPANCY_THRESHOLD = 0.5              # grid 0.4/0.5/0.6/0.7
QUANTILE_PAIR = (0.10, 0.90)           # grid (.10,.90)/(.25,.75)/(.05,.95)
CLUSTER_TAU = 0.10                     # grid 0.05/0.10/0.20  -- complete-linkage, FRAME-normalised
CLUSTER_DOMINANCE_FRAC = 0.70          # grid 0.60/0.70/0.80  -- the ONLY cluster gate

METHOD_MEDIAN = "coordinate_median"
METHOD_OCCUPANCY = "conditional_occupancy_vote"
METHOD_QUANTILE = "robust_quantile_envelope"
METHOD_CLUSTER = "largest_positional_cluster"
METHODS = (METHOD_MEDIAN, METHOD_OCCUPANCY, METHOD_QUANTILE, METHOD_CLUSTER)


@dataclass(frozen=True)
class RoutedObservation:
    """One routed-positive observation, already reduced past the RGB gate."""

    frame_id: str
    t: float
    box: BBoxXYXY


@dataclass(frozen=True)
class ThinnedEvidence:
    """Temporally thinned routed-positive evidence.

    One representative per temporal bin. Protocol section 2: 300 consecutive routed
    frames spanning 10 s become ~10 bins, so a long episode cannot outvote a short
    one purely by frame count.
    """

    bins: tuple[RoutedObservation, ...]
    bin_indices: tuple[int, ...]
    n_input_frames: int
    time_span_sec: float
    bin_sec: float


def thin_by_time(observations: list[RoutedObservation],
                 bin_sec: float = TEMPORAL_BIN_SEC) -> ThinnedEvidence:
    """Keep ONE representative observation per temporal bin.

    The representative is the observation closest to its bin centre, chosen because
    it is deterministic and does not favour whichever frame happened to arrive first.
    Ties break on frame_id so the result never depends on input ordering.
    """
    if bin_sec <= 0:
        raise ValueError("bin_sec must be positive")
    if not observations:
        return ThinnedEvidence((), (), 0, 0.0, bin_sec)

    ordered = sorted(observations, key=lambda o: (o.t, o.frame_id))
    t0 = ordered[0].t
    buckets: dict[int, list[RoutedObservation]] = defaultdict(list)
    for o in ordered:
        buckets[int(math.floor((o.t - t0) / bin_sec))].append(o)

    reps: list[RoutedObservation] = []
    idxs: list[int] = []
    for b in sorted(buckets):
        centre = t0 + (b + 0.5) * bin_sec
        reps.append(min(buckets[b], key=lambda o: (abs(o.t - centre), o.frame_id)))
        idxs.append(b)

    span = ordered[-1].t - ordered[0].t
    return ThinnedEvidence(tuple(reps), tuple(idxs), len(ordered), float(span), bin_sec)


# ---------------------------------------------------------------- estimators


def _coords(bins: tuple[RoutedObservation, ...]) -> np.ndarray:
    return np.asarray([o.box for o in bins], dtype=np.float64)


def _as_box(vals) -> BBoxXYXY:
    x1, y1, x2, y2 = (int(round(float(v))) for v in vals)
    # A degenerate box is a bug, not a valid product; widen by one pixel rather than
    # emit x2 <= x1, which would silently produce an empty crop downstream.
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def estimate_median(bins: tuple[RoutedObservation, ...]) -> BBoxXYXY:
    return _as_box(np.median(_coords(bins), axis=0))


def estimate_quantile(bins: tuple[RoutedObservation, ...],
                      q: tuple[float, float] = QUANTILE_PAIR) -> BBoxXYXY:
    """Robust envelope: low quantile for the near edges, high for the far edges.

    Applying q_lo to x1/y1 and q_hi to x2/y2 keeps the box on the OUTSIDE of the
    bulk of the observations. Using the same quantile for all four would shift the
    box rather than size it.
    """
    q_lo, q_hi = q
    if not 0.0 <= q_lo < q_hi <= 1.0:
        raise ValueError("quantile pair must satisfy 0 <= q_lo < q_hi <= 1")
    c = _coords(bins)
    return _as_box((np.quantile(c[:, 0], q_lo), np.quantile(c[:, 1], q_lo),
                    np.quantile(c[:, 2], q_hi), np.quantile(c[:, 3], q_hi)))


def estimate_occupancy(bins: tuple[RoutedObservation, ...],
                       threshold: float = OCCUPANCY_THRESHOLD) -> BBoxXYXY | None:
    """Vote in the box domain with a CONDITIONAL denominator.

    The denominator is the number of routed-positive BINS, not the 24 calibration
    frames and not every frame in the recording. Protocol section 4: using the full
    frame count would erase an intermittent PiP that is genuinely present whenever
    it appears.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("occupancy threshold must be in (0, 1]")
    c = _coords(bins).astype(np.int64)
    x_lo, y_lo = int(c[:, 0].min()), int(c[:, 1].min())
    x_hi, y_hi = int(c[:, 2].max()), int(c[:, 3].max())
    w, h = x_hi - x_lo, y_hi - y_lo
    if w <= 0 or h <= 0:
        return None

    # 2-D difference array -> O(n + area) instead of O(n * area).
    acc = np.zeros((h + 1, w + 1), dtype=np.int32)
    for x1, y1, x2, y2 in c:
        acc[y1 - y_lo, x1 - x_lo] += 1
        acc[y1 - y_lo, x2 - x_lo] -= 1
        acc[y2 - y_lo, x1 - x_lo] -= 1
        acc[y2 - y_lo, x2 - x_lo] += 1
    occ = np.cumsum(np.cumsum(acc, axis=0), axis=1)[:h, :w]

    keep = occ >= math.ceil(threshold * len(c))
    if not keep.any():
        return None
    ys, xs = np.where(keep)
    return (int(xs.min()) + x_lo, int(ys.min()) + y_lo,
            int(xs.max()) + 1 + x_lo, int(ys.max()) + 1 + y_lo)


def _complete_linkage(points: np.ndarray, tau: float) -> list[list[int]]:
    """Agglomerative complete-linkage clustering, cut at `tau`.

    COMPLETE, not single, linkage. Single-linkage chains: a PiP that slides from one
    corner to another leaves intermediate observations that bridge the two ends, so
    single-linkage merges everything into one cluster, dominance reads 1.0, and a
    genuinely ambiguous session passes the gate. Complete-linkage bounds cluster
    DIAMETER by tau, which is exactly the property that blocks the chain.

    O(n^3) worst case. n is a bin count (tens), not a frame count, so this is fine
    and is preferred over pulling in scipy for one function.
    """
    n = len(points)
    clusters = [[i] for i in range(n)]
    if n <= 1:
        return clusters
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    while len(clusters) > 1:
        best, best_pair = None, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # complete linkage: the distance between clusters is their FARTHEST pair
                dist = d[np.ix_(clusters[a], clusters[b])].max()
                if best is None or dist < best:
                    best, best_pair = dist, (a, b)
        if best is None or best > tau:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    return clusters


def estimate_cluster(bins: tuple[RoutedObservation, ...], frame_width: int,
                     frame_height: int, tau: float = CLUSTER_TAU,
                     dominance: float = CLUSTER_DOMINANCE_FRAC
                     ) -> tuple[BBoxXYXY | None, dict]:
    """Largest positional cluster, or fail closed.

    Centres are normalised by FRAME size, not by the session-box diagonal. The
    diagonal is not available here: clustering runs in order to BUILD the session
    box, so measuring distance in units of that box would be circular. Frame size is
    known before clustering and constant for the recording.

    The normalisation is anisotropic -- on a 16:9 frame a horizontal 0.1 is a longer
    physical distance than a vertical 0.1. PiP relocation is typically a large
    cross-quadrant move, so this does not change the verdict, but it is recorded.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if not 0.0 < dominance <= 1.0:
        raise ValueError("dominance must be in (0, 1]")

    c = _coords(bins)
    centres = np.stack([(c[:, 0] + c[:, 2]) / 2.0 / frame_width,
                        (c[:, 1] + c[:, 3]) / 2.0 / frame_height], axis=1)
    clusters = sorted(_complete_linkage(centres, tau), key=len, reverse=True)
    fracs = [len(g) / len(bins) for g in clusters]
    summary = {"n_clusters": len(clusters), "cluster_fractions": [round(f, 4) for f in fracs],
               "tau": tau, "dominance_required": dominance,
               "normalisation": "frame_width_height_anisotropic", "linkage": "complete"}

    if fracs[0] < dominance:
        # The ONLY gate. `competing_pip_clusters` is attached as a diagnostic inside
        # this failure -- it never causes the failure. See types.py.
        margin = fracs[0] - (fracs[1] if len(fracs) > 1 else 0.0)
        summary["top_two_margin"] = round(margin, 4)
        summary["diagnostic"] = (DIAGNOSTIC_COMPETING_CLUSTERS
                                 if len(fracs) > 1 and margin < COMPETING_CLUSTER_DIAGNOSTIC_MARGIN
                                 else "")
        return None, summary

    winner = [bins[i] for i in clusters[0]]
    return estimate_median(tuple(winner)), summary


# ---------------------------------------------------------------- driver


def derive_session_pip_geometry(
    observations: list[RoutedObservation],
    *,
    frame_width: int,
    frame_height: int,
    method: str = METHOD_MEDIAN,
    bin_sec: float = TEMPORAL_BIN_SEC,
    min_routed_frames: int = MIN_ROUTED_FRAMES,
    min_temporal_bins: int = MIN_TEMPORAL_BINS,
    min_positive_time_span_sec: float = MIN_POSITIVE_TIME_SPAN_SEC,
    occupancy_threshold: float = OCCUPANCY_THRESHOLD,
    quantile_pair: tuple[float, float] = QUANTILE_PAIR,
    cluster_tau: float = CLUSTER_TAU,
    cluster_dominance: float = CLUSTER_DOMINANCE_FRAC,
    popup_detection_available: bool = True,
) -> SessionPipGeometry:
    """Thin, check minimum evidence, estimate -- or abstain with a reason code.

    Returns `available=False` for every insufficiency. That is the normal
    fail-closed path, not an error: inner + full are still emitted by the frozen
    solver, which this function never touches.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not popup_detection_available:
        # A 3-class model cannot distinguish "no PiP" from "cannot see PiP". Asserting
        # absence from it would be a claim the model cannot support.
        return unavailable("model_has_no_class3_channel", method=method)

    routed = [o for o in observations if o.box is not None]
    if len(routed) < min_routed_frames:
        return unavailable(REASON_INSUFFICIENT_TEMPORAL, method=method,
                           qualifying_frames=len(routed))

    ev = thin_by_time(routed, bin_sec)
    if len(ev.bins) < min_temporal_bins or ev.time_span_sec < min_positive_time_span_sec:
        # Adjacent frames inside one bin are not independent evidence: three frames
        # 30 ms apart are one observation of one instant, not three.
        return unavailable(REASON_INSUFFICIENT_TEMPORAL, method=method,
                           qualifying_frames=len(routed),
                           qualifying_time_bins=len(ev.bins))

    summary: dict = {}
    if method == METHOD_MEDIAN:
        box = estimate_median(ev.bins)
    elif method == METHOD_QUANTILE:
        box = estimate_quantile(ev.bins, quantile_pair)
    elif method == METHOD_OCCUPANCY:
        box = estimate_occupancy(ev.bins, occupancy_threshold)
    else:
        box, summary = estimate_cluster(ev.bins, frame_width, frame_height,
                                        cluster_tau, cluster_dominance)

    if box is None:
        reason = (REASON_AMBIGUOUS_POSITION if method == METHOD_CLUSTER
                  else REASON_INSUFFICIENT_TEMPORAL)
        return unavailable(reason, method=method, diagnostic=summary.get("diagnostic", ""),
                           qualifying_frames=len(routed),
                           qualifying_time_bins=len(ev.bins), cluster_summary=summary)

    return SessionPipGeometry(
        available=True, box=box, qualifying_frames=len(routed),
        qualifying_time_bins=len(ev.bins),
        evidence_frame_ids=tuple(o.frame_id for o in ev.bins),
        method=method, stability=None, reason_code=REASON_OK,
        cluster_summary=summary)


# ========================================================================
# from colfov/pip/stability.py
# ========================================================================










FAIL_BOX_IOU = DEFAULT_THRESHOLDS.fail_box_iou        # 0.95, frozen
FAIL_AREA_RATIO = DEFAULT_THRESHOLDS.fail_area_ratio  # 0.97, frozen


def box_iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def folded_area_ratio(a: BBoxXYXY, b: BBoxXYXY) -> float:
    """min(area)/max(area) -- symmetric, so split order cannot change the verdict."""
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if max(area_a, area_b) == 0:
        return 0.0
    return float(min(area_a, area_b) / max(area_a, area_b))


def _estimate(bins: tuple[RoutedObservation, ...], method: str, frame_width: int,
              frame_height: int, **kw) -> BBoxXYXY | None:
    if method == METHOD_MEDIAN:
        return estimate_median(bins)
    if method == METHOD_QUANTILE:
        return estimate_quantile(bins, kw.get("quantile_pair", (0.10, 0.90)))
    if method == METHOD_OCCUPANCY:
        return estimate_occupancy(bins, kw.get("occupancy_threshold", 0.5))
    if method == METHOD_CLUSTER:
        return estimate_cluster(bins, frame_width, frame_height,
                                kw.get("cluster_tau", 0.10),
                                kw.get("cluster_dominance", 0.70))[0]
    raise ValueError(f"unknown method {method!r}")


def evaluate_pip_stability(
    evidence: ThinnedEvidence,
    *,
    frame_width: int,
    frame_height: int,
    method: str = METHOD_MEDIAN,
    min_bins_per_half: int = MIN_TEMPORAL_BINS,
    fail_box_iou: float = FAIL_BOX_IOU,
    fail_area_ratio: float = FAIL_AREA_RATIO,
    **estimator_kw,
) -> PipStabilityVerdict:
    """Split-half over bins, using the frozen split family.

    A split whose half falls below `min_bins_per_half` is counted as
    `n_splits_insufficient` and does NOT contribute to the worst-case statistics --
    a two-bin half would otherwise produce a meaningless IoU that could pass.
    """
    bins = evidence.bins
    n = len(bins)
    if n < 2 * min_bins_per_half:
        return PipStabilityVerdict(
            stable=False, n_bins=n, worst_split_iou=None, worst_folded_area_ratio=None,
            n_splits_evaluated=0, n_splits_insufficient=len(calibration_splits(max(n, 1))),
            reason_code=REASON_INSUFFICIENT_SPLIT)

    worst_iou, worst_area = 1.0, 1.0
    evaluated = insufficient = 0
    for _name, idx_a, idx_b in calibration_splits(n):
        if len(idx_a) < min_bins_per_half or len(idx_b) < min_bins_per_half:
            insufficient += 1
            continue
        box_a = _estimate(tuple(bins[i] for i in idx_a), method, frame_width,
                          frame_height, **estimator_kw)
        box_b = _estimate(tuple(bins[i] for i in idx_b), method, frame_width,
                          frame_height, **estimator_kw)
        if box_a is None or box_b is None:
            # A half that cannot produce a box is not evidence of instability; it is
            # absence of evidence. Recorded separately.
            insufficient += 1
            continue
        evaluated += 1
        worst_iou = min(worst_iou, box_iou(box_a, box_b))
        worst_area = min(worst_area, folded_area_ratio(box_a, box_b))

    if evaluated == 0:
        return PipStabilityVerdict(
            stable=False, n_bins=n, worst_split_iou=None, worst_folded_area_ratio=None,
            n_splits_evaluated=0, n_splits_insufficient=insufficient,
            reason_code=REASON_INSUFFICIENT_SPLIT)

    stable = worst_iou >= fail_box_iou and worst_area >= fail_area_ratio
    return PipStabilityVerdict(
        stable=stable, n_bins=n, worst_split_iou=float(worst_iou),
        worst_folded_area_ratio=float(worst_area), n_splits_evaluated=evaluated,
        n_splits_insufficient=insufficient,
        reason_code=REASON_OK if stable else REASON_UNSTABLE)


def temporal_segment_boxes(evidence: ThinnedEvidence, method: str, frame_width: int,
                           frame_height: int, **kw) -> dict[str, BBoxXYXY | None]:
    """Early/late and thirds, which a random split cannot see.

    Random halves interleave time, so a box that drifts monotonically can still agree
    across random splits. These contiguous segments are what expose drift.
    """
    bins, n = evidence.bins, len(evidence.bins)
    out: dict[str, BBoxXYXY | None] = {}
    if n >= 2:
        out["early"] = _estimate(bins[: n // 2], method, frame_width, frame_height, **kw)
        out["late"] = _estimate(bins[n // 2:], method, frame_width, frame_height, **kw)
    if n >= 3:
        t = n // 3
        out["first_third"] = _estimate(bins[:t], method, frame_width, frame_height, **kw)
        out["middle_third"] = _estimate(bins[t:2 * t], method, frame_width, frame_height, **kw)
        out["last_third"] = _estimate(bins[2 * t:], method, frame_width, frame_height, **kw)
    return out


def time_block_bootstrap_boxes(evidence: ThinnedEvidence, method: str, frame_width: int,
                               frame_height: int, n_resamples: int = 1000,
                               block_bins: int = 3, seed: int = SPLIT_RNG_SEED,
                               **kw) -> list[BBoxXYXY]:
    """Block bootstrap over bins.

    Blocks, not i.i.d. bins: consecutive bins are correlated, and resampling them
    independently would understate the confidence interval.
    """
    bins, n = evidence.bins, len(evidence.bins)
    if n < block_bins:
        return []
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_bins))
    out: list[BBoxXYXY] = []
    for _ in range(n_resamples):
        picked: list[RoutedObservation] = []
        for start in rng.integers(0, max(1, n - block_bins + 1), size=n_blocks):
            picked.extend(bins[int(start): int(start) + block_bins])
        box = _estimate(tuple(picked[:n]), method, frame_width, frame_height, **kw)
        if box is not None:
            out.append(box)
    return out


# ========================================================================
# from colfov/pip/state.py
# ========================================================================










MAX_QUALIFICATION_BINS = 60
DRIFT_WINDOW_BINS = 30
EVIDENCE_ID_RING_SIZE = 64
DRIFT_PATIENCE_SECONDS = 1.0            # grid 0.5 / 1.0 / 2.0
DRIFT_MIN_CONSECUTIVE_OBSERVATIONS = 2  # fixed floor, not a grid
TTL_MULTIPLIER = 1.5                    # primary; 1.0 / 2.0 are sensitivity arms

STATE_CALIBRATING = "CALIBRATING"
STATE_LOCKED = "LOCKED"
STATE_FAILED = "FAILED"


@dataclass
class PipQualificationState:
    """Rolling PiP state for one recording. Every container is bounded."""

    monitor_interval_sec: float
    drift_patience_seconds: float = DRIFT_PATIENCE_SECONDS
    drift_min_consecutive: int = DRIFT_MIN_CONSECUTIVE_OBSERVATIONS
    ttl_multiplier: float = TTL_MULTIPLIER
    drift_window_bins: int = DRIFT_WINDOW_BINS
    evidence_id_ring_size: int = EVIDENCE_ID_RING_SIZE

    session_box: BBoxXYXY | None = None
    locked: bool = False

    # --- bounded containers -------------------------------------------------
    # NOTE: there is deliberately NO `qualification_bins` here any more. Pre-lock
    # evidence lives in exactly one place -- `PipQualifier._evidence`, which is what
    # the estimator reads and where `max_qualification_bins` is enforced. Keeping a
    # second, independently capped copy here meant the documented bound was applied to
    # a container nothing estimated from.
    drift_window: deque = field(default_factory=deque)
    evidence_ids: deque = field(default_factory=deque)

    # --- aggregate counters (O(1), never grow) ------------------------------
    n_observations: int = 0
    n_routed: int = 0
    n_locks: int = 0
    n_unlocks: int = 0
    n_ttl_inferred: int = 0
    n_expired: int = 0

    # --- drift tracking -----------------------------------------------------
    drift_run_count: int = 0
    drift_run_started_t: float | None = None
    last_positive_t: float | None = None

    def __post_init__(self) -> None:
        self.drift_window = deque(self.drift_window, maxlen=self.drift_window_bins)
        self.evidence_ids = deque(self.evidence_ids, maxlen=self.evidence_id_ring_size)
        if self.monitor_interval_sec <= 0:
            raise ValueError("monitor_interval_sec must be positive")
        if self.drift_min_consecutive < 1:
            raise ValueError("drift_min_consecutive must be >= 1")

    # ------------------------------------------------------------------ TTL

    @property
    def ttl_sec(self) -> float:
        return self.ttl_multiplier * self.monitor_interval_sec

    def pip_state_valid_at(self, t: float) -> bool:
        """Is the last positive observation still within its TTL?

        Fail-closed: unknown or expired means no PiP output. A negative observation
        clears `last_positive_t` immediately, so this returns False at once rather
        than coasting to the end of the TTL.
        """
        if self.last_positive_t is None:
            return False
        return (t - self.last_positive_t) <= self.ttl_sec

    # ------------------------------------------------------- observation intake

    def observe(self, obs: RoutedObservation | None, t: float,
                drift: bool = False) -> str:
        """Fold one monitoring observation into the state.

        Returns the transition that happened: "" (none), "lock", or "unlock".
        `obs is None` means the monitor looked and found no routable PiP.
        """
        self.n_observations += 1

        if obs is None:
            # Negative observation stops PiP output immediately -- it does not wait
            # for the TTL. Protocol section 6.1.
            self.last_positive_t = None
            self._reset_drift()
            return ""

        self.n_routed += 1
        self.last_positive_t = t
        self.evidence_ids.append(obs.frame_id)

        if not self.locked:
            # Pre-lock evidence is the qualifier's responsibility, not this object's.
            return ""

        self.drift_window.append(obs)
        if drift:
            if self.drift_run_started_t is None:
                self.drift_run_started_t = t
            self.drift_run_count += 1
            elapsed = t - self.drift_run_started_t
            if (elapsed >= self.drift_patience_seconds
                    and self.drift_run_count >= self.drift_min_consecutive):
                self.unlock()
                return "unlock"
        else:
            self._reset_drift()
        return ""

    def _reset_drift(self) -> None:
        self.drift_run_count = 0
        self.drift_run_started_t = None

    # ------------------------------------------------------------ transitions

    def lock(self, box: BBoxXYXY) -> None:
        self.session_box = box
        self.locked = True
        self.n_locks += 1
        self._reset_drift()
        self.drift_window.clear()

    def unlock(self) -> None:
        self.session_box = None
        self.locked = False
        self.n_unlocks += 1
        self._reset_drift()
        self.drift_window.clear()

    # ------------------------------------------------------------- reporting

    def state_size(self) -> dict[str, int]:
        """Everything that could grow. Asserted flat across run length in tests."""
        return {"drift_window": len(self.drift_window),
                "evidence_ids": len(self.evidence_ids)}

    def as_manifest(self) -> dict:
        return {
            "locked": self.locked,
            "session_box": list(self.session_box) if self.session_box else None,
            "n_observations": self.n_observations, "n_routed": self.n_routed,
            "n_locks": self.n_locks, "n_unlocks": self.n_unlocks,
            "n_ttl_inferred": self.n_ttl_inferred, "n_expired": self.n_expired,
            "caps": {"drift_window_bins": self.drift_window_bins,
                     "evidence_id_ring_size": self.evidence_id_ring_size},
            "ttl_sec": self.ttl_sec,
            "drift_patience_seconds": self.drift_patience_seconds,
            "drift_min_consecutive": self.drift_min_consecutive,
            "state_size": self.state_size(),
        }


# ========================================================================
# from colfov/pip/qualification.py
# ========================================================================












@dataclass
class PipQualifier:
    """Accumulates routed evidence and decides when a session box may lock.

    Bounded by construction: `max_evidence` caps the retained observation list, so a
    60-minute recording does not grow this object without limit.
    """

    frame_width: int
    frame_height: int
    method: str = METHOD_MEDIAN
    bin_sec: float = TEMPORAL_BIN_SEC
    min_routed_frames: int = MIN_ROUTED_FRAMES
    min_temporal_bins: int = MIN_TEMPORAL_BINS
    min_positive_time_span_sec: float = MIN_POSITIVE_TIME_SPAN_SEC
    occupancy_threshold: float = OCCUPANCY_THRESHOLD
    quantile_pair: tuple[float, float] = QUANTILE_PAIR
    cluster_tau: float = CLUSTER_TAU
    cluster_dominance: float = CLUSTER_DOMINANCE_FRAC
    popup_detection_available: bool = True
    require_stability: bool = True
    # THE preregistered cap, enforced on the evidence the
    # estimator actually reads. There is no second, independently capped store: an
    # earlier version also kept `PipQualificationState.qualification_bins`, capped at
    # 60, which the estimator never read -- so the documented bound was enforced on the
    # wrong container while the real one held 500+ observations.
    max_qualification_bins: int = 60
    # Absolute backstop on raw observations, so a very high monitor rate cannot make
    # the retained list large even inside the bin window.
    max_raw_observations: int = 4000

    _evidence: list[RoutedObservation] = field(default_factory=list, repr=False)
    last_geometry: SessionPipGeometry | None = None
    last_rotation_reason: str = ""
    n_attempts: int = 0
    epoch: int = 0
    n_epoch_rotations: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.max_qualification_bins < 1:
            raise ValueError("max_qualification_bins must be >= 1")

    # ------------------------------------------------------------------ intake

    def add(self, obs: RoutedObservation) -> None:
        self._evidence.append(obs)
        self._trim()

    def _trim(self) -> None:
        """Keep only the newest `max_qualification_bins` bins' worth of evidence.

        Trimming by TIME WINDOW rather than by observation count is what makes the cap
        mean what the protocol says. `max_qualification_bins = 60` with a 1 s bin is
        "the last 60 seconds of routed evidence" regardless of whether the monitor ran
        at 5 Hz or at native rate; an observation-count cap would mean a different
        amount of history at every cadence.
        """
        if not self._evidence:
            return
        newest = self._evidence[-1].t
        window = self.max_qualification_bins * self.bin_sec
        cutoff = newest - window
        if self._evidence[0].t <= cutoff:
            self._evidence = [o for o in self._evidence if o.t > cutoff]
        if len(self._evidence) > self.max_raw_observations:
            del self._evidence[: len(self._evidence) - self.max_raw_observations]

    @property
    def n_evidence(self) -> int:
        return len(self._evidence)

    @property
    def n_bins(self) -> int:
        """Bins the estimator would actually see. This is what the cap bounds."""
        return len(thin_by_time(list(self._evidence), self.bin_sec).bins)

    def clear(self) -> None:
        self._evidence.clear()

    def rotate_epoch(self, reason: str = "") -> int:
        """Discard all evidence and start a new qualification epoch.

        Called on a confirmed drift unlock or any other episode discontinuity. Without
        this, the observations that were valid for the OLD location survive the unlock
        and are still eligible to re-lock it -- which is exactly the thrash the
        pre-fix implementation produced (4 locks / 4 unlocks on one relocation).

        Requalification must start from new-episode observations only.
        """
        self._evidence.clear()
        self.epoch += 1
        self.n_epoch_rotations += 1
        self.last_geometry = None
        self.last_rotation_reason = reason
        return self.epoch

    # -------------------------------------------------------------- evaluation

    def attempt(self) -> SessionPipGeometry:
        """Try to qualify a session box. Returns an unavailable geometry on any gate failure."""
        self.n_attempts += 1

        geom = derive_session_pip_geometry(
            list(self._evidence), frame_width=self.frame_width,
            frame_height=self.frame_height, method=self.method, bin_sec=self.bin_sec,
            min_routed_frames=self.min_routed_frames,
            min_temporal_bins=self.min_temporal_bins,
            min_positive_time_span_sec=self.min_positive_time_span_sec,
            occupancy_threshold=self.occupancy_threshold,
            quantile_pair=self.quantile_pair, cluster_tau=self.cluster_tau,
            cluster_dominance=self.cluster_dominance,
            popup_detection_available=self.popup_detection_available)

        if not geom.available:
            self.last_geometry = geom
            return geom

        # Gate 0.5: the estimator produced something -- is it a legal box at all?
        v = validate_box(geom.box, self.frame_width, self.frame_height)
        if not v.valid:
            geom = unavailable(v.reason_code, method=self.method,
                               qualifying_frames=geom.qualifying_frames,
                               qualifying_time_bins=geom.qualifying_time_bins)
            self.last_geometry = geom
            return geom

        if not self.require_stability:
            self.last_geometry = geom
            return geom

        # Gate 2: stability over the SAME thinned bins the box came from.
        ev = thin_by_time(list(self._evidence), self.bin_sec)
        verdict = evaluate_pip_stability(
            ev, frame_width=self.frame_width, frame_height=self.frame_height,
            method=self.method, min_bins_per_half=self.min_temporal_bins,
            quantile_pair=self.quantile_pair,
            occupancy_threshold=self.occupancy_threshold,
            cluster_tau=self.cluster_tau, cluster_dominance=self.cluster_dominance)

        if not verdict.stable:
            # Preserve WHY: insufficient_split_evidence != session_box_failed_stability.
            geom = SessionPipGeometry(
                available=False, box=None,
                qualifying_frames=geom.qualifying_frames,
                qualifying_time_bins=geom.qualifying_time_bins,
                evidence_frame_ids=geom.evidence_frame_ids, method=self.method,
                stability=verdict, reason_code=verdict.reason_code or REASON_UNSTABLE,
                cluster_summary=geom.cluster_summary)
            self.last_geometry = geom
            return geom

        geom = SessionPipGeometry(
            available=True, box=geom.box, qualifying_frames=geom.qualifying_frames,
            qualifying_time_bins=geom.qualifying_time_bins,
            evidence_frame_ids=geom.evidence_frame_ids, method=self.method,
            stability=verdict, reason_code=REASON_OK,
            cluster_summary=geom.cluster_summary)
        self.last_geometry = geom
        return geom

    def as_manifest(self) -> dict:
        return {"method": self.method, "n_evidence": self.n_evidence,
                "n_bins": self.n_bins, "n_attempts": self.n_attempts,
                "max_qualification_bins": self.max_qualification_bins,
                "max_raw_observations": self.max_raw_observations,
                "bin_sec": self.bin_sec,
                "epoch": self.epoch, "n_epoch_rotations": self.n_epoch_rotations,
                "last_rotation_reason": self.last_rotation_reason,
                "require_stability": self.require_stability,
                "last_geometry": (self.last_geometry.as_manifest()
                                  if self.last_geometry else None)}


# ========================================================================
# from colfov/pip/transition.py
# ========================================================================










# A locked box that a new routed observation disagrees with by more than this is
# evidence of relocation, not noise. Frozen at 0.50.
DRIFT_IOU_THRESHOLD = 0.50

UNLOCK_REASON = "confirmed_drift_unlock"

REASON_OK = "ok"
REASON_NO_SESSION_BOX = "pip_session_box_unavailable"
REASON_TTL_EXPIRED = "pip_ttl_expired"


@dataclass(frozen=True)
class TransitionOutcome:
    """What one observation did. Counters stay with the caller that owns them."""

    transition: str            # "", "lock" or "unlock", as PipQualificationState says
    box_rejected: bool         # the observation carried an out-of-frame box
    epoch_rotated: bool        # a confirmed drift unlock rotated the qualifier epoch
    locked_now: bool           # this observation is the one that locked the session
    attempted: bool            # qualification was attempted on this observation
    reason_code: str = ""      # qualifier's reason when it attempted and declined
    # The stability verdict the qualifier produced on this attempt, if it made one.
    # Carried here so an offline replay can record stability AT THE MOMENT OF LOCK
    # rather than re-deriving it later from evidence the lock never saw.
    stability: dict | None = None


def is_drift_locked(obs, pip_state) -> bool:
    """Does this observation disagree with the locked box enough to count as drift?

    Only meaningful while locked: before a lock there is nothing to drift from.
    """
    if obs is None or not pip_state.locked or pip_state.session_box is None:
        return False
    return box_iou(obs.box, pip_state.session_box) < DRIFT_IOU_THRESHOLD


def apply_observation(obs, t: float, *, pip_state, qualifier,
                      frame_width: int, frame_height: int) -> TransitionOutcome:
    """Fold ONE observation into the session state, in causal order.

    The ordering is load-bearing and is the reason this is not inlined anywhere:

    1. An out-of-frame box is discarded and the tick becomes a NEGATIVE observation,
       not a skipped one. Dropping the tick entirely would stop the TTL clock.
    2. Drift is measured against the currently locked box, BEFORE the state folds the
       observation in -- afterwards the comparison is against a box that has already
       moved.
    3. A confirmed unlock rotates the qualifier epoch and RETURNS WITHOUT ATTEMPTING.
       The evidence that justified the old location is still in the qualifier; if it
       may attempt in the same tick it re-locks the position that was just abandoned.
       The pre-fix runtime produced 4 locks and 4 unlocks on one relocation this way.
    4. Qualification is attempted only while UNLOCKED. A locked session does not
       re-estimate itself from later evidence.
    """
    box_rejected = False
    if obs is not None:
        v = validate_box(obs.box, frame_width, frame_height)
        if not v.valid:
            box_rejected = True
            obs = None

    drift = is_drift_locked(obs, pip_state)
    transition = pip_state.observe(obs, t, drift=drift)

    if transition == "unlock":
        qualifier.rotate_epoch(UNLOCK_REASON)
        return TransitionOutcome(transition=transition, box_rejected=box_rejected,
                                 epoch_rotated=True, locked_now=False,
                                 attempted=False)

    if obs is None:
        return TransitionOutcome(transition=transition, box_rejected=box_rejected,
                                 epoch_rotated=False, locked_now=False,
                                 attempted=False)

    qualifier.add(obs)
    if pip_state.locked:
        return TransitionOutcome(transition=transition, box_rejected=box_rejected,
                                 epoch_rotated=False, locked_now=False,
                                 attempted=False)

    geom = qualifier.attempt()
    verdict = geom.stability.as_manifest() if geom.stability is not None else None
    if geom.available and geom.box is not None:
        pip_state.lock(geom.box)
        return TransitionOutcome(transition=transition, box_rejected=box_rejected,
                                 epoch_rotated=False, locked_now=True, attempted=True,
                                 reason_code=geom.reason_code, stability=verdict)
    return TransitionOutcome(transition=transition, box_rejected=box_rejected,
                             epoch_rotated=False, locked_now=False, attempted=True,
                             reason_code=geom.reason_code, stability=verdict)


def active_session_box(pip_state, t: float, frame_width: int, frame_height: int, *,
                       expected_resolution=None) -> tuple[BBoxXYXY | None, str]:
    """The box that is genuinely live at instant `t`, or None and why not.

    Fail-closed in three separate ways, kept distinct because they mean different
    things when they show up in an evaluation: never locked, locked but stale, and
    locked but no longer a legal box for this frame size.

    Mutates only `pip_state`'s own counters, exactly as the runtime does.
    """
    if not pip_state.locked or pip_state.session_box is None:
        return None, REASON_NO_SESSION_BOX
    if not pip_state.pip_state_valid_at(t):
        pip_state.n_expired += 1
        return None, REASON_TTL_EXPIRED
    v = validate_box(pip_state.session_box, frame_width, frame_height,
                     expected_resolution=expected_resolution)
    if not v.valid:
        return None, v.reason_code
    pip_state.n_ttl_inferred += 1
    return pip_state.session_box, REASON_OK
