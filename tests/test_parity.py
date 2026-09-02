"""Causal state-transition contract.

`fixtures/session_trace.json` is a deterministic synthetic contract fixture: an
observation sequence built from two fixed box positions that covers lock, TTL expiry,
unlock and relock, together with the state this package must produce at every tick.
No real recording, timestamp or identifier is involved. The test replays the sequence
and requires an exact match on every field.
"""

from __future__ import annotations

import json
from pathlib import Path

from colfov.session_pip import (
    METHOD_MEDIAN, PipQualificationState, PipQualifier, RoutedObservation,
    active_session_box, apply_observation,
)

TRACE = json.loads((Path(__file__).resolve().parent / "fixtures" /
                    "session_trace.json").read_text(encoding="utf-8"))


def replay():
    W, H = TRACE["frame_width"], TRACE["frame_height"]
    q = PipQualifier(frame_width=W, frame_height=H, method=METHOD_MEDIAN)
    s = PipQualificationState(monitor_interval_sec=TRACE["monitor_interval_sec"])
    locks = unlocks = 0
    out = []
    for rec in TRACE["records"]:
        t, b = rec["timestamp_sec"], rec["observation_box"]
        obs = (RoutedObservation(frame_id=f"f{int(t*100):06d}", t=t, box=tuple(b))
               if b else None)
        o = apply_observation(obs, t, pip_state=s, qualifier=q,
                              frame_width=W, frame_height=H)
        box, reason = active_session_box(s, t, W, H)
        ev = None
        if o.locked_now:
            locks += 1
            ev = "relock" if unlocks else "lock"
        elif o.epoch_rotated:
            unlocks += 1
            ev = "unlock"
        out.append({"timestamp_sec": t,
                    "active_session_pip_box": list(box) if box else None,
                    "pip_state": "locked" if s.locked else "unlocked",
                    "pip_epoch": q.epoch, "pip_event": ev, "pip_reason": reason})
    return out, locks, unlocks, q.epoch


def test_the_fixture_covers_lock_unlock_and_relock():
    events = [r["pip_event"] for r in TRACE["records"] if r["pip_event"]]
    assert events == ["lock", "unlock", "relock"], events
    assert TRACE["final_epoch"] >= 1


def test_every_tick_matches_the_verified_implementation():
    got, locks, unlocks, epoch = replay()
    assert len(got) == len(TRACE["records"])
    for mine, want in zip(got, TRACE["records"]):
        for key in ("timestamp_sec", "active_session_pip_box", "pip_state",
                    "pip_epoch", "pip_event", "pip_reason"):
            assert mine[key] == want[key], (want["timestamp_sec"], key,
                                            mine[key], want[key])
    assert (locks, unlocks, epoch) == (TRACE["n_locks"], TRACE["n_unlocks"],
                                       TRACE["final_epoch"])
