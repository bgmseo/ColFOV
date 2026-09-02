"""Causal session-PiP behaviour: lock, unlock, relock, epoch, TTL, no future evidence."""

from __future__ import annotations

import pytest

from colfov.session_pip import (
    METHOD_MEDIAN, PipQualificationState, PipQualifier, RoutedObservation,
    active_session_box, apply_observation,
)

W, H = 1280, 720
A = (900, 460, 1100, 610)
B = (200, 120, 400, 270)


def drive(seq, *, interval=0.2, probe_all=True):
    q = PipQualifier(frame_width=W, frame_height=H, method=METHOD_MEDIAN)
    s = PipQualificationState(monitor_interval_sec=interval)
    out = []
    for t, box in seq:
        obs = RoutedObservation(frame_id=f"f{int(t*100):06d}", t=t, box=box) if box else None
        o = apply_observation(obs, t, pip_state=s, qualifier=q,
                              frame_width=W, frame_height=H)
        live, reason = active_session_box(s, t, W, H)
        out.append({"t": t, "box": list(live) if live else None, "reason": reason,
                    "locked": s.locked, "epoch": q.epoch, "locked_now": o.locked_now,
                    "rotated": o.epoch_rotated})
    return out


def test_a_session_with_no_pip_never_locks():
    rec = drive([(i * 0.2, None) for i in range(200)])
    assert not any(r["locked_now"] for r in rec)
    assert all(r["box"] is None for r in rec)
    assert {r["reason"] for r in rec} == {"pip_session_box_unavailable"}


def test_stable_positives_lock_only_after_the_evidence_gates():
    rec = drive([(i * 0.2, A) for i in range(60)])
    locks = [r for r in rec if r["locked_now"]]
    assert len(locks) == 1
    assert locks[0]["t"] > rec[0]["t"], "must not lock on the first observation"


def test_no_active_box_before_the_lock():
    rec = drive([(i * 0.2, A) for i in range(60)])
    t_lock = next(r["t"] for r in rec if r["locked_now"])
    assert all(r["box"] is None for r in rec if r["t"] < t_lock)


def test_a_negative_observation_clears_liveness():
    seq = [(i * 0.2, A) for i in range(60)] + [(12.2, None)]
    assert drive(seq)[-1]["box"] is None
    assert drive(seq)[-1]["reason"] == "pip_ttl_expired"


def test_ttl_expiry_after_a_silent_gap():
    seq = [(i * 0.2, A) for i in range(60)] + [(900.0, None)]
    assert drive(seq)[-1]["reason"] == "pip_ttl_expired"


def test_relocation_unlocks_rotates_the_epoch_and_relocks_elsewhere():
    seq = ([(round(i * 0.2, 6), A) for i in range(60)]
           + [(round(20.0 + i * 0.2, 6), B) for i in range(60)])
    rec = drive(seq)
    assert any(r["rotated"] for r in rec), "a confirmed relocation must rotate the epoch"
    assert rec[-1]["epoch"] >= 1
    assert rec[-1]["box"] == list(B), "must relock on the NEW position"
    assert rec[-1]["box"] != list(A)


def test_the_old_position_is_not_relocked_from_stale_evidence():
    seq = ([(round(i * 0.2, 6), A) for i in range(60)]
           + [(round(20.0 + i * 0.2, 6), B) for i in range(60)])
    rec = drive(seq)
    rot = next(i for i, r in enumerate(rec) if r["rotated"])
    assert rec[rot]["box"] is None, "unlock must not hand back a box in the same tick"


def test_an_early_timestamp_is_never_scored_with_a_later_box():
    """The whole point of causal replay."""
    rec = drive([(i * 0.2, A) for i in range(60)])
    assert rec[0]["box"] is None
    assert rec[-1]["box"] == list(A)


def test_an_out_of_frame_box_is_rejected_and_becomes_a_negative_tick():
    bad = (W - 10, H - 10, W + 500, H + 500)
    rec = drive([(i * 0.2, bad) for i in range(40)])
    assert all(r["box"] is None for r in rec)
