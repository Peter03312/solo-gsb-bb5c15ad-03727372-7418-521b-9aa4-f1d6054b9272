"""End-to-end tests for multi-stop trips (跨回收点行程).

A trip atomically books 2..5 reservations at distinct recycling points in
visiting order; all stops are created or none are.
"""

from __future__ import annotations

import secrets
import threading
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_point(client, tz="UTC"):
    r = client.post("/points", json={"name": "行程测试点", "timezone": tz})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def open_point(client, weekly=None, makeup=None, tz="UTC"):
    """Point with a published rule covering tomorrow 08:00-12:00 by default."""
    pid = create_point(client, tz=tz)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": weekly
                if weekly is not None
                else [
                    {
                        "weekday": tomorrow.isoweekday(),
                        "start_time": "08:00",
                        "end_time": "12:00",
                    }
                ],
                "closed_dates": [],
                "makeup": makeup or [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return pid


def open_points(client, n, **kw):
    return [open_point(client, **kw) for _ in range(n)]


def at(hour, minute=0, day_offset=1):
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def trip(client, key, stops):
    return client.post(
        "/trips",
        json={
            "idempotency_key": key,
            "stops": [
                {"point_id": pid, "start": iso(s), "end": iso(e)}
                for pid, s, e in stops
            ],
        },
    )


def three_stop_stops(pids, *, first=9):
    """Three chronologically ordered half-hour stops on distinct points."""
    return [
        (pids[0], at(first), at(first, 30)),
        (pids[1], at(first + 1), at(first + 1, 30)),
        (pids[2], at(first + 2), at(first + 2, 30)),
    ]


def book_trip(client, pids=None, key=None, **kw):
    pids = pids or open_points(client, 3)
    stops = three_stop_stops(pids, **kw)
    r = trip(client, key or f"trip-{secrets.token_hex(8)}", stops)
    assert r.status_code == 201, r.text
    return pids, stops, r.json()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_trip_books_all_stops_atomically_in_input_order(client):
    p1, p2, p3 = open_points(client, 3)
    stops = [
        (p1, at(9), at(9, 30)),
        (p2, at(10), at(10, 30)),
        (p3, at(11), at(11, 30)),
    ]
    r = trip(client, "trip-ok", stops)
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["stops"]) == 3
    for index, ((pid, s, e), stop) in enumerate(zip(stops, body["stops"])):
        assert stop["index"] == index
        assert stop["point_id"] == pid
        assert stop["start"] == iso(s)
        assert stop["end"] == iso(e)
        assert stop["status"] == "active"
        assert stop["revision"] == 1
        assert stop["rules_version"] == 1
        assert stop["cancel_token"]
        assert stop["cancelled_at"] is None
    # Reservation numbers differ and follow input order in the response.
    ids = [s["id"] for s in body["stops"]]
    assert len(set(ids)) == 3


def test_trip_accepts_two_and_five_stops(client):
    pids2 = open_points(client, 2)
    r = trip(client, "trip-two", [(pids2[0], at(9), at(9, 30)),
                                  (pids2[1], at(10), at(10, 30))])
    assert r.status_code == 201, r.text
    assert len(r.json()["stops"]) == 2

    pids5 = open_points(client, 5)
    slots = [(8, 0), (8, 10), (8, 20), (8, 30), (8, 40)]
    stops = [
        (pid, at(h, m), at(h, m + 5))
        for pid, (h, m) in zip(pids5, slots)
    ]
    r = trip(client, "trip-five", stops)
    assert r.status_code == 201, r.text
    assert len(r.json()["stops"]) == 5


def test_trip_stops_that_touch_are_allowed(client):
    p1, p2 = open_points(client, 2)
    r = trip(client, "trip-touch", [(p1, at(9), at(10)),
                                    (p2, at(10), at(11))])
    assert r.status_code == 201, r.text


def test_trip_stop_reservations_occupy_their_windows(client):
    pids, _stops, body = book_trip(client)
    clash = reserve(client, pids[1], "squatter", at(10, 15), at(10, 45))
    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "reservation_overlap"
    # Touching is still fine.
    neighbor = reserve(client, pids[1], "neighbor", at(10, 30), at(11))
    assert neighbor.status_code == 201, neighbor.text


def test_trip_stop_reservations_are_cancelable_with_their_own_tokens(client):
    pids, _stops, body = book_trip(client)
    stop = body["stops"][0]
    r = cancel(client, pids[0], stop["id"], stop["cancel_token"])
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # The freed window at that point is bookable again; other stops stand.
    again = reserve(client, pids[0], "replacement", at(9), at(9, 30))
    assert again.status_code == 201, again.text
    clash = reserve(client, pids[2], "squatter", at(11), at(11, 30))
    assert clash.status_code == 409


def test_trip_records_each_points_rules_version(client):
    p1 = open_point(client)
    p2 = open_point(client)
    weekday = at(0).isoweekday()
    # Bump point 2 to a second version; its stop must report version 2.
    pub = client.post(
        f"/points/{p2}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": weekday, "start_time": "07:00", "end_time": "14:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert pub.status_code == 201, pub.text
    r = trip(client, "trip-ver", [(p1, at(9), at(9, 30)),
                                  (p2, at(9, 30), at(10))])
    assert r.status_code == 201, r.text
    stops = r.json()["stops"]
    assert stops[0]["rules_version"] == 1
    assert stops[1]["rules_version"] == 2


# ---------------------------------------------------------------------------
# Structural / cross-stop validation
# ---------------------------------------------------------------------------

def test_trip_requires_between_two_and_five_stops(client):
    pids = open_points(client, 5)
    for n in (0, 1, 6):
        stops = [
            (pids[i % 5], at(8, i * 10), at(8, i * 10 + 5))
            for i in range(n)
        ]
        r = trip(client, f"trip-badlen-{n}", stops)
        assert r.status_code == 422, (n, r.text)
        assert r.json()["error"]["code"] == "invalid_request"


def test_trip_rejects_duplicate_points(client):
    p1, p2 = open_points(client, 2)
    r = trip(client, "trip-dup", [
        (p1, at(9), at(9, 30)),
        (p2, at(10), at(10, 30)),
        (p1, at(11), at(11, 30)),  # point 1 repeats
    ])
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"]["stop_index"] == 2
    assert err["details"]["reason"] == "duplicate_point"
    assert err["details"]["point_id"] == p1
    # Nothing was booked.
    free = reserve(client, p1, "free", at(9), at(9, 30))
    assert free.status_code == 201, free.text


def test_trip_rejects_non_chronological_stops(client):
    p1, p2 = open_points(client, 2)
    r = trip(client, "trip-order", [
        (p1, at(10), at(11)),
        (p2, at(10, 30), at(11, 30)),  # starts before stop 0 ends
    ])
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["reason"] == "stops_not_chronological"
    assert err["details"]["previous_end"] == iso(at(11))


def test_trip_naive_timestamps_are_invalid_request(client):
    p1, p2 = open_points(client, 2)
    r = client.post("/trips", json={
        "idempotency_key": "trip-naive",
        "stops": [
            {"point_id": p1, "start": "2026-10-01T09:00:00",
             "end": "2026-10-01T09:30:00"},
            {"point_id": p2, "start": "2026-10-01T10:00:00Z",
             "end": "2026-10-01T10:30:00Z"},
        ],
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_trip_accepts_timezone_offsets_and_interprets_as_utc(client):
    # 17:00+08:00 == 09:00Z; both stops land inside the 08:00-12:00 UTC day.
    p1, p2 = open_points(client, 2)
    r = client.post("/trips", json={
        "idempotency_key": "trip-offset",
        "stops": [
            {"point_id": p1, "start": "2099-01-01T09:00:00+00:00",
             "end": "2099-01-01T09:30:00+00:00"},
            {"point_id": p2, "start": "2099-01-01T10:00:00+00:00",
             "end": "2099-01-01T10:30:00+00:00"},
        ],
    })
    # Beyond the 30-day horizon: per-stop invalid_period must survive.
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"

    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    day = tomorrow.isoformat()
    r = client.post("/trips", json={
        "idempotency_key": "trip-offset-ok",
        "stops": [
            {"point_id": p1,
             "start": f"{day}T09:00:00+00:00",
             "end": f"{day}T09:30:00+00:00"},
            {"point_id": p2,
             "start": f"{day}T18:00:00+08:00",  # == 10:00Z
             "end": f"{day}T18:30:00+08:00"},
        ],
    })
    assert r.status_code == 201, r.text
    assert r.json()["stops"][1]["start"] == f"{day}T10:00:00Z"


# ---------------------------------------------------------------------------
# Per-stop validation — original error reason + failing stop index, and
# all-or-nothing persistence
# ---------------------------------------------------------------------------

def test_trip_unknown_point_is_404_with_stop_index(client):
    p1 = open_point(client)
    r = trip(client, "trip-ghost", [
        (p1, at(9), at(9, 30)),
        (999999, at(10), at(10, 30)),
    ])
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "point_not_found"
    assert err["details"]["stop_index"] == 1


def test_trip_stop_without_published_rules_conflicts(client):
    p1 = open_point(client)
    p2 = create_point(client)  # no rules
    r = trip(client, "trip-norules", [(p1, at(9), at(9, 30)),
                                      (p2, at(10), at(10, 30))])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "no_published_version"
    assert err["details"]["stop_index"] == 1


def test_trip_invalid_window_is_invalid_period_with_stop_index(client):
    p1, p2, p3 = open_points(client, 3)
    # Third stop ends before it starts.
    r = trip(client, "trip-badwindow", [
        (p1, at(9), at(9, 30)),
        (p2, at(10), at(10, 30)),
        (p3, at(11, 30), at(11)),
    ])
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 2


def test_trip_outside_open_hours_reports_stop(client):
    p1, p2 = open_points(client, 2)
    r = trip(client, "trip-outside", [
        (p1, at(9), at(9, 30)),
        (p2, at(13), at(13, 30)),  # calendar only opens 08:00-12:00
    ])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "outside_open_hours"
    assert err["details"]["stop_index"] == 1


def test_trip_validates_against_each_points_own_timezone(client):
    p1 = open_point(client)
    # Shanghai point open 08:00-12:00 local: 09:00Z is 17:00 local — outside.
    p2 = open_point(client, tz="Asia/Shanghai")
    r = trip(client, "trip-tz", [(p1, at(9), at(9, 30)),
                                 (p2, at(9, 30), at(10))])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "outside_open_hours"
    assert err["details"]["stop_index"] == 1
    # Visiting the Shanghai point first (01:00Z == 09:00 local, inside its
    # 08:00-12:00 opening) then the UTC point works.
    ok = trip(client, "trip-tz-ok", [(p2, at(1), at(1, 30)),
                                     (p1, at(8), at(8, 30))])
    assert ok.status_code == 201, ok.text


def test_trip_overlap_reports_stop_and_conflicting_ids(client):
    p1, p2, p3 = open_points(client, 3)
    holder = reserve(client, p2, "holder", at(10), at(10, 30))
    assert holder.status_code == 201
    r = trip(client, "trip-overlap", [
        (p1, at(9), at(9, 30)),
        (p2, at(10, 15), at(10, 45)),
        (p3, at(11), at(11, 30)),
    ])
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["conflicting_reservation_ids"] == [holder.json()["id"]]


def test_failed_trip_persists_nothing(client):
    p1, p2, p3 = open_points(client, 3)
    reserve(client, p2, "holder", at(10), at(10, 30))
    r = trip(client, "trip-rollback", [
        (p1, at(9), at(9, 30)),
        (p2, at(10, 15), at(10, 45)),
        (p3, at(11), at(11, 30)),
    ])
    assert r.status_code == 409
    # The unheld stop windows remain free (no partial booking).
    for pid, s, e in ((p1, at(9), at(9, 30)),
                      (p2, at(11), at(11, 30)),
                      (p3, at(11), at(11, 30))):
        free = reserve(client, pid, f"free-{pid}", s, e)
        assert free.status_code == 201, (pid, free.text)


def test_failed_trip_can_be_retried_with_fixed_content_same_key(client):
    p1, p2, p3 = open_points(client, 3)
    bad = trip(client, "trip-fix", [
        (p1, at(9), at(9, 30)),
        (p2, at(10), at(10, 30)),
        (p3, at(13), at(13, 30)),  # outside hours; chronological check passes
    ])
    assert bad.status_code == 409
    assert bad.json()["error"]["details"]["stop_index"] == 2
    # Fix the bad stop (move it back inside opening hours) and retry under
    # the same key: no idempotency conflict because nothing was stored.
    ok = trip(client, "trip-fix", [
        (p1, at(9), at(9, 30)),
        (p2, at(10), at(10, 30)),
        (p3, at(11), at(11, 30)),
    ])
    assert ok.status_code == 201, ok.text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_trip_same_key_same_content_replays_first_snapshot(client):
    pids, stops, first = book_trip(client, key="trip-idem")
    replay = trip(client, "trip-idem", stops)
    assert replay.status_code == 200, replay.text
    assert replay.json() == first
    # No second set of reservations: stop windows are held exactly once.
    clash = reserve(client, pids[0], "squatter", at(9, 15), at(9, 45))
    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "reservation_overlap"


def test_trip_replay_after_stop_cancel_reports_live_status(client):
    pids, stops, first = book_trip(client, key="trip-cancel-replay")
    token = first["stops"][1]["cancel_token"]
    rid = first["stops"][1]["id"]
    assert cancel(client, pids[1], rid, token).status_code == 200

    replay = trip(client, "trip-cancel-replay", stops)
    assert replay.status_code == 200
    body = replay.json()
    assert body["id"] == first["id"]
    # Snapshot window unchanged ...
    assert body["stops"][1]["start"] == iso(at(10))
    assert body["stops"][1]["revision"] == 1
    assert body["stops"][1]["rules_version"] == 1
    # ... but the status reflects the live reservation.
    assert body["stops"][1]["status"] == "cancelled"
    assert body["stops"][1]["cancelled_at"] is not None
    assert body["stops"][0]["status"] == "active"


def test_trip_replay_after_stop_reschedule_reports_original_snapshot(client):
    pids, stops, first = book_trip(client, key="trip-move-replay")
    token = first["stops"][0]["cancel_token"]
    rid = first["stops"][0]["id"]
    moved = client.post(
        f"/points/{pids[0]}/reservations/{rid}/reschedule",
        json={
            "cancel_token": token,
            "idempotency_key": "move-stop-0",
            "expected_revision": 1,
            "start": iso(at(8)),
            "end": iso(at(8, 30)),
        },
    )
    assert moved.status_code == 200, moved.text

    replay = trip(client, "trip-move-replay", stops)
    assert replay.status_code == 200
    stop0 = replay.json()["stops"][0]
    # First-creation snapshot: original window, revision 1.
    assert stop0["start"] == iso(at(9))
    assert stop0["end"] == iso(at(9, 30))
    assert stop0["revision"] == 1
    assert stop0["status"] == "active"
    # The old window was freed by the reschedule.
    free = reserve(client, pids[0], "old-window", at(9), at(9, 30))
    assert free.status_code == 201, free.text


def test_trip_same_key_different_content_conflicts(client):
    pids, stops, first = book_trip(client, key="trip-diff")

    # Different window at one stop.
    changed = [
        (s[0], s[1], s[2]) for s in stops
    ]
    changed[1] = (pids[1], at(10, 15), at(10, 45))
    r = trip(client, "trip-diff", changed)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "idempotency_conflict"
    assert err["details"]["trip_id"] == first["id"]
    stored = err["details"]["stops"]
    assert [s["point_id"] for s in stored] == pids
    assert stored[1]["start"] == iso(at(10))

    # A different number of stops also conflicts.
    r = trip(client, "trip-diff", stops[:2])
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "idempotency_conflict"

    # A different point at one stop also conflicts.
    extra = open_point(client)
    changed2 = list(stops)
    changed2[2] = (extra, at(11), at(11, 30))
    r = trip(client, "trip-diff", changed2)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "idempotency_conflict"

    # The conflict created nothing extra.
    clash = reserve(client, extra, "still-free", at(11), at(11, 30))
    assert clash.status_code == 201, clash.text


def test_trip_idempotency_keys_are_global_scope(client):
    # The trip key namespace is global (not per point): two unrelated trips
    # may share a key only if their ordered stop content is identical, in
    # which case the second call is just a replay.
    pids = open_points(client, 3)
    stops = three_stop_stops(pids, first=8)
    first = trip(client, "global-key", stops)
    assert first.status_code == 201, first.text
    # Same content again (even though the points are "another caller's"):
    # replay rather than a second booking.
    replay = trip(client, "global-key", stops)
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    # Different content under the same global key conflicts.
    other = open_points(client, 3)
    r = trip(client, "global-key", three_stop_stops(other, first=8))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "idempotency_conflict"


# ---------------------------------------------------------------------------
# Interaction with waitlist / reassign
# ---------------------------------------------------------------------------

def test_cancelling_a_trip_stop_promotes_that_points_waitlist(client):
    pids, _stops, body = book_trip(client, key="trip-wait")
    # Someone waits for stop 1's window at point pids[1].
    w = client.post(
        f"/points/{pids[1]}/waitlist",
        json={"idempotency_key": "w-stop-1",
              "start": iso(at(10)), "end": iso(at(10, 30))},
    )
    assert w.status_code == 201, w.text
    wid, wtoken = w.json()["id"], w.json()["query_token"]

    stop1 = body["stops"][1]
    r = cancel(client, pids[1], stop1["id"], stop1["cancel_token"])
    assert r.status_code == 200
    state = client.get(
        f"/points/{pids[1]}/waitlist/{wid}", params={"query_token": wtoken}
    ).json()
    assert state["status"] == "promoted"
    assert state["reservation"]["start"] == iso(at(10))


def test_trip_replay_after_stop_reassigned_reports_original_snapshot(client):
    pids, stops, body = book_trip(client, key="trip-reassign-replay")
    target = open_point(client)
    stop0 = body["stops"][0]
    r = client.post(
        f"/points/{pids[0]}/reservations/{stop0['id']}/reassign",
        json={
            "cancel_token": stop0["cancel_token"],
            "idempotency_key": "reassign-stop-0",
            "expected_revision": 1,
            "target_point_id": target,
            "start": iso(at(8)),
            "end": iso(at(8, 30)),
        },
    )
    assert r.status_code == 200, r.text

    replay = trip(client, "trip-reassign-replay", stops)
    assert replay.status_code == 200
    out0 = replay.json()["stops"][0]
    # The trip stop still points at the original reservation id, now
    # cancelled, with its first-created snapshot intact.
    assert out0["id"] == stop0["id"]
    assert out0["point_id"] == pids[0]
    assert out0["status"] == "cancelled"
    assert out0["start"] == iso(at(9))
    assert out0["revision"] == 1


# ---------------------------------------------------------------------------
# Concurrency — atomicity, no double occupancy, no deadlock
# ---------------------------------------------------------------------------

def test_concurrent_same_key_trip_requests_exactly_one_books(client):
    pids = open_points(client, 3)
    stops = three_stop_stops(pids)
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt():
        barrier.wait()
        outcomes.append(trip(client, "trip-race-key", stops))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(r.status_code for r in outcomes)
    assert statuses == [200, 201], [r.text for r in outcomes]
    bodies = [r.json() for r in outcomes]
    assert bodies[0]["id"] == bodies[1]["id"]
    # Exactly one reservation per window.
    clash = reserve(client, pids[0], "squatter", at(9), at(9, 30))
    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "reservation_overlap"


def test_concurrent_trip_and_single_booking_no_double_occupancy(client):
    p1, p2, p3 = open_points(client, 3)
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip():
        barrier.wait()
        outcomes["trip"] = trip(client, "trip-vs-book", [
            (p1, at(9), at(9, 30)),
            (p2, at(10), at(10, 30)),
        ])

    def do_book():
        barrier.wait()
        outcomes["book"] = reserve(client, p2, "racer", at(10), at(10, 30))

    threads = [threading.Thread(target=do_trip), threading.Thread(target=do_book)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trip_r, book_r = outcomes["trip"], outcomes["book"]
    if trip_r.status_code == 201:
        assert book_r.status_code == 409
        assert book_r.json()["error"]["code"] == "reservation_overlap"
    else:
        assert trip_r.status_code == 409
        err = trip_r.json()["error"]
        assert err["code"] == "reservation_overlap"
        assert err["details"]["stop_index"] == 1
        assert book_r.status_code == 201
        # The failed trip booked nothing at its other point either.
        free = reserve(client, p1, "free-after-fail", at(9), at(9, 30))
        assert free.status_code == 201, free.text


def test_concurrent_overlapping_trips_exactly_one_wins(client):
    pids = open_points(client, 3)
    stops = three_stop_stops(pids)
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip(name, key):
        barrier.wait()
        outcomes[name] = trip(client, key, stops)

    threads = [
        threading.Thread(target=do_trip, args=("a", "trip-a")),
        threading.Thread(target=do_trip, args=("b", "trip-b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(outcomes[k].status_code for k in ("a", "b"))
    assert statuses == [201, 409]
    loser = outcomes["a"] if outcomes["a"].status_code == 409 else outcomes["b"]
    assert loser.json()["error"]["code"] == "reservation_overlap"


def test_concurrent_opposite_order_trips_do_not_deadlock(client):
    # Two trips touch the same two points but list them in opposite input
    # order (point locks are nonetheless taken in ascending id order) and
    # book disjoint windows, so both must commit.  Repeated a few times to
    # give lock-order inversions a chance to surface.
    pids = open_points(client, 3)
    low, high = min(pids), max(pids)
    for round_no in range(5):
        barrier = threading.Barrier(2)
        outcomes = {}
        a_windows = [(at(8, round_no * 5), at(8, round_no * 5 + 3)),
                     (at(9, round_no * 5), at(9, round_no * 5 + 3))]
        d_windows = [(at(10, round_no * 5), at(10, round_no * 5 + 3)),
                     (at(11, round_no * 5), at(11, round_no * 5 + 3))]

        def ascending():
            barrier.wait()
            outcomes["asc"] = trip(client, f"trip-asc-{round_no}", [
                (low, a_windows[0][0], a_windows[0][1]),
                (high, a_windows[1][0], a_windows[1][1]),
            ])

        def descending():
            barrier.wait()
            outcomes["desc"] = trip(client, f"trip-desc-{round_no}", [
                (high, d_windows[0][0], d_windows[0][1]),
                (low, d_windows[1][0], d_windows[1][1]),
            ])

        threads = [
            threading.Thread(target=ascending),
            threading.Thread(target=descending),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "deadlock suspected"
        assert outcomes["asc"].status_code == 201, outcomes["asc"].text
        assert outcomes["desc"].status_code == 201, outcomes["desc"].text


def test_concurrent_trip_and_reassign_over_shared_points_no_deadlock(client):
    # A reservation on point A gets reassigned A -> B while a trip books
    # windows at A and B (plus C).  Both must finish; exactly one holds any
    # contested window.
    pids = open_points(client, 3)
    p_a, p_b = pids[0], pids[1]
    held = reserve(client, p_a, "held", at(8), at(8, 30))
    assert held.status_code == 201
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_reassign():
        barrier.wait()
        outcomes["reassign"] = client.post(
            f"/points/{p_a}/reservations/{held.json()['id']}/reassign",
            json={
                "cancel_token": held.json()["cancel_token"],
                "idempotency_key": "mv",
                "expected_revision": 1,
                "target_point_id": p_b,
                "start": iso(at(10)),
                "end": iso(at(10, 30)),
            },
        )

    def do_trip():
        barrier.wait()
        outcomes["trip"] = trip(client, "trip-vs-reassign", [
            (p_a, at(9), at(9, 30)),
            (p_b, at(10), at(10, 30)),
            (pids[2], at(11), at(11, 30)),
        ])

    threads = [threading.Thread(target=do_reassign), threading.Thread(target=do_trip)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "deadlock suspected"

    rs, tr = outcomes["reassign"], outcomes["trip"]
    assert rs.status_code in (200, 409)
    assert tr.status_code in (201, 409)
    # The contested B window is held exactly once.
    if rs.status_code == 200:
        assert tr.status_code == 409
        assert tr.json()["error"]["code"] == "reservation_overlap"
    else:
        assert tr.status_code == 201
        assert rs.json()["error"]["code"] == "reservation_overlap"


def test_concurrent_trip_and_breaking_publish_exactly_one_wins(client):
    p1, p2 = open_points(client, 2)
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip():
        barrier.wait()
        outcomes["trip"] = trip(client, "trip-vs-publish", [
            (p1, at(9), at(9, 30)),
            (p2, at(10), at(10, 30)),
        ])

    def do_publish():
        barrier.wait()
        # Empty calendar at point 2 would un-cover its stop.
        outcomes["publish"] = client.post(
            f"/points/{p2}/rules",
            json={
                "expected_version": 1,
                "rules": {"weekly": [], "closed_dates": [], "makeup": []},
            },
        )

    threads = [threading.Thread(target=do_trip), threading.Thread(target=do_publish)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    tr, pub = outcomes["trip"], outcomes["publish"]
    if tr.status_code == 201:
        assert pub.status_code == 409
        assert pub.json()["error"]["code"] == "reservation_change_conflict"
    else:
        assert pub.status_code == 201
        assert tr.status_code == 409
        assert tr.json()["error"]["code"] == "outside_open_hours"
        assert tr.json()["error"]["details"]["stop_index"] == 1
        # Nothing booked at point 1 either.
        free = reserve(client, p1, "free", at(9), at(9, 30))
        assert free.status_code == 201, free.text
