from scripts.audit_batched_pld_task_isolation import audit_events


def _event(task_id="task/1", emitted=None, verifier=None, drafts=None, accepted_drafts=0):
    emitted = emitted or [7]
    verifier = verifier or [7]
    drafts = drafts or []
    return {
        "event": "verify_scatter",
        "verifier_batch_id": 1,
        "task_id": task_id,
        "batch_slot": 0,
        "prefix_len_before": 10,
        "prefix_len_after": 10 + len(emitted),
        "cache_len_after": 9 + len(emitted),
        "kv_cache_task_id_or_cache_handle": f"{task_id}:1234",
        "finished_flag_before": False,
        "emitted_tokens": emitted,
        "accepted_tokens": emitted,
        "verifier_output_tokens": verifier,
        "draft_tokens": drafts,
        "accepted_drafts": accepted_drafts,
    }


def test_task_isolation_audit_accepts_verified_correction_token():
    report = audit_events([_event()])
    assert report["passed"]
    assert report["unverified_token_violations"] == 0


def test_task_isolation_audit_accepts_verified_draft_prefix():
    event = _event(emitted=[4, 5, 9], verifier=[4, 5, 9], drafts=[4, 5], accepted_drafts=2)
    report = audit_events([event])
    assert report["passed"]


def test_task_isolation_audit_rejects_unverified_emission():
    event = _event(emitted=[8], verifier=[7])
    report = audit_events([event])
    assert not report["passed"]
    assert report["unverified_token_violations"] == 1


def test_task_isolation_audit_detects_duplicate_batch_slot():
    a = _event(task_id="a")
    b = _event(task_id="b")
    b["batch_slot"] = a["batch_slot"]
    report = audit_events([a, b])
    assert not report["passed"]
    assert any(v["type"] == "duplicate_batch_slot" for v in report["violations"])


def test_task_isolation_audit_detects_finished_reentry():
    finish = {"event": "task_finish", "task_id": "a"}
    event = _event(task_id="a")
    report = audit_events([finish, event])
    assert not report["passed"]
    assert report["finished_task_violations"] == 1
