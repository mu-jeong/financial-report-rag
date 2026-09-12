"""Session-isolation regressions for process-local chat job events."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from apps.gui import chat_jobs


def _registry(*events: dict) -> dict:
    return {
        "running_job_ids": set(),
        "events": list(events),
        "lock": threading.Lock(),
    }


def test_chat_job_events_are_consumed_only_by_their_session_owner(monkeypatch):
    owner_a_events = [
        {"owner_id": "owner-a", "status": "progress", "event_id": "progress"},
        {"owner_id": "owner-a", "status": "succeeded", "event_id": "success"},
        {"owner_id": "owner-a", "status": "failed", "event_id": "failure"},
        {
            "owner_id": "owner-a",
            "status": "failed",
            "event_id": "start-failure",
        },
    ]
    owner_b_events = [
        {**event, "owner_id": "owner-b", "event_id": f"b-{event['event_id']}"}
        for event in owner_a_events
    ]
    registry = _registry(*(owner_a_events + owner_b_events))
    monkeypatch.setattr(chat_jobs, "_chat_job_registry", lambda: registry)

    assert chat_jobs.consume_chat_job_events("owner-b") == owner_b_events
    assert registry["events"] == owner_a_events
    assert chat_jobs.consume_chat_job_events("owner-a") == owner_a_events
    assert registry["events"] == []


def test_legacy_unowned_events_never_drain_session_owned_events(monkeypatch):
    unowned = {"status": "succeeded", "event_id": "legacy"}
    owned = {
        "owner_id": "owner-a",
        "status": "succeeded",
        "event_id": "owned",
    }
    registry = _registry(unowned, owned)
    monkeypatch.setattr(chat_jobs, "_chat_job_registry", lambda: registry)

    assert chat_jobs.consume_chat_job_events() == [unowned]
    assert registry["events"] == [owned]


def test_chat_job_event_retention_is_bounded_for_closed_sessions():
    registry = _registry()
    registry["event_retention_limit"] = 3

    for index in range(5):
        chat_jobs._record_chat_job_event(
            {
                "owner_id": f"closed-owner-{index}",
                "status": "succeeded",
                "event_id": index,
            },
            registry,
        )

    assert [event["event_id"] for event in registry["events"]] == [2, 3, 4]


def test_notification_renderer_never_displays_another_sessions_events(monkeypatch):
    owner_a_events = [
        {
            "owner_id": "owner-a",
            "status": "progress",
            "thread_id": "thread-a",
            "assistant_message_id": 1,
            "engine_queue_released": True,
        },
        {
            "owner_id": "owner-a",
            "status": "succeeded",
            "thread_id": "thread-a",
            "thread_name": "대화 A 성공",
            "assistant_message_id": 2,
            "message": "A success",
        },
        {
            "owner_id": "owner-a",
            "status": "failed",
            "thread_id": "thread-a",
            "thread_name": "대화 A 실패",
            "assistant_message_id": 3,
            "message": "A failure",
        },
        {
            "owner_id": "owner-a",
            "status": "failed",
            "thread_id": "thread-a",
            "thread_name": "대화 A 시작 실패",
            "assistant_message_id": 4,
            "message": "A start failure",
        },
    ]
    owner_b_event = {
        "owner_id": "owner-b",
        "status": "succeeded",
        "thread_id": "thread-b",
        "thread_name": "대화 B",
        "assistant_message_id": 5,
        "message": "B success",
    }
    registry = _registry(*(owner_a_events + [owner_b_event]))

    class SessionState(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, value):
            self[name] = value

    class FakeStreamlit:
        def __init__(self):
            self.session_state = SessionState(_chat_job_owner_id="owner-b")
            self.toasts: list[tuple[str, str]] = []
            self.reruns: list[str | None] = []

        def toast(self, message, *, icon):
            self.toasts.append((message, icon))

        def rerun(self, *, scope=None):
            self.reruns.append(scope)

    fake_st = FakeStreamlit()
    monkeypatch.setattr(chat_jobs, "st", fake_st)
    monkeypatch.setattr(chat_jobs, "_chat_job_registry", lambda: registry)

    renderer = getattr(
        chat_jobs.render_chat_job_notifications,
        "__wrapped__",
        chat_jobs.render_chat_job_notifications,
    )
    renderer("thread-b")

    assert fake_st.toasts == [("B success", "✅")]
    assert all(event["owner_id"] == "owner-a" for event in registry["events"])
    assert fake_st.reruns == ["app"]


def _patch_start_dependencies(monkeypatch, registry: dict, session_state: dict) -> None:
    monkeypatch.setattr(chat_jobs, "st", SimpleNamespace(session_state=session_state))
    monkeypatch.setattr(chat_jobs, "_chat_job_registry", lambda: registry)
    monkeypatch.setattr(chat_jobs, "get_chat_history", lambda _thread_id: [])
    monkeypatch.setattr(
        chat_jobs.search_engine,
        "get_search_engine_status",
        lambda: {"state": "ready"},
    )
    monkeypatch.setattr(
        chat_jobs,
        "append_pending_exchange",
        lambda *_args, **_kwargs: (11, 12),
    )


def test_start_captures_session_owner_before_creating_background_thread(monkeypatch):
    registry = _registry()
    session_state: dict = {}
    created_threads: list[object] = []
    _patch_start_dependencies(monkeypatch, registry, session_state)

    class CapturingThread:
        def __init__(self, *, target, kwargs, name, daemon):
            self.target = target
            self.kwargs = kwargs
            self.name = name
            self.daemon = daemon
            created_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(chat_jobs.threading, "Thread", CapturingThread)

    assert chat_jobs.start_chat_response_job(
        thread_id="thread-a",
        thread_name="대화 A",
        user_query="질문",
    ) == 12

    owner_id = session_state["_chat_job_owner_id"]
    assert isinstance(owner_id, str) and owner_id
    assert created_threads[0].kwargs["owner_id"] == owner_id


def test_thread_start_failure_event_belongs_to_the_starting_session(monkeypatch):
    registry = _registry()
    session_state: dict = {}
    updates: list[tuple] = []
    _patch_start_dependencies(monkeypatch, registry, session_state)
    monkeypatch.setattr(
        chat_jobs,
        "update_message",
        lambda *args: updates.append(args),
    )

    class FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(chat_jobs.threading, "Thread", FailingThread)

    chat_jobs.start_chat_response_job(
        thread_id="thread-a",
        thread_name="대화 A",
        user_query="질문",
    )

    owner_id = session_state["_chat_job_owner_id"]
    assert updates[0][2]["status"] == "failed"
    assert registry["events"] == [
        {
            "owner_id": owner_id,
            "status": "failed",
            "thread_id": "thread-a",
            "thread_name": "대화 A",
            "assistant_message_id": 12,
            "message": "'대화 A' 답변 작업을 시작하지 못했습니다.",
            "engine_queue_released": False,
        }
    ]
