"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.text_mode = "internal"
    a.recovery_enabled = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(
            fresh=Mock(return_value=True),
            observe=Mock(return_value=p),
            target_id="target-test",
            connection_name="default",
        ),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
        "base_goal": "Find a book",
        "plan": ["Find a book"],
        "recoveries": [],
        "block_counts": {},
        "recovery_attempts": 0,
        "handoff": None,
        "block_reason": None,
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def test_caller_text_mode_pauses_without_calling_internal_helper(runner, monkeypatch):
    runner.text_mode = "caller"
    helper = Mock(side_effect=AssertionError("caller mode must not generate text"))
    monkeypatch.setattr(loop, "field_text", helper)

    result = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})

    assert result["status"] == "need_text"
    assert result["field"] == {"label": "Search", "role": "textbox", "current_value": ""}
    assert result["context"]["goal"] == "Find a book"
    assert result["resume_token"]
    helper.assert_not_called()
    runner.state["browser"].act.assert_not_called()


def test_caller_text_resume_executes_once_and_consumes_token(runner):
    runner.text_mode = "caller"
    request = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})

    result = runner.resume_text(request["resume_token"], "book")

    assert result["status"] == "ready"
    runner.state["browser"].act.assert_called_once()
    assert runner.state["history"][-1]["text"] == "book"
    with pytest.raises(ValueError, match="resume token"):
        runner.resume_text(request["resume_token"], "book")
    runner.state["browser"].act.assert_called_once()


def test_stale_caller_text_resume_fails_before_mutation(runner):
    runner.text_mode = "caller"
    runner.state["browser"].fresh.side_effect = [True, False]
    request = runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})

    with pytest.raises(StalePage, match="resume"):
        runner.resume_text(request["resume_token"], "book")

    runner.state["browser"].act.assert_not_called()
    assert runner.pending_text is None
    assert runner.state["status"] == "ready"


def test_internal_text_mode_preserves_helper_path(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 1, "usage": {}}))
    monkeypatch.setattr(loop, "field_text", helper)

    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})

    helper.assert_called_once()
    runner.state["browser"].act.assert_called_once()


def test_recovery_revises_goal_then_equivalent_second_block_requires_handoff(runner, monkeypatch):
    runner.recovery_enabled = True
    runner.state["status"] = "blocked"
    runner.state["block_reason"] = "model_blocked"
    recovery = Mock(return_value={
        "diagnosis": "The current route has no useful control.",
        "revised_subgoal": "Open the search form first.",
        "avoid": ["Do not wait on the unchanged page."],
    })
    monkeypatch.setattr(loop, "recovery_guidance", recovery)

    first = runner.recover()
    assert first["status"] == "ready"
    assert runner.state["recovery_attempts"] == 1
    assert "Open the search form first." in runner.state["goal"]

    runner.state["status"] = "blocked"
    runner.state["block_reason"] = "model_blocked"
    second = runner.recover()
    assert second["status"] == "handoff_required"
    assert second["handoff"]["reason"] == "repeated_block"
    assert second["handoff"]["target_id"] == runner.state["browser"].target_id
    assert second["handoff"]["block_count"] == 2
    assert second["handoff"]["recovery_attempts"] == 1
    assert recovery.call_count == 1


def test_total_recovery_budget_stops_different_blocks(runner, monkeypatch):
    runner.recovery_enabled = True
    recovery = Mock(return_value={"diagnosis": "blocked", "revised_subgoal": "Try another route", "avoid": []})
    monkeypatch.setattr(loop, "recovery_guidance", recovery)

    for index in range(loop.MAX_TOTAL_RECOVERIES):
        runner.state["page"]["fingerprint"] = f"fingerprint-{index}"
        runner.state["status"] = "blocked"
        runner.state["block_reason"] = f"block-{index}"
        assert runner.recover()["status"] == "ready"

    runner.state["page"]["fingerprint"] = "fingerprint-exhausted"
    runner.state["status"] = "blocked"
    runner.state["block_reason"] = "another-block"
    result = runner.recover()
    assert result["status"] == "handoff_required"
    assert result["handoff"]["reason"] == "recovery_budget_exhausted"
    assert recovery.call_count == loop.MAX_TOTAL_RECOVERIES


def test_recovery_unavailable_stops_explicitly(runner, monkeypatch):
    runner.recovery_enabled = True
    runner.state["status"] = "blocked"
    runner.state["block_reason"] = "model_blocked"
    monkeypatch.setattr(loop, "recovery_guidance", Mock(side_effect=ValueError("missing recovery credential")))

    result = runner.recover()

    assert result["status"] == "recovery_unavailable"
    assert result["recovery_error"] == "missing recovery credential"
    runner.state["browser"].act.assert_not_called()


def test_recovery_partial_override_fails_before_sending_text_credentials(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "text-provider-secret")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://text-provider.test/v1")
    monkeypatch.setenv("TEXT_MODEL", "text-model")
    monkeypatch.setenv("RECOVERY_MODEL_BASE_URL", "https://other-provider.test/v1")
    monkeypatch.delenv("RECOVERY_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("RECOVERY_MODEL", raising=False)
    post = Mock()
    monkeypatch.setattr(model, "post_json", post)

    with pytest.raises(ValueError, match="configuration is incomplete"):
        model.recovery_guidance({"current_page": {"url": "https://example.test"}})

    post.assert_not_called()


def test_recovery_text_fallback_uses_one_complete_provider_bundle(monkeypatch):
    for name in ("RECOVERY_MODEL_API_KEY", "RECOVERY_MODEL_BASE_URL", "RECOVERY_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "text-secret")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://text-provider.test/v1")
    monkeypatch.setenv("TEXT_MODEL", "text-model")
    post = Mock(return_value={
        "choices": [{"message": {"content": json.dumps({
            "diagnosis": "The page did not change.",
            "revised_subgoal": "Use another visible path.",
            "avoid": [],
        })}}],
    })
    monkeypatch.setattr(model, "post_json", post)

    model.recovery_guidance({"current_page": {"url": "https://example.test"}})

    assert post.call_args.args[0] == "https://text-provider.test/v1/chat/completions"
    assert post.call_args.args[1] == "text-secret"
    assert post.call_args.args[2]["model"] == "text-model"


def test_recovery_helper_uses_provider_neutral_openai_compatible_config(monkeypatch):
    monkeypatch.setenv("RECOVERY_MODEL_API_KEY", "test")
    monkeypatch.setenv("RECOVERY_MODEL_BASE_URL", "https://provider.test/v1")
    monkeypatch.setenv("RECOVERY_MODEL", "reasoner")
    post = Mock(return_value={
        "choices": [{"message": {"content": json.dumps({
            "diagnosis": "The page did not change.",
            "revised_subgoal": "Use a different visible control.",
            "avoid": ["Repeating the prior wait"],
        })}}],
        "usage": {"total_tokens": 12},
    })
    monkeypatch.setattr(model, "post_json", post)

    result = model.recovery_guidance({"page": {"url": "https://example.test"}})

    assert result["revised_subgoal"] == "Use a different visible control."
    assert post.call_args.args[0] == "https://provider.test/v1/chat/completions"
    assert post.call_args.args[2]["model"] == "reasoner"


def terminal_decision(choice):
    return {
        "choice": choice,
        "operation": choice,
        "target": None,
        "confidence": 1.0,
        "probabilities": {choice: 1.0},
        "latency_ms": 1,
        "usage": {},
    }


def test_e2e_click_then_done_uses_jev_only(runner, monkeypatch):
    runner.state["decision"] = None
    choices = [decision("e3"), terminal_decision("DONE")]
    monkeypatch.setattr(loop, "choose", Mock(side_effect=choices))

    results = list(runner.run())

    assert results[-1]["status"] == "done"
    assert runner.state["browser"].act.call_count == 1


def test_e2e_caller_text_resume_then_done(runner, monkeypatch):
    runner.text_mode = "caller"
    runner.state["decision"] = None
    monkeypatch.setattr(loop, "choose", Mock(side_effect=[decision(), terminal_decision("DONE")]))

    request = runner.step()
    assert request["status"] == "need_text"
    runner.resume_text(request["resume_token"], "book")
    result = runner.step()

    assert result["status"] == "done"
    assert runner.state["history"][0]["text"] == "book"
    runner.state["browser"].act.assert_called_once()


def test_e2e_internal_text_helper_then_done(runner, monkeypatch):
    runner.state["decision"] = None
    monkeypatch.setattr(loop, "choose", Mock(side_effect=[decision(), terminal_decision("DONE")]))
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 1, "usage": {}}))
    monkeypatch.setattr(loop, "field_text", helper)

    results = list(runner.run())

    assert results[-1]["status"] == "done"
    helper.assert_called_once()
    assert runner.state["history"][0]["text_helper"] == "test"


def test_e2e_first_block_recovers_then_succeeds(runner, monkeypatch):
    runner.recovery_enabled = True
    runner.state["decision"] = None
    monkeypatch.setattr(loop, "choose", Mock(side_effect=[terminal_decision("BLOCKED"), terminal_decision("DONE")]))
    recovery = Mock(return_value={"diagnosis": "Try another route", "revised_subgoal": "Open search", "avoid": []})
    monkeypatch.setattr(loop, "recovery_guidance", recovery)

    results = list(runner.run())

    assert results[-1]["status"] == "done"
    assert recovery.call_count == 1
    runner.state["browser"].act.assert_not_called()


def test_e2e_equivalent_second_block_requires_handoff(runner, monkeypatch):
    runner.recovery_enabled = True
    runner.state["decision"] = None
    monkeypatch.setattr(
        loop,
        "choose",
        Mock(side_effect=[terminal_decision("BLOCKED"), terminal_decision("BLOCKED")]),
    )
    monkeypatch.setattr(
        loop,
        "recovery_guidance",
        Mock(return_value={"diagnosis": "blocked", "revised_subgoal": "Try again", "avoid": []}),
    )

    results = list(runner.run())

    assert results[-1]["status"] == "handoff_required"
    assert results[-1]["handoff"]["target_id"] == "target-test"
    assert results[-1]["handoff"]["browser_connection"] == {
        "kind": "browser-harness-cdp",
        "name": "default",
    }
    runner.state["browser"].act.assert_not_called()


def test_model_call_budget_enters_blocked_recovery_path_without_mutation(runner, monkeypatch):
    runner.state["decision"] = None
    runner.state["decisions"] = [{} for _ in range(loop.MAX_STEPS * 2)]
    choose = Mock(side_effect=AssertionError("budget exhaustion must stop before another Jev call"))
    monkeypatch.setattr(loop, "choose", choose)

    result = runner.step()

    assert result["status"] == "blocked"
    assert result["block_reason"] == "model_call_budget"
    choose.assert_not_called()
    runner.state["browser"].act.assert_not_called()


def test_handoff_uses_actual_browser_harness_connection_name(runner):
    runner.state["browser"].connection_name = "recovery-profile"
    signature = runner._block_signature()

    handoff = runner._require_handoff("repeated_block", signature, 2)["handoff"]

    assert handoff["browser_connection"] == {
        "kind": "browser-harness-cdp",
        "name": "recovery-profile",
    }


def test_recovery_packet_projects_decisions_without_raw_requests(runner):
    runner.state["history"].append({
        "action": "Go",
        "kind": "click",
        "operation": "CLICK",
        "target": "2",
        "page_changed": False,
        "url": "https://example.test/",
        "confidence": 0.9,
        "usage": {"private": "omit"},
        "text": "omit",
    })
    runner.state["decisions"].append({
        "operation": "CLICK",
        "target": "2",
        "confidence": 0.9,
        "target_confidence": 0.8,
        "fingerprint": "fp",
        "elapsed_ms": 12,
        "request": {"state": {"page": {"text": "duplicate payload"}}},
        "raw_answers": {"operation": "omit"},
        "usage": {"omit": True},
    })

    packet = runner._recovery_packet()

    assert set(packet["history"][-1]) == {
        "action", "kind", "operation", "target", "page_changed", "url", "confidence"
    }
    assert set(packet["decisions"][-1]) == {
        "operation", "target", "confidence", "target_confidence", "fingerprint", "elapsed_ms"
    }
    assert "request" not in packet["decisions"][-1]
    assert "raw_answers" not in packet["decisions"][-1]


def test_duplicate_urls_use_target_id_as_handoff_identity(runner):
    signature = runner._block_signature()
    first = runner._require_handoff("repeated_block", signature, 2)["handoff"]

    runner.state["status"] = "blocked"
    runner.state["handoff"] = None
    runner.state["browser"].target_id = "target-other"
    second = runner._require_handoff("repeated_block", signature, 2)["handoff"]

    assert first["url"] == second["url"]
    assert first["target_id"] != second["target_id"]
