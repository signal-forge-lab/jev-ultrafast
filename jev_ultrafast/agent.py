"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import hashlib
import json
import secrets
import time
from pathlib import Path

from .browser import Browser, StalePage
from .model import action_space, choose, field_context, field_text, recovery_guidance
from .questions import MAX_STEPS, MAX_TOTAL_RECOVERIES


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False, text_mode="internal", recovery=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        if text_mode not in {"caller", "internal"}:
            raise ValueError("text_mode must be 'caller' or 'internal'")
        plan = [task]
        self.text_mode = text_mode
        self.recovery_enabled = recovery
        self.pending_text = None
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
            base_goal=task,
            recoveries=[],
            block_counts={},
            recovery_attempts=0,
            handoff=None,
            block_reason=None,
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        result = {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }
        if self.state["status"] == "need_text" and isinstance(self.pending_text, dict):
            context = self.pending_text["context"]
            result.update(
                field={
                    "label": context["field"]["label"],
                    "role": context["field"].get("role"),
                    "current_value": context["field"].get("value", ""),
                },
                context={
                    "goal": context["goal"],
                    "page_title": context["page"]["title"],
                    "visible_text": context["page"]["text"],
                    "recent_actions": context["recent_actions"],
                },
                resume_token=self.pending_text["token"],
            )
        return result

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                if state["status"] == "blocked":
                    return self.snapshot()
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked", "need_text", "handoff_required", "recovery_unavailable"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                state["status"] = "blocked"
                state["block_reason"] = "model_call_budget"
                return self.snapshot()
            state["decision"] = choose(state["page"], state["goal"], state["history"])
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["block_reason"] = "model_blocked" if selected == "BLOCKED" else None
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                state["block_reason"] = "action_budget"
                return self.snapshot()
            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.text_mode == "caller":
                    self.pending_text = {
                        "token": secrets.token_urlsafe(24),
                        "context": context,
                        "action": action,
                        "decision": decision,
                        "fingerprint": page["fingerprint"],
                    }
                    state["status"] = "need_text"
                    return self.snapshot()
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            return self._execute(action, page, decision, text, helper)
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def _execute(self, action, page, decision, text=None, helper=None):
        state = self.state
        # Browser.act checks freshness immediately before input, including after text generation.
        state["browser"].act(action, page, text=text)
        self.pending_text = None
        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        # Record execution before observing. A stale post-action observation must not erase the action.
        state["history"].append(
            {
                "step": len(state["history"]) + 1,
                "action": action["label"],
                "kind": action["kind"],
                "choice": decision["choice"],
                "probability": decision["probabilities"][decision["choice"]],
                "confidence": decision["confidence"],
                "latency_ms": decision["latency_ms"],
                "text": text,
                "text_helper": helper["model"] if helper else None,
                "text_latency_ms": helper["latency_ms"] if helper else 0,
                "operation": decision["operation"],
                "target": decision["target"],
                "page_changed": None,
                "url": page["url"],
                "usage": decision["usage"],
                "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                "elapsed_ms": state["elapsed_ms"],
            }
        )
        state["page"] = state["browser"].observe(screenshot=self.screenshots)
        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        state["history"][-1].update(
            page_changed=state["page"]["fingerprint"] != page["fingerprint"],
            url=state["page"]["url"],
            elapsed_ms=state["elapsed_ms"],
        )
        if state["record"]:
            (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                base64.b64decode(state["page"]["screenshot"])
            )
        repeated = state["history"][-3:]
        blocked = len(repeated) == 3 and all(
            h["page_changed"] is False and h["kind"] != "wait" for h in repeated
        )
        state["status"] = "blocked" if blocked else "ready"
        state["block_reason"] = "no_progress" if blocked else None
        return self.snapshot()

    def resume_text(self, token, text):
        state, pending = self.state, self.pending_text
        if state["status"] != "need_text" or not isinstance(pending, dict) or not secrets.compare_digest(
            str(token), pending["token"]
        ):
            raise ValueError("Invalid or consumed resume token; nothing typed.")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("Caller text must be a non-empty string of at most 2000 characters; nothing typed.")
        action, page = pending["action"], state["page"]
        valid_action = (
            page["fingerprint"] == pending["fingerprint"]
            and action["kind"] == "fill"
            and any(candidate == action for candidate in page["actions"])
        )
        try:
            fresh = valid_action and state["browser"].fresh(page, action)
        except StalePage:
            fresh = False
        if not fresh:
            self.pending_text = None
            state["status"] = "ready"
            raise StalePage("Page or target changed before text resume; nothing typed.")
        decision = pending["decision"]
        self.pending_text = None  # Consume before the only allowed mutation attempt.
        state["status"] = "ready"
        try:
            return self._execute(action, page, decision, text)
        except StalePage:
            state["status"] = "ready"
            raise StalePage("Page or target changed before text resume; nothing typed.") from None

    def _block_signature(self):
        state = self.state
        recent = [
            {k: action.get(k) for k in ("action", "kind", "page_changed")}
            for action in state["history"]
            if action.get("kind") != "wait"
        ][-3:]
        value = {
            "fingerprint": state["page"]["fingerprint"],
            "url": state["page"]["url"],
            "title": state["page"]["title"],
            "reason": state.get("block_reason") or "blocked",
            "recent_non_wait_actions": recent,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _recovery_packet(self):
        state = self.state
        page = state["page"]
        return {
            "goal": state["base_goal"],
            "current_page": {
                "url": page["url"],
                "title": page["title"],
                "visible_text": page["text"][:6000],
                "scroll": page["scroll"],
                "actions": page["actions"],
                "fingerprint": page["fingerprint"],
                "guards": page.get("guards", {}),
            },
            "history": [
                {
                    k: item.get(k)
                    for k in ("action", "kind", "operation", "target", "page_changed", "url", "confidence")
                }
                for item in state["history"][-6:]
            ],
            "decisions": [
                {
                    k: item.get(k)
                    for k in ("operation", "target", "confidence", "target_confidence", "fingerprint", "elapsed_ms")
                }
                for item in state["decisions"][-3:]
            ],
            "block_reason": state.get("block_reason") or "blocked",
        }

    def _require_handoff(self, reason, signature, block_count):
        state = self.state
        page = state["page"]
        state["handoff"] = {
            "required": True,
            "reason": reason,
            "browser_backend": None,
            "browser_connection": {"kind": "browser-harness-cdp", "name": state["browser"].connection_name},
            "target_id": state["browser"].target_id,
            "url": page["url"],
            "title": page["title"],
            "fingerprint": page["fingerprint"],
            "status": "blocked",
            "block_signature": signature,
            "block_count": block_count,
            "recovery_attempts": state["recovery_attempts"],
        }
        state["status"] = "handoff_required"
        return self.snapshot()

    def recover(self):
        state = self.state
        if state["status"] != "blocked":
            raise ValueError("Recovery is only available for a blocked run")
        signature = self._block_signature()
        count = state["block_counts"].get(signature, 0) + 1
        state["block_counts"][signature] = count
        if count >= 2:
            return self._require_handoff("repeated_block", signature, count)
        if state["recovery_attempts"] >= MAX_TOTAL_RECOVERIES:
            return self._require_handoff("recovery_budget_exhausted", signature, count)
        try:
            guidance = recovery_guidance(self._recovery_packet())
        except (RuntimeError, ValueError) as error:
            state["status"] = "recovery_unavailable"
            state["recovery_error"] = str(error)
            return self.snapshot()
        state["recovery_attempts"] += 1
        state["recoveries"].append({"block_signature": signature, **guidance})
        avoid = "; ".join(guidance["avoid"])
        recovery_goal = f"Current subgoal: {guidance['revised_subgoal']}"
        if avoid:
            recovery_goal += f"\nAvoid: {avoid}"
        state["plan"] = [state["base_goal"], recovery_goal]
        state["goal"] = "\n".join(state["plan"])
        state["status"] = "ready"
        return self.snapshot()

    def step(self):
        result = self.command("tick")
        if self.recovery_enabled and self.state["status"] == "blocked":
            return self.recover()
        return result

    def run(self):
        terminal = {"done", "blocked", "need_text", "handoff_required", "recovery_unavailable"}
        while self.state["status"] not in terminal:
            yield self.step()

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
