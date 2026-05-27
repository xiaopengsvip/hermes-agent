"""
anti-stall plugin v1 — Agent loop protection + crash/disconnect logging.

Features:
- Detects "reasoning loops" (N consecutive API calls with no tool calls)
- Progressive intervention: warn → inject prompt → abort
- Logs all API calls with timing, tool counts, token usage
- Detects and logs SSH disconnects (session_finalize without session_end)
- Detects and logs crashes (unexpected session termination)
- Daily JSON-line log files for post-mortem analysis
- CLI command: hermes anti-stall report [date]

Author: Everett (https://github.com/xiaopengsvip)
"""

import json
import os
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# ─── Configuration ──────────────────────────────────────────────────

LOG_DIR = Path(os.environ.get(
    "ANTISTALL_LOG_DIR",
    os.path.expanduser("~/.hermes/anti-stall-logs")
))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Stall detection thresholds
STALL_CONSECUTIVE_NO_TOOLS = int(os.environ.get("ANTISTALL_WARN_AFTER", "3"))
STALL_HARD_STOP = int(os.environ.get("ANTISTALL_STOP_AFTER", "6"))
STALL_TIME_LIMIT_SEC = int(os.environ.get("ANTISTALL_TIME_LIMIT", "300"))  # 5 min

# ─── Per-session state (thread-local) ───────────────────────────────

_state = threading.local()

def _get_state() -> Dict[str, Any]:
    """Get or init per-thread session state."""
    if not hasattr(_state, "data"):
        _state.data = {
            "session_id": "",
            "platform": "",
            "session_start_time": 0.0,
            "api_calls": [],
            "consecutive_no_tools": 0,
            "total_api_calls": 0,
            "total_reasoning_only": 0,
            "last_tool_call_time": 0.0,
            "stall_level": 0,  # 0=ok, 1=warn, 2=inject, 3=stop
            "session_ended_cleanly": False,
        }
    return _state.data


# ─── Logging ────────────────────────────────────────────────────────

_tz_cst = timezone(timedelta(hours=8))

def _log_event(event_type: str, **data):
    """Append a JSON-line event to today's log file."""
    now = datetime.now(_tz_cst)
    log_file = LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"
    entry = {
        "ts": now.isoformat(),
        "event": event_type,
        "session_id": _get_state().get("session_id", ""),
        "platform": _get_state().get("platform", ""),
        **data,
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never crash the agent from logging


# ─── Hook callbacks ─────────────────────────────────────────────────

def on_session_start(session_id: str = "", platform: str = "", **kwargs):
    """Initialize tracking state for a new session."""
    state = _get_state()
    state["session_id"] = session_id
    state["platform"] = platform
    state["session_start_time"] = time.time()
    state["api_calls"] = []
    state["consecutive_no_tools"] = 0
    state["total_api_calls"] = 0
    state["total_reasoning_only"] = 0
    state["last_tool_call_time"] = time.time()
    state["stall_level"] = 0
    state["session_ended_cleanly"] = False

    _log_event("session_start",
               session_id=session_id,
               platform=platform,
               model=kwargs.get("model", ""))


def on_pre_api_request(
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    api_call_count: int = 0,
    message_count: int = 0,
    tool_count: int = 0,
    **kwargs,
):
    """Track API call start."""
    state = _get_state()
    state["session_id"] = session_id or state["session_id"]
    state["platform"] = platform or state["platform"]

    call_record = {
        "start_time": time.time(),
        "api_call_count": api_call_count,
        "message_count": message_count,
        "tool_count": tool_count,
        "model": model,
        "provider": provider,
    }
    state["api_calls"].append(call_record)
    state["total_api_calls"] += 1

    # Check time limit
    elapsed = time.time() - state["session_start_time"]
    if elapsed > STALL_TIME_LIMIT_SEC and state["stall_level"] < 2:
        state["stall_level"] = 2
        _log_event("stall_time_limit",
                   elapsed_sec=round(elapsed, 1),
                   total_calls=state["total_api_calls"],
                   reasoning_only=state["total_reasoning_only"])


def on_post_api_request(
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    api_call_count: int = 0,
    api_duration: float = 0.0,
    finish_reason: str = "",
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    usage: Optional[Dict] = None,
    **kwargs,
):
    """Detect reasoning-only responses (no tool calls)."""
    state = _get_state()
    now = time.time()

    # Update last call record
    if state["api_calls"]:
        state["api_calls"][-1].update({
            "end_time": now,
            "duration": api_duration,
            "finish_reason": finish_reason,
            "content_chars": assistant_content_chars,
            "tool_calls": assistant_tool_call_count,
            "usage": usage,
        })

    # Detect stall
    if assistant_tool_call_count == 0 and assistant_content_chars > 0:
        state["consecutive_no_tools"] += 1
        state["total_reasoning_only"] += 1
    else:
        state["consecutive_no_tools"] = 0
        state["last_tool_call_time"] = now

    # Update stall level
    c = state["consecutive_no_tools"]
    if c >= STALL_HARD_STOP:
        new_level = 3
    elif c >= STALL_CONSECUTIVE_NO_TOOLS:
        new_level = 2
    elif c >= 2:
        new_level = 1
    else:
        new_level = 0

    if new_level > state["stall_level"]:
        state["stall_level"] = new_level
        _log_event("stall_level_change",
                   level=new_level,
                   consecutive_no_tools=c,
                   total_calls=state["total_api_calls"],
                   elapsed_sec=round(now - state["session_start_time"], 1))

    # Log every API call (compact)
    _log_event("api_call",
               call_num=api_call_count,
               duration_ms=round(api_duration * 1000),
               content_chars=assistant_content_chars,
               tool_calls=assistant_tool_call_count,
               finish_reason=finish_reason,
               consecutive_no_tools=state["consecutive_no_tools"],
               stall_level=state["stall_level"],
               prompt_tokens=usage.get("prompt_tokens", 0) if usage else 0,
               completion_tokens=usage.get("completion_tokens", 0) if usage else 0)

    return None  # don't modify anything


def on_session_end(session_id: str = "", **kwargs):
    """Session ended cleanly."""
    state = _get_state()
    state["session_ended_cleanly"] = True
    elapsed = time.time() - state["session_start_time"]

    _log_event("session_end",
               session_id=session_id,
               elapsed_sec=round(elapsed, 1),
               total_api_calls=state["total_api_calls"],
               total_reasoning_only=state["total_reasoning_only"],
               end_reason="clean")


def on_session_finalize(session_id: str = "", platform: str = "", **kwargs):
    """Session finalized — detect if it was a clean end or crash/disconnect."""
    state = _get_state()
    elapsed = time.time() - state["session_start_time"]

    if state["session_ended_cleanly"]:
        _log_event("session_finalize",
                   session_id=session_id,
                   elapsed_sec=round(elapsed, 1),
                   end_reason="clean")
    else:
        # Session finalized without clean end = crash or SSH disconnect
        _log_event("session_crash",
                   session_id=session_id,
                   platform=platform,
                   elapsed_sec=round(elapsed, 1),
                   total_api_calls=state["total_api_calls"],
                   total_reasoning_only=state["total_reasoning_only"],
                   last_stall_level=state["stall_level"],
                   end_reason="unclean_finalize",
                   note="Session finalized without clean end — likely crash or SSH disconnect")

    # Reset state
    _state.data = {
        "session_id": "",
        "platform": "",
        "session_start_time": 0.0,
        "api_calls": [],
        "consecutive_no_tools": 0,
        "total_api_calls": 0,
        "total_reasoning_only": 0,
        "last_tool_call_time": 0.0,
        "stall_level": 0,
        "session_ended_cleanly": False,
    }


# ─── Plugin registration ───────────────────────────────────────────

def register(ctx):
    """Register hooks and tools."""
    # Hooks
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)

    # Tool
    try:
        from .tools import TOOLS, anti_stall_report_handler
        for tool_def in TOOLS:
            ctx.register_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                handler=anti_stall_report_handler,
            )
    except Exception:
        pass  # hooks still work even if tool registration fails
