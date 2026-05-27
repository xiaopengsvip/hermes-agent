"""
Anti-stall log analysis tool for Hermes Agent.
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

LOG_DIR = Path(os.path.expanduser("~/.hermes/anti-stall-logs"))
TZ_CST = timezone(timedelta(hours=8))

TOOLS = [
    {
        "name": "anti_stall_report",
        "description": "View anti-stall logs: crash/disconnect events, stall patterns, session timelines. Use for debugging agent loop issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Report mode: 'today' (default), 'crashes' (all crash/disconnect events), 'date' (specific date), 'session' (specific session)",
                    "enum": ["today", "crashes", "date", "session"],
                    "default": "today",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (for mode='date')",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (for mode='session')",
                },
            },
        },
    },
]


def _load_log(date_str: str) -> list:
    log_file = LOG_DIR / f"{date_str}.jsonl"
    if not log_file.exists():
        return []
    events = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _load_all_logs() -> list:
    events = []
    if not LOG_DIR.exists():
        return events
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        events.extend(_load_log(f.stem))
    return events


def _format_report(date_str: str) -> str:
    events = _load_log(date_str)
    if not events:
        return f"No events found for {date_str}"

    sessions = {}
    crashes = []
    stalls = []
    api_calls = []

    for e in events:
        etype = e.get("event", "")
        sid = e.get("session_id", "")

        if etype == "session_start":
            sessions[sid] = {
                "start": e["ts"], "platform": e.get("platform", ""),
                "model": e.get("model", ""), "api_calls": 0,
                "reasoning_only": 0, "end": None, "end_reason": None,
                "max_stall_level": 0, "elapsed": 0,
            }
        elif etype == "api_call":
            api_calls.append(e)
            if sid in sessions:
                sessions[sid]["api_calls"] += 1
                if e.get("tool_calls", 0) == 0:
                    sessions[sid]["reasoning_only"] += 1
                sessions[sid]["max_stall_level"] = max(
                    sessions[sid]["max_stall_level"], e.get("stall_level", 0))
        elif etype in ("session_end", "session_finalize"):
            if sid in sessions:
                sessions[sid]["end"] = e["ts"]
                sessions[sid]["end_reason"] = e.get("end_reason", "")
                sessions[sid]["elapsed"] = e.get("elapsed_sec", 0)
        elif etype == "session_crash":
            crashes.append(e)
            if sid in sessions:
                sessions[sid]["end"] = e["ts"]
                sessions[sid]["end_reason"] = "CRASH/DISCONNECT"
                sessions[sid]["elapsed"] = e.get("elapsed_sec", 0)
        elif etype == "stall_level_change":
            stalls.append(e)

    lines = [f"Anti-Stall Report: {date_str}"]
    lines.append(f"Sessions: {len(sessions)} | API Calls: {len(api_calls)} | "
                 f"Stalls: {len(stalls)} | Crashes: {len(crashes)}")

    for sid, info in sessions.items():
        status = "OK" if info["end_reason"] in ("clean", None) else "CRASH"
        lines.append(f"  [{status}] {sid[:36]} — {info['api_calls']} calls "
                     f"({info['reasoning_only']} reasoning-only), "
                     f"stall={info['max_stall_level']}, "
                     f"{info['elapsed']}s, {info['end_reason'] or 'active'}")

    if crashes:
        lines.append(f"\nCRASHES ({len(crashes)}):")
        for c in crashes:
            lines.append(f"  [{c['ts']}] {c.get('session_id', '?')[:30]} — "
                         f"{c.get('elapsed_sec', '?')}s, "
                         f"{c.get('total_api_calls', '?')} calls, "
                         f"stall={c.get('last_stall_level', '?')}")

    return "\n".join(lines)


def _format_crashes() -> str:
    events = _load_all_logs()
    crashes = [e for e in events if e.get("event") == "session_crash"]
    if not crashes:
        return "No crashes or disconnects found."

    lines = [f"All Crashes/Disconnects ({len(crashes)} total):"]
    for c in crashes:
        lines.append(f"  [{c['ts']}] {c.get('session_id', '?')[:30]} — "
                     f"{c.get('platform', '?')}, {c.get('elapsed_sec', '?')}s, "
                     f"{c.get('total_api_calls', '?')} calls, "
                     f"stall={c.get('last_stall_level', '?')}")
    return "\n".join(lines)


def _format_session(session_id: str) -> str:
    events = _load_all_logs()
    session_events = [e for e in events if session_id in e.get("session_id", "")]
    if not session_events:
        return f"No events found for session: {session_id}"

    lines = [f"Session Timeline: {session_id}"]
    for e in session_events:
        etype = e.get("event", "")
        ts = e.get("ts", "")
        if etype == "session_start":
            lines.append(f"  [{ts}] START — {e.get('platform', '')} {e.get('model', '')}")
        elif etype == "api_call":
            stall = e.get("stall_level", 0)
            marker = "!!" if stall > 0 else "  "
            lines.append(f"  [{ts}] {marker}API#{e.get('call_num', '?')} — "
                         f"{e.get('duration_ms', 0)}ms, {e.get('tool_calls', 0)} tools, "
                         f"stall={stall}")
        elif etype == "stall_level_change":
            lines.append(f"  [{ts}] STALL→{e.get('level', '?')} "
                         f"({e.get('consecutive_no_tools', '?')} consecutive)")
        elif etype in ("session_end", "session_finalize"):
            lines.append(f"  [{ts}] END — {e.get('end_reason', '?')}, {e.get('elapsed_sec', '?')}s")
        elif etype == "session_crash":
            lines.append(f"  [{ts}] CRASH — {e.get('elapsed_sec', '?')}s, "
                         f"{e.get('total_api_calls', '?')} calls")
    return "\n".join(lines)


def anti_stall_report_handler(params: dict) -> str:
    mode = params.get("mode", "today")
    try:
        if mode == "today":
            today = datetime.now(TZ_CST).strftime("%Y-%m-%d")
            result = _format_report(today)
        elif mode == "crashes":
            result = _format_crashes()
        elif mode == "date":
            date_str = params.get("date", datetime.now(TZ_CST).strftime("%Y-%m-%d"))
            result = _format_report(date_str)
        elif mode == "session":
            sid = params.get("session_id", "")
            if not sid:
                result = "Error: session_id required for mode='session'"
            else:
                result = _format_session(sid)
        else:
            result = f"Unknown mode: {mode}"
    except Exception as e:
        result = f"Error: {e}"

    return json.dumps({"report": result})
