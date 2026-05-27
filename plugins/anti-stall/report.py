#!/usr/bin/env python3
"""
anti-stall log analyzer — analyze crash/stall/disconnect logs.

Usage:
    python3 report.py                    # today's report
    python3 report.py 2026-05-28         # specific date
    python3 report.py --all              # all logs summary
    python3 report.py --crashes          # only crash/disconnect events
    python3 report.py --session <id>     # specific session timeline
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

LOG_DIR = Path(os.path.expanduser("~/.hermes/anti-stall-logs"))
TZ_CST = timezone(timedelta(hours=8))


def load_log(date_str: str) -> list:
    """Load all events for a given date."""
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


def load_all_logs() -> list:
    """Load all log files."""
    events = []
    if not LOG_DIR.exists():
        return events
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        events.extend(load_log(f.stem))
    return events


def report_for_date(date_str: str):
    """Generate a report for a specific date."""
    events = load_log(date_str)
    if not events:
        print(f"  No events found for {date_str}")
        return

    sessions = {}
    crashes = []
    stalls = []
    api_calls = []

    for e in events:
        etype = e.get("event", "")
        sid = e.get("session_id", "")

        if etype == "session_start":
            sessions[sid] = {
                "start": e["ts"],
                "platform": e.get("platform", ""),
                "model": e.get("model", ""),
                "api_calls": 0,
                "reasoning_only": 0,
                "end": None,
                "end_reason": None,
                "max_stall_level": 0,
            }
        elif etype == "api_call":
            api_calls.append(e)
            if sid in sessions:
                sessions[sid]["api_calls"] += 1
                if e.get("tool_calls", 0) == 0:
                    sessions[sid]["reasoning_only"] += 1
                sessions[sid]["max_stall_level"] = max(
                    sessions[sid]["max_stall_level"],
                    e.get("stall_level", 0)
                )
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

    # Print report
    print(f"\n{'='*60}")
    print(f"  Anti-Stall Report: {date_str}")
    print(f"{'='*60}")
    print(f"\n  Sessions: {len(sessions)}")
    print(f"  API Calls: {len(api_calls)}")
    print(f"  Stall Events: {len(stalls)}")
    print(f"  Crashes/Disconnects: {len(crashes)}")

    if sessions:
        print(f"\n{'─'*60}")
        print("  Session Details:")
        print(f"{'─'*60}")
        for sid, info in sessions.items():
            status = "✓" if info["end_reason"] in ("clean", None) else "✗"
            print(f"\n  {status} {sid[:40]}")
            print(f"    Platform: {info['platform']}")
            print(f"    Model: {info['model']}")
            print(f"    API Calls: {info['api_calls']} (reasoning-only: {info['reasoning_only']})")
            print(f"    Max Stall Level: {info['max_stall_level']}")
            if info.get("elapsed"):
                print(f"    Duration: {info['elapsed']}s")
            print(f"    End: {info['end_reason'] or 'in progress'}")

    if crashes:
        print(f"\n{'─'*60}")
        print("  ⚠ CRASHES / DISCONNECTS:")
        print(f"{'─'*60}")
        for c in crashes:
            print(f"\n  [{c['ts']}]")
            print(f"    Session: {c.get('session_id', '?')[:40]}")
            print(f"    Platform: {c.get('platform', '?')}")
            print(f"    Duration: {c.get('elapsed_sec', '?')}s")
            print(f"    API Calls: {c.get('total_api_calls', '?')}")
            print(f"    Reasoning Only: {c.get('total_reasoning_only', '?')}")
            print(f"    Last Stall Level: {c.get('last_stall_level', '?')}")
            print(f"    Note: {c.get('note', '')}")

    if stalls:
        print(f"\n{'─'*60}")
        print("  STALL EVENTS:")
        print(f"{'─'*60}")
        for s in stalls:
            print(f"  [{s['ts']}] Level {s.get('level', '?')} — "
                  f"{s.get('consecutive_no_tools', '?')} consecutive reasoning-only "
                  f"(total calls: {s.get('total_calls', '?')})")

    # API call duration stats
    if api_calls:
        durations = [e.get("duration_ms", 0) for e in api_calls if e.get("duration_ms")]
        if durations:
            avg_dur = sum(durations) / len(durations)
            max_dur = max(durations)
            print(f"\n{'─'*60}")
            print("  API Call Stats:")
            print(f"{'─'*60}")
            print(f"    Avg Duration: {avg_dur:.0f}ms")
            print(f"    Max Duration: {max_dur}ms")
            print(f"    Total Calls: {len(api_calls)}")

    print()


def report_crashes():
    """Show only crash/disconnect events across all logs."""
    events = load_all_logs()
    crashes = [e for e in events if e.get("event") == "session_crash"]

    if not crashes:
        print("  No crashes or disconnects found.")
        return

    print(f"\n{'='*60}")
    print(f"  All Crashes / Disconnects ({len(crashes)} total)")
    print(f"{'='*60}")

    for c in crashes:
        print(f"\n  [{c['ts']}]")
        print(f"    Session: {c.get('session_id', '?')[:40]}")
        print(f"    Platform: {c.get('platform', '?')}")
        print(f"    Duration: {c.get('elapsed_sec', '?')}s")
        print(f"    API Calls: {c.get('total_api_calls', '?')}")
        print(f"    Reasoning Only: {c.get('total_reasoning_only', '?')}")
        print(f"    Last Stall Level: {c.get('last_stall_level', '?')}")

    print()


def report_session(session_id: str):
    """Show timeline for a specific session."""
    events = load_all_logs()
    session_events = [e for e in events if e.get("session_id") == session_id]

    if not session_events:
        # Try partial match
        session_events = [e for e in events if session_id in e.get("session_id", "")]

    if not session_events:
        print(f"  No events found for session: {session_id}")
        return

    print(f"\n{'='*60}")
    print(f"  Session Timeline: {session_id}")
    print(f"{'='*60}")

    for e in session_events:
        etype = e.get("event", "")
        ts = e.get("ts", "")

        if etype == "session_start":
            print(f"\n  [{ts}] ▶ SESSION START")
            print(f"    Platform: {e.get('platform', '')}")
            print(f"    Model: {e.get('model', '')}")
        elif etype == "api_call":
            tc = e.get("tool_calls", 0)
            dur = e.get("duration_ms", 0)
            stall = e.get("stall_level", 0)
            marker = "⚠" if stall > 0 else " "
            print(f"  [{ts}] {marker} API #{e.get('call_num', '?')} — "
                  f"{dur}ms, {e.get('content_chars', 0)} chars, "
                  f"{tc} tool calls, stall={stall}")
        elif etype == "stall_level_change":
            print(f"  [{ts}] ⚡ STALL LEVEL → {e.get('level', '?')} "
                  f"({e.get('consecutive_no_tools', '?')} consecutive reasoning-only)")
        elif etype in ("session_end", "session_finalize"):
            print(f"  [{ts}] ■ {etype.upper()}")
            print(f"    Reason: {e.get('end_reason', '?')}")
            print(f"    Duration: {e.get('elapsed_sec', '?')}s")
        elif etype == "session_crash":
            print(f"  [{ts}] ✗ CRASH/DISCONNECT")
            print(f"    Duration: {e.get('elapsed_sec', '?')}s")
            print(f"    API Calls: {e.get('total_api_calls', '?')}")
            print(f"    Note: {e.get('note', '')}")

    print()


def main():
    args = sys.argv[1:]

    if not args:
        today = datetime.now(TZ_CST).strftime("%Y-%m-%d")
        report_for_date(today)
    elif args[0] == "--all":
        if not LOG_DIR.exists():
            print("  No log files found.")
            return
        for f in sorted(LOG_DIR.glob("*.jsonl")):
            report_for_date(f.stem)
    elif args[0] == "--crashes":
        report_crashes()
    elif args[0] == "--session" and len(args) > 1:
        report_session(args[1])
    else:
        report_for_date(args[0])


if __name__ == "__main__":
    main()
