# anti-stall

Hermes Agent plugin for **agent loop protection** and **crash/disconnect logging**.

## Problem

When the LLM gets stuck in a "reasoning loop" — repeatedly saying "let me write this file now" without actually calling any tools — the session wastes tokens, hangs indefinitely, and may eventually crash or disconnect via SSH timeout.

## Features

- **Stall Detection** — Monitors consecutive API calls with no tool calls
- **Progressive Intervention** — Warns at level 1, injects prompts at level 2, logs critical at level 3
- **Crash Logging** — Detects sessions that end without clean `session_end` (crash/SSH disconnect)
- **Structured JSON Logs** — Daily `.jsonl` files for post-mortem analysis
- **CLI Report** — `python3 ~/.hermes/plugins/anti-stall/report.py` for analysis
- **Agent Tool** — `anti_stall_report` tool for in-session log queries

## Configuration

Enable in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - anti-stall
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTISTALL_LOG_DIR` | `~/.hermes/anti-stall-logs` | Log directory |
| `ANTISTALL_WARN_AFTER` | `3` | Consecutive reasoning-only calls before warning |
| `ANTISTALL_STOP_AFTER` | `6` | Consecutive calls before hard stop |
| `ANTISTALL_TIME_LIMIT` | `300` | Wall-clock seconds before escalating |

## Usage

### Log Files

Logs are written to `~/.hermes/anti-stall-logs/YYYY-MM-DD.jsonl` (one JSON object per line).

Events logged:
- `session_start` — New session begins
- `api_call` — Every LLM API call with timing, tool counts, stall level
- `stall_level_change` — Stall severity increases
- `session_end` — Clean session end
- `session_finalize` — Session finalization (clean or not)
- `session_crash` — Session ended without clean end (crash/SSH disconnect)

### CLI Report

```bash
# Today's report
python3 ~/.hermes/plugins/anti-stall/report.py

# Specific date
python3 ~/.hermes/plugins/anti-stall/report.py 2026-05-28

# All crashes/disconnects
python3 ~/.hermes/plugins/anti-stall/report.py --crashes

# Specific session timeline
python3 ~/.hermes/plugins/anti-stall/report.py --session 20260528_011927_4cbb7f

# All logs
python3 ~/.hermes/plugins/anti-stall/report.py --all
```

### Agent Tool

Ask the agent:
- "Show me the anti-stall report"
- "Any crashes today?"
- "Show me the timeline for session XXX"

## How It Works

1. `on_session_start` initializes tracking state
2. `pre_api_request` records API call start time
3. `post_api_request` checks if the response had tool calls
   - If `assistant_tool_call_count == 0` for N consecutive calls → stall detected
4. `on_session_end` marks clean session end
5. `on_session_finalize` detects unclean endings (crash/SSH disconnect)

## Author

**Everett** — [GitHub](https://github.com/xiaopengsvip)
