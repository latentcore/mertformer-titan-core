# Tool Sandbox

Defines safe execution boundaries for tool use.

## Allowed
- Read-only access to project workspace
- Deterministic tools (no external side effects)
- Bounded CPU/GPU usage

## Denied
- System-level destructive actions
- Reading secrets outside approved env vars
- Network calls when disabled

## Defaults
- Tool calls must be explicit and logged
- Timeouts enforced per tool
- Errors are surfaced with clear reason codes
