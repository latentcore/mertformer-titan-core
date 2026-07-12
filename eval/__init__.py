"""MertFormer Titan evaluation harnesses (checkpoint-bound benchmark/probe scripts).

[2026-07-12] Was a namespace package (no __init__.py). Per PEP 420, a
namespace package loses to any REGULAR module of the same name found later
in sys.path -- and scripts/eval.py (an unrelated benchmark-suite CLI) sits
right next to it, so any code run as `python3 scripts/<something>.py` (which
puts scripts/ on sys.path ahead of nothing, but the namespace resolution
still prefers the later regular-module match per PEP 420) could get `eval`
silently rebound to scripts/eval.py instead of this package, breaking
`from eval.<submodule> import ...` with a confusing "'eval' is not a
package" error. Making this a real (non-namespace) package removes the
ambiguity entirely.
"""
