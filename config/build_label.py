"""Canonical build-epoch label — single source of truth.

Distinct from mertformer_sdk.__version__ (SDK semver, tracks the
distributable package's own API surface, matches pyproject.toml). This
module tracks the internal closure/build milestone banner shown by CLI
scripts and module __version__ attributes across the repo. The two are
deliberately not unified: SDK semver should track real API-breaking
changes, not internal closure passes. See the reference doc's own
"KASITLI — DOKUNMA" list. layers/qinn.py and scripts/chat.py are
intentional exceptions that use the SDK scheme instead of this one.
"""

BUILD_LABEL = "1.0-BUILD30-V2"
