"""Circuit breaker for upstream LLM API calls — shared component.

Track consecutive transient failures (ReadTimeout, APITimeoutError,
RateLimitError, etc.) and stop retrying once a threshold is crossed.
This is a *reusable* component: every LLM call site in hermes — the
main conversation loop, the auxiliary client (vision/browser/etc.
through ccr), subagent dispatch, and any future caller — shares the
same breaker state and contributes to / benefits from the same
protection.

Two usage styles are supported:

1. **Direct instance** (for callers that need a private breaker):

       from agent.circuit_breaker import CircuitBreaker, CircuitOpenError
       breaker = CircuitBreaker(threshold=3, cooldown_s=60.0)
       try:
           response = await raw_call(...)
           breaker.record_success()
       except Exception as e:
           if breaker.record_failure(e):
               raise CircuitOpenError(*breaker.snapshot()) from e

2. **Module-level shared instance** (recommended for normal call sites):

       from agent.circuit_breaker import get_breaker, guarded_call
       if get_breaker("main").is_open():
           raise CircuitOpenError(...)
       try:
           response = await raw_call(...)
           get_breaker("main").record_success()
       except Exception as e:
           get_breaker("main").record_failure(e)

The "main" / "auxiliary" / "subagent" scopes each get their own
breaker so, e.g., a stuck vision endpoint doesn't trip the main
chat loop.  Scopes are created on first access.

Pre-existing behavior: hermes had no breaker and would retry the same
upstream for 30+ minutes on persistent failures.  See
docs/incidents/2026-06-08-hermes-retry-loop/ for the postmortem.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Error types that should trip the breaker.
# Other errors (ValidationError, UnicodeEncodeError, etc.) are likely
# deterministic client-side bugs and retrying them amplifies waste.
BREAKER_TRIGGER_ERRORS: frozenset[str] = frozenset({
    "ReadTimeout",
    "APITimeoutError",
    "TimeoutError",
    "RateLimitError",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
})

# Subclass relationships for robust matching — covers cases where the
# provider's SDK wraps the error in a subclass (e.g. openai.APIError →
# openai.APITimeoutError).  We check isinstance first (handles subclass
# transparently), then fall back to the class-name match.
# Importing is wrapped in try/except so this module stays usable in
# lightweight test environments where openai/httpx aren't installed.
def _build_breaker_subclass_map() -> dict:
    mapping: dict = {}
    try:
        import httpx
        mapping[httpx.ReadTimeout] = "ReadTimeout"
        mapping[httpx.ConnectTimeout] = "ConnectTimeout"
        mapping[httpx.PoolTimeout] = "PoolTimeout"
        mapping[httpx.TimeoutException] = "TimeoutError"
    except Exception:
        pass
    try:
        import openai
        mapping[openai.APITimeoutError] = "APITimeoutError"
        mapping[openai.RateLimitError] = "RateLimitError"
        mapping[openai.APIConnectionError] = "APIConnectionError"
        mapping[openai.InternalServerError] = "InternalServerError"
        # openai.APIStatusError covers 5xx, with InternalServerError as a
        # specific 500 — we still want to count *all* 5xx as triggerable.
        for _cls in (openai.APIStatusError,):
            if _cls not in mapping.values():
                mapping[_cls] = "APIStatusError"
    except Exception:
        pass
    return mapping


_BREAKER_SUBCLASS_MAP: dict = _build_breaker_subclass_map()


def _classify_error(error: BaseException) -> Optional[str]:
    """Return the breaker trigger name for ``error``, or None if not a trigger.

    Three-tier classification:
    1. Subclass map (``isinstance`` check) — handles SDK wrappers where
       ``openai.APITimeoutError(openai.APIError)`` is the actual class.
    2. MRO walk + class-name match — handles plain ``TimeoutError``
       subclasses, ``httpx.ReadTimeout`` aliases, and any custom
       exception that extends one of the trigger base classes.
    3. Direct class-name match — handles third-party libraries that
       don't follow Python class hierarchy conventions.
    """
    for cls, name in _BREAKER_SUBCLASS_MAP.items():
        try:
            if isinstance(error, cls):
                return name
        except TypeError:
            continue

    for cls in type(error).__mro__:
        if cls.__name__ in BREAKER_TRIGGER_ERRORS:
            return cls.__name__

    return None

# Module-level knobs (overridable via env for ops debugging).
DEFAULT_THRESHOLD = int(os.environ.get("HERMES_CB_THRESHOLD", "3"))
DEFAULT_COOLDOWN_S = float(os.environ.get("HERMES_CB_COOLDOWN", "60.0"))


class CircuitOpenError(Exception):
    """Raised when the breaker is open and caller should stop retrying."""

    def __init__(self, error_type: str, cooldown_remaining: float, count: int, scope: str = "default"):
        super().__init__(
            f"circuit_breaker[{scope}] open: {count} consecutive {error_type}, "
            f"{cooldown_remaining:.1f}s remaining"
        )
        self.error_type = error_type
        self.cooldown_remaining = cooldown_remaining
        self.count = count
        self.scope = scope


class CircuitBreaker:
    """Per-scope consecutive-failure counter with cooldown.

    Thread-safe — all public methods acquire an internal lock. The
    counters are tiny (a handful of dict entries) so the lock is
    uncontended in normal use; the lock guards against the rare
    multi-threaded case (e.g. subagent dispatch + main loop racing).
    """

    def __init__(self, threshold: int = DEFAULT_THRESHOLD, cooldown_s: float = DEFAULT_COOLDOWN_S):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._consecutive: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}
        self._last_type: Optional[str] = None

    def is_open(self) -> bool:
        with self._lock:
            if self._last_type is None:
                return False
            return time.monotonic() < self._open_until.get(self._last_type, 0.0)

    def cooldown_remaining(self) -> float:
        with self._lock:
            if self._last_type is None:
                return 0.0
            return max(0.0, self._open_until.get(self._last_type, 0.0) - time.monotonic())

    def snapshot(self) -> tuple[str, float, int]:
        """Return (error_type, cooldown_remaining, count) for diagnostics."""
        with self._lock:
            return (
                self._last_type or "unknown",
                self._cooldown_remaining_unlocked(),
                self._consecutive.get(self._last_type or "", 0),
            )

    def _cooldown_remaining_unlocked(self) -> float:
        if self._last_type is None:
            return 0.0
        return max(0.0, self._open_until.get(self._last_type, 0.0) - time.monotonic())

    def record_failure(self, error: BaseException) -> bool:
        """Record a failure. Returns True if the breaker just tripped.

        Errors not classified as a breaker trigger are ignored (return
        False) because retrying them is pointless or actively harmful.
        """
        error_type = _classify_error(error)
        if error_type is None:
            return False

        with self._lock:
            self._last_type = error_type
            self._consecutive[error_type] = self._consecutive.get(error_type, 0) + 1
            count = self._consecutive[error_type]

            if count >= self.threshold:
                self._open_until[error_type] = time.monotonic() + self.cooldown_s
                logger.warning(
                    "circuit_breaker: %d consecutive %s, opening for %.1fs",
                    count, error_type, self.cooldown_s,
                )
                return True
        return False

    def record_success(self) -> None:
        with self._lock:
            if self._consecutive:
                logger.debug(
                    "circuit_breaker: success, reset %d error types",
                    len(self._consecutive),
                )
            self._consecutive.clear()
            self._open_until.clear()
            self._last_type = None

    def reset(self) -> None:
        """Force-close the breaker (e.g. after operator intervention)."""
        with self._lock:
            self._consecutive.clear()
            self._open_until.clear()
            self._last_type = None
            logger.info("circuit_breaker: manually reset")


# ── Module-level scope registry ──────────────────────────────────────
# Every LLM call site in hermes can share state via a named scope.
# This is what makes the breaker a *component* rather than a per-loop
# ad-hoc hack.  Add new scopes as new call sites get wired up.

_DEFAULT_SCOPES = ("main", "auxiliary", "subagent", "vision")
_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(scope: str = "default") -> CircuitBreaker:
    """Return the breaker for a given scope, creating it on first use.

    Standard scopes (initialized eagerly so they show up in diagnostics):
    - "main"      — primary conversation_loop LLM calls
    - "auxiliary" — auxiliary_client LLM calls (vision, browser, etc.)
    - "subagent"  — subagent dispatches
    - "vision"    — vision_analyze / browser_vision specifically
    """
    if scope in _breakers:
        return _breakers[scope]
    with _breakers_lock:
        if scope not in _breakers:
            _breakers[scope] = CircuitBreaker()
            logger.debug("circuit_breaker: created scope=%r", scope)
        return _breakers[scope]


def reset_all() -> None:
    """Reset every scope.  Useful for tests."""
    with _breakers_lock:
        for b in _breakers.values():
            b.reset()


def list_scopes() -> Dict[str, tuple[bool, float, int, str]]:
    """Diagnostic snapshot of every scope.

    Returns: {scope: (is_open, cooldown_remaining, count, last_error_type)}
    """
    out = {}
    with _breakers_lock:
        for name, b in _breakers.items():
            err_type, remaining, count = b.snapshot()
            out[name] = (b.is_open(), remaining, count, err_type)
    return out


# Eagerly create the standard scopes so they appear in `list_scopes()`
# from the start of the process.
for _scope in _DEFAULT_SCOPES:
    get_breaker(_scope)
