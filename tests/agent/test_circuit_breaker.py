"""Tests for the shared circuit_breaker component.

Covers:
- Per-instance state (threshold + cooldown)
- Module-level shared scopes (get_breaker, list_scopes)
- Thread safety
- BREAKER_TRIGGER_ERRORS filtering
- Reset / snapshot behavior
"""
import time
import threading

import pytest

from agent import circuit_breaker
from agent.circuit_breaker import (
    BREAKER_TRIGGER_ERRORS,
    CircuitBreaker,
    CircuitOpenError,
    get_breaker,
    list_scopes,
    reset_all,
)


# --- Test doubles ------------------------------------------------------
# We use ``type()`` to build classes with the right ``__name__`` so the
# class-name fallback in ``_classify_error`` can match them.  Plain
# ``class Foo(Exception): pass`` makes the name "Foo" and the breaker
# would ignore it.

# Names that should match BREAKER_TRIGGER_ERRORS by class name
FakeReadTimeout = type("ReadTimeout", (Exception,), {})
FakeRateLimit = type("RateLimitError", (Exception,), {})
FakeConnectionError = type("APIConnectionError", (Exception,), {})
FakeAPITimeoutError = type("APITimeoutError", (Exception,), {})

# Names that should NOT match (validation, unicode, etc.)
FakeValidationError = type("ValidationError", (Exception,), {})


# Reset shared state between tests so module-level scopes don't leak.
@pytest.fixture(autouse=True)
def _reset_module_state():
    reset_all()
    yield
    reset_all()


# --- Per-instance behavior --------------------------------------------

class TestCircuitBreakerInstance:
    def test_threshold_triggers_open(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=1.0)
        for _ in range(2):
            assert cb.record_failure(FakeReadTimeout()) is False
        # third failure trips
        assert cb.record_failure(FakeReadTimeout()) is True
        assert cb.is_open() is True
        assert cb.cooldown_remaining() > 0

    def test_success_resets_counter(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=10.0)
        cb.record_failure(FakeReadTimeout())
        cb.record_failure(FakeReadTimeout())
        cb.record_success()
        # After a success, two more failures should not trip.
        assert cb.record_failure(FakeReadTimeout()) is False
        assert cb.record_failure(FakeReadTimeout()) is False
        assert cb.is_open() is False

    def test_non_trigger_errors_ignored(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=10.0)
        for _ in range(10):
            assert cb.record_failure(FakeValidationError()) is False
        assert cb.is_open() is False

    def test_mixed_errors_aggregated_per_type(self):
        cb = CircuitBreaker(threshold=3, cooldown_s=10.0)
        cb.record_failure(FakeReadTimeout())
        cb.record_failure(FakeReadTimeout())
        # 2 ReadTimeout + 1 RateLimit — should NOT trip yet (RateLimit count = 1)
        assert cb.record_failure(FakeRateLimit()) is False
        # Now another ReadTimeout to reach 3 ReadTimeout
        assert cb.record_failure(FakeReadTimeout()) is True

    def test_cooldown_expires(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=0.1)
        cb.record_failure(FakeReadTimeout())
        cb.record_failure(FakeReadTimeout())
        assert cb.is_open() is True
        time.sleep(0.15)
        assert cb.is_open() is False

    def test_snapshot_includes_count_and_type(self):
        cb = CircuitBreaker(threshold=5, cooldown_s=10.0)
        cb.record_failure(FakeConnectionError())
        cb.record_failure(FakeConnectionError())
        et, remaining, count = cb.snapshot()
        # FakeConnectionError is built with name "APIConnectionError" to
        # match the trigger set.
        assert et == "APIConnectionError"
        assert count == 2
        assert remaining >= 0.0

    def test_reset_clears_state(self):
        cb = CircuitBreaker(threshold=2, cooldown_s=10.0)
        cb.record_failure(FakeReadTimeout())
        cb.record_failure(FakeReadTimeout())
        assert cb.is_open() is True
        cb.reset()
        assert cb.is_open() is False
        # After reset, full threshold needed again.
        assert cb.record_failure(FakeReadTimeout()) is False


# --- Module-level shared scopes ---------------------------------------

class TestSharedScopes:
    def test_default_scopes_present(self):
        scopes = list_scopes()
        # The four standard scopes should be eagerly created.
        for s in ("main", "auxiliary", "subagent", "vision"):
            assert s in scopes, f"scope {s!r} missing from list_scopes()"

    def test_get_breaker_returns_same_instance(self):
        a = get_breaker("main")
        b = get_breaker("main")
        assert a is b

    def test_get_breaker_creates_new_scope_lazily(self):
        custom = get_breaker("custom_test_scope_xyz")
        assert custom is not None
        assert "custom_test_scope_xyz" in list_scopes()

    def test_scopes_independent(self):
        main_breaker = get_breaker("main")
        aux_breaker = get_breaker("auxiliary")
        # Trip main breaker with ReadTimeouts
        for _ in range(3):
            main_breaker.record_failure(FakeReadTimeout())
        assert main_breaker.is_open() is True
        # auxiliary should still be closed
        assert aux_breaker.is_open() is False

    def test_reset_all_clears_everything(self):
        main_breaker = get_breaker("main")
        for _ in range(3):
            main_breaker.record_failure(FakeReadTimeout())
        assert main_breaker.is_open() is True
        reset_all()
        assert main_breaker.is_open() is False


# --- Thread safety -----------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_failure(self):
        """100 threads each recording one failure → counter is exactly 100."""
        cb = CircuitBreaker(threshold=1000, cooldown_s=0.0)

        def hammer():
            for _ in range(10):
                cb.record_failure(FakeReadTimeout())

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The exact count of "FakeReadTimeout" should be 100.
        _, _, count = cb.snapshot()
        assert count == 100, f"expected 100, got {count}"

    def test_concurrent_open_close(self):
        cb = CircuitBreaker(threshold=50, cooldown_s=0.05)

        def tripper():
            for _ in range(60):
                cb.record_failure(FakeReadTimeout())

        def success():
            for _ in range(60):
                cb.record_success()

        t1 = threading.Thread(target=tripper)
        t2 = threading.Thread(target=success)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Should not raise; final state is one of the two.
        assert isinstance(cb.is_open(), bool)


# --- CircuitOpenError --------------------------------------------------

class TestCircuitOpenError:
    def test_message_includes_context(self):
        e = CircuitOpenError(
            error_type="ReadTimeout",
            cooldown_remaining=42.5,
            count=3,
            scope="main",
        )
        msg = str(e)
        assert "main" in msg
        assert "ReadTimeout" in msg
        assert "3" in msg
        assert "42.5" in msg

    def test_attributes_set(self):
        e = CircuitOpenError("RateLimitError", 1.5, 5, scope="auxiliary")
        assert e.error_type == "RateLimitError"
        assert e.cooldown_remaining == 1.5
        assert e.count == 5
        assert e.scope == "auxiliary"


# --- BREAKER_TRIGGER_ERRORS sanity ------------------------------------

def test_breaker_triggers_covers_known_transient_errors():
    """The expected transient error types are covered."""
    expected = {
        "ReadTimeout",
        "APITimeoutError",
        "TimeoutError",
        "RateLimitError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
    }
    assert expected.issubset(BREAKER_TRIGGER_ERRORS)


# --- Env var overrides -------------------------------------------------

def test_env_threshold_override(monkeypatch):
    monkeypatch.setenv("HERMES_CB_THRESHOLD", "5")
    monkeypatch.setenv("HERMES_CB_COOLDOWN", "120.0")
    # Re-import to pick up env changes
    import importlib
    importlib.reload(circuit_breaker)
    try:
        cb = circuit_breaker.CircuitBreaker()
        assert cb.threshold == 5
        assert cb.cooldown_s == 120.0
    finally:
        # Restore defaults
        monkeypatch.delenv("HERMES_CB_THRESHOLD", raising=False)
        monkeypatch.delenv("HERMES_CB_COOLDOWN", raising=False)
        importlib.reload(circuit_breaker)
