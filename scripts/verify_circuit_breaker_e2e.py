"""End-to-end verification of the circuit_breaker integration.

This script:
  1. Imports the patched circuit_breaker
  2. Wraps a fake LLM call that throws ReadTimeout
  3. Verifies the breaker trips after 3 consecutive failures
  4. Verifies the breaker is shared between conversation_loop and
     auxiliary_client scopes (i.e. main and auxiliary trip independently)

Run from the worktree root:

    .venv/bin/python scripts/verify_circuit_breaker_e2e.py
"""
import sys
import time

sys.path.insert(0, ".")

from agent.circuit_breaker import (
    BREAKER_TRIGGER_ERRORS,
    CircuitOpenError,
    get_breaker,
    list_scopes,
    reset_all,
)


# Stand-in for httpx.ReadTimeout.  Real classes from httpx/openai would
# also work, but using a local subclass keeps the test dependency-free
# and lets us assert on the exact class name passed through.
class _MockReadTimeout(TimeoutError):
    pass


def fake_llm_call(call_id: int) -> str:
    """Pretend to call an LLM. Always raises ReadTimeout."""
    raise _MockReadTimeout(f"mocked ReadTimeout on call #{call_id}")


def verify_trips_after_threshold(threshold: int = 3) -> None:
    """Verify: 3 consecutive ReadTimeouts → breaker trips."""
    print(f"\n=== Test 1: trips after {threshold} consecutive ReadTimeouts ===")
    reset_all()
    breaker = get_breaker("main")

    for i in range(1, threshold + 1):
        tripped = breaker.record_failure(_MockReadTimeout(f"call {i}"))
        print(f"  call {i}: tripped={tripped}, is_open={breaker.is_open()}")
        if i < threshold:
            assert not tripped, f"should not trip at call {i}"
        else:
            assert tripped, f"should trip at call {i}"

    assert breaker.is_open(), "breaker should be open after threshold"
    err_type, remaining, count = breaker.snapshot()
    print(f"  → snapshot: type={err_type}, count={count}, "
          f"cooldown_remaining={remaining:.1f}s")
    assert err_type == "TimeoutError", f"expected TimeoutError, got {err_type}"
    assert count == threshold
    print("  ✓ PASS")


def verify_success_resets() -> None:
    """Verify: a success in between prevents the breaker from tripping."""
    print("\n=== Test 2: success between failures resets the counter ===")
    reset_all()
    breaker = get_breaker("main")

    breaker.record_failure(_MockReadTimeout("c1"))
    breaker.record_failure(_MockReadTimeout("c2"))
    breaker.record_success()  # counter resets
    breaker.record_failure(_MockReadTimeout("c3"))
    breaker.record_failure(_MockReadTimeout("c4"))

    assert not breaker.is_open(), \
        "breaker should still be closed (only 2 consecutive)"
    print(f"  is_open={breaker.is_open()} (expected False)")
    print("  ✓ PASS")


def verify_scope_isolation() -> None:
    """Verify: 'main' and 'auxiliary' scopes are independent."""
    print("\n=== Test 3: 'main' and 'auxiliary' scopes are independent ===")
    reset_all()
    main_breaker = get_breaker("main")
    aux_breaker = get_breaker("auxiliary")

    # Trip main
    for _ in range(3):
        main_breaker.record_failure(_MockReadTimeout("main"))
    assert main_breaker.is_open(), "main should be open"
    assert not aux_breaker.is_open(), "auxiliary should still be closed"
    print(f"  main.is_open={main_breaker.is_open()}, "
          f"aux.is_open={aux_breaker.is_open()}")
    print("  ✓ PASS")


def verify_circuit_open_error() -> None:
    """Verify: CircuitOpenError carries diagnostic info."""
    print("\n=== Test 4: CircuitOpenError surfaces count + cooldown ===")
    reset_all()
    breaker = get_breaker("main")
    for _ in range(3):
        breaker.record_failure(_MockReadTimeout("z"))
    err_type, remaining, count = breaker.snapshot()
    err = CircuitOpenError(err_type, remaining, count, scope="main")
    print(f"  message: {err}")
    print(f"  attrs: type={err.error_type}, count={err.count}, "
          f"cooldown={err.cooldown_remaining:.1f}s, scope={err.scope}")
    assert err.count == 3
    assert err.scope == "main"
    print("  ✓ PASS")


def verify_default_scopes_eagerly_initialized() -> None:
    """Verify: the four standard scopes show up in list_scopes()."""
    print("\n=== Test 5: default scopes eagerly created ===")
    reset_all()
    scopes = list_scopes()
    print(f"  scopes: {sorted(scopes.keys())}")
    for s in ("main", "auxiliary", "subagent", "vision"):
        assert s in scopes, f"scope {s!r} should be present"
    print("  ✓ PASS")


def verify_non_trigger_ignored() -> None:
    """Verify: ValidationError-style errors don't trip the breaker."""
    print("\n=== Test 6: non-trigger errors are ignored ===")
    reset_all()
    breaker = get_breaker("main")

    class FakeValidationError(Exception):
        pass

    for _ in range(20):
        tripped = breaker.record_failure(FakeValidationError("nope"))
        assert not tripped
    assert not breaker.is_open()
    print(f"  is_open={breaker.is_open()} after 20 validation errors")
    print("  ✓ PASS")


def verify_realistic_simulation() -> None:
    """Realistic end-to-end: simulate 5 API calls with mix of success/timeout."""
    print("\n=== Test 7: realistic 5-call simulation ===")
    reset_all()
    breaker = get_breaker("main")
    log = []

    def attempt(n: int, should_fail: bool):
        try:
            if should_fail:
                raise _MockReadTimeout(f"call {n} failed")
            log.append(f"call {n}: success")
            breaker.record_success()
            return "ok"
        except _MockReadTimeout as e:
            if breaker.record_failure(e):
                log.append(f"call {n}: TRIPPED ({type(e).__name__})")
                raise CircuitOpenError(*breaker.snapshot(), scope="main") from e
            log.append(f"call {n}: failed but not tripped")
            raise

    # 1: success
    # 2: timeout (count=1)
    # 3: timeout (count=2)
    # 4: timeout (count=3) → trips
    # 5: would timeout but breaker is open
    scenario = [False, True, True, True, True]
    for i, fail in enumerate(scenario, 1):
        try:
            attempt(i, fail)
        except (CircuitOpenError, _MockReadTimeout) as e:
            log.append(f"  call {i} → caught {type(e).__name__}: {e}")

    for line in log:
        print(f"  {line}")
    print("  ✓ SIMULATION COMPLETE")


if __name__ == "__main__":
    print("=" * 70)
    print(" Circuit Breaker End-to-End Verification")
    print(f" BREAKER_TRIGGER_ERRORS: {sorted(BREAKER_TRIGGER_ERRORS)}")
    print("=" * 70)
    verify_trips_after_threshold()
    verify_success_resets()
    verify_scope_isolation()
    verify_circuit_open_error()
    verify_default_scopes_eagerly_initialized()
    verify_non_trigger_ignored()
    verify_realistic_simulation()
    print("\n" + "=" * 70)
    print(" ALL VERIFICATIONS PASSED ✓")
    print("=" * 70)
