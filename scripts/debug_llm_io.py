"""Direct test of circuit_breaker DEBUG_LLM_IN/OUT/CB.

Bypasses hermes' run_oneshot (which calls logging.disable(CRITICAL) and
redirects stdout to devnull) so we can actually SEE the debug output the
patch writes to agent.log.

Wires up the same AIAgent path as oneshot, but runs conversation_loop
directly with logging intact. Points the LLM at the local mock server
so the ReadTimeout exception is raised and the circuit breaker trips.
"""
import logging
import os
import sys
import time
from pathlib import Path

WORKTREE = Path("/Users/baiju/Documents/git-workspace/hermes-agent/.worktrees/fix-circuit-breaker")
sys.path.insert(0, str(WORKTREE))

# --- env: point LLM at the mock on port 9999, force short timeouts ---
os.environ["GLM_BASE_URL"] = "http://localhost:9999/v1"
os.environ["HERMES_STREAM_READ_TIMEOUT"] = "3"
os.environ["HERMES_API_TIMEOUT"] = "3"
os.environ.setdefault("HERMES_HOME", "/tmp/cb_test_home")
os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")

# --- import: same path as hermes_cli.oneshot._run_agent ---
from hermes_cli.config import load_config
from run_agent import AIAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    stream=sys.stdout,
)
# Make sure error+ shows
logging.getLogger().setLevel(logging.INFO)

print("=" * 60)
print(" Direct debug test: AIAgent against mock glm @ :9999")
print("=" * 60)

cfg = load_config()
print(f"[cfg] keys={list(cfg.keys())[:8]}")
_model = cfg.get("model", "")
if isinstance(_model, str):
    print(f"[cfg] model={_model}")
else:
    print(f"[cfg] model.default={_model.get('default')}")
    print(f"[cfg] model.provider={_model.get('provider')}")
    print(f"[cfg] model.base_url={_model.get('base_url')}")
print(f"[env] GLM_BASE_URL={os.environ.get('GLM_BASE_URL')}")
print(f"[env] HERMES_STREAM_READ_TIMEOUT={os.environ.get('HERMES_STREAM_READ_TIMEOUT')}")

# Build agent
try:
    agent = AIAgent(
        model="glm-5.1",
        provider="zai",
        base_url="http://localhost:9999/v1",
        quiet_mode=True,
        verbose_logging=True,
    )
    print(f"[agent] OK  type={type(agent).__name__}")
    print(f"[agent] model={agent.model}, base_url={agent.base_url}")
except Exception as e:
    print(f"[agent] FAILED to construct: {e!r}")
    raise

# Reset breaker to start clean
from agent.circuit_breaker import get_breaker, list_scopes, reset_all
reset_all()
print(f"[breaker] scopes at start: {sorted(list_scopes().keys())}")

# Run one conversation turn.  The mock hangs 5s; our stream timeout is
# 3s so we'll see a ReadTimeout, then DEBUG_CB should fire.
print("\n[run] Calling agent.chat('Say hi in one word') ...")
print("-" * 60)
try:
    result = agent.chat("Say hi in one word")
    print(f"\n[run] Returned (type={type(result).__name__})")
    if isinstance(result, str):
        print(f"      text: {result[:200]!r}")
    elif isinstance(result, dict):
        print(f"      keys: {list(result.keys())}")
        print(f"      final_response: {str(result.get('final_response'))[:200]!r}")
        print(f"      error: {str(result.get('error'))[:200]!r}")
except Exception as e:
    print(f"\n[run] EXCEPTION: {type(e).__name__}: {e}")

print("-" * 60)
# Final breaker state
print(f"\n[breaker] final state: {list_scopes()}")
print("\nDone.")
