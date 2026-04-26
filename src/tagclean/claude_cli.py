"""Headless `claude -p` subprocess wrapper for tagclean's LLM calls.

Lifted in spirit from chronicle's claude_cli.py but trimmed for a batch
tool: no subprocess registry, no daemon-shutdown machinery. What stays:

1. Binary resolution that works under launchd/systemd minimal PATH.
2. Subprocess env stripping so subscription auth wins over any
   ANTHROPIC_API_KEY / AUTH_TOKEN / BASE_URL the user has set.
3. Three-way error classification (INFRA / TRANSIENT / PARSE) so the
   caller's retry budget only burns on transient failures.
4. Subtype-aware envelope parsing — claude returns valid JSON for many
   failures (`error_during_execution`, `error_max_structured_output_retries`,
   `error_max_turns`); treating "is_error=false" as the only success
   signal misses real failures and lets the pipeline silently false-keep.
5. Telemetry fields on ClaudeResult so the caller can persist
   per-call cost + cache token counts for postmortem.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional


# Env vars that route Claude calls away from the user's Claude.ai subscription.
# Stripped before every `claude -p` invocation so a stray ANTHROPIC_API_KEY
# in the shell doesn't silently flip billing to API-key spend.
_STRIP_ENV_VARS = frozenset({
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
})

# Subtypes that the claude `-p --output-format json` envelope can carry.
# Anything not in _SUCCESS_SUBTYPES is treated as a model-side failure
# even if the JSON parses cleanly and `is_error` is false.
_SUCCESS_SUBTYPES = frozenset({"success"})
_TRANSIENT_SUBTYPES = frozenset({
    "error_during_execution",
    "error_max_structured_output_retries",
    "error_max_turns",
})


def _fallback_bin_dirs() -> list[Path]:
    """Dirs to search when shutil.which("claude") misses. Per-call so test
    fixtures that monkeypatch HOME are honored."""
    return [
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ]


def _standard_path_dirs() -> list[Path]:
    """Dirs prepended to subprocess PATH so any nested claude tool calls
    can still find their helpers under launchd/systemd minimal env."""
    return [
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
    ]


class ErrorKind(Enum):
    INFRA = "infra"          # Binary missing, perm denied, auth — don't retry
    TRANSIENT = "transient"  # Timeout, claude reported error subtype — retry
    PARSE = "parse"          # Stdout unparseable — retry once, then fail


class ClaudeNotFound(RuntimeError):
    pass


@dataclass
class ClaudeResult:
    """Outcome of a single `claude -p` call.

    On success: `structured_output` (when --json-schema was used) or
    `result_text` (when not) holds the payload. On failure: `error_kind`
    + `error_message` describe what went wrong; `subtype` is the
    envelope-level claim from claude itself when available.
    """
    structured_output: Optional[dict] = None
    result_text: str = ""
    subtype: str = ""
    total_cost_usd: float = 0.0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    num_turns: int = 0
    duration_ms: int = 0
    session_id: str = ""
    error_kind: Optional[ErrorKind] = None
    error_message: str = ""
    raw_envelope: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error_kind is None


# ---------- Binary resolution ----------

_cached_claude_path: Optional[Path] = None


def resolve_claude_binary(force_refresh: bool = False) -> Path:
    """Absolute path to `claude`. Raises ClaudeNotFound if absent.

    Searches PATH (via shutil.which), then a small set of fallback dirs.
    Caches the hit for the process lifetime.
    """
    global _cached_claude_path
    if _cached_claude_path is not None and not force_refresh:
        if _cached_claude_path.exists():
            return _cached_claude_path
        _cached_claude_path = None

    hit = shutil.which("claude")
    if hit:
        _cached_claude_path = Path(hit).resolve()
        return _cached_claude_path

    for d in _fallback_bin_dirs():
        candidate = d / "claude"
        if candidate.exists() and os.access(candidate, os.X_OK):
            _cached_claude_path = candidate.resolve()
            return _cached_claude_path

    searched = [os.environ.get("PATH", "")] + [str(d) for d in _fallback_bin_dirs()]
    raise ClaudeNotFound(
        "Could not find `claude` binary. Searched: "
        + " | ".join(searched)
        + ". Install Claude Code CLI (https://claude.com/claude-code) and "
        "ensure `claude` is on PATH."
    )


def try_resolve_claude_binary() -> Optional[Path]:
    try:
        return resolve_claude_binary()
    except ClaudeNotFound:
        return None


def build_subprocess_env(base: Optional[dict] = None) -> dict:
    """Env dict suitable for spawning `claude -p`.

    Strips ANTHROPIC_API_KEY/AUTH_TOKEN/BASE_URL so subscription routing
    wins; prepends standard bin dirs so nested invocations still resolve
    helpers under minimal-PATH parents.
    """
    src = base if base is not None else os.environ
    env = {k: v for k, v in src.items() if k not in _STRIP_ENV_VARS}
    existing = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    extra = [str(d) for d in _standard_path_dirs() if d.exists()]
    seen: set[str] = set()
    merged: list[str] = []
    for p in extra + existing:
        if p and p not in seen:
            seen.add(p)
            merged.append(p)
    env["PATH"] = os.pathsep.join(merged)
    return env


# ---------- Core invocation ----------

async def spawn_claude(
    prompt: str,
    *,
    model: str,
    fallback_model: str = "sonnet",
    effort: str = "high",
    json_schema: Optional[dict] = None,
    extra_flags: Iterable[str] = (),
    timeout: float = 300.0,
) -> ClaudeResult:
    """Run `claude -p` headlessly and return a classified ClaudeResult.

    Never raises for expected failure paths — callers branch on
    `result.error_kind`. INFRA failures should not count against retry
    budgets; TRANSIENT and PARSE should.
    """
    try:
        claude_bin = resolve_claude_binary()
    except ClaudeNotFound as e:
        return ClaudeResult(error_kind=ErrorKind.INFRA, error_message=str(e))

    args = [
        str(claude_bin), "-p",
        "--model", model,
        "--effort", effort,
        "--output-format", "json",
        "--no-session-persistence",
    ]
    # Claude CLI rejects --fallback-model when it equals --model
    # ("Fallback model cannot be the same as the main model"). Omit the
    # flag in that case — there's no useful fallback to a peer of yourself.
    if fallback_model and fallback_model != model:
        args += ["--fallback-model", fallback_model]
    if json_schema is not None:
        args += ["--json-schema", json.dumps(json_schema)]
    args += list(extra_flags)

    env = build_subprocess_env()

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as e:
        return ClaudeResult(
            error_kind=ErrorKind.INFRA,
            error_message=f"claude binary vanished before spawn: {e}",
        )
    except PermissionError as e:
        return ClaudeResult(
            error_kind=ErrorKind.INFRA,
            error_message=f"permission denied spawning claude: {e}",
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        return ClaudeResult(
            error_kind=ErrorKind.TRANSIENT,
            error_message=f"claude -p timed out after {timeout:.0f}s",
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Non-zero exit: probe stderr/stdout for auth-shaped infra failures
    # before classifying as transient.
    if proc.returncode != 0:
        msg = (stderr or stdout or f"exit {proc.returncode}").strip()[:300]
        combined = (stderr + " " + stdout).lower()
        infra_hints = (
            "command not found", "no such file",
            "not authenticated", "authentication required",
            "unauthorized", "please run", "please log in", "not logged in",
        )
        if any(h in combined for h in infra_hints):
            return ClaudeResult(error_kind=ErrorKind.INFRA, error_message=msg)
        # Try to extract a partial envelope so we can still record cost.
        partial = _try_parse_envelope(stdout)
        if partial is not None:
            res = _result_from_envelope(partial)
            res.error_kind = ErrorKind.TRANSIENT
            res.error_message = msg
            return res
        return ClaudeResult(error_kind=ErrorKind.TRANSIENT, error_message=msg)

    envelope = _try_parse_envelope(stdout)
    if envelope is None:
        return ClaudeResult(
            error_kind=ErrorKind.PARSE,
            error_message=f"outer JSON parse failed: {stdout[:200]}",
        )

    return _result_from_envelope(envelope)


def _try_parse_envelope(stdout: str) -> Optional[dict]:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def _result_from_envelope(envelope: dict) -> ClaudeResult:
    """Map a parsed envelope to a ClaudeResult with subtype-aware classification.

    The envelope shape (output_format=json) is approximately:
      {
        "type": "result",
        "subtype": "success" | "error_during_execution" | ...,
        "is_error": bool,
        "result": "<assistant text>" or "<error message>",
        "structured_output": {...}  // only when --json-schema was used
        "session_id": "...",
        "total_cost_usd": float,
        "duration_ms": int,
        "num_turns": int,
        "usage": {
          "cache_creation_input_tokens": int,
          "cache_read_input_tokens": int,
          ...
        }
      }
    """
    usage = envelope.get("usage") or {}
    res = ClaudeResult(
        structured_output=envelope.get("structured_output"),
        result_text=str(envelope.get("result") or ""),
        subtype=str(envelope.get("subtype") or ""),
        total_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        num_turns=int(envelope.get("num_turns") or 0),
        duration_ms=int(envelope.get("duration_ms") or 0),
        session_id=str(envelope.get("session_id") or ""),
        raw_envelope=envelope,
    )

    # Subtype is the canonical success signal. Envelope-level is_error
    # may be missing/inconsistent across CLI versions.
    if res.subtype and res.subtype not in _SUCCESS_SUBTYPES:
        res.error_kind = ErrorKind.TRANSIENT
        res.error_message = (
            f"claude reported subtype={res.subtype}: "
            + (res.result_text[:240] or "(no result text)")
        )
        return res

    # Defensive fallback when subtype field is absent.
    if envelope.get("is_error"):
        res.error_kind = ErrorKind.TRANSIENT
        res.error_message = (res.result_text[:240] or "claude reported is_error=true")

    return res


# ---------- Auth probe ----------

async def probe_auth(timeout: float = 30.0) -> tuple[bool, str]:
    """Cheap startup probe: confirm `claude -p` works under stripped env.

    Stripping ANTHROPIC_API_KEY forces subscription routing, but a user
    who isn't logged in via `claude /login` will get an INFRA failure on
    the first real call — much better to find out before launching a
    multi-hour pipeline run.

    Returns (ok, diagnostic_message).
    """
    # No --fallback-model on the probe (would force --model != --fallback);
    # spawn_claude already omits the flag when they match.
    res = await spawn_claude(
        prompt="Reply with exactly the word OK.",
        model="sonnet",
        fallback_model="sonnet",  # spawn_claude sees ==, drops the flag
        effort="low",
        timeout=timeout,
    )
    if res.ok:
        return True, f"auth ok (cost ${res.total_cost_usd:.4f}, {res.duration_ms}ms)"
    return False, f"auth probe failed [{res.error_kind.value if res.error_kind else 'unknown'}]: {res.error_message}"


# ---------- Test hooks ----------

def _reset_cache_for_tests() -> None:
    global _cached_claude_path
    _cached_claude_path = None
