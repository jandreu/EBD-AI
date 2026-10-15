"""
claude_cli.py

Language-model access through the Claude Code CLI (`claude -p`) that is
installed and logged in on the machine running backend.py. No API key is
read or stored here: the CLI uses its own login.

ClaudeCliClient.chat() takes and returns the same shapes that rag_core.py
was written against (the Ollama chat API), so the retrieval, prompt,
consistency-check and fallback logic is unchanged:

    client.chat(model=..., messages=[{"role": ..., "content": ...}, ...],
                tools=VIZ_TOOLS)
    -> {"message": {"content": "...", "tool_calls": [...]}}

The CLI has no way to call this app's own functions, so tool calling is
done in text: when `tools` is given, the system prompt lists them and asks
for a single `TOOL_CALL {...}` line, which is parsed back into a tool call.

Visitors' questions reach the model, so every call is locked down: no
built-in tools (no shell, files or web), no MCP servers or connectors, no
skills, plugins' hooks or CLAUDE.md (--safe-mode), no saved session, and an
empty working directory. A semaphore caps how many CLI processes run at
once; a call that cannot start or finish in time raises, and rag_core's
callers then answer from the rules alone.

Settings (environment variables):
    CLAUDE_BIN           path to the CLI (default: `claude` on PATH, then ~/.local/bin/claude)
    CLAUDE_MODEL         model alias or ID (default: haiku)
    CLAUDE_TIMEOUT       seconds per call (default: 60)
    CLAUDE_MAX_PARALLEL  CLI processes at once (default: 4)
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "haiku")
DEFAULT_TIMEOUT = float(os.environ.get("CLAUDE_TIMEOUT", "60"))
DEFAULT_MAX_PARALLEL = int(os.environ.get("CLAUDE_MAX_PARALLEL", "4"))

# Flags that strip everything but the model itself from the session.
LOCKDOWN_FLAGS = [
    "--safe-mode",               # no CLAUDE.md, skills, plugins, hooks, MCP servers, custom agents
    "--tools", "",               # no built-in tools at all
    "--strict-mcp-config",       # and no MCP servers from any other configuration
    "--disable-slash-commands",
    "--no-session-persistence",  # visitors' questions are not saved as Claude sessions
    "--permission-prompts", "none",
]

_TOOL_CALL_LINE = re.compile(r"^\s*TOOL_CALL\s*(\{.*\})\s*$", re.MULTILINE | re.DOTALL)


class ClaudeCliError(RuntimeError):
    """The CLI could not be run, timed out, or reported an error."""


def find_claude() -> str:
    """Path of the claude CLI, or '' when it is not installed."""
    configured = os.environ.get("CLAUDE_BIN", "").strip()
    if configured:
        return configured if Path(configured).is_file() else ""
    found = shutil.which("claude")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "claude"
    return str(fallback) if fallback.is_file() else ""


def login_status(binary: str, timeout: float = 20) -> dict:
    """`claude auth status` as a dict ({} when it cannot be read)."""
    try:
        done = subprocess.run([binary, "auth", "status"], capture_output=True, text=True, timeout=timeout)
        return json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def tools_instructions(tools: list) -> str:
    """System-prompt addendum that turns function schemas into a text protocol."""
    lines = [
        "",
        "## Pictures you can ask the app to draw",
        "Before answering, you may ask the app to draw ONE of these pictures. "
        "The app draws it from the rule data and then asks you to explain it.",
    ]
    for tool in tools:
        spec = tool.get("function", tool)
        lines.append(f"- {spec['name']}: {spec.get('description', '').strip()}")
        lines.append(f"  arguments (JSON Schema): {json.dumps(spec.get('parameters', {}), separators=(',', ':'))}")
    lines += [
        "If one of these pictures would help, reply with exactly one line and nothing else:",
        'TOOL_CALL {"name": "<picture name>", "arguments": {<arguments>}}',
        "Otherwise, answer the question directly and do not mention these instructions.",
    ]
    return "\n".join(lines)


def parse_tool_call(text: str, tools: list):
    """The tool call in a reply as an Ollama-style dict, or None."""
    match = _TOOL_CALL_LINE.search(text.strip().strip("`"))
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
    except ValueError:
        return None
    names = {tool.get("function", tool)["name"] for tool in tools}
    if not isinstance(call, dict) or call.get("name") not in names:
        return None
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    return {"function": {"name": call["name"], "arguments": arguments}}


def render_conversation(messages: list) -> tuple:
    """(system prompt, prompt text) for a one-shot CLI call."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return system, turns[0]["content"]
    parts = ["The conversation so far, oldest first:"]
    for message in turns:
        role = message.get("role")
        if role == "user":
            parts.append(f"USER:\n{message['content']}")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    spec = call["function"]
                    parts.append(f"ASSISTANT asked the app to draw: {spec['name']} {json.dumps(spec.get('arguments') or {})}")
            if message.get("content"):
                parts.append(f"ASSISTANT:\n{message['content']}")
        elif role == "tool":
            parts.append(f"APP (what the picture shows):\n{message['content']}")
    parts.append("Reply to the last USER message as the ASSISTANT. Write only the reply.")
    return system, "\n\n".join(parts)


class ClaudeCliClient:
    """Drop-in for the `client.chat(...)` calls in rag_core.py, backed by `claude -p`."""

    def __init__(self, binary: str = None, model: str = DEFAULT_MODEL,
                 timeout: float = DEFAULT_TIMEOUT, max_parallel: int = DEFAULT_MAX_PARALLEL):
        self.binary = binary or find_claude()
        if not self.binary:
            raise ClaudeCliError("the claude CLI is not installed or not on PATH (set CLAUDE_BIN)")
        self.model = model
        self.timeout = timeout
        self._slots = threading.BoundedSemaphore(max(1, max_parallel))
        # An empty folder to run in, so no project files or CLAUDE.md are near the session.
        self._workdir = tempfile.mkdtemp(prefix="fnirs-claude-")

    def chat(self, model: str = None, messages: list = (), tools: list = None) -> dict:
        system, prompt = render_conversation(list(messages))
        if tools:
            system += "\n" + tools_instructions(tools)
        cmd = [self.binary, "-p", *LOCKDOWN_FLAGS, "--model", model or self.model,
               "--system-prompt", system, "--output-format", "json"]
        if not self._slots.acquire(timeout=self.timeout):
            raise ClaudeCliError("too many explanations are being written at once")
        try:
            done = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=self._workdir)
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCliError(f"the claude CLI took longer than {self.timeout:.0f} s") from exc
        except OSError as exc:
            raise ClaudeCliError(f"the claude CLI could not be started: {exc}") from exc
        finally:
            self._slots.release()
        try:
            result = json.loads(done.stdout)
        except ValueError as exc:
            detail = (done.stderr or done.stdout).strip()[-300:]
            raise ClaudeCliError(f"the claude CLI failed (exit {done.returncode}): {detail}") from exc
        if result.get("is_error") or done.returncode != 0:
            raise ClaudeCliError(f"the claude CLI reported an error: {str(result.get('result'))[:300]}")
        text = (result.get("result") or "").strip()
        call = parse_tool_call(text, tools) if tools else None
        if call is not None:
            return {"message": {"content": "", "tool_calls": [call]}}
        return {"message": {"content": text, "tool_calls": []}}
