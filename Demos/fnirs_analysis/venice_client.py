"""
venice_client.py

Language-model access through the Venice API (https://venice.ai), an
OpenAI-compatible chat service. backend.py uses it for visitors who are not
logged in with the admin password; the admin's answers come from the Claude
CLI (claude_cli.py).

VeniceClient.chat() has the same interface and the same text protocol for
tool calls as ClaudeCliClient, so both models see identical prompts:

    client.chat(model=..., messages=[...], tools=VIZ_TOOLS)
    -> {"message": {"content": "...", "tool_calls": [...]}}

Venice's own system prompt is switched off and the model's reasoning trace is
neither requested nor shown. A semaphore caps the requests in flight; a
request that cannot start or finish in time raises, and rag_core's callers
then answer from the rules alone.

Settings (environment variables; keep the key out of the repository):
    VENICE_API_KEY        API key (required)
    VENICE_MODEL          model ID (default: zai-org-glm-4.7-flash)
    VENICE_TIMEOUT        seconds per request (default: 60)
    VENICE_MAX_PARALLEL   requests at once (default: 8)
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request

from claude_cli import parse_tool_call, render_conversation, tools_instructions

API_URL = "https://api.venice.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("VENICE_MODEL", "zai-org-glm-4.7-flash")
DEFAULT_TIMEOUT = float(os.environ.get("VENICE_TIMEOUT", "60"))
DEFAULT_MAX_PARALLEL = int(os.environ.get("VENICE_MAX_PARALLEL", "8"))

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class VeniceError(RuntimeError):
    """The Venice API could not be reached, timed out, or returned an error."""


class VeniceClient:
    """Drop-in for the `client.chat(...)` calls in rag_core.py, backed by the Venice API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT,
                 max_parallel: int = DEFAULT_MAX_PARALLEL, url: str = API_URL):
        if not api_key:
            raise VeniceError("VENICE_API_KEY is not set")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = url
        self._slots = threading.BoundedSemaphore(max(1, max_parallel))

    def __repr__(self) -> str:  # never show the key
        return f"VeniceClient(model={self.model!r})"

    def chat(self, model: str = None, messages: list = (), tools: list = None) -> dict:
        system, prompt = render_conversation(list(messages))
        if tools:
            system += "\n" + tools_instructions(tools)
        body = {
            "model": model or self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2048,
            "venice_parameters": {"include_venice_system_prompt": False,
                                  "strip_thinking_response": True, "disable_thinking": True},
        }
        request = urllib.request.Request(
            self.url, json.dumps(body).encode(),
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        if not self._slots.acquire(timeout=self.timeout):
            raise VeniceError("too many explanations are being written at once")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise VeniceError(f"the Venice API answered {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise VeniceError(f"the Venice API could not be reached: {exc}") from exc
        finally:
            self._slots.release()
        try:
            text = reply["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VeniceError("the Venice API returned an unexpected reply") from exc
        text = _THINKING.sub("", text).strip()
        call = parse_tool_call(text, tools) if tools else None
        if call is not None:
            return {"message": {"content": "", "tool_calls": [call]}}
        return {"message": {"content": text, "tool_calls": []}}
