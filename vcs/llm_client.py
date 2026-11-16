"""Unified LLM/VLM client supporting both LiteLLM and OpenAI SDK conventions."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

try:
    import litellm
except Exception:  # pragma: no cover - optional at import time
    litellm = None

try:
    from openai import OpenAI, OpenAIError
except Exception:  # pragma: no cover
    OpenAI = None

    class OpenAIError(Exception):
        pass


_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


class _CompatToolCall:
    """OpenAI-compatible tool_call wrapper with ``model_dump`` support."""

    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _CompatMessage:
    """OpenAI-compatible assistant message wrapper with ``model_dump`` support."""

    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[_CompatToolCall]] = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self) -> Dict[str, Any]:
        dump: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            dump["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return dump


def _normalize_tool_calls(raw_tool_calls: Any) -> List[_CompatToolCall]:
    if not raw_tool_calls:
        return []
    out: List[_CompatToolCall] = []
    for idx, tc in enumerate(raw_tool_calls):
        if isinstance(tc, dict):
            tc_id = tc.get("id") or f"call_{idx}"
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
        else:
            tc_id = getattr(tc, "id", None) or f"call_{idx}"
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn is not None else ""
            args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
        if not isinstance(args, str):
            import json

            args = json.dumps(args, ensure_ascii=False)
        out.append(_CompatToolCall(call_id=str(tc_id), name=str(name), arguments=args))
    return out


def _normalize_response(raw_resp: Any) -> Any:
    """Normalize LiteLLM/OpenAI response into a consistent OpenAI-like shape."""
    choices = getattr(raw_resp, "choices", None) or []
    if not choices:
        return raw_resp
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return raw_resp
    content = getattr(msg, "content", None)
    tool_calls = _normalize_tool_calls(getattr(msg, "tool_calls", None))
    compat_msg = _CompatMessage(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=compat_msg)],
        usage=getattr(raw_resp, "usage", None),
        model=getattr(raw_resp, "model", None),
        raw=raw_resp,
    )


def resolve_api_credentials(
    *,
    for_tool: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve API key, base URL, and API version supporting both LiteLLM and OpenAI env names."""
    if for_tool:
        resolved_key = (
            api_key
            or os.getenv("TOOL_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base = (
            base_url
            or os.getenv("TOOL_OPENAI_API_BASE")
            or os.getenv("TOOL_OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
        resolved_version = (
            api_version
            or os.getenv("TOOL_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
        )
    else:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
        resolved_version = api_version or os.getenv("OPENAI_API_VERSION")

    return {
        "api_key": resolved_key,
        "base_url": resolved_base,
        "api_version": resolved_version,
    }


def split_model_and_effort(model_name: str, reasoning_effort: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Split optional ``-<effort>`` suffix from model name."""
    if reasoning_effort is not None:
        return model_name, reasoning_effort
    parts = str(model_name).rsplit("-", 1)
    if len(parts) == 2 and parts[1] in _REASONING_EFFORTS:
        return parts[0], parts[1]
    return model_name, None


class _LiteLLMCompletions:
    """Drop-in ``client.chat.completions`` implementation backed by ``litellm.completion``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = 900,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout

    def create(self, **kwargs: Any) -> Any:
        if litellm is None:
            raise RuntimeError("litellm is not installed. Install it with `pip install litellm`.")

        req = dict(kwargs)
        model = str(req.pop("model"))
        explicit_effort = req.pop("reasoning_effort", None)
        base_model, parsed_effort = split_model_and_effort(model, explicit_effort)

        call_kwargs: Dict[str, Any] = {
            "model": base_model,
            "messages": req.pop("messages"),
            "timeout": req.pop("timeout", self.timeout),
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_version:
            call_kwargs["api_version"] = self.api_version
        if parsed_effort:
            call_kwargs["reasoning_effort"] = parsed_effort
        if "max_tokens" in req and "max_completion_tokens" not in req:
            max_tok = req.pop("max_tokens")
            if max_tok is not None:
                call_kwargs["max_completion_tokens"] = max_tok

        # Forward remaining supported options (tools, tool_choice, response_format, temperature, seed, extra_body, etc.)
        for k, v in req.items():
            if v is not None:
                call_kwargs[k] = v

        try:
            raw_resp = litellm.completion(**call_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "temperature" in call_kwargs and "temperature" in msg:
                call_kwargs.pop("temperature", None)
                raw_resp = litellm.completion(**call_kwargs)
            else:
                raise
        return _normalize_response(raw_resp)


class LiteLLMOpenAIClient:
    """OpenAI-SDK-compatible client powered by ``litellm.completion``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = 900,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.chat = SimpleNamespace(
            completions=_LiteLLMCompletions(
                api_key=api_key,
                api_base=base_url,
                api_version=api_version,
                timeout=timeout,
            )
        )


def build_llm_client(
    *,
    for_tool: bool = False,
    backend: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    timeout: int = 900,
) -> Any:
    """Construct either a LiteLLM-backed or OpenAI-SDK-backed chat client.

    Parameters
    ----------
    for_tool:
        When True, prefer ``TOOL_OPENAI_*`` environment variables with fallback to ``OPENAI_*``.
    backend:
        ``"litellm"``, ``"openai"``, or ``"auto"`` (defaults to ``"litellm"`` if available, else ``"openai"``).
    """
    creds = resolve_api_credentials(
        for_tool=for_tool,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )
    selected = (backend or os.getenv("LENSWALK_LLM_BACKEND", "auto")).lower()
    if selected == "auto":
        selected = "litellm" if litellm is not None else "openai"

    if selected == "litellm":
        return LiteLLMOpenAIClient(
            api_key=creds["api_key"],
            base_url=creds["base_url"],
            api_version=creds["api_version"],
            timeout=timeout,
        )
    if selected == "openai":
        if OpenAI is None:
            raise RuntimeError("The openai SDK is not installed. Install it with `pip install openai`.")
        return OpenAI(api_key=creds["api_key"], base_url=creds["base_url"], timeout=timeout)

    raise ValueError(f"Unsupported LLM backend '{selected}'. Expected 'auto', 'litellm', or 'openai'.")
