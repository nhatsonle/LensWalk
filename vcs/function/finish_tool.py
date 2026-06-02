from __future__ import annotations
from typing import Any, Dict

from utils.tool_base import ToolBase
from utils.tool_registry import register_tool

from ..vcs_utils import StopException

@register_tool("finish")
class FinishTool(ToolBase):
    """
    A special tool that stops execution by raising StopException.
    Used to signal that the conversation has been completed.
    """

    DEFAULT_ARGUMENTS: Dict[str, Any] = {
        "answer": None,
    }

    def __init__(
        self,
        *,
        _type = 'func',
        name: str = "finish",
        arguments: Dict[str, Any] | None = None,
        timeout: int = 5,
        **kwargs
    ) -> None:
        merged_args = {**self.DEFAULT_ARGUMENTS, **(arguments or {})}
        super().__init__(
            name=name,
            type=_type,
            arguments=merged_args,
            timeout=timeout,
        )

    def _url_call(self, params: Dict[str, Any], *, identity=None, bench=None) -> Any:
        return {"error": "FinishTool does not support URL mode."}

    def _func_call(self, params: Dict[str, Any]) -> Any:
        answer = params["answer"]
        raise StopException(answer)


__all__ = ["FinishTool", "StopException"]
