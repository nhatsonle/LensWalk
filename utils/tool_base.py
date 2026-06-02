from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ToolBase(ABC):
    """Abstract base class for callable tools."""

    name: str
    type: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    url: Optional[str] = None
    timeout: int = 60

    def __call__(self, params: Dict[str, Any], *, identity: Optional[str] = None, bench: Optional[str]= None) -> Any:
        """Validate parameters and dispatch by tool type."""
        self._validate(params)
        if self.type == "url":
            return self._url_call(params, identity=identity, bench=bench)
        if self.type == "func":
            return self._func_call(params)
        raise RuntimeError(f"Unknown tool type '{self.type}' for '{self.name}'. Tool type must be 'url' or 'func'.")

    def _validate(self, params: Dict[str, Any]) -> None:
        """Validate that all declared arguments are present."""
        missing = [k for k in self.arguments if k not in params and not getattr(self, k, False)]
        if missing:
            raise ValueError(f"Tool '{self.name}': Missing required argument(s): {', '.join(missing)}")

    @abstractmethod
    def _url_call(self, params: Dict[str, Any], *, identity: Optional[str], bench: Optional[str]) -> Any:
        """Handle URL-backed tool calls."""
        raise NotImplementedError

    @abstractmethod
    def _func_call(self, params: Dict[str, Any], *, identity: Optional[str], bench: Optional[str]) -> Any:
        """Handle local function-backed tool calls."""
        raise NotImplementedError
