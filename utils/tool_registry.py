from __future__ import annotations
from typing import Dict, Type

_TOOL_CLASS_REGISTRY: Dict[str, Type] = {}

def register_tool(name: str):
    """Register a tool class under a config-facing name."""
    def decorator(cls):
        _TOOL_CLASS_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_tool_class(name: str):
    return _TOOL_CLASS_REGISTRY.get(name.lower())

def all_tools():
    return list(_TOOL_CLASS_REGISTRY)
