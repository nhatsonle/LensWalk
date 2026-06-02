"""Config-driven tool dispatcher used by LensWalk VCS tools."""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .tool_registry import get_tool_class
from .tool_base import ToolBase

# Importing this package registers the VCS tool classes via decorators.
import vcs.function  # noqa: F401


def _safe_literal_eval(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return text


def _extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        text = "None"
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        inner = re.search(r"\b([A-Z])\b", m.group(1))
        if inner:
            return inner.group(1)
    boxed = re.search(r"\\?\s*boxed\s*\{\s*([A-Z])\s*\}", text, flags=re.IGNORECASE)
    if boxed:
        return boxed.group(1).upper()
    global_first = re.search(r"[A-Z]", text)
    if global_first:
        return global_first.group(0)

    return None


class ToolDispatcher:
    def __init__(self) -> None:
        self._registry: Dict[str, ToolBase] = {}
        self._bench: Optional[str] = None

    def load_config(self, path: str | Path) -> None:
        cfg = yaml.safe_load(Path(path).read_text())
        tools_cfg = cfg.get("tools", cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(tools_cfg, list):
            raise ValueError("Config must be a list or contain 'tools' list")
        self._bench = cfg.get("bench") if isinstance(cfg, dict) else None

        for entry in tools_cfg:
            tool_name = entry["name"]

            class_path = entry.get("class", tool_name)
            tool_cls = get_tool_class(class_path)
            if tool_cls is None:
                raise ValueError(f"Tool class is not registered: {class_path}")

            params = {k: v for k, v in entry.items() if k != "class"}

            tool_instance = tool_cls(**params)  # type: ignore[arg-type]
            self._registry[tool_name.lower()] = tool_instance

    @staticmethod
    def get_query(text: str) -> Optional[str]:
        m = re.findall(r"<tool>(.*?)</tool>", text, flags=re.DOTALL)
        return m[-1] if m else None

    @staticmethod
    def parse_tool_call(raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            if "arguments" in raw and not isinstance(raw["arguments"], dict):
                raw["arguments"] = _safe_literal_eval(raw["arguments"])
            return raw
        if isinstance(raw, str):
            raw = raw.strip()
            obj = _safe_literal_eval(raw)
            if isinstance(obj, dict) and "name" in obj:
                if not isinstance(obj.get("arguments"), dict):
                    obj["arguments"] = _safe_literal_eval(obj.get("arguments", "{}"))
                return obj
            name_m = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw)
            if not name_m:
                return None
            args_m = re.search(r"['\"]arguments['\"]\s*:\s*({.*})", raw, flags=re.DOTALL)
            args = _safe_literal_eval(args_m.group(1)) if args_m else {}
            if not isinstance(args, dict):
                args = {}
            return {"name": name_m.group(1), "arguments": args}
        return None

    def call(self, query_info: Dict[str, Any] | None, *, identity: Optional[str] = None) -> Any:
        if not query_info:
            return ""
        tool = self._registry.get(query_info["name"].lower())
        if tool is None:
            return f"Invalid tool name: {query_info['name']}."
        try:
            return tool(query_info.get("arguments", {}), identity=identity, bench=self._bench)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Tool '%s' failed", tool.name)
            return f"Tool call error: {exc}"

    @staticmethod
    def compute_score(pred: str, gt: str) -> float:
        a = _extract_answer(pred)
        return 1.0 if a and a.lower() == gt.strip().lower() else 0.0
