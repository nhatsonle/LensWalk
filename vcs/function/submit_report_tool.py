"""Structured report submission tool used by Actor Agents."""
from __future__ import annotations

from typing import Any, Dict, Optional

from utils.tool_base import ToolBase
from utils.tool_registry import register_tool


class ReportSubmittedException(Exception):
    """Signal that an Actor Agent has submitted its structured ActorReport."""

    def __init__(self, report: Dict[str, Any]) -> None:
        super().__init__("Actor report submitted.")
        self.report = report


@register_tool("submit_report")
class SubmitReportTool(ToolBase):
    """Terminate an Actor loop by submitting a structured evidence report."""

    DEFAULT_ARGUMENTS: Dict[str, Any] = {
        "status": "NOT_FOUND",
        "candidate_answer": None,
        "confidence": "low",
        "evidences": [],
        "uncertainties": "",
    }

    def __init__(
        self,
        *,
        _type: str = "func",
        name: str = "submit_report",
        arguments: Optional[Dict[str, Any]] = None,
        timeout: int = 5,
        **kwargs: Any,
    ) -> None:
        merged_args = {**self.DEFAULT_ARGUMENTS, **(arguments or {})}
        super().__init__(
            name=name,
            type=_type,
            arguments=merged_args,
            timeout=timeout,
        )

    def _validate(self, params: Dict[str, Any]) -> None:
        # Allow optional fields like candidate_answer or uncertainties to default gracefully.
        if "status" not in params:
            raise ValueError(f"Tool '{self.name}': Missing required argument 'status'.")

    def _url_call(self, params: Dict[str, Any], *, identity=None, bench=None) -> Any:
        return {"error": "SubmitReportTool does not support URL mode."}

    def _func_call(self, params: Dict[str, Any]) -> Any:
        status = str(params.get("status", "NOT_FOUND")).upper()
        if status not in {"FOUND", "PARTIAL", "NOT_FOUND"}:
            status = "PARTIAL"
        candidate_answer = params.get("candidate_answer")
        if status == "NOT_FOUND":
            candidate_answer = None
        report = {
            "status": status,
            "candidate_answer": candidate_answer,
            "confidence": str(params.get("confidence") or "low").lower(),
            "evidences": params.get("evidences") or [],
            "uncertainties": str(params.get("uncertainties") or ""),
        }
        raise ReportSubmittedException(report)
