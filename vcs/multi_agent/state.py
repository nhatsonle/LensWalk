"""Shared state, interval utilities, and LLM JSON/retry helpers for multi-agent inference."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from ..data_models import ActorReport, SubjectRegistryState, TimeInterval


_TIMESTAMP_PATTERNS = (
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:m|min|mins|minute|minutes)\s*(?:\d+\s*(?:s|sec|seconds?))?\b", re.IGNORECASE),
)

_TRANSIENT_MARKERS = (
    "rate limit",
    "usage limit",
    "reset after",
    "timed out",
    "timeout",
    "too many requests",
    "forbidden for url",
    "internal",
    "server had an error",
    "service unavailable",
    "serviceunavailableerror",
    "connection",
    "429",
    "502",
    "503",
)

_RESET_AFTER_RE = re.compile(r"reset after\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def has_explicit_timestamp(question: str) -> bool:
    """Return True if the question already references a concrete timestamp."""
    if not question:
        return False
    return any(pat.search(question) for pat in _TIMESTAMP_PATTERNS)


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from raw LLM text, stripping markdown fences if present."""
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def call_chat_with_retry(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    max_retries: int = 5,
) -> Any:
    """Invoke ``client.chat.completions.create`` with exponential backoff and format fallback."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None and tools:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if "qwen" in str(model).lower():
        kwargs["extra_body"] = {"enable_thinking": False}

    delay = 1.5
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "response_format" in kwargs and (
                "response_format" in msg or "json_schema" in msg or "json_object" in msg or "unsupported" in msg
            ):
                kwargs.pop("response_format", None)
                continue
            is_transient = any(tok in msg for tok in _TRANSIENT_MARKERS) or "openai" in type(exc).__name__.lower()
            if not is_transient or attempt == max_retries - 1:
                raise
            wait_sec = min(20.0, delay)
            reset_match = _RESET_AFTER_RE.search(str(exc))
            if reset_match:
                wait_sec = min(120.0, float(reset_match.group(1)) + 2.0)
            print(f"[LLM Retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc} (sleeping {wait_sec:.1f}s)", flush=True)
            time.sleep(wait_sec)
            delay *= 2.0
    return None


def merge_intervals(intervals: List[TimeInterval], gap_tol: float = 1.0) -> List[TimeInterval]:
    """Merge overlapping or nearly adjacent intervals into sorted disjoint intervals."""
    valid = [
        TimeInterval(start_sec=float(iv.start_sec), end_sec=float(iv.end_sec))
        for iv in intervals
        if float(iv.end_sec) > float(iv.start_sec)
    ]
    if not valid:
        return []
    valid.sort(key=lambda iv: (iv.start_sec, iv.end_sec))
    merged: List[TimeInterval] = [valid[0]]
    for cur in valid[1:]:
        prev = merged[-1]
        if cur.start_sec <= prev.end_sec + gap_tol:
            merged[-1] = TimeInterval(
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, cur.end_sec),
            )
        else:
            merged.append(cur)
    return merged


def compute_uncovered_intervals(
    duration_sec: float,
    covered_intervals: List[TimeInterval],
    *,
    min_span: float = 10.0,
) -> List[TimeInterval]:
    """Return uncovered segments in ``[0, duration_sec]`` with span >= ``min_span``."""
    dur = max(0.0, float(duration_sec))
    if dur <= 0.0:
        return []
    merged = merge_intervals(covered_intervals, gap_tol=1.0)
    gaps: List[TimeInterval] = []
    cursor = 0.0
    for iv in merged:
        s = max(0.0, min(iv.start_sec, dur))
        e = max(0.0, min(iv.end_sec, dur))
        if s - cursor >= min_span:
            gaps.append(TimeInterval(start_sec=round(cursor, 2), end_sec=round(s, 2)))
        cursor = max(cursor, e)
    if dur - cursor >= min_span:
        gaps.append(TimeInterval(start_sec=round(cursor, 2), end_sec=round(dur, 2)))
    return gaps


def format_intervals(intervals: List[TimeInterval]) -> str:
    """Format interval list as compact ``[s-e]`` text."""
    if not intervals:
        return "None"
    return ", ".join(f"[{iv.start_sec:.1f}s-{iv.end_sec:.1f}s]" for iv in intervals)


@dataclass
class SharedVideoState:
    """Thread-safe shared state across Orchestrator, Actors, and Verification."""

    duration_sec: float
    total_frame_budget: int
    overview_frames: int = 0
    actor_frames: int = 0
    verify_frames: int = 0
    fallback_frames: int = 0
    covered_intervals: List[TimeInterval] = field(default_factory=list)
    actor_reports: List[ActorReport] = field(default_factory=list)
    subject_registry: SubjectRegistryState = field(default_factory=SubjectRegistryState)
    replan_count: int = 0
    has_stitched_verified: bool = False
    verbose: bool = False
    tool_call_logs: List[Dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def log(self, message: str) -> None:
        """Thread-safe verbose logger."""
        if not self.verbose:
            return
        with self._lock:
            print(message, flush=True)

    @property
    def used_frames(self) -> int:
        return self.overview_frames + self.actor_frames + self.verify_frames + self.fallback_frames

    @property
    def remaining_budget(self) -> int:
        return max(0, self.total_frame_budget - self.used_frames)

    def record_usage(
        self,
        *,
        phase: str,
        frames: int,
        coverage: Optional[List[TimeInterval]] = None,
        mark_covered: bool = True,
        subject_updates: Optional[Dict[str, Any]] = None,
        log_entry: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            n = max(0, int(frames))
            if phase == "overview":
                self.overview_frames += n
            elif phase == "actor":
                self.actor_frames += n
            elif phase == "verify":
                self.verify_frames += n
            elif phase == "fallback":
                self.fallback_frames += n
            if mark_covered and coverage:
                self.covered_intervals = merge_intervals(self.covered_intervals + list(coverage))
            if subject_updates:
                merged_reg = dict(self.subject_registry.subject_registry)
                for key, val in subject_updates.items():
                    merged_reg[key] = val
                self.subject_registry = SubjectRegistryState(subject_registry=merged_reg)
            if log_entry is not None:
                self.tool_call_logs.append(log_entry)

    def add_reports(self, reports: List[ActorReport]) -> None:
        with self._lock:
            self.actor_reports.extend(reports)

    def get_covered_intervals(self) -> List[TimeInterval]:
        with self._lock:
            return list(self.covered_intervals)

    def get_uncovered_intervals(self, min_span: float = 10.0) -> List[TimeInterval]:
        with self._lock:
            return compute_uncovered_intervals(
                self.duration_sec,
                self.covered_intervals,
                min_span=min_span,
            )

    def budget_breakdown(self) -> Dict[str, int]:
        return {
            "total_frame_budget": self.total_frame_budget,
            "overview_frames": self.overview_frames,
            "actor_frames": self.actor_frames,
            "verify_frames": self.verify_frames,
            "fallback_frames": self.fallback_frames,
            "total_frames_used": self.used_frames,
            "remaining_budget": self.remaining_budget,
        }
