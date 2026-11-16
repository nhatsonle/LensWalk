"""Actor Agent runner for bounded hypothesis exploration."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from ..data_models import ActorEvidenceItem, ActorReport, Hypothesis, TimeInterval
from ..function.submit_report_tool import ReportSubmittedException
from ..prompts.multi_agent import ACTOR_SYSTEM_PROMPT, ACTOR_USER_PROMPT
from ..vcs_dispatcher import VCSDispatcher
from .state import SharedVideoState, call_chat_with_retry, extract_json_object, merge_intervals


def _extract_payload_and_text(result: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    if isinstance(result, tuple):
        text = ""
        payload = None
        for part in result:
            if isinstance(part, str) and not text:
                text = part
            elif isinstance(part, dict) and payload is None:
                payload = part
        return text or json.dumps(result, ensure_ascii=False), payload
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False), result
    return str(result), None


def _extract_subject_updates(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return merged
    for obs in payload.get("observations") or []:
        if isinstance(obs, dict) and isinstance(obs.get("subject_registry"), dict):
            merged.update(obs["subject_registry"])
    return merged


def _build_report_from_dict(
    raw: Dict[str, Any],
    *,
    hypothesis: Hypothesis,
    covered: List[TimeInterval],
    used_frames: int,
) -> ActorReport:
    status = str(raw.get("status", "NOT_FOUND")).upper()
    if status not in {"FOUND", "PARTIAL", "NOT_FOUND"}:
        status = "PARTIAL"
    candidate_answer = raw.get("candidate_answer")
    if status == "NOT_FOUND":
        candidate_answer = None

    confidence = str(raw.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    evidences: List[ActorEvidenceItem] = []
    for item in raw.get("evidences") or []:
        if not isinstance(item, dict):
            continue
        itv_raw = item.get("interval") or {
            "start_sec": hypothesis.interval.start_sec,
            "end_sec": hypothesis.interval.end_sec,
        }
        try:
            itv = TimeInterval(
                start_sec=float(itv_raw.get("start_sec", hypothesis.interval.start_sec)),
                end_sec=float(itv_raw.get("end_sec", hypothesis.interval.end_sec)),
            )
        except Exception:
            itv = hypothesis.interval
        rel = str(item.get("relevance", "partial")).lower()
        if rel not in {"direct", "partial", "none"}:
            rel = "partial"
        evidences.append(
            ActorEvidenceItem(
                interval=itv,
                visual_fact=str(item.get("visual_fact", "")),
                relevance=rel,  # type: ignore[arg-type]
            )
        )

    return ActorReport(
        hypothesis_id=hypothesis.hypothesis_id,
        status=status,  # type: ignore[arg-type]
        candidate_answer=str(candidate_answer) if candidate_answer is not None else None,
        confidence=confidence,  # type: ignore[arg-type]
        evidences=evidences,
        uncertainties=str(raw.get("uncertainties") or ""),
        covered_intervals=merge_intervals(covered),
        used_frames=used_frames,
    )


class ActorRunner:
    """Executes one hypothesis inside a bounded time window and frame budget."""

    def __init__(
        self,
        *,
        dispatcher: VCSDispatcher,
        llm_client: Any,
        model_name: str,
        max_turns: int = 3,
        scan_min_span_sec: float = 60.0,
    ) -> None:
        self.dispatcher = dispatcher
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_turns = max(1, int(max_turns))
        self.scan_min_span_sec = float(scan_min_span_sec)

    def run(
        self,
        *,
        hypothesis: Hypothesis,
        question: str,
        video_path: str,
        duration_sec: float,
        state: SharedVideoState,
        session_id: Any = None,
        subtitles: Any = None,
        phase: str = "actor",
    ) -> Tuple[ActorReport, List[Dict[str, Any]]]:
        """Investigate ``hypothesis`` and return ``(ActorReport, chat_history)``."""
        h_start = max(0.0, float(hypothesis.interval.start_sec))
        h_end = min(float(duration_sec), float(hypothesis.interval.end_sec))
        if h_end <= h_start:
            return (
                ActorReport(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status="NOT_FOUND",
                    candidate_answer=None,
                    confidence="low",
                    uncertainties="Invalid or empty hypothesis interval.",
                    covered_intervals=[],
                    used_frames=0,
                ),
                [],
            )

        span = h_end - h_start
        allocated = max(1, int(hypothesis.allocated_frames))
        actor_used_frames = 0
        actor_covered: List[TimeInterval] = []
        observations_seen: List[str] = []

        sys_prompt = ACTOR_SYSTEM_PROMPT.format(
            h_start=h_start,
            h_end=h_end,
            max_turns=self.max_turns,
            allocated_frames=allocated,
        )
        usr_prompt = ACTOR_USER_PROMPT.format(
            hypothesis_id=hypothesis.hypothesis_id,
            h_start=h_start,
            h_end=h_end,
            span=span,
            allocated_frames=allocated,
            question=question,
            sub_query=hypothesis.sub_query,
            rationale=hypothesis.rationale or "N/A",
        )
        chat_history: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]

        allow_scan = span >= self.scan_min_span_sec and allocated >= 16
        state.log(
            f"[Actor {hypothesis.hypothesis_id}] Start window=[{h_start:.1f}s-{h_end:.1f}s] "
            f"budget={allocated} frames | sub_query={hypothesis.sub_query}"
        )

        for turn in range(self.max_turns):
            remaining_actor_budget = max(0, allocated - actor_used_frames)
            is_last_turn = turn == self.max_turns - 1 or remaining_actor_budget <= 0

            if is_last_turn:
                allowed_tools = {"submit_report"}
                chat_history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Turn {turn + 1}/{self.max_turns} | Remaining actor budget: {remaining_actor_budget} frames] "
                            "You MUST call `submit_report` now based strictly on observed evidence. "
                            "If evidence is absent or insufficient, set status='NOT_FOUND' and candidate_answer=null."
                        ),
                    }
                )
            else:
                allowed_tools = {"dynamic_focus", "submit_report"}
                if allow_scan:
                    allowed_tools.add("scan_observer")

            tool_schemas = self.dispatcher.get_tool_schemas(allowed_tools)
            resp = call_chat_with_retry(
                self.llm_client,
                model=self.model_name,
                messages=chat_history,
                tools=tool_schemas or None,
                tool_choice="required" if is_last_turn and tool_schemas else None,
            )
            msg = resp.choices[0].message
            msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {
                "role": "assistant",
                "content": getattr(msg, "content", None),
            }
            chat_history.append(msg_dict)

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                parsed_json = extract_json_object(getattr(msg, "content", "") or "")
                if parsed_json and "status" in parsed_json:
                    report = _build_report_from_dict(
                        parsed_json,
                        hypothesis=hypothesis,
                        covered=actor_covered,
                        used_frames=actor_used_frames,
                    )
                    return report, chat_history
                if is_last_turn:
                    break
                chat_history.append(
                    {
                        "role": "user",
                        "content": "Please invoke `dynamic_focus`, `scan_observer`, or `submit_report`.",
                    }
                )
                continue

            for tc in tool_calls:
                parsed = self.dispatcher.parse_tool_call(tc)
                if isinstance(parsed, tuple):
                    parsed_call, tc_json = parsed
                else:
                    parsed_call, tc_json = parsed, tc
                tc_id = (
                    tc_json.get("id")
                    if isinstance(tc_json, dict)
                    else getattr(tc, "id", "call_actor")
                )
                if not parsed_call:
                    chat_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": "unknown",
                            "content": "[Tool error] Could not parse tool call JSON arguments.",
                        }
                    )
                    continue

                tool_name = str(parsed_call.get("name", ""))
                remaining_actor_budget = max(0, allocated - actor_used_frames)
                t0 = time.time()
                err_msg: Optional[str] = None
                frames_used = 0
                tool_cov: List[TimeInterval] = []
                obs_text = ""
                payload: Optional[Dict[str, Any]] = None

                try:
                    state.log(
                        f"[Actor {hypothesis.hypothesis_id} | Turn {turn + 1}/{self.max_turns}] "
                        f"Calling {tool_name}({json.dumps(parsed_call.get('arguments') or {}, ensure_ascii=False)})"
                    )
                    raw_res = self.dispatcher.call_tool(
                        parsed_call,
                        video_path=video_path,
                        subtitles=subtitles,
                        duration_sec=duration_sec,
                        allowed_interval=(h_start, h_end),
                        max_budget_cap=max(1, remaining_actor_budget) if remaining_actor_budget > 0 else 1,
                    )
                    obs_text, payload = _extract_payload_and_text(raw_res)
                except ReportSubmittedException as submitted:
                    t1 = time.time()
                    state.record_usage(
                        phase=phase,
                        frames=0,
                        log_entry={
                            "session_id": session_id,
                            "phase": phase,
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "tool_name": tool_name,
                            "tool_args": parsed_call.get("arguments"),
                            "frames_used": 0,
                            "duration": t1 - t0,
                            "error": None,
                        },
                    )
                    report = _build_report_from_dict(
                        submitted.report,
                        hypothesis=hypothesis,
                        covered=actor_covered,
                        used_frames=actor_used_frames,
                    )
                    state.log(
                        f"[Actor {hypothesis.hypothesis_id}] Report submitted: "
                        f"status={report.status} confidence={report.confidence} "
                        f"used_frames={report.used_frames}/{allocated}"
                    )
                    return report, chat_history
                except Exception as exc:
                    err_msg = str(exc) or repr(exc)
                    obs_text = f"[Tool error] {err_msg}"

                t1 = time.time()
                if isinstance(payload, dict):
                    frames_used = int(
                        (payload.get("used_budget") or {}).get("sampled_frames_count") or 0
                    )
                    for cv_item in payload.get("coverage") or []:
                        if isinstance(cv_item, dict):
                            tool_cov.append(TimeInterval(**cv_item))
                    actor_used_frames += frames_used
                    actor_covered.extend(tool_cov)
                    subject_updates = _extract_subject_updates(payload)
                else:
                    subject_updates = {}

                state.log(
                    f"[Actor {hypothesis.hypothesis_id} | Turn {turn + 1}/{self.max_turns}] "
                    f"Done {tool_name} in {t1 - t0:.1f}s | frames={frames_used} "
                    f"({actor_used_frames}/{allocated})"
                    + (f" | error={err_msg}" if err_msg else "")
                )

                state.record_usage(
                    phase=phase,
                    frames=frames_used,
                    coverage=tool_cov,
                    mark_covered=True,
                    subject_updates=subject_updates,
                    log_entry={
                        "session_id": session_id,
                        "phase": phase,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "tool_name": tool_name,
                        "tool_args": parsed_call.get("arguments"),
                        "interval": [c.model_dump() for c in tool_cov],
                        "frames_used": frames_used,
                        "duration": t1 - t0,
                        "error": err_msg,
                    },
                )
                observations_seen.append(obs_text)
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tool_name,
                        "content": (
                            f"{obs_text}\n\n"
                            f"[Actor Budget Status: used {actor_used_frames}/{allocated} frames]"
                        ),
                    }
                )

            if actor_used_frames >= allocated:
                # Next turn will force submit_report
                continue

        # Fallback if Actor exhausted turns without calling submit_report
        return (
            ActorReport(
                hypothesis_id=hypothesis.hypothesis_id,
                status="PARTIAL" if observations_seen else "NOT_FOUND",
                candidate_answer=None,
                confidence="low",
                evidences=[
                    ActorEvidenceItem(
                        interval=TimeInterval(start_sec=h_start, end_sec=h_end),
                        visual_fact=obs[:600],
                        relevance="partial",
                    )
                    for obs in observations_seen[-2:]
                ],
                uncertainties="Actor reached turn/budget limit before explicit submit_report.",
                covered_intervals=merge_intervals(actor_covered),
                used_frames=actor_used_frames,
            ),
            chat_history,
        )
