"""Orchestrator Agent coordinating Overview -> Parallel Actors -> Stitched Verify / Fallback."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ..data_models import (
    ActorReport,
    CoordinatorDecision,
    Hypothesis,
    OrchestratorPlan,
    TimeInterval,
    VerifySegmentSpec,
)
from ..prompts.multi_agent import (
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_POST_VERIFY_PROMPT,
    ORCHESTRATOR_SYNTHESIS_PROMPT,
)
from ..utils import probe_video_metadata
from ..vcs_dispatcher import VCSDispatcher
from .actor import ActorRunner, _extract_payload_and_text, _extract_subject_updates
from .state import (
    SharedVideoState,
    call_chat_with_retry,
    extract_json_object,
    format_intervals,
    has_explicit_timestamp,
)


def _format_subtitles_for_planner(subtitles: Any, max_chars: int = 8000) -> str:
    if not subtitles:
        return "No subtitles provided."
    lines: List[str] = []
    for row in subtitles:
        if not isinstance(row, dict):
            continue
        s = row.get("start", row.get("start_time", ""))
        e = row.get("end", row.get("end_time", ""))
        txt = row.get("line", row.get("subtitle", row.get("text", ""))) or ""
        txt = str(txt).strip()
        if txt:
            lines.append(f"[{s}-{e}] {txt}")
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n...[subtitles truncated]..."
    return joined or "No subtitles provided."


def _deduplicate_overlapping_hypotheses(
    hypotheses: List[Hypothesis],
    duration_sec: float,
    min_span: float = 5.0,
) -> List[Hypothesis]:
    """Ensure hypothesis windows stay within video bounds and avoid >50% overlap."""
    cleaned: List[Hypothesis] = []
    for idx, h in enumerate(hypotheses[:3]):
        s = max(0.0, min(float(h.interval.start_sec), duration_sec))
        e = max(0.0, min(float(h.interval.end_sec), duration_sec))
        if e - s < min_span:
            continue

        # Trim against previously accepted disjoint intervals when heavily overlapping
        for prev in cleaned:
            ps, pe = prev.interval.start_sec, prev.interval.end_sec
            inter = max(0.0, min(e, pe) - max(s, ps))
            shorter = min(e - s, pe - ps)
            if shorter > 0 and (inter / shorter) > 0.5:
                if s < ps and ps - s >= min_span:
                    e = ps
                elif e > pe and e - pe >= min_span:
                    s = pe
                else:
                    s, e = 0.0, 0.0
                    break
        if e - s < min_span:
            continue

        cleaned.append(
            Hypothesis(
                hypothesis_id=h.hypothesis_id or f"H{idx + 1}",
                interval=TimeInterval(start_sec=round(s, 2), end_sec=round(e, 2)),
                sub_query=h.sub_query,
                rationale=h.rationale,
                allocated_frames=max(1, int(h.allocated_frames)),
            )
        )
    return cleaned


def _normalize_budget_split(
    hypotheses: List[Hypothesis],
    allocatable_budget: int,
) -> List[Hypothesis]:
    """Scale ``allocated_frames`` proportionally so total does not exceed ``allocatable_budget``."""
    if not hypotheses:
        return []
    budget = max(len(hypotheses), int(allocatable_budget))
    total_req = sum(max(1, h.allocated_frames) for h in hypotheses)
    if total_req <= budget:
        return hypotheses

    scaled: List[Hypothesis] = []
    assigned = 0
    for idx, h in enumerate(hypotheses):
        if idx == len(hypotheses) - 1:
            alloc = max(1, budget - assigned)
        else:
            alloc = max(1, int(round(budget * (h.allocated_frames / total_req))))
            alloc = min(alloc, budget - assigned - (len(hypotheses) - idx - 1))
        assigned += alloc
        scaled.append(h.model_copy(update={"allocated_frames": alloc}))
    return scaled


class MultiAgentCoordinator:
    """Coordinates Conditional Overview -> Parallel Actors -> Cross-Verification."""

    def __init__(
        self,
        *,
        dispatcher: VCSDispatcher,
        llm_client: Any,
        orchestrator_model: str,
        actor_model: Optional[str] = None,
        total_frame_budget: int = 192,
        overview_frames: int = 64,
        verify_reserve_frames: int = 32,
        min_actor_frames: int = 16,
        max_actor_turns: int = 3,
        max_replans: int = 1,
        conditional_overview: bool = True,
        max_actor_workers: int = 3,
        verbose: bool = False,
    ) -> None:
        self.dispatcher = dispatcher
        self.llm_client = llm_client
        self.orchestrator_model = orchestrator_model
        self.actor_model = actor_model or orchestrator_model
        self.total_frame_budget = int(total_frame_budget)
        self.overview_frames = int(overview_frames)
        self.verify_reserve_frames = int(verify_reserve_frames)
        self.min_actor_frames = int(min_actor_frames)
        self.max_actor_turns = int(max_actor_turns)
        self.max_replans = int(max_replans)
        self.conditional_overview = bool(conditional_overview)
        self.max_actor_workers = max(1, int(max_actor_workers))
        self.verbose = bool(verbose)

        self.actor_runner = ActorRunner(
            dispatcher=self.dispatcher,
            llm_client=self.llm_client,
            model_name=self.actor_model,
            max_turns=self.max_actor_turns,
        )

    def _run_conditional_overview(
        self,
        *,
        question: str,
        video_path: str,
        subtitles: Any,
        state: SharedVideoState,
        session_id: Any,
        force_overview: Optional[bool] = None,
    ) -> Tuple[str, Optional[str]]:
        if force_overview is False:
            return "Overview disabled by configuration.", "disabled"
        if force_overview is None and self.conditional_overview and has_explicit_timestamp(question):
            return "Overview skipped because the question already contains explicit timestamps.", "explicit_timestamp"
        if state.remaining_budget < self.overview_frames + self.min_actor_frames:
            return "Overview skipped to preserve frame budget for focused inspection.", "insufficient_budget"

        t0 = time.time()
        err_msg: Optional[str] = None
        frames_used = 0
        overview_text = "Overview unavailable."
        try:
            res = self.dispatcher.call_tool(
                {"name": "overview_observer", "arguments": {"num_frames": self.overview_frames}},
                video_path=video_path,
                subtitles=subtitles,
                duration_sec=state.duration_sec,
            )
            obs_text, payload = _extract_payload_and_text(res)
            overview_text = obs_text
            if isinstance(payload, dict):
                frames_used = int((payload.get("used_budget") or {}).get("sampled_frames_count") or 0)
        except Exception as exc:
            err_msg = str(exc) or repr(exc)
            overview_text = f"Overview failed: {err_msg}"
            state.log(f"[Phase 0: Overview] ERROR: {err_msg}")

        t1 = time.time()
        # Coarse whole-video overview does NOT mark segments as covered for fallback.
        state.record_usage(
            phase="overview",
            frames=frames_used,
            coverage=[],
            mark_covered=False,
            log_entry={
                "session_id": session_id,
                "phase": "overview",
                "hypothesis_id": None,
                "tool_name": "overview_observer",
                "tool_args": {"num_frames": self.overview_frames},
                "frames_used": frames_used,
                "duration": t1 - t0,
                "error": err_msg,
            },
        )
        return overview_text, None

    def _plan_hypotheses(
        self,
        *,
        question: str,
        overview_text: str,
        subtitles: Any,
        state: SharedVideoState,
    ) -> Tuple[OrchestratorPlan, List[Dict[str, Any]]]:
        rem = state.remaining_budget
        reserve = min(self.verify_reserve_frames, max(0, rem - self.min_actor_frames))
        allocatable = max(1, rem - reserve)

        prompt = ORCHESTRATOR_PLAN_PROMPT.format(
            duration_sec=state.duration_sec,
            remaining_budget=rem,
            verify_reserve=reserve,
            allocatable_budget=allocatable,
            overview_summary=overview_text,
            subtitles_summary=_format_subtitles_for_planner(subtitles),
            question=question,
            min_actor_frames=min(self.min_actor_frames, allocatable),
        )
        messages = [{"role": "user", "content": prompt}]
        resp = call_chat_with_retry(
            self.llm_client,
            model=self.orchestrator_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw_content = getattr(resp.choices[0].message, "content", "") or ""
        parsed = extract_json_object(raw_content)

        hypotheses: List[Hypothesis] = []
        reasoning = ""
        if isinstance(parsed, dict):
            reasoning = str(parsed.get("reasoning") or "")
            for idx, h_raw in enumerate(parsed.get("hypotheses") or []):
                if not isinstance(h_raw, dict):
                    continue
                itv_raw = h_raw.get("interval") or {}
                try:
                    h = Hypothesis(
                        hypothesis_id=str(h_raw.get("hypothesis_id") or f"H{idx + 1}"),
                        interval=TimeInterval(
                            start_sec=float(itv_raw.get("start_sec", 0.0)),
                            end_sec=float(itv_raw.get("end_sec", state.duration_sec)),
                        ),
                        sub_query=str(h_raw.get("sub_query") or question),
                        rationale=str(h_raw.get("rationale") or ""),
                        allocated_frames=int(h_raw.get("allocated_frames") or max(1, allocatable)),
                    )
                    hypotheses.append(h)
                except Exception:
                    continue

        hypotheses = _deduplicate_overlapping_hypotheses(hypotheses, state.duration_sec)
        if not hypotheses:
            hypotheses = [
                Hypothesis(
                    hypothesis_id="H1",
                    interval=TimeInterval(start_sec=0.0, end_sec=round(state.duration_sec, 2)),
                    sub_query=question,
                    rationale="Fallback full-video hypothesis when plan parsing returned no valid intervals.",
                    allocated_frames=allocatable,
                )
            ]
        hypotheses = _normalize_budget_split(hypotheses, allocatable)
        plan = OrchestratorPlan(reasoning=reasoning or "Normalized plan.", hypotheses=hypotheses)
        return plan, messages + [{"role": "assistant", "content": raw_content}]

    def _run_actors_parallel(
        self,
        *,
        hypotheses: List[Hypothesis],
        question: str,
        video_path: str,
        state: SharedVideoState,
        session_id: Any,
        subtitles: Any,
        phase: str = "actor",
    ) -> Tuple[List[ActorReport], Dict[str, List[Dict[str, Any]]]]:
        reports: List[ActorReport] = []
        actor_histories: Dict[str, List[Dict[str, Any]]] = {}

        if len(hypotheses) == 1:
            rep, hist = self.actor_runner.run(
                hypothesis=hypotheses[0],
                question=question,
                video_path=video_path,
                duration_sec=state.duration_sec,
                state=state,
                session_id=session_id,
                subtitles=subtitles,
                phase=phase,
            )
            reports.append(rep)
            actor_histories[hypotheses[0].hypothesis_id] = hist
        else:
            workers = min(self.max_actor_workers, len(hypotheses))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(
                        self.actor_runner.run,
                        hypothesis=h,
                        question=question,
                        video_path=video_path,
                        duration_sec=state.duration_sec,
                        state=state,
                        session_id=session_id,
                        subtitles=subtitles,
                        phase=phase,
                    ): h.hypothesis_id
                    for h in hypotheses
                }
                for fut, hid in future_map.items():
                    rep, hist = fut.result()
                    reports.append(rep)
                    actor_histories[hid] = hist

        reports.sort(key=lambda r: r.hypothesis_id)
        state.add_reports(reports)
        return reports, actor_histories

    def _synthesize_decision(
        self,
        *,
        question: str,
        state: SharedVideoState,
    ) -> Tuple[CoordinatorDecision, str]:
        uncovered = state.get_uncovered_intervals(min_span=10.0)
        covered = state.get_covered_intervals()
        reports_payload = [r.model_dump() for r in state.actor_reports]

        prompt = ORCHESTRATOR_SYNTHESIS_PROMPT.format(
            question=question,
            duration_sec=state.duration_sec,
            remaining_budget=state.remaining_budget,
            replan_count=state.replan_count,
            max_replans=self.max_replans,
            has_stitched_verified=state.has_stitched_verified,
            covered_intervals=format_intervals(covered),
            uncovered_intervals=format_intervals(uncovered),
            subject_registry=json.dumps(state.subject_registry.subject_registry, ensure_ascii=False),
            actor_reports_json=json.dumps(reports_payload, ensure_ascii=False, indent=2),
        )
        resp = call_chat_with_retry(
            self.llm_client,
            model=self.orchestrator_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_text = getattr(resp.choices[0].message, "content", "") or ""
        parsed = extract_json_object(raw_text) or {}

        action = str(parsed.get("action", "FINISH")).upper()
        if action not in {"FINISH", "STITCHED_VERIFY", "FALLBACK_REPLAN"}:
            action = "FINISH"

        verify_segments: Optional[List[VerifySegmentSpec]] = None
        if action == "STITCHED_VERIFY":
            verify_segments = []
            for seg in parsed.get("verify_segments") or []:
                if not isinstance(seg, dict):
                    continue
                itv = seg.get("interval") if isinstance(seg.get("interval"), dict) else seg
                try:
                    verify_segments.append(
                        VerifySegmentSpec(
                            interval=TimeInterval(
                                start_sec=float(itv.get("start_sec", 0.0)),
                                end_sec=float(itv.get("end_sec", 0.0)),
                            ),
                            fps=float(seg.get("fps") or 1.0),
                        )
                    )
                except Exception:
                    continue

        fallback_intervals: Optional[List[TimeInterval]] = None
        if action == "FALLBACK_REPLAN":
            fallback_intervals = []
            for itv in parsed.get("fallback_intervals") or []:
                if not isinstance(itv, dict):
                    continue
                try:
                    fallback_intervals.append(
                        TimeInterval(
                            start_sec=float(itv.get("start_sec", 0.0)),
                            end_sec=float(itv.get("end_sec", 0.0)),
                        )
                    )
                except Exception:
                    continue

        decision = CoordinatorDecision(
            action=action,  # type: ignore[arg-type]
            reasoning=str(parsed.get("reasoning") or raw_text),
            final_answer=parsed.get("final_answer"),
            verify_segments=verify_segments,
            verify_query=parsed.get("verify_query"),
            fallback_intervals=fallback_intervals,
            fallback_query=parsed.get("fallback_query"),
        )
        return decision, raw_text

    def _run_stitched_verify(
        self,
        *,
        decision: CoordinatorDecision,
        question: str,
        video_path: str,
        subtitles: Any,
        state: SharedVideoState,
        session_id: Any,
    ) -> Tuple[str, str]:
        """Run final ``stitched_observer`` verification and finalize the answer."""
        state.has_stitched_verified = True
        segments_arg = [
            {
                "interval": seg.interval.model_dump(),
                "fps": seg.fps,
            }
            for seg in (decision.verify_segments or [])
        ]
        verify_query = decision.verify_query or question
        verify_cap = max(8, min(state.remaining_budget, 64))

        t0 = time.time()
        err_msg: Optional[str] = None
        frames_used = 0
        stitched_obs_text = "Stitched verification failed."
        try:
            res = self.dispatcher.call_tool(
                {
                    "name": "stitched_observer",
                    "arguments": {
                        "segments": segments_arg,
                        "query": verify_query,
                        "max_total_frames": verify_cap,
                    },
                },
                video_path=video_path,
                subtitles=subtitles,
                duration_sec=state.duration_sec,
                max_budget_cap=verify_cap,
            )
            obs_text, payload = _extract_payload_and_text(res)
            stitched_obs_text = obs_text
            cov: List[TimeInterval] = []
            subject_updates: Dict[str, Any] = {}
            if isinstance(payload, dict):
                frames_used = int((payload.get("used_budget") or {}).get("sampled_frames_count") or 0)
                for cv_item in payload.get("coverage") or []:
                    if isinstance(cv_item, dict):
                        cov.append(TimeInterval(**cv_item))
                subject_updates = _extract_subject_updates(payload)
            t1 = time.time()
            state.record_usage(
                phase="verify",
                frames=frames_used,
                coverage=cov,
                mark_covered=True,
                subject_updates=subject_updates,
                log_entry={
                    "session_id": session_id,
                    "phase": "verify",
                    "hypothesis_id": None,
                    "tool_name": "stitched_observer",
                    "tool_args": {"segments": segments_arg, "query": verify_query},
                    "interval": [c.model_dump() for c in cov],
                    "frames_used": frames_used,
                    "duration": t1 - t0,
                    "error": None,
                },
            )
        except Exception as exc:
            err_msg = str(exc) or repr(exc)
            t1 = time.time()
            state.record_usage(
                phase="verify",
                frames=0,
                log_entry={
                    "session_id": session_id,
                    "phase": "verify",
                    "hypothesis_id": None,
                    "tool_name": "stitched_observer",
                    "tool_args": {"segments": segments_arg, "query": verify_query},
                    "frames_used": 0,
                    "duration": t1 - t0,
                    "error": err_msg,
                },
            )
            stitched_obs_text = f"[Stitched verify error] {err_msg}"

        reports_payload = [r.model_dump() for r in state.actor_reports]
        post_prompt = ORCHESTRATOR_POST_VERIFY_PROMPT.format(
            question=question,
            actor_reports_json=json.dumps(reports_payload, ensure_ascii=False, indent=2),
            stitched_observation=stitched_obs_text,
        )
        resp = call_chat_with_retry(
            self.llm_client,
            model=self.orchestrator_model,
            messages=[{"role": "user", "content": post_prompt}],
            response_format={"type": "json_object"},
        )
        raw_text = getattr(resp.choices[0].message, "content", "") or ""
        parsed = extract_json_object(raw_text) or {}
        final_answer = str(parsed.get("final_answer") or raw_text).strip()
        return final_answer, stitched_obs_text

    def run(
        self,
        *,
        question: str,
        video_path: str,
        duration_sec: Optional[float] = None,
        subtitles: Any = None,
        session_id: Any = None,
        force_overview: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run full Orchestrator - Parallel Actors - Stitched Verify pipeline on one video QA."""
        if duration_sec is None or float(duration_sec) <= 0:
            _, _, probed_dur = probe_video_metadata(video_path)
            duration_sec = probed_dur

        state = SharedVideoState(
            duration_sec=float(duration_sec),
            total_frame_budget=self.total_frame_budget,
            verbose=self.verbose,
        )
        state.log(
            f"[Session {session_id or '0'}] Video duration={state.duration_sec:.1f}s | "
            f"total_frame_budget={self.total_frame_budget}"
        )

        # 1. Conditional Overview
        state.log("[Phase 0: Overview] Evaluating whether overview is needed...")
        overview_text, overview_skip_reason = self._run_conditional_overview(
            question=question,
            video_path=video_path,
            subtitles=subtitles,
            state=state,
            session_id=session_id,
            force_overview=force_overview,
        )
        if overview_skip_reason:
            state.log(f"[Phase 0: Overview] Skipped ({overview_skip_reason})")
        else:
            state.log(
                f"[Phase 0: Overview] Completed | frames={state.overview_frames} | "
                f"remaining_budget={state.remaining_budget}"
            )

        # 2. Orchestrator Phase 1: Hypothesis Generation
        state.log("[Phase 1: Orchestrator Plan] Generating hypotheses...")
        plan, plan_messages = self._plan_hypotheses(
            question=question,
            overview_text=overview_text,
            subtitles=subtitles,
            state=state,
        )
        hyp_desc = ", ".join(
            f"{h.hypothesis_id}[{h.interval.start_sec:.1f}-{h.interval.end_sec:.1f}s, {h.allocated_frames}f]"
            for h in plan.hypotheses
        )
        state.log(f"[Phase 1: Orchestrator Plan] K={len(plan.hypotheses)} -> {hyp_desc}")

        # 3. Parallel Actor Execution
        state.log(f"[Phase 1b: Actors] Launching {len(plan.hypotheses)} Actor(s)...")
        _, actor_histories = self._run_actors_parallel(
            hypotheses=plan.hypotheses,
            question=question,
            video_path=video_path,
            state=state,
            session_id=session_id,
            subtitles=subtitles,
            phase="actor",
        )

        # 4. Orchestrator Phase 2: Synthesis, Fallback Replan, and Stitched Verify
        decision_trace: List[Dict[str, Any]] = []
        stitched_observation: Optional[str] = None
        final_answer: Optional[str] = None

        for _ in range(self.max_replans + 2):
            state.log("[Phase 2: Synthesis] Reviewing Actor reports...")
            decision, raw_decision = self._synthesize_decision(question=question, state=state)
            state.log(f"[Phase 2: Synthesis] Decision={decision.action} | {decision.reasoning}")
            decision_trace.append(
                {
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                    "raw": raw_decision,
                }
            )

            if decision.action == "FALLBACK_REPLAN":
                uncovered = state.get_uncovered_intervals(min_span=10.0)
                if (
                    state.replan_count < self.max_replans
                    and state.remaining_budget >= self.min_actor_frames
                    and uncovered
                ):
                    state.replan_count += 1
                    chosen_intervals = decision.fallback_intervals or uncovered[:2]
                    fb_query = decision.fallback_query or question
                    fb_hypotheses = [
                        Hypothesis(
                            hypothesis_id=f"F{state.replan_count}_{idx + 1}",
                            interval=itv,
                            sub_query=fb_query,
                            rationale="Fallback exploration of previously uncovered intervals.",
                            allocated_frames=max(
                                1,
                                min(state.remaining_budget, state.remaining_budget // min(2, len(chosen_intervals))),
                            ),
                        )
                        for idx, itv in enumerate(chosen_intervals[:2])
                    ]
                    fb_hypotheses = _deduplicate_overlapping_hypotheses(fb_hypotheses, state.duration_sec)
                    if fb_hypotheses:
                        fb_hypotheses = _normalize_budget_split(fb_hypotheses, state.remaining_budget)
                        state.log(
                            f"[Phase 2: Fallback Replan #{state.replan_count}] "
                            f"Running {len(fb_hypotheses)} fallback Actor(s)..."
                        )
                        _, fb_hist = self._run_actors_parallel(
                            hypotheses=fb_hypotheses,
                            question=fb_query,
                            video_path=video_path,
                            state=state,
                            session_id=session_id,
                            subtitles=subtitles,
                            phase="fallback",
                        )
                        actor_histories.update(fb_hist)
                        continue

            if (
                decision.action == "STITCHED_VERIFY"
                and not state.has_stitched_verified
                and state.remaining_budget >= 8
                and decision.verify_segments
            ):
                segs_desc = ", ".join(
                    f"[{s.interval.start_sec:.1f}-{s.interval.end_sec:.1f}s@{s.fps}fps]"
                    for s in decision.verify_segments
                )
                state.log(f"[Phase 2: Stitched Verify] Calling stitched_observer on {segs_desc}...")
                final_answer, stitched_observation = self._run_stitched_verify(
                    decision=decision,
                    question=question,
                    video_path=video_path,
                    subtitles=subtitles,
                    state=state,
                    session_id=session_id,
                )
                state.log(f"[Phase 2: Stitched Verify] Completed | verify_frames={state.verify_frames}")
                break

            # FINISH (or fallback when guardrails prevent extra actions)
            if decision.final_answer is not None and str(decision.final_answer).strip():
                final_answer = str(decision.final_answer).strip()
                break

            # If Orchestrator requested a disallowed action without final_answer, pick best candidate.
            candidates = [
                r.candidate_answer
                for r in state.actor_reports
                if r.status != "NOT_FOUND" and r.candidate_answer
            ]
            final_answer = candidates[0] if candidates else decision.reasoning
            break

        return {
            "final_answer": final_answer or "",
            "overview_skipped_reason": overview_skip_reason,
            "overview_text": overview_text,
            "plan": plan.model_dump(),
            "plan_messages": plan_messages,
            "actor_reports": [r.model_dump() for r in state.actor_reports],
            "actor_histories": actor_histories,
            "decision_trace": decision_trace,
            "stitched_observation": stitched_observation,
            "budget": state.budget_breakdown(),
            "covered_intervals": [iv.model_dump() for iv in state.get_covered_intervals()],
            "replan_count": state.replan_count,
            "has_stitched_verified": state.has_stitched_verified,
            "subject_registry": state.subject_registry.subject_registry,
            "tool_call_logs": state.tool_call_logs,
        }
