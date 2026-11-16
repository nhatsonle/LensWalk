"""Dynamic Focus observer unifying coarse skimming and dense visual inspection."""
from __future__ import annotations

from typing import Any, Dict, Optional

from utils.tool_registry import register_tool
from ..data_models import Observation, Provenance, TimeInterval, ToolOutput
from ..prompts import (
    DEFAULT_BIN_TIME_RANGE_PROMPT,
    DEFAULT_GROUP_TIME_RANGE_PROMPT,
    DEFAULT_USER_PROMPT,
    SINGLE_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from ..utils import extract_and_resize_frames
from .base_observer import BaseVideoObserver


_DETAIL_EDGE_MAP = {
    "low": 320,
    "high": 720,
}


@register_tool("dynamic_focus_observer")
class DynamicFocusObserver(BaseVideoObserver):
    """Unified segment observer supporting dynamic FPS and selectable spatial detail."""

    def __init__(
        self,
        *,
        base_fps: float = 1.0,
        base_frame_num: int = 32,
        base_max_edge_length: int = 320,
        low_edge_length: int = 320,
        high_edge_length: int = 720,
        base_prompt_time_range: float = 10.0,
        openai_base_url: Optional[str] = None,
        openai_key: Optional[str] = None,
        mllm_backend: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        group_time_range_prompt: Optional[str] = None,
        bin_time_range_prompt: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        **base_kwargs: Any,
    ) -> None:
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.user_prompt = DEFAULT_USER_PROMPT if user_prompt is None else user_prompt
        self.group_time_range_prompt = (
            DEFAULT_GROUP_TIME_RANGE_PROMPT if group_time_range_prompt is None else group_time_range_prompt
        )
        self.bin_time_range_prompt = (
            DEFAULT_BIN_TIME_RANGE_PROMPT if bin_time_range_prompt is None else bin_time_range_prompt
        )
        self.detail_edge_map = {
            "low": int(low_edge_length or _DETAIL_EDGE_MAP["low"]),
            "high": int(high_edge_length or _DETAIL_EDGE_MAP["high"]),
        }

        merged_arguments = {
            "interval": "Time interval to inspect.",
            "query": "Focused question for the MLLM.",
            "video_path": "Path to the video file.",
            "fps": None,
            "detail_level": None,
            "max_edge_length": None,
            "max_total_frames": None,
            "num_workers": None,
            "subject_registry": None,
        }
        if arguments:
            merged_arguments.update(arguments)

        self.anyof_groups = []
        if base_kwargs.get("name") is None:
            base_kwargs["name"] = "dynamic_focus"
        super().__init__(
            arguments=merged_arguments,
            base_fps=base_fps,
            base_frame_num=base_frame_num,
            base_max_edge_length=base_max_edge_length,
            openai_base_url=openai_base_url,
            openai_key=openai_key,
            mllm_backend=mllm_backend,
            **base_kwargs,
        )
        self.base_prompt_time_range = float(base_prompt_time_range)

    def _resolve_detail(self, params: Dict[str, Any]) -> tuple[str, int]:
        detail_raw = str(params.get("detail_level") or "low").strip().lower()
        detail_level = detail_raw if detail_raw in self.detail_edge_map else "low"
        explicit_edge = params.get("max_edge_length")
        if explicit_edge is not None and int(explicit_edge) > 0:
            return detail_level, int(explicit_edge)
        return detail_level, self.detail_edge_map[detail_level]

    def _func_call(self, params: Dict[str, Any]) -> Any:
        interval = TimeInterval(**params["interval"])
        detail_level, max_edge_length = self._resolve_detail(params)
        num_workers: Optional[int] = params.get("num_workers")

        fps_param = params.get("fps")
        if fps_param is None or float(fps_param) <= 0:
            fps_param = float(self.base_fps)
        fps_param = float(fps_param)

        max_total_frames = params.get("max_total_frames")
        if max_total_frames is None:
            max_total_frames = int(self.base_frame_num)
        max_total_frames = max(1, min(int(max_total_frames), int(self.base_frame_num)))

        segments_to_sample = [{"interval": interval.model_dump(), "fps": fps_param}]

        prompt_time_range = None
        if self.use_time_prompt:
            user_ptr = params.get("prompt_time_range")
            try:
                user_ptr_val = float(user_ptr) if user_ptr is not None else None
            except Exception:
                user_ptr_val = None
            if user_ptr_val is not None and user_ptr_val > 0:
                prompt_time_range = user_ptr_val
            else:
                seg_len = max(0.0, interval.end_sec - interval.start_sec)
                prompt_time_range = max(
                    self.base_prompt_time_range,
                    (seg_len / 5.0) if seg_len > 0 else self.base_prompt_time_range,
                )

        frames, budget, meta = extract_and_resize_frames(
            video_path=params["video_path"],
            segments=segments_to_sample,
            max_edge_length=max_edge_length,
            max_total_frames=max_total_frames,
            num_workers=num_workers,
            prompt_time_range=prompt_time_range,
        )

        use_subject_registry = bool(params.get("subject_registry", False))
        mllm_results = self.call_mllm_backend(
            frame_groups=[frames],
            query=params["query"],
            group_intervals=[interval],
            meta=meta,
            subject_registry=use_subject_registry,
        )

        span = max(1e-6, interval.end_sec - interval.start_sec)
        actual_frames = int(budget.sampled_frames_count)
        actual_fps = actual_frames / span

        observations = []
        for res in mllm_results:
            obs = Observation(**res)
            obs.metadata.update(
                {
                    "requested_fps": fps_param,
                    "actual_fps": round(actual_fps, 4),
                    "detail_level": detail_level,
                    "max_edge_length": max_edge_length,
                }
            )
            observations.append(obs)

        output = ToolOutput(
            observations=observations,
            coverage=[interval],
            used_budget=budget,
            provenance=Provenance(tool_name=self.name, tool_params=params),
        )
        header = (
            f"[Observed: {interval.start_sec:.1f}s-{interval.end_sec:.1f}s | "
            f"Requested FPS: {fps_param:.2f} | Actual FPS: {actual_fps:.2f} "
            f"({actual_frames} frames) | Detail: {detail_level} ({max_edge_length}px)]"
        )
        body = "\n".join(o.description for o in sorted(observations, key=lambda o: o.interval.start_sec))
        observe_text = f"{header}\n{body}"

        subtitles = params.get("subtitles")
        if subtitles is not None:
            sub_txt = self._retrieve_subtitles(interval, subtitles)
            if sub_txt is not None:
                observe_text += f"\n[subtitle]\n{sub_txt}"
        return observe_text, output.model_dump()
