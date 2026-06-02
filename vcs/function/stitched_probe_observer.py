from typing import Dict, Any, List, Optional

from .base_observer import BaseVideoObserver
from ..data_models import ToolOutput, TimeInterval, Observation, Provenance
from ..utils import extract_and_resize_frames
from ..prompts import (
    DEFAULT_BIN_TIME_RANGE_PROMPT,
    DEFAULT_GROUP_TIME_RANGE_PROMPT,
    DEFAULT_USER_PROMPT,
    STITCH_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from utils.tool_registry import register_tool


@register_tool("stitched_segment_observer")
class StitchedProbeObserver(BaseVideoObserver):
    def __init__(
        self,
        *,
        base_fps: float = 0.5,
        base_segment_fps: float = 1.0,
        base_frame_num: int = 160,
        base_max_edge_length: int = 720,
        base_prompt_time_range: float = 10.0,
        openai_base_url: Optional[str] = None,
        openai_key: Optional[str] = None,
        mllm_backend: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        group_time_range_prompt: Optional[str] = None,
        bin_time_range_prompt: Optional[str] = None,
        arguments: Optional[dict] = None,
        **base_kwargs
    ):
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.user_prompt = DEFAULT_USER_PROMPT if user_prompt is None else user_prompt

        self.group_time_range_prompt = (
            DEFAULT_GROUP_TIME_RANGE_PROMPT if group_time_range_prompt is None else group_time_range_prompt
        )
        self.bin_time_range_prompt = (
            DEFAULT_BIN_TIME_RANGE_PROMPT if bin_time_range_prompt is None else bin_time_range_prompt
        )

        merged_arguments = {
            "segments": "List of segments, each with interval and optional fps.",
            "query": "Unified question for all selected intervals.",
            "video_path": "Path to the video file.",
            "global_interval": None,
            "max_edge_length": None,
            "max_total_frames": None,
            "num_workers": None,
            "fps": None,
            "subject_registry": None,
        }
        if arguments:
            merged_arguments.update(arguments)
        self.anyof_groups = []

        self.base_sgement_fps = base_segment_fps

        if base_kwargs.get('name', None) is None:
            base_kwargs['name'] = "stitched_probe"
        super().__init__(
            arguments=merged_arguments,
            base_fps=base_fps,
            base_frame_num=base_frame_num,
            base_max_edge_length=base_max_edge_length,
            openai_base_url=openai_base_url,
            openai_key=openai_key,
            mllm_backend=mllm_backend,
            **base_kwargs
        )
        self.base_prompt_time_range = float(base_prompt_time_range)

    def _postprocess_output(self, output: ToolOutput) -> str:
        """Convert tool output into a text observation."""
        def _first_bounds(interval):
            if isinstance(interval, list):
                if not interval:
                    return float("inf"), float("inf")
                iv = interval[0]
            else:
                iv = interval
            s = getattr(iv, "start_sec", None)
            e = getattr(iv, "end_sec",   None)
            if s is None or e is None:
                return float("inf"), float("inf")
            return float(s), float(e)

        obs_sorted = sorted(output.observations, key=lambda o: _first_bounds(o.interval)[0])

        lines = []
        for o in obs_sorted:
            lines.append(f"{o.description}")
        return "\n".join(lines)
    
    def _func_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_edge_length = params.get("max_edge_length", self.base_max_edge_length)
        num_workers: Optional[int] = params.get("num_workers", None)

        segments_filled: List[Dict[str, Any]] = []
        for seg in params["segments"]:
            seg_fps = seg.pop("fps", None)
            if seg_fps is None or float(seg_fps) <= 0:
                seg_fps = float(self.base_sgement_fps)
            segments_filled.append({"interval": seg.get("interval",seg), "fps": float(seg_fps)})

        use_global = ("global_interval" in params and params["global_interval"] is not None)

        max_total_frames = params.get("max_total_frames", None)
        if max_total_frames is None:
            max_total_frames = int(self.base_frame_num)
        max_total_frames = min(max_total_frames, self.base_frame_num)

        prompt_time_range = None
        if self.use_time_prompt:
            user_ptr = params.get('prompt_time_range', None)
            try:
                user_ptr_val = float(user_ptr) if user_ptr is not None else None
            except Exception:
                user_ptr_val = None
            if user_ptr_val is not None and user_ptr_val > 0:
                prompt_time_range = user_ptr_val
            else:
                if use_global:
                    gi = TimeInterval(**params["global_interval"])  # type: ignore[arg-type]
                    total_len = max(0.0, gi.end_sec - gi.start_sec)
                    basis = total_len
                else:
                    lens = []
                    for seg in segments_filled:
                        itv = TimeInterval(**seg["interval"])  # type: ignore[arg-type]
                        lens.append(max(0.0, itv.end_sec - itv.start_sec))
                    basis = min(lens) if lens else 0.0
                prompt_time_range = max(self.base_prompt_time_range, (basis / 5.0) if basis > 0 else self.base_prompt_time_range)

        frames, budget, meta = extract_and_resize_frames(
            video_path=params["video_path"],
            segments=segments_filled,
            max_edge_length=max_edge_length,
            global_interval=params.get("global_interval") if use_global else None,
            base_fps=float(self.base_fps) if use_global else None,
            max_total_frames=max_total_frames,
            num_workers=num_workers,
            prompt_time_range=prompt_time_range
        )

        if use_global:
            coverage = [TimeInterval(**params["global_interval"])]
        else:
            coverage = [TimeInterval(**seg["interval"]) for seg in segments_filled]

        use_subject_registry = bool(params.get("subject_registry", False))

        mllm_results = self.call_mllm_backend(
            frame_groups=[frames],
            query=params["query"],
            group_intervals=coverage if len(coverage)==1 else [coverage],
            meta=meta,
            subject_registry=use_subject_registry,
        )

        observations = [Observation(**res) for res in mllm_results]
        output = ToolOutput(
            observations=observations,
            coverage=coverage,
            used_budget=budget,
            provenance=Provenance(tool_name=self.name, tool_params=params)
        )
        observe_text_message = self._postprocess_output(output)
        subtitles = params.get("subtitles")
        if subtitles is not None:
            subtitles = self._retrieve_subtitles(coverage, subtitles)
            if subtitles is not None:
                observe_text_message += f"[subtitle]{subtitles}"
        return observe_text_message, output.model_dump()
