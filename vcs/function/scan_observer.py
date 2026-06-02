from typing import Dict, Any, List, Optional

from .base_observer import BaseVideoObserver
from ..data_models import ToolOutput, TimeInterval, Observation, Budget, Provenance
from ..utils import extract_and_resize_frames
from ..prompts import (
    DEFAULT_BIN_TIME_RANGE_PROMPT,
    DEFAULT_GROUP_TIME_RANGE_PROMPT,
    DEFAULT_USER_PROMPT,
    SCAN_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from utils.tool_registry import register_tool


@register_tool("per_segment_scan_observer")
class ScanObserver(BaseVideoObserver):
    """Scan a long interval by splitting it into smaller observation slices."""

    def __init__(
        self,
        *,
        base_fps: float = 1.0,
        base_frame_num: int = 128,
        base_max_edge_length: int = 512,
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
            "global_interval": "Global time interval to scan.",
            "num_slices": None,
            "slice_duration_sec": None,
            "query": "Object, event, or scene to search for.",
            "video_path": "Path to the video file.",
            "fps": None,
            "max_edge_length": None,
            "max_total_frames": None,
            "num_workers": None,
            "subject_registry": None,
        }
        if arguments:
            merged_arguments.update(arguments)
        self.anyof_groups = [
            ("num_slices", "slice_duration_sec"),
        ]
        if base_kwargs.get('name', None) is None:
            base_kwargs['name'] = "partitioned_parallel"
        super().__init__(
            arguments=merged_arguments,
            base_fps=base_fps,
            base_frame_num=base_frame_num,
            base_max_edge_length=base_max_edge_length,
            mllm_backend=mllm_backend,
            **base_kwargs
        )
        self.base_prompt_time_range = float(base_prompt_time_range)

    def _split_intervals(
        self,
        gi: TimeInterval,
        num_slices: Optional[int],
        slice_duration_sec: Optional[float]
    ) -> List[TimeInterval]:
        """Split the global interval into observation slices."""
        start = gi.start_sec
        end = gi.end_sec
        if end <= start:
            return []

        slices: List[TimeInterval] = []
        if num_slices is not None:
            n = int(num_slices)
            if n <= 0:
                raise ValueError("ScanObserver: num_slices must be a positive integer.")
            width = (end - start) / n
            for i in range(n):
                s = start + i * width
                e = start + (i + 1) * width
                slices.append(TimeInterval(start_sec=s, end_sec=e))
        else:
            dur = float(slice_duration_sec)
            if dur <= 0:
                raise ValueError("ScanObserver: slice_duration_sec must be positive.")
            s = start
            while s < end:
                e = min(s + dur, end)
                slices.append(TimeInterval(start_sec=s, end_sec=e))
                s = e
        return slices
    def _postprocess_output(self, output: ToolOutput) -> str:
        """Convert tool output into sorted interval descriptions."""
        obs_sorted = sorted(output.observations, key=lambda o: o.interval.start_sec)
        return "\n".join(
            f"[{o.interval.start_sec:g}, {o.interval.end_sec:g}]: {o.description}"
            for o in obs_sorted
        )
    
    def _func_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        gi = TimeInterval(**params["global_interval"])
        max_edge_length = params.get("max_edge_length", self.base_max_edge_length)
        num_workers: Optional[int] = params.get("num_workers", None)

        num_slices = params.get("num_slices", None)
        slice_duration_sec = params.get("slice_duration_sec", None)
        if (num_slices is None and slice_duration_sec is None) or \
           (num_slices is not None and slice_duration_sec is not None):
            raise ValueError("ScanObserver requires exactly one of num_slices or slice_duration_sec.")

        fps_param = params.get("fps", None)
        if fps_param is None or float(fps_param) <= 0:
            fps_param = float(self.base_fps)
        fps_param = float(fps_param)

        max_total_frames = params.get("max_total_frames", None)
        if max_total_frames is None:
            max_total_frames = int(self.base_frame_num)
        max_total_frames = min(max_total_frames, self.base_frame_num)
        
        intervals = self._split_intervals(gi, num_slices, slice_duration_sec)
        if not intervals:
            return ToolOutput(
                observations=[],
                coverage=[],
                used_budget=Budget(sampled_frames_count=0, frame_indices=[]),
                provenance=Provenance(tool_name=self.name, tool_params=params)
            ).model_dump()

        segments = [{"interval": itv.model_dump(), "fps": fps_param} for itv in intervals]

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
                lens = [max(0.0, itv.end_sec - itv.start_sec) for itv in intervals]
                basis = min(lens) if lens else 0.0
                prompt_time_range = max(self.base_prompt_time_range, (basis / 5.0) if basis > 0 else self.base_prompt_time_range)

        frames_by_segment, budget, meta = extract_and_resize_frames(
            video_path=params["video_path"],
            segments=segments,
            max_edge_length=max_edge_length,
            max_total_frames=max_total_frames,
            num_workers=num_workers,
            keep_segment=True,
            prompt_time_range=prompt_time_range
        )

        use_subject_registry = bool(params.get("subject_registry", False))

        mllm_results = self.call_mllm_backend(
            frame_groups=frames_by_segment,
            query=params["query"],
            group_intervals=intervals,
            meta=meta,
            subject_registry=use_subject_registry,
        )

        observations = [Observation(**res) for res in mllm_results]
        output = ToolOutput(
            observations=observations,
            coverage=intervals,
            used_budget=budget,
            provenance=Provenance(tool_name=self.name, tool_params=params)
        )
        subtitles = params.get("subtitles")
        observe_text_message = self._postprocess_output(output)
        if subtitles is not None:
            subtitles = self._retrieve_subtitles(intervals, subtitles)
            if subtitles is not None:
                observe_text_message += f"[subtitle]{subtitles}"
        return observe_text_message, output.model_dump()
