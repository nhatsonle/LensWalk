"""Grid-packed whole-video overview observer inspired by VideoSeek."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

from utils.tool_registry import register_tool
from ..data_models import Budget, Observation, Provenance, TimeInterval, ToolOutput
from ..prompts import OVERVIEW_GRID_PROMPT
from ..utils import sample_uniform_frames_rgb
from .base_observer import BaseVideoObserver


def _parse_overview_json(raw_text: str) -> List[Dict[str, str]]:
    """Parse ``{"frames": [{"timestamp": ..., "description": ...}]}`` with graceful fallback."""
    text = (raw_text or "").strip()
    if not text:
        return []
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("frames"), list):
        items: List[Dict[str, str]] = []
        for entry in parsed["frames"]:
            if not isinstance(entry, dict):
                continue
            ts = str(entry.get("timestamp", "")).strip()
            desc = str(entry.get("description", "")).strip()
            if ts or desc:
                items.append({"timestamp": ts, "description": desc})
        return items
    return []


@register_tool("overview_observer")
class OverviewObserver(BaseVideoObserver):
    """Uniformly sample the whole video into 2x4 grid sheets for a low-cost storyline summary."""

    def __init__(
        self,
        *,
        base_fps: float = 0.1,
        base_frame_num: int = 64,
        base_max_edge_length: int = 256,
        short_side: int = 256,
        openai_base_url: Optional[str] = None,
        openai_key: Optional[str] = None,
        mllm_backend: Optional[Dict[str, Any]] = None,
        arguments: Optional[Dict[str, Any]] = None,
        **base_kwargs: Any,
    ) -> None:
        self.short_side = int(short_side or base_max_edge_length or 256)
        merged_arguments = {
            "video_path": "Path to the video file.",
            "num_frames": None,
            "short_side": None,
        }
        if arguments:
            merged_arguments.update(arguments)
        self.anyof_groups = []
        if base_kwargs.get("name") is None:
            base_kwargs["name"] = "overview_observer"
        super().__init__(
            arguments=merged_arguments,
            base_fps=base_fps,
            base_frame_num=base_frame_num,
            base_max_edge_length=base_max_edge_length,
            openai_base_url=openai_base_url,
            openai_key=openai_key,
            mllm_backend=mllm_backend,
            use_time_prompt=False,
            **base_kwargs,
        )

    def _func_call(self, params: Dict[str, Any]) -> Any:
        video_path = params["video_path"]
        req_frames = int(params.get("num_frames") or self.base_frame_num)
        num_frames = max(8, int(np.ceil(req_frames / 8.0) * 8))
        short_side = int(params.get("short_side") or self.short_side)

        frames_rgb, indices, timestamps, duration = sample_uniform_frames_rgb(
            video_path=video_path,
            num_frames=num_frames,
            short_side=short_side,
        )
        if not frames_rgb:
            raise RuntimeError(f"Failed to sample frames from video: {video_path}")

        actual_n = len(frames_rgb)
        usable = (actual_n // 8) * 8
        if usable == 0:
            # Pad small mock videos (< 8 frames) to 1 grid of 8 by repeating last frame.
            pad_needed = 8 - actual_n
            frames_rgb = frames_rgb + [frames_rgb[-1]] * pad_needed
            timestamps = timestamps + [timestamps[-1]] * pad_needed
            usable = 8
        else:
            frames_rgb = frames_rgb[:usable]
            timestamps = timestamps[:usable]

        arr = np.stack(frames_rgb, axis=0)  # (T, H, W, C) RGB
        num_groups = usable // 8
        _, h, w, c = arr.shape
        grids = (
            arr.reshape(num_groups, 2, 4, h, w, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(num_groups, 2 * h, 4 * w, c)
        )
        ts_grid = np.array(timestamps, dtype=np.float32).reshape(num_groups, 2, 4)
        timestamps_str = ", ".join(f"{t:.1f}s" for t in timestamps)

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"The video covers 0.0s - {duration:.1f}s.\n"
                    "The following 2x4 grid images contain uniformly sampled frames.\n"
                ),
            }
        ]
        for grid_img, ts_mat in zip(grids, ts_grid):
            row_1 = ", ".join(f"{t:.1f}s" for t in ts_mat[0])
            row_2 = ", ".join(f"{t:.1f}s" for t in ts_mat[1])
            matrix_str = f"[[{row_1}],\n[{row_2}]]"
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[{ts_mat[0, 0]:.1f}s - {ts_mat[-1, -1]:.1f}s]:\n"
                        f"Timestamp Matrix:\n{matrix_str}\n"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._img_to_data_url(grid_img, is_rgb=True)},
                }
            )

        content.append(
            {
                "type": "text",
                "text": OVERVIEW_GRID_PROMPT.format(timestamps_str=timestamps_str),
            }
        )

        client, model_name = self._build_api_client()
        messages = [
            {"role": "system", "content": "You are a concise and factual video summarization assistant."},
            {"role": "user", "content": content},
        ]
        resp = self._call_with_backoff(
            self._api_chat_completions,
            client,
            model_name,
            messages,
            response_format={"type": "json_object"},
        )
        raw_text = self._extract_text_from_response(resp) if resp is not None else ""
        parsed_frames = _parse_overview_json(raw_text)
        if parsed_frames:
            summary_text = "\n".join(
                f"{item['timestamp']}: {item['description']}" for item in parsed_frames
            )
        else:
            summary_text = raw_text or "(overview returned empty response)"

        coverage = [TimeInterval(start_sec=0.0, end_sec=round(duration, 2))]
        budget = Budget(sampled_frames_count=len(indices), frame_indices=indices)
        obs = Observation(
            interval=coverage[0],
            description=summary_text,
            metadata={"grid_count": num_groups, "timestamps": timestamps},
        )
        output = ToolOutput(
            observations=[obs],
            coverage=coverage,
            used_budget=budget,
            provenance=Provenance(tool_name=self.name, tool_params=params),
        )
        header = (
            f"[Overview: 0.0s-{duration:.1f}s | Sampled: {len(indices)} frames "
            f"packed into {num_groups} 2x4 grids]"
        )
        return f"{header}\n{summary_text}", output.model_dump()
