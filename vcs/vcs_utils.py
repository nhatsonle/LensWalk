import base64
import random
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


class StopException(Exception):
    """Signal that the agent has produced the final answer."""


def encode_video(
    video_path: str,
    for_get_frames_num: int,
    *,
    fps: Union[int, float, None] = None,
    max_size: Tuple[int, int] = (512, 512),
    shuffle_frames: bool = False,
) -> List[str]:
    """Sample, resize, and encode video frames as base64 JPEG strings."""
    if for_get_frames_num == 0:
        return []

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = vr.get_avg_fps()

    if fps is not None and fps > 0:
        interval = video_fps / fps
        idx = np.arange(0, total_frame_num, interval).astype(int)
        if len(idx) > for_get_frames_num:
            idx = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
    else:
        idx = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

    frames = vr.get_batch(idx.tolist()).asnumpy()

    def _to_b64(arr):
        img = Image.fromarray(arr)
        img.thumbnail(max_size)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    base64_frames = [_to_b64(f) for f in frames]

    if shuffle_frames:
        random.shuffle(base64_frames)
    return base64_frames


def strip_image_urls(history):
    """Remove image data URLs from chat history before writing logs."""
    cleaned = []
    for msg in history:
        content = msg["content"]
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        remaining = [
            part for part in content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]
        if remaining:
            cleaned.append({**msg, "content": remaining})

    return cleaned
