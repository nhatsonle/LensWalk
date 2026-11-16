try:
    import cv2
except Exception:  # pragma: no cover - optional when decord+PIL fallback is used
    cv2 = None

try:
    from decord import VideoReader, cpu as decord_cpu
except Exception:  # pragma: no cover
    VideoReader = None
    decord_cpu = None

from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from multiprocessing import cpu_count

from .data_models import Budget, TimeInterval

def _resize_frame(frame: np.ndarray, max_edge: int) -> np.ndarray:
    """Resize a frame while keeping its aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_edge:
        return frame
    if h >= w:
        new_h = max_edge
        new_w = max(1, int(round(w * (max_edge / h))))
    else:
        new_w = max_edge
        new_h = max(1, int(round(h * (max_edge / w))))
    if cv2 is not None:
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.array(Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC))


def probe_video_metadata(video_path: str) -> Tuple[float, int, float]:
    """Return ``(fps, total_frames, duration_sec)`` using OpenCV or Decord."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if video_fps <= 0:
            raise ValueError("Cannot read video FPS.")
        duration = total_frames / video_fps if total_frames > 0 else 0.0
        return video_fps, total_frames, duration
    if VideoReader is not None:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        video_fps = float(vr.get_avg_fps() or 0.0)
        total_frames = len(vr)
        if video_fps <= 0:
            raise ValueError("Cannot read video FPS.")
        return video_fps, total_frames, total_frames / video_fps
    raise RuntimeError("Neither OpenCV (cv2) nor decord is installed to read videos.")


def _read_indices_decord(
    video_path: str,
    indices: List[int],
    *,
    max_edge_length: Optional[int] = None,
    short_side: Optional[int] = None,
) -> List[np.ndarray]:
    """Read RGB frames at ``indices`` with Decord and optionally resize."""
    if not indices:
        return []
    if VideoReader is None:
        raise RuntimeError("decord is required when OpenCV (cv2) is not installed.")
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    total = len(vr)
    if total == 0:
        return []
    clamped = [min(max(0, int(idx)), total - 1) for idx in indices]
    batch = vr.get_batch(clamped).asnumpy()
    out: List[np.ndarray] = []
    for frame in batch:
        if short_side is not None:
            h, w = frame.shape[:2]
            scale = short_side / min(h, w)
            th = max(1, int(round(h * scale)))
            tw = max(1, int(round(w * scale)))
            frame = np.array(Image.fromarray(frame).resize((tw, th), Image.BICUBIC))
        elif max_edge_length is not None:
            frame = _resize_frame(frame, max_edge_length)
        out.append(frame)
    return out


def sample_uniform_frames_rgb(
    video_path: str,
    num_frames: int,
    short_side: int = 256,
) -> Tuple[List[np.ndarray], List[int], List[float], float]:
    """Uniformly sample ``num_frames`` RGB frames across the entire video."""
    video_fps, total_frames, duration = probe_video_metadata(video_path)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    timestamps = [round(idx / video_fps, 1) for idx in indices]

    if VideoReader is not None:
        frames = _read_indices_decord(video_path, indices, short_side=short_side)
        return frames, indices, timestamps, duration

    if cv2 is None:
        raise RuntimeError("Neither decord nor cv2 is available to decode frames.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    frames: List[np.ndarray] = []
    last_pos = -1
    for idx in indices:
        if idx != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = short_side / min(h, w)
        th = max(1, int(round(h * scale)))
        tw = max(1, int(round(w * scale)))
        frame_rgb = np.array(Image.fromarray(frame_rgb).resize((tw, th), Image.BICUBIC))
        frames.append(frame_rgb)
        last_pos = idx
    cap.release()
    return frames, indices, timestamps, duration


def _build_disjoint_intervals_with_fps(
    global_interval: TimeInterval,
    override_segments: List[Dict[str, Any]],
    base_fps: float
) -> List[Tuple[float, float, float]]:
    """Split a global interval into disjoint pieces with per-piece fps."""
    g0, g1 = global_interval.start_sec, global_interval.end_sec

    segs: List[Tuple[float, float, float]] = []
    boundaries = {g0, g1}
    for seg in override_segments:
        itv = seg["interval"]
        s0 = getattr(itv, "start_sec", itv["start_sec"])
        e0 = getattr(itv, "end_sec",   itv["end_sec"])
        s, e = max(g0, s0), min(g1, e0)
        if e > s:
            seg_fps = float(seg["fps"])
            segs.append((s, e, seg_fps))
            boundaries.update((s, e))

    b_sorted = sorted(boundaries)

    pieces: List[Tuple[float, float, float]] = []
    for i in range(len(b_sorted) - 1):
        a, b = b_sorted[i], b_sorted[i + 1]
        if b <= a:
            continue
        fps_list = [fps for s, e, fps in segs if not (e <= a or s >= b)]
        fps_here = max(fps_list) if fps_list else float(base_fps)
        pieces.append((a, b, fps_here))

    return pieces


def _estimate_expected_frames(pieces: List[Tuple[float, float, float]]) -> float:
    """Estimate sampled frame count as duration * fps."""
    total = 0.0
    for (s, e, fps) in pieces:
        if fps > 0 and e > s:
            total += (e - s) * fps
    return max(total, 0.0)


def _generate_indices_for_pieces(
    pieces: List[Tuple[float, float, float]],
    video_fps: float
) -> List[int]:
    """Generate sorted unique frame indices for disjoint pieces."""
    all_indices = []
    for (s, e, fps) in pieces:
        if fps <= 0 or e <= s:
            continue
        step = int(round(video_fps / fps)) if fps > 0 else 10**9
        if step <= 0:
            step = 1
        start_frame = int(math.floor(s * video_fps))
        end_frame = int(math.floor(e * video_fps))
        if end_frame < start_frame:
            end_frame = start_frame
        indices = np.arange(start_frame, end_frame + 1, step, dtype=int)
        if indices.size == 0:
            mid_f = int(round((s + e) / 2.0 * video_fps))
            indices = np.array([mid_f], dtype=int)
        all_indices.append(indices)
    if not all_indices:
        return []
    merged = np.unique(np.concatenate(all_indices))
    return merged.tolist()


def _generate_indices_per_piece(
    pieces: List[Tuple[float, float, float]],
    video_fps: float
) -> List[List[int]]:
    """Generate frame indices for each piece while preserving piece order."""
    per_piece: List[List[int]] = []
    for (s, e, fps) in pieces:
        if fps <= 0 or e <= s:
            per_piece.append([])
            continue
        step = int(round(video_fps / fps)) if fps > 0 else 10**9
        if step <= 0:
            step = 1
        start_frame = int(math.floor(s * video_fps))
        end_frame = int(math.floor(e * video_fps))
        if end_frame < start_frame:
            end_frame = start_frame
        indices = np.arange(start_frame, end_frame + 1, step, dtype=int)
        if indices.size == 0:
            mid_f = int(round((s + e) / 2.0 * video_fps))
            indices = np.array([mid_f], dtype=int)
        per_piece.append(indices.astype(int).tolist())
    return per_piece


def _multithread_read_and_resize(
    video_path: str,
    sorted_indices: List[int],
    max_edge_length: int,
    num_workers: Optional[int] = None
) -> List[np.ndarray]:
    """Read and resize selected frames with a bounded thread pool."""
    if not sorted_indices:
        return []

    if cv2 is None:
        return _read_indices_decord(video_path, sorted_indices, max_edge_length=max_edge_length)

    if num_workers is None or num_workers <= 0:
        try:
            hw_threads = cpu_count()
        except Exception:
            hw_threads = 4
        num_workers = min(4, hw_threads)
    num_workers = max(1, int(num_workers))
    if len(sorted_indices) < num_workers * 8:
        num_workers = max(1, len(sorted_indices) // 8) or 1

    chunks: List[List[int]] = []
    n = len(sorted_indices)
    base = n // num_workers
    rem = n % num_workers
    start = 0
    for i in range(num_workers):
        sz = base + (1 if i < rem else 0)
        if sz == 0:
            continue
        end = start + sz
        chunks.append(sorted_indices[start:end])
        start = end

    def worker(indices: List[int]) -> List[Tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        results: List[Tuple[int, np.ndarray]] = []
        if not cap.isOpened():
            return results
        last_pos = -1
        for idx in indices:
            if idx != last_pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            resized = _resize_frame(frame, max_edge_length)
            results.append((idx, resized))
            last_pos = idx
        cap.release()
        return results

    ordered: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(worker, ch) for ch in chunks]
        for fut in as_completed(futures):
            for idx, img in fut.result():
                ordered[idx] = img

    frames: List[np.ndarray] = []
    for idx in sorted_indices:
        if idx in ordered:
            frames.append(ordered[idx])
    return frames


def _multithread_read_grouped(
    video_path: str,
    indices_per_piece: List[List[int]],
    max_edge_length: int,
    num_workers: Optional[int] = None
) -> List[List[np.ndarray]]:
    """Read and resize grouped frame indices with a bounded thread pool."""
    piece_count = len(indices_per_piece)
    if piece_count == 0:
        return []

    if cv2 is None:
        flat_unique = sorted({idx for group in indices_per_piece for idx in group})
        decoded = _read_indices_decord(video_path, flat_unique, max_edge_length=max_edge_length)
        idx_to_img = {idx: img for idx, img in zip(flat_unique, decoded)}
        return [[idx_to_img[idx] for idx in group if idx in idx_to_img] for group in indices_per_piece]

    if num_workers is None or num_workers <= 0:
        try:
            hw_threads = cpu_count()
        except Exception:
            hw_threads = 4
        num_workers = min(4, hw_threads)
    num_workers = max(1, int(num_workers))

    piece_ids = list(range(piece_count))
    chunks: List[List[int]] = []
    base = piece_count // num_workers
    rem = piece_count % num_workers
    start = 0
    for i in range(num_workers):
        sz = base + (1 if i < rem else 0)
        if sz == 0:
            continue
        end = start + sz
        chunks.append(piece_ids[start:end])
        start = end

    def worker(pid_list: List[int]) -> List[Tuple[int, List[np.ndarray]]]:
        cap = cv2.VideoCapture(video_path)
        results: List[Tuple[int, List[np.ndarray]]] = []
        if not cap.isOpened():
            return results
        for pid in pid_list:
            frames_here: List[np.ndarray] = []
            last_pos = -1
            for idx in indices_per_piece[pid]:
                if idx != last_pos + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frames_here.append(_resize_frame(frame, max_edge_length))
                last_pos = idx
            results.append((pid, frames_here))
        cap.release()
        return results

    by_piece: Dict[int, List[np.ndarray]] = {i: [] for i in range(piece_count)}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(worker, ch) for ch in chunks]
        for fut in as_completed(futures):
            for pid, frames_here in fut.result():
                by_piece[pid] = frames_here

    return [by_piece[i] for i in range(piece_count)]


def extract_and_resize_frames(
    video_path: str,
    segments: List[Dict[str, Any]],
    max_edge_length: int,
    *,
    global_interval: Optional[Dict[str, float]] = None,
    base_fps: Optional[float] = None,
    max_total_frames: Optional[int] = None,
    num_workers: Optional[int] = None,
    keep_segment: bool = False,
    prompt_time_range: Optional[float] = None
) -> Tuple[Any, Budget]:
    """Sample and resize frames for segment, stitched, or scan observers."""
    video_fps, _, _ = probe_video_metadata(video_path)

    pieces: List[Tuple[float, float, float]]
    if global_interval is not None and base_fps is not None:
        gi = TimeInterval(**global_interval)
        ov = []
        for seg in segments:
            itv = TimeInterval(**seg["interval"])
            if itv.end_sec > gi.start_sec and itv.start_sec < gi.end_sec:
                ov.append(seg)
        pieces = _build_disjoint_intervals_with_fps(gi, ov, float(base_fps))
    else:
        recs = []
        boundaries = set()
        for i, seg in enumerate(segments):
            itv = seg["interval"]
            s = float(getattr(itv, "start_sec", itv["start_sec"]))
            e = float(getattr(itv, "end_sec",   itv["end_sec"]))
            fps = float(seg["fps"])
            if e > s and fps > 0:
                recs.append((s, e, fps, i, s))
                boundaries.add(s); boundaries.add(e)

        b_sorted = sorted(boundaries)
        pieces = []
        prev_owner = None
        prev_end = None
        eps = 1e-9

        def pick_owner(active, prev_owner):
            """
            active: list of (s, e, fps, id, s0) covering [a,b)
            1) highest fps; 2) prefer the previous owner on ties; 3) earlier
            start; 4) smaller id.
            """
            max_fps = max(r[2] for r in active)
            cand = [r for r in active if abs(r[2] - max_fps) < 1e-12]
            if prev_owner is not None:
                for (s, e, fps, rid, s0) in cand:
                    if rid == prev_owner:
                        return rid, fps
            cand.sort(key=lambda r: (r[4], r[3]))  # by s0 then id
            s, e, fps, rid, s0 = cand[0]
            return rid, fps

        for i in range(len(b_sorted) - 1):
            a, b = b_sorted[i], b_sorted[i + 1]
            if b <= a + eps:
                continue

            active = [(s, e, fps, rid, s0) for (s, e, fps, rid, s0) in recs
                    if not (e <= a + eps or s >= b - eps)]
            if not active:
                continue

            owner_id, owner_fps = pick_owner(active, prev_owner)

            if pieces and abs(prev_end - a) < eps and pieces[-1][3] == owner_id:
                pieces[-1] = (pieces[-1][0], b, owner_fps, owner_id)
            else:
                pieces.append((a, b, owner_fps, owner_id))

            prev_owner = owner_id
            prev_end = b

        pieces = [(s, e, fps) for (s, e, fps, owner_id) in pieces]

    if not pieces:
        return ([] if not keep_segment else []), Budget(sampled_frames_count=0, frame_indices=[]), None

    if max_total_frames is not None and max_total_frames > 0:
        expected = _estimate_expected_frames(pieces)
        if expected > max_total_frames:
            scale = max_total_frames / expected
            pieces = [(s, e, max(fps * scale, 1e-6)) for (s, e, fps) in pieces]

    def _build_bins(start_sec: float, end_sec: float, bin_size_sec: float) -> List[Tuple[float, float]]:
        if bin_size_sec <= 0:
            return [(start_sec, end_sec)]
        bins = []
        cur = float(start_sec)
        while cur + bin_size_sec <= end_sec:
            bins.append((cur, cur + bin_size_sec))
            cur += bin_size_sec
        if cur < end_sec:
            bins.append((cur, float(end_sec)))
        return bins

    def _which_group(ts: float, bins: List[Tuple[float, float]]) -> int:
        n = len(bins)
        if n == 0:
            return 0
        for i, (bs, be) in enumerate(bins[:-1]):
            if ts >= bs and ts < be:
                return i
        bs, be = bins[-1]
        if ts >= bs and ts <= be:
            return n - 1
        return n - 1

    if keep_segment:
        indices_per_piece = _generate_indices_per_piece(pieces, video_fps)
        frames_grouped = _multithread_read_grouped(
            video_path=video_path,
            indices_per_piece=indices_per_piece,
            max_edge_length=max_edge_length,
            num_workers=num_workers
        )
        flat_indices = sorted(set(i for group in indices_per_piece for i in group))
        budget = Budget(sampled_frames_count=sum(len(g) for g in frames_grouped),
                        frame_indices=flat_indices)

        if prompt_time_range is None:
            return frames_grouped, budget, None

        meta_list: List[Dict[str, Any]] = []
        bin_size = float(prompt_time_range) if prompt_time_range is not None else 0.0
        for (s, e, _), idx_list, frames_here in zip(pieces, indices_per_piece, frames_grouped):
            k = len(frames_here)
            planned = idx_list[:k]
            bins = _build_bins(s, e, bin_size)
            group_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(bins))} if bins else {0: []}
            for local_i, frame_idx in enumerate(planned):
                ts = frame_idx / float(video_fps)
                gid = _which_group(ts, bins) if bins else 0
                group_to_indices[gid].append(local_i)

            meta_list.append({
                'start_sec': float(s),
                'end_sec': float(e),
                'bin_size_sec': float(bin_size),
                'bins': [{'start_sec': bs, 'end_sec': be} for (bs, be) in bins] if bins else [{'start_sec': float(s), 'end_sec': float(e)}],
                'group_to_indices': group_to_indices
            })
        return frames_grouped, budget, meta_list

    else:
        sorted_indices = _generate_indices_for_pieces(pieces, video_fps)
        frames = _multithread_read_and_resize(
            video_path=video_path,
            sorted_indices=sorted_indices,
            max_edge_length=max_edge_length,
            num_workers=num_workers
        )
        budget = Budget(sampled_frames_count=len(frames), frame_indices=sorted_indices)

        if prompt_time_range is None:
            return frames, budget,None

        g_start = min(s for (s, _, _) in pieces)
        g_end = max(e for (_, e, _) in pieces)
        bin_size = float(prompt_time_range)
        bins = _build_bins(g_start, g_end, bin_size)

        k = len(frames)
        planned = sorted_indices[:k]
        group_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(bins))} if bins else {0: []}
        for ret_i, frame_idx in enumerate(planned):
            ts = frame_idx / float(video_fps)
            gid = _which_group(ts, bins) if bins else 0
            group_to_indices[gid].append(ret_i)

        meta = {
            'start_sec': float(g_start),
            'end_sec': float(g_end),
            'bin_size_sec': float(bin_size),
            'bins': [{'start_sec': bs, 'end_sec': be} for (bs, be) in bins] if bins else [{'start_sec': float(g_start), 'end_sec': float(g_end)}],
            'group_to_indices': group_to_indices
        }
        return frames, budget, [meta]
