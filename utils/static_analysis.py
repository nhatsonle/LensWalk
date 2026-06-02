"""Summarize LensWalk tool-call logs into CSV tables."""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def _try_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        cur = d
        ok = True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default

def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str) and x.strip() != "":
            return int(float(x))
    except Exception:
        return None
    return None

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() != "":
            return float(x)
    except Exception:
        return None
    return None

def _extract_session_id(rec: Dict[str, Any]) -> str:
    for k in ["session_id", "session", "sid", "trace_id", "conversation_id", "run_id"]:
        v = _try_get(rec, [k])
        if v is not None:
            return str(v)
    return "UNKNOWN_SESSION"

def _extract_example_id(rec: Dict[str, Any], fallback: str) -> str:
    for k in ["example_id", "id", "index", "sample_id", "qid",
              "Question ID", "question_id", "video_id", "video ID", "run_id"]:
        v = _try_get(rec, [k])
        if v is not None:
            return str(v)
    return fallback

def _extract_tool_name(rec: Dict[str, Any]) -> str:
    v = _try_get(rec, ["tool_name"])
    if isinstance(v, str) and v.strip():
        return v
    v = _try_get(rec, ["tool", "name", "observer", "observer_name", "function", "call_type", "type"])
    return str(v or "UNKNOWN")

def _extract_model(rec: Dict[str, Any]) -> Optional[str]:
    m = _try_get(rec, ["model", "mllm_backend.model", "response.model", "result.model_name"])
    return str(m) if m is not None else None

def _extract_tokens(rec: Dict[str, Any]) -> Optional[int]:
    vals = [
        _to_int(_try_get(rec, ["_token_used", "token_used", "token",
                               "usage.total_tokens", "result.usage.total_tokens",
                               "response.usage.total_tokens", "mllm_results.usage.total_tokens"])),
        _to_int(_try_get(rec, ["stats.tokens", "usage.tokens"])),
    ]
    for v in vals:
        if v is not None:
            return v
    return None

def _extract_error(rec: Dict[str, Any]) -> Optional[str]:
    err = _try_get(rec, ["error", "exception", "err", "message", "status_message", "result.error"])
    if isinstance(err, str) and err.strip():
        return err.strip()
    status = _try_get(rec, ["status", "result.status"])
    if isinstance(status, str) and status.lower() in {"error", "failed", "exception"}:
        return status
    return None

def _extract_frames(rec: Dict[str, Any]) -> Optional[int]:
    """Extract frame counts from common scalar or list-shaped fields."""
    candidates = [
        _try_get(rec, ["num_frames", "frame_count", "frames_used", "frames_count", "n_frames", "stats.frames"]),
        _try_get(rec, ["frames"]),
        _try_get(rec, ["group_frames"]),
        _try_get(rec, ["result.frames"]),
    ]
    for c in candidates:
        if isinstance(c, int):
            return c
        if isinstance(c, list):
            if c and isinstance(c[0], list):
                try:
                    return int(sum(len(x) for x in c))
                except Exception:
                    return None
            return int(len(c))
    return None

def _extract_intervals(rec: Dict[str, Any]) -> List[Tuple[Optional[float], Optional[float]]]:
    """Best-effort extraction of [start, end] intervals from a log row."""
    intervals: List[Tuple[Optional[float], Optional[float]]] = []

    s = _to_float(_try_get(rec, ["start", "interval.start", "global_interval.start", "time.start", "meta.start"]))
    e = _to_float(_try_get(rec, ["end", "interval.end", "global_interval.end", "time.end", "meta.end"]))
    if s is not None and e is not None:
        intervals.append((s, e))

    gi = _try_get(rec, ["group_intervals", "intervals", "bins", "bin_intervals", "meta.group_intervals"])
    if isinstance(gi, list):
        for it in gi:
            if isinstance(it, dict):
                ss = _to_float(_try_get(it, ["start"]))
                ee = _to_float(_try_get(it, ["end"]))
                if ss is not None and ee is not None:
                    intervals.append((ss, ee))

    tr = _try_get(rec, ["time_ranges", "time_intervals", "covered_intervals"])
    if isinstance(tr, list):
        for it in tr:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                ss = _to_float(it[0])
                ee = _to_float(it[1])
                if ss is not None and ee is not None:
                    intervals.append((ss, ee))

    return intervals

def _union_duration(intervals: List[Tuple[Optional[float], Optional[float]]]) -> float:
    """Compute the union duration in seconds, skipping invalid intervals."""
    clean = [(s, e) for s, e in intervals if s is not None and e is not None and e >= s]
    if not clean:
        return 0.0
    clean.sort()
    merged = []
    cs, ce = clean[0]
    for s, e in clean[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return float(sum(e - s for s, e in merged))


def summarize_tool_calls(
    input_path: str,
    outdir: str,
    excel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a JSONL tool log, write CSV summaries, and return output paths."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception:
                continue

    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(raw, start=1):
        intervals = _extract_intervals(rec)
        row = {
            "row_id": i,
            "session_id": _extract_session_id(rec),
            "example_id": _extract_example_id(rec, fallback=f"row-{i}"),
            "tool_name": _extract_tool_name(rec),
            "model": _extract_model(rec),
            "tokens": _extract_tokens(rec),
            "frames": _extract_frames(rec),
            "error": _extract_error(rec) or "",
            "interval_count": int(len(intervals)),
            "interval_union_sec": round(_union_duration(intervals), 6),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(outdir, exist_ok=True)
    session_overview_csv = os.path.join(outdir, "session_overview.csv")
    session_tool_calls_csv = os.path.join(outdir, "session_tool_calls_pivot.csv")
    session_tool_frames_csv = os.path.join(outdir, "session_tool_frames_pivot.csv")
    session_tool_tokens_csv = os.path.join(outdir, "session_tool_tokens_pivot.csv")
    tool_frame_dist_csv = os.path.join(outdir, "tool_frame_distribution.csv")
    calls_normalized_csv = os.path.join(outdir, "tool_calls_normalized.csv")

    if df.empty:
        for p in [session_overview_csv, session_tool_calls_csv, session_tool_frames_csv,
                  session_tool_tokens_csv, tool_frame_dist_csv, calls_normalized_csv]:
            pd.DataFrame().to_csv(p, index=False)
        if excel_path:
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                for name in ["session_overview", "session_tool_calls", "session_tool_frames",
                             "session_tool_tokens", "tool_frame_distribution", "tool_calls_normalized"]:
                    pd.DataFrame().to_excel(writer, sheet_name=name, index=False)
        return {
            "rows": 0,
            "sessions": 0,
            "outputs": {
                "session_overview_csv": session_overview_csv,
                "session_tool_calls_pivot_csv": session_tool_calls_csv,
                "session_tool_frames_pivot_csv": session_tool_frames_csv,
                "session_tool_tokens_pivot_csv": session_tool_tokens_csv,
                "tool_frame_distribution_csv": tool_frame_dist_csv,
                "tool_calls_normalized_csv": calls_normalized_csv,
                "excel": excel_path or "",
            }
        }

    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
    df["frames"] = pd.to_numeric(df["frames"], errors="coerce")
    df["interval_union_sec"] = pd.to_numeric(df["interval_union_sec"], errors="coerce")

    wanted_cols = [
        "row_id", "session_id", "example_id", "tool_name", "model",
        "tokens", "frames", "interval_count", "interval_union_sec", "error"
    ]
    df[wanted_cols].to_csv(calls_normalized_csv, index=False)

    df["has_error"] = df["error"].apply(lambda x: isinstance(x, str) and x.strip() != "")

    session_overview = (
        df.groupby("session_id", as_index=False)
          .agg(tool_calls=("row_id", "count"),
               tokens_total=("tokens", "sum"),
               frames_total=("frames", "sum"),
               covered_sec_union_sum=("interval_union_sec", "sum"),
               error_calls=("has_error", "sum"))
    )
    tools_by_session = (
        df.groupby("session_id")["tool_name"]
          .agg(lambda s: sorted(set([t for t in s if isinstance(t, str) and t.strip() != "" and t != "UNKNOWN"])))
          .rename("tools")
    )
    session_overview = session_overview.merge(tools_by_session, on="session_id", how="left")
    session_overview["distinct_tools"] = session_overview["tools"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    session_overview = session_overview.sort_values(["tool_calls", "tokens_total"], ascending=[False, False])

    session_tool_calls = (
        df.pivot_table(index="session_id", columns="tool_name", values="row_id",
                       aggfunc="count", fill_value=0).reset_index()
    )
    session_tool_frames = (
        df.pivot_table(index="session_id", columns="tool_name", values="frames",
                       aggfunc="sum", fill_value=0).reset_index()
    )
    session_tool_tokens = (
        df.pivot_table(index="session_id", columns="tool_name", values="tokens",
                       aggfunc="sum", fill_value=0).reset_index()
    )

    stats_rows = []
    for tool, sub in df.groupby("tool_name"):
        frames = pd.to_numeric(sub["frames"], errors="coerce")
        frames_clean = frames.dropna().astype(float)
        row = {
            "tool_name": tool,
            "count": int(frames.shape[0]),
            "frames_sum": float(frames_clean.sum()) if not frames_clean.empty else 0.0,
            "frames_mean": float(frames_clean.mean()) if not frames_clean.empty else np.nan,
            "frames_std": float(frames_clean.std(ddof=1)) if frames_clean.shape[0] >= 2 else np.nan,
            "frames_min": float(frames_clean.min()) if not frames_clean.empty else np.nan,
            "frames_max": float(frames_clean.max()) if not frames_clean.empty else np.nan,
        }
        q = frames_clean.quantile([0.25, 0.5, 0.75, 0.95]) if not frames_clean.empty else None
        for p, key in [(0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.95, "p95")]:
            row[key] = float(q.loc[p]) if q is not None else np.nan
        stats_rows.append(row)
    tool_frame_stats = pd.DataFrame(stats_rows).sort_values("frames_sum", ascending=False)

    session_overview.to_csv(session_overview_csv, index=False)
    session_tool_calls.to_csv(session_tool_calls_csv, index=False)
    session_tool_frames.to_csv(session_tool_frames_csv, index=False)
    session_tool_tokens.to_csv(session_tool_tokens_csv, index=False)
    tool_frame_stats.to_csv(tool_frame_dist_csv, index=False)

    if excel_path:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            session_overview.to_excel(writer, sheet_name="session_overview", index=False)
            session_tool_calls.to_excel(writer, sheet_name="session_tool_calls", index=False)
            session_tool_frames.to_excel(writer, sheet_name="session_tool_frames", index=False)
            session_tool_tokens.to_excel(writer, sheet_name="session_tool_tokens", index=False)
            tool_frame_stats.to_excel(writer, sheet_name="tool_frame_distribution", index=False)
            df[wanted_cols].to_excel(writer, sheet_name="tool_calls_normalized", index=False)

    return {
        "rows": int(df.shape[0]),
        "sessions": int(session_overview.shape[0]),
        "outputs": {
            "session_overview_csv": session_overview_csv,
            "session_tool_calls_pivot_csv": session_tool_calls_csv,
            "session_tool_frames_pivot_csv": session_tool_frames_csv,
            "session_tool_tokens_pivot_csv": session_tool_tokens_csv,
            "tool_frame_distribution_csv": tool_frame_dist_csv,
            "tool_calls_normalized_csv": calls_normalized_csv,
            "excel": excel_path or "",
        }
    }
