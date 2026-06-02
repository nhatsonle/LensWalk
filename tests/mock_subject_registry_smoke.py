from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
VIDEO_DIR = FIXTURE_DIR / "videos"
VIDEO_PATH = VIDEO_DIR / "mock_video.mp4"


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[FakeToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls] if self.tool_calls else None,
        }


class FakeChatCompletions:
    planner_calls = 0
    registry_seen_by_planner = False

    def create(self, **kwargs):
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools")

        if tools:
            FakeChatCompletions.planner_calls += 1
            joined = json.dumps(messages, ensure_ascii=False)
            if "[Subject Registry]" in joined and "blue synthetic background" in joined:
                FakeChatCompletions.registry_seen_by_planner = True

            if FakeChatCompletions.planner_calls == 1:
                msg = FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            "call_segment_1",
                            "segment_observer",
                            (
                                '{"interval":{"start_sec":0,"end_sec":2},'
                                '"query":"Identify the dominant background color.",'
                                '"fps":1,"max_total_frames":2}'
                            ),
                        )
                    ]
                )
            else:
                msg = FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            "call_finish_1",
                            "finish",
                            '{"answer":"A"}',
                        )
                    ]
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        system_text = ""
        if messages and isinstance(messages[0], dict):
            system_text = str(messages[0].get("content") or "")

        if "maintaining a `subject_registry`" in system_text:
            content = json.dumps(
                {
                    "updated_subject_registry": {
                        "S1": {
                            "description": ["blue synthetic background"],
                            "appeared_intervals": ["[0, 2]"],
                        }
                    }
                }
            )
        else:
            content = json.dumps(
                {
                    "observation": "The sampled frames show a blue synthetic background.",
                    "subject_registry": {
                        "S1": {
                            "description": ["blue synthetic background"],
                            "appeared_intervals": ["[0, 2]"],
                        }
                    },
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def _make_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 2.0, (96, 64))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create fixture video: {path}")
    for i in range(4):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)
        cv2.putText(frame, str(i), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    _make_video(VIDEO_PATH)
    shutil.rmtree(ROOT / "results", ignore_errors=True)

    import eval.infer_bench_summ_vcs as infer_mod
    import vcs.function.base_observer as base_observer

    FakeChatCompletions.planner_calls = 0
    FakeChatCompletions.registry_seen_by_planner = False
    infer_mod.OpenAI = FakeOpenAI
    base_observer.OpenAI = FakeOpenAI

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("OPENAI_BASE_URL", "http://test.local/v1")
    os.environ.setdefault("TOOL_OPENAI_API_KEY", "test-tool-key")
    os.environ.setdefault("TOOL_OPENAI_BASE_URL", "http://test.local/v1")
    os.environ.setdefault("MLLM_BACKEND", "mock-vision-model")

    sys.argv = [
        "infer_bench_summ_vcs.py",
        "--tool_config",
        "tool_configs/vcs_standard.yaml",
        "--bench_name",
        str(FIXTURE_DIR / "mock_videoqa.jsonl"),
        "--video_root",
        str(VIDEO_DIR) + "/",
        "--model_name_or_path",
        "mock-planner",
        "--max_turns",
        "3",
        "--batch_size",
        "1",
        "--workers",
        "1",
        "--data_start",
        "0",
        "--data_end",
        "1",
        "--custom_name",
        "subject_registry_smoke",
        "--subject_registry",
        "True",
    ]
    infer_mod.main()

    conversation_files = sorted((ROOT / "results").glob("run_lods/**/all_conversations.jsonl"))
    if not conversation_files:
        raise AssertionError("Subject-registry smoke test did not produce conversations.")
    record = json.loads(conversation_files[-1].read_text(encoding="utf-8").strip().splitlines()[-1])
    registry = record.get("subject_registry") or {}
    if "S1" not in registry:
        raise AssertionError(f"Expected S1 in subject registry, got: {registry}")
    if not FakeChatCompletions.registry_seen_by_planner:
        raise AssertionError("Planner did not receive the updated subject registry.")
    print(f"Subject-registry smoke test passed: {conversation_files[-1]}")


if __name__ == "__main__":
    main()
