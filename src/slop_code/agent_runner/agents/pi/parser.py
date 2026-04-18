"""PI trajectory parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slop_code.agent_runner.trajectory import AgentStep
from slop_code.agent_runner.trajectory import ThinkingStep
from slop_code.agent_runner.trajectory import ToolUseStep
from slop_code.agent_runner.trajectory import Trajectory
from slop_code.agent_runner.trajectory import TrajectoryStep
from slop_code.agent_runner.trajectory_parsing import ParseError
from slop_code.agent_runner.trajectory_parsing import TrajectoryParser

_PI_EVENT_TYPES = {
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
}


class PiParser(TrajectoryParser):
    """Parser for PI JSON event trajectories."""

    def can_parse(self, artifact_dir: Path) -> bool:
        jsonl_file = self._find_jsonl_file(artifact_dir)
        if not jsonl_file:
            return False

        try:
            with jsonl_file.open() as handle:
                for _, raw in zip(range(40), handle):
                    line = raw.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("type") in _PI_EVENT_TYPES:
                        return True
        except (json.JSONDecodeError, OSError):
            return False

        return False

    def parse(self, artifact_dir: Path) -> Trajectory:
        jsonl_file = self._find_jsonl_file(artifact_dir)
        if not jsonl_file:
            raise ParseError(f"No JSONL file found in {artifact_dir}")

        steps: list[TrajectoryStep] = []
        metadata: dict[str, Any] = {}
        pending_tools: dict[str, ToolUseStep] = {}

        with jsonl_file.open() as handle:
            for line_num, raw in enumerate(handle, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Invalid JSON at line {line_num}: {e}")

                event_type = event.get("type")
                if event_type == "message_end":
                    self._process_message_end(
                        event=event,
                        steps=steps,
                        metadata=metadata,
                        pending_tools=pending_tools,
                    )
                elif event_type == "tool_execution_start":
                    self._process_tool_start(
                        event=event,
                        steps=steps,
                        pending_tools=pending_tools,
                    )
                elif event_type == "tool_execution_update":
                    self._process_tool_update(
                        event=event,
                        pending_tools=pending_tools,
                    )
                elif event_type == "tool_execution_end":
                    self._process_tool_end(
                        event=event,
                        steps=steps,
                        metadata=metadata,
                        pending_tools=pending_tools,
                    )
                else:
                    continue

        return Trajectory(agent_type="pi", steps=steps, metadata=metadata)

    def _process_message_end(
        self,
        event: dict[str, Any],
        steps: list[TrajectoryStep],
        metadata: dict[str, Any],
        pending_tools: dict[str, ToolUseStep],
    ) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return

        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            steps.append(AgentStep(content=text))
                    elif block_type == "thinking":
                        thinking = block.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            steps.append(ThinkingStep(content=thinking))
                    elif block_type == "toolCall":
                        step = ToolUseStep(
                            type=str(block.get("name") or "unknown"),
                            arguments=(
                                block.get("arguments")
                                if isinstance(block.get("arguments"), dict)
                                else {}
                            ),
                            result=None,
                        )
                        steps.append(step)
                        call_id = block.get("id")
                        if isinstance(call_id, str) and call_id:
                            pending_tools[call_id] = step

            usage = message.get("usage")
            if isinstance(usage, dict):
                metadata["last_usage"] = usage
            provider = message.get("provider")
            model = message.get("model")
            if isinstance(provider, str):
                metadata["provider"] = provider
            if isinstance(model, str):
                metadata["model"] = model
            stop_reason = message.get("stopReason")
            if stop_reason in {"error", "aborted"}:
                errors = metadata.setdefault("errors", [])
                if isinstance(errors, list):
                    errors.append(
                        {
                            "event": "message_end",
                            "stopReason": stop_reason,
                            "errorMessage": message.get("errorMessage"),
                        }
                    )

        elif role == "toolResult":
            tool_call_id = message.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id in pending_tools:
                pending_tools[tool_call_id].result = self._content_to_text(
                    message.get("content")
                )

    def _process_tool_start(
        self,
        event: dict[str, Any],
        steps: list[TrajectoryStep],
        pending_tools: dict[str, ToolUseStep],
    ) -> None:
        args = event.get("args")
        step = ToolUseStep(
            type=str(event.get("toolName") or "unknown"),
            arguments=args if isinstance(args, dict) else {},
            result=None,
        )
        steps.append(step)

        tool_call_id = event.get("toolCallId")
        if isinstance(tool_call_id, str) and tool_call_id:
            pending_tools[tool_call_id] = step

    def _process_tool_update(
        self,
        event: dict[str, Any],
        pending_tools: dict[str, ToolUseStep],
    ) -> None:
        tool_call_id = event.get("toolCallId")
        if not isinstance(tool_call_id, str):
            return
        if tool_call_id not in pending_tools:
            return
        pending_tools[tool_call_id].result = self._content_to_text(
            event.get("partialResult")
        )

    def _process_tool_end(
        self,
        event: dict[str, Any],
        steps: list[TrajectoryStep],
        metadata: dict[str, Any],
        pending_tools: dict[str, ToolUseStep],
    ) -> None:
        tool_call_id = event.get("toolCallId")
        step: ToolUseStep | None = None

        if isinstance(tool_call_id, str):
            step = pending_tools.get(tool_call_id)

        if step is None:
            step = ToolUseStep(
                type=str(event.get("toolName") or "unknown"),
                arguments={},
                result=None,
            )
            steps.append(step)

        step.result = self._content_to_text(event.get("result"))

        if event.get("isError"):
            errors = metadata.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append(
                    {
                        "event": "tool_execution_end",
                        "toolCallId": tool_call_id,
                        "toolName": event.get("toolName"),
                        "result": step.result,
                    }
                )

        if isinstance(tool_call_id, str):
            pending_tools.pop(tool_call_id, None)

    def _content_to_text(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "content" in value:
                return self._content_to_text(value.get("content"))
            return json.dumps(value)
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if value is None:
            return None
        return str(value)

    def _find_jsonl_file(self, artifact_dir: Path) -> Path | None:
        if artifact_dir.is_file() and artifact_dir.suffix == ".jsonl":
            return artifact_dir

        stdout_file = artifact_dir / "stdout.jsonl"
        if stdout_file.exists():
            return stdout_file

        messages_file = artifact_dir / "messages.jsonl"
        if messages_file.exists():
            return messages_file

        jsonl_files = list(artifact_dir.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
        return None

