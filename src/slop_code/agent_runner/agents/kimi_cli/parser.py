"""Kimi CLI trajectory parser and wire helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from slop_code.agent_runner.trajectory import AgentStep
from slop_code.agent_runner.trajectory import ThinkingStep
from slop_code.agent_runner.trajectory import ToolUseStep
from slop_code.agent_runner.trajectory import Trajectory
from slop_code.agent_runner.trajectory import TrajectoryStep
from slop_code.agent_runner.trajectory_parsing import ParseError
from slop_code.agent_runner.trajectory_parsing import TrajectoryParser


@dataclass(slots=True)
class _PendingToolCall:
    call_id: str
    name: str
    arguments_buffer: str


@dataclass(slots=True)
class _WireStep:
    n: int
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    token_usage: dict[str, Any] | None = None
    pending_tool: _PendingToolCall | None = None

    def finalize_pending_tool(self) -> None:
        if self.pending_tool is None:
            return

        raw_arguments = self.pending_tool.arguments_buffer
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = {"raw": raw_arguments} if raw_arguments else {}

        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        self.tool_calls.append(
            {
                "id": self.pending_tool.call_id,
                "name": self.pending_tool.name,
                "arguments": arguments,
            }
        )
        self.pending_tool = None


def _iter_jsonrpc_messages(raw_text: str) -> list[str]:
    messages: list[str] = []
    buffer = ""

    for line in raw_text.splitlines():
        if line.lstrip().startswith('{"jsonrpc"'):
            if buffer:
                messages.append(buffer)
            buffer = line
            continue
        if buffer:
            buffer += "\n" + line

    if buffer:
        messages.append(buffer)

    return messages


def parse_wire_events(raw_text: str) -> list[dict[str, Any]]:
    """Parse Kimi --wire output into event payloads.

    ToolResult payloads can include unescaped tab/newline control characters,
    so we reconstruct full JSON-RPC messages first, then decode with
    ``strict=False``.
    """
    events: list[dict[str, Any]] = []
    for raw in _iter_jsonrpc_messages(raw_text):
        try:
            message = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            continue
        if message.get("method") != "event":
            continue
        params = message.get("params")
        if isinstance(params, dict):
            events.append(params)
    return events


def has_final_result(raw_text: str) -> bool:
    """Return whether wire output contains the final JSON-RPC result."""
    for raw in _iter_jsonrpc_messages(raw_text):
        try:
            message = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            continue
        if message.get("id") in {"1", 1} and "result" in message:
            return True
    return False


def group_events_into_steps(events: list[dict[str, Any]]) -> list[_WireStep]:
    """Group Kimi wire events into step-level structures."""
    steps: list[_WireStep] = []
    current: _WireStep | None = None

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "StepBegin":
            if current is not None:
                current.finalize_pending_tool()
                steps.append(current)
            n = payload.get("n", len(steps) + 1)
            current = _WireStep(
                n=int(n) if isinstance(n, int) else len(steps) + 1
            )
            continue

        if current is None:
            continue

        if event_type == "ContentPart":
            content_type = payload.get("type")
            if content_type == "text":
                text = payload.get("text")
                if isinstance(text, str) and text:
                    current.text_parts.append(text)
            elif content_type == "think":
                think = payload.get("think")
                if isinstance(think, str) and think:
                    current.reasoning_parts.append(think)
            continue

        if event_type == "ToolCall":
            current.finalize_pending_tool()
            function_data = payload.get("function", {})
            if not isinstance(function_data, dict):
                function_data = {}
            arguments = function_data.get("arguments")
            current.pending_tool = _PendingToolCall(
                call_id=str(payload.get("id", "")),
                name=str(function_data.get("name", "")),
                arguments_buffer=str(arguments)
                if arguments is not None
                else "",
            )
            continue

        if event_type == "ToolCallPart":
            if current.pending_tool is not None:
                arguments_part = payload.get("arguments_part")
                if isinstance(arguments_part, str):
                    current.pending_tool.arguments_buffer += arguments_part
            continue

        if event_type == "ToolResult":
            current.finalize_pending_tool()
            tool_call_id = payload.get("tool_call_id")
            return_value = payload.get("return_value")
            if isinstance(tool_call_id, str) and isinstance(return_value, dict):
                current.tool_results[tool_call_id] = return_value
            continue

        if event_type == "StatusUpdate":
            token_usage = payload.get("token_usage")
            if isinstance(token_usage, dict):
                current.token_usage = token_usage
            continue

        if event_type == "TurnEnd":
            current.finalize_pending_tool()
            steps.append(current)
            current = None

    if current is not None:
        current.finalize_pending_tool()
        steps.append(current)

    return steps


def _render_tool_result(result_payload: dict[str, Any] | None) -> str | None:
    if not result_payload:
        return None

    output = result_payload.get("output")
    if output is None:
        return None
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


class KimiCliParser(TrajectoryParser):
    """Parser for Kimi CLI wire artifacts."""

    def can_parse(self, artifact_dir: Path) -> bool:
        wire_file = self._find_wire_file(artifact_dir)
        if wire_file is None:
            return False
        try:
            raw = wire_file.read_text(encoding="utf-8")
        except OSError:
            return False
        return bool(parse_wire_events(raw))

    def parse(self, artifact_dir: Path) -> Trajectory:
        wire_file = self._find_wire_file(artifact_dir)
        if wire_file is None:
            raise ParseError(f"No Kimi wire output found in {artifact_dir}")

        raw = wire_file.read_text(encoding="utf-8")
        events = parse_wire_events(raw)
        wire_steps = group_events_into_steps(events)

        steps: list[TrajectoryStep] = []
        for wire_step in wire_steps:
            thinking = "".join(wire_step.reasoning_parts).strip()
            if thinking:
                steps.append(ThinkingStep(content=thinking))

            text = "".join(wire_step.text_parts).strip()
            if text:
                steps.append(AgentStep(content=text))

            for tool_call in wire_step.tool_calls:
                call_id = str(tool_call.get("id", ""))
                arguments = tool_call.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                result_payload = wire_step.tool_results.get(call_id)
                steps.append(
                    ToolUseStep(
                        type=str(tool_call.get("name", "unknown")),
                        arguments=arguments,
                        result=_render_tool_result(result_payload),
                    )
                )

        return Trajectory(
            agent_type="kimi_cli",
            steps=steps,
            metadata={
                "wire_events": len(events),
                "wire_steps": len(wire_steps),
            },
        )

    def _find_wire_file(self, artifact_dir: Path) -> Path | None:
        if artifact_dir.is_file() and artifact_dir.name == "kimi-cli.txt":
            return artifact_dir

        candidate = artifact_dir / "kimi-cli.txt"
        if candidate.exists():
            return candidate

        return None
