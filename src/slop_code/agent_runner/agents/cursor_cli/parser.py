"""Cursor CLI trajectory parser."""

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


class CursorCliParser(TrajectoryParser):
    """Parser for Cursor CLI stream-json trajectory format."""

    def can_parse(self, artifact_dir: Path) -> bool:
        """Check if directory contains Cursor CLI trajectory."""
        jsonl_file = self._find_jsonl_file(artifact_dir)
        if not jsonl_file:
            return False

        try:
            with jsonl_file.open() as f:
                for i, line in enumerate(f):
                    if i > 10:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        event.get("type") == "system"
                        and event.get("subtype") == "init"
                        and "apiKeySource" in event
                        and "permissionMode" in event
                    ):
                        return True
            return False
        except OSError:
            return False

    def parse(self, artifact_dir: Path) -> Trajectory:
        """Parse Cursor CLI trajectory."""
        jsonl_file = self._find_jsonl_file(artifact_dir)
        if not jsonl_file:
            raise ParseError(f"No JSONL file found in {artifact_dir}")

        steps: list[TrajectoryStep] = []
        metadata: dict[str, Any] = {}

        with jsonl_file.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Invalid JSON at line {line_num}: {e}")

                event_type = event.get("type")

                if (
                    event_type == "system"
                    and event.get("subtype") == "init"
                ):
                    metadata = {
                        "model": event.get("model"),
                        "session_id": event.get("session_id"),
                        "cwd": event.get("cwd"),
                        "permission_mode": event.get("permissionMode"),
                    }
                    continue

                if event_type == "thinking":
                    text = event.get("text")
                    if isinstance(text, str) and text.strip():
                        steps.append(ThinkingStep(content=text.strip()))
                    continue

                if event_type == "assistant":
                    message = event.get("message", {})
                    content_blocks = message.get("content", [])
                    if isinstance(content_blocks, list):
                        text_parts = []
                        for block in content_blocks:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") != "text":
                                continue
                            text = block.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                        text = "".join(text_parts).strip()
                        if text:
                            steps.append(AgentStep(content=text))
                    continue

                if (
                    event_type == "tool_call"
                    and event.get("subtype") == "completed"
                ):
                    tool_call = event.get("tool_call", {})
                    if not isinstance(tool_call, dict):
                        continue
                    for tool_name, detail in tool_call.items():
                        if not isinstance(tool_name, str):
                            continue
                        if not isinstance(detail, dict):
                            continue
                        args = detail.get("args")
                        if not isinstance(args, dict):
                            args = {}
                        result = detail.get("result")
                        if result is None:
                            rendered_result = None
                        elif isinstance(result, str):
                            rendered_result = result
                        else:
                            rendered_result = json.dumps(result)
                        steps.append(
                            ToolUseStep(
                                type=tool_name,
                                arguments=args,
                                result=rendered_result,
                            )
                        )

        return Trajectory(
            agent_type="cursor_cli",
            steps=steps,
            metadata=metadata,
        )

    def _find_jsonl_file(self, artifact_dir: Path) -> Path | None:
        """Find the JSONL trajectory file."""
        if artifact_dir.is_file() and artifact_dir.suffix == ".jsonl":
            return artifact_dir

        stdout_file = artifact_dir / "stdout.jsonl"
        if stdout_file.exists():
            return stdout_file

        jsonl_files = list(artifact_dir.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]

        return None

