import json
import os
import queue
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from slop_code.agent_runner import runner
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.models import AgentRunSpec
from slop_code.agent_runner.resume import ResumeInfo
from slop_code.evaluation import ProblemConfig
from slop_code.execution import Session

from slop_code.agent_runner.models import UsageTracker
from slop_code.common.llms import TokenUsage
def _usage(cost: float = 0.0, steps: int = 0):
    return UsageTracker(
        cost=cost,
        steps=steps,
        net_tokens=TokenUsage(),
        current_tokens=TokenUsage()
    )

class StubCheckpoint:
    def __init__(self, name: str, desc: str):
        self.name = name
        self.description = desc

def test_between_checkpoint_hook_execution(tmp_path: Path) -> None:
    agent = Mock(spec=Agent)
    agent.usage = _usage()

    run_spec = Mock(spec=AgentRunSpec)
    run_spec.problem = Mock(spec=ProblemConfig)
    run_spec.problem.name = "test_prob"
    run_spec.problem.checkpoints = {
        "checkpoint_1": StubCheckpoint("checkpoint_1", ""),
        "checkpoint_2": StubCheckpoint("checkpoint_2", ""),
    }
    run_spec.problem.get_checkpoint_spec = Mock(return_value="test_spec")
    run_spec.problem.entry_file = "test.py"
    run_spec.environment = Mock()
    run_spec.template = "test_template"
    run_spec.skip_evaluation = True
    run_spec.pass_policy = Mock()
    run_spec.between_checkpoint_hook = str(tmp_path / "test_hook.sh")
    run_spec.model_name = "test_model"
    run_spec.agent_type = "test_agent"
    run_spec.agent_version = "1.0"

    # Create dummy hook
    hook_path = Path(run_spec.between_checkpoint_hook)
    hook_path.write_text("#!/bin/bash\necho 'HOOK_PREFIX'")
    hook_path.chmod(0o755)

    ar = runner.AgentRunner(
        run_spec=run_spec,
        agent=agent,
        output_path=tmp_path,
        progress_queue=queue.Queue(),
    )
    mock_session = Mock()
    mock_session.workspace = Mock()
    mock_session.workspace.working_dir = tmp_path / "workspace"
    # Overwrite the session property on the instance using a patch or by ignoring the property descriptor.
    # Actually AgentRunner creates session inside run(), so we can just patch it.
    # the property returns self._session
    ar._session = mock_session

    checkpoints = [
        (StubCheckpoint("checkpoint_1", ""), tmp_path / "checkpoint_1"),
        (StubCheckpoint("checkpoint_2", ""), tmp_path / "checkpoint_2"),
    ]

    summary_1 = Mock()
    summary_1.checkpoint_name = "checkpoint_1"
    summary_1.path = tmp_path / "checkpoint_1"
    summary_1.snapshot_dir = tmp_path / "checkpoint_1" / "snapshot"
    summary_1.usage = agent.usage

    summary_2 = Mock()
    summary_2.checkpoint_name = "checkpoint_2"
    summary_2.path = tmp_path / "checkpoint_2"
    summary_2.snapshot_dir = tmp_path / "checkpoint_2" / "snapshot"
    summary_2.usage = agent.usage

    with (
        patch("slop_code.agent_runner.runner.get_checkpoints", return_value=iter(checkpoints)),
        patch.object(runner.AgentRunner, "_run_checkpoint", side_effect=[summary_1, summary_2]) as mock_run_checkpoint,
    ):
        ar._run_problem()
        
    assert mock_run_checkpoint.call_count == 2
    # First call: prompt_prefix = None
    assert mock_run_checkpoint.call_args_list[0].kwargs.get("prompt_prefix") is None
    # Second call: prompt_prefix = 'HOOK_PREFIX'
    assert mock_run_checkpoint.call_args_list[1].kwargs.get("prompt_prefix") == "HOOK_PREFIX"

    # Verify that hook prefix was written to the directory
    prefix_file = tmp_path / "checkpoint_2" / "hook_prefix.txt"
    assert prefix_file.exists()
    assert prefix_file.read_text() == "HOOK_PREFIX"

